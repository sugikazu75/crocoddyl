///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModelRollingTpl<Scalar>::ContactModelRollingTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar radius, const SE3& pref, const std::size_t nu,
    const Vector2s& gains, const Vector3s& axis)
    : Base(state, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 5, nu),
      radius_(radius),
      pref_(pref),
      axis_(axis.normalized()),
      gains_(gains),
      sRo_(pref.rotation().transpose()) {
  id_ = id;
  if (radius < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "radius has to be a positive value");
  }
}

template <typename Scalar>
ContactModelRollingTpl<Scalar>::ContactModelRollingTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Scalar radius, const SE3& pref, const Vector2s& gains,
    const Vector3s& axis)
    : Base(state, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 5),
      radius_(radius),
      pref_(pref),
      axis_(axis.normalized()),
      gains_(gains),
      sRo_(pref.rotation().transpose()) {
  id_ = id;
  if (radius < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "radius has to be a positive value");
  }
}

template <typename Scalar>
void ContactModelRollingTpl<Scalar>::calc(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::ModelTpl<Scalar>& model = *state_->get_pinocchio().get();
  pinocchio::updateFramePlacement(model, *d->pinocchio, id_);
  pinocchio::getFrameJacobian(model, *d->pinocchio, id_, pinocchio::LOCAL,
                              d->fJf);
  d->v = pinocchio::getFrameVelocity(model, *d->pinocchio, id_);
  d->a0_local = pinocchio::getFrameClassicalAcceleration(model, *d->pinocchio,
                                                         id_, pinocchio::LOCAL);

  // Placement of the reference frame w.r.t. the contact frame C, whose origin
  // lies at the contact point and whose axes are those of the support. Its
  // translation is the constant (0, 0, radius) set in the data constructor.
  const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
  d->Rc.noalias() = sRo_ * oRf;
  d->cMf.rotation(d->Rc);

  // Contact Jacobian: the material point at the contact location shifted by
  // cMf, expressed in the support axes.
  d->Jc_full.noalias() = d->cMf.toActionMatrix() * d->fJf;

  // Contact drift, i.e. d/dt(Jc*v). The offset between the frame origin and the
  // contact point is constant in the world frame -- it is -r*n, with n the
  // support normal -- so the transfer only brings the alpha x k term, which is
  // already carried by cMf. There is no centripetal contribution: the contact
  // point is not a fixed material point of the sole.
  d->a0_full = d->cMf.act(d->a0_local);
  d->omega_c.noalias() = d->Rc * d->v.angular();

  if (gains_[0] != Scalar(0.)) {
    // The position error is only holonomic along h and n, and in pitch and yaw;
    // the foot legitimately drifts along b as it rolls.
    d->dp = d->pinocchio->oMf[id_].translation() -
            radius_ * pref_.rotation().col(2) - pref_.translation();
    d->dp_c.noalias() = sRo_ * d->dp;
    d->axis_c.noalias() = d->Rc * axis_;
    d->a0_full.linear()[0] += gains_[0] * d->dp_c[0];
    d->a0_full.linear()[2] += gains_[0] * d->dp_c[2];
    // The orientation error is the rotation vector h x axis_c, whose h
    // component vanishes identically: rolling is left free by construction.
    d->a0_full.angular()[1] -= gains_[0] * d->axis_c[2];
    d->a0_full.angular()[2] += gains_[0] * d->axis_c[1];
  }
  if (gains_[1] != Scalar(0.)) {
    d->v_c = d->cMf.act(d->v);
    d->a0_full += gains_[1] * d->v_c;
  }

  // Keep the five constrained rows: the three linear ones plus pitch and yaw.
  d->Jc.template topRows<3>() = d->Jc_full.template topRows<3>();
  d->Jc.row(3) = d->Jc_full.row(4);
  d->Jc.row(4) = d->Jc_full.row(5);
  d->a0.template head<3>() = d->a0_full.linear();
  d->a0[3] = d->a0_full.angular()[1];
  d->a0[4] = d->a0_full.angular()[2];
}

template <typename Scalar>
void ContactModelRollingTpl<Scalar>::calcDiff(
    const std::shared_ptr<ContactDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::ModelTpl<Scalar>& model = *state_->get_pinocchio().get();
  const pinocchio::JointIndex joint = model.frames[d->frame].parentJoint;
  pinocchio::getJointAccelerationDerivatives(
      model, *d->pinocchio, joint, pinocchio::LOCAL, d->v_partial_dq,
      d->a_partial_dq, d->a_partial_dv, d->a_partial_da);
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->v.linear(), d->vv_skew);
  pinocchio::skew(d->v.angular(), d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq;
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq;
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv;

  // Derivatives of the LOCAL classical acceleration of the frame origin.
  d->da0_local_dx.leftCols(nv).template topRows<3>() =
      d->fXjda_dq.template topRows<3>();
  d->da0_local_dx.leftCols(nv).template topRows<3>().noalias() +=
      d->vw_skew * d->fXjdv_dq.template topRows<3>();
  d->da0_local_dx.leftCols(nv).template topRows<3>().noalias() -=
      d->vv_skew * d->fXjdv_dq.template bottomRows<3>();
  d->da0_local_dx.leftCols(nv).template bottomRows<3>() =
      d->fXjda_dq.template bottomRows<3>();
  d->da0_local_dx.rightCols(nv).template topRows<3>() =
      d->fXjda_dv.template topRows<3>();
  d->da0_local_dx.rightCols(nv).template topRows<3>().noalias() +=
      d->vw_skew * d->fJf.template topRows<3>();
  d->da0_local_dx.rightCols(nv).template topRows<3>().noalias() -=
      d->vv_skew * d->fJf.template bottomRows<3>();
  d->da0_local_dx.rightCols(nv).template bottomRows<3>() =
      d->fXjda_dv.template bottomRows<3>();

  // Recalculate the constrained acceleration after imposing the contact
  // constraints. This is necessary for the forward-dynamics case.
  d->a0_local = pinocchio::getFrameClassicalAcceleration(model, *d->pinocchio,
                                                         id_, pinocchio::LOCAL);

  d->RcJv.noalias() = d->Rc * d->fJf.template topRows<3>();
  d->RcJw.noalias() = d->Rc * d->fJf.template bottomRows<3>();

  // Rotate the drift into the support axes. The rotation itself depends on q,
  // which contributes the -skew(.) * RcJw terms.
  d->mr_lin.noalias() = d->Rc * d->a0_local.linear();
  d->mr_ang.noalias() = d->Rc * d->a0_local.angular();
  pinocchio::skew(d->mr_lin, d->mr_lin_skew);
  pinocchio::skew(d->mr_ang, d->mr_ang_skew);
  d->dlin_dx.noalias() = d->Rc * d->da0_local_dx.template topRows<3>();
  d->dang_dx.noalias() = d->Rc * d->da0_local_dx.template bottomRows<3>();
  d->dlin_dx.leftCols(nv).noalias() -= d->mr_lin_skew * d->RcJw;
  d->dang_dx.leftCols(nv).noalias() -= d->mr_ang_skew * d->RcJw;

  if (gains_[1] != Scalar(0.)) {
    // The damping term is the contact-point velocity, so it goes through the
    // same rotation and offset transfer as the drift.
    // Derivative of the angular velocity expressed in the support axes.
    pinocchio::skew(d->omega_c, d->omega_c_skew);
    d->domega_c_dx.leftCols(nv).noalias() =
        d->Rc * d->fXjdv_dq.template bottomRows<3>();
    d->domega_c_dx.leftCols(nv).noalias() -= d->omega_c_skew * d->RcJw;
    d->domega_c_dx.rightCols(nv) = d->RcJw;
    d->vr_lin.noalias() = d->Rc * d->v.linear();
    pinocchio::skew(d->vr_lin, d->vr_lin_skew);
    d->dlin_dx.leftCols(nv).noalias() +=
        gains_[1] * d->Rc * d->fXjdv_dq.template topRows<3>();
    d->dlin_dx.leftCols(nv).noalias() -= gains_[1] * d->vr_lin_skew * d->RcJw;
    d->dlin_dx.rightCols(nv).noalias() += gains_[1] * d->RcJv;
    d->dang_dx.noalias() += gains_[1] * d->domega_c_dx;
  }

  // Transfer the linear rows from the frame origin to the contact point. Only
  // the terms above are affected, since the ones below are already expressed at
  // the contact point.
  d->dlin_dx.noalias() += d->pT_skew * d->dang_dx;

  if (gains_[0] != Scalar(0.)) {
    d->dlin_dx.leftCols(nv).row(0).noalias() += gains_[0] * d->RcJv.row(0);
    d->dlin_dx.leftCols(nv).row(2).noalias() += gains_[0] * d->RcJv.row(2);
    pinocchio::skew(d->axis_c, d->axis_c_skew);
    d->daxis_c_dq.noalias() = -d->axis_c_skew * d->RcJw;
    d->dang_dx.leftCols(nv).row(1).noalias() -=
        gains_[0] * d->daxis_c_dq.row(2);
    d->dang_dx.leftCols(nv).row(2).noalias() +=
        gains_[0] * d->daxis_c_dq.row(1);
  }

  // Keep the five constrained rows.
  d->da0_dx.template topRows<3>() = d->dlin_dx;
  d->da0_dx.row(3) = d->dang_dx.row(1);
  d->da0_dx.row(4) = d->dang_dx.row(2);
}

template <typename Scalar>
void ContactModelRollingTpl<Scalar>::updateForce(
    const std::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) {
  if (force.size() != 5) {
    throw_pretty(
        "Invalid argument: " << "lambda has wrong dimension (it should be 5)");
  }
  Data* d = static_cast<Data*>(data.get());
  // A line contact carries no torque about the rolling axis h.
  d->f.linear() = force.template head<3>();
  d->f.angular() << Scalar(0.), force[3], force[4];
  d->f_local = d->cMf.actInv(d->f);
  d->fext = d->jMf.act(d->f_local);
  // Both components of f_local are oRf^T times a wrench held constant in the
  // support axes, so their q-derivative has the same form as in the 6d contact.
  pinocchio::skew(d->f_local.linear(), d->fv_skew);
  pinocchio::skew(d->f_local.angular(), d->fw_skew);
  d->fJf_df.template topRows<3>().noalias() =
      d->fv_skew * d->fJf.template bottomRows<3>();
  d->fJf_df.template bottomRows<3>().noalias() =
      d->fw_skew * d->fJf.template bottomRows<3>();
  d->dtau_dq.noalias() = -d->fJf.transpose() * d->fJf_df;
}

template <typename Scalar>
std::shared_ptr<ContactDataAbstractTpl<Scalar> >
ContactModelRollingTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ContactModelRollingTpl<NewScalar> ContactModelRollingTpl<Scalar>::cast() const {
  typedef ContactModelRollingTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()), id_,
      scalar_cast<NewScalar>(radius_), pref_.template cast<NewScalar>(), nu_,
      gains_.template cast<NewScalar>(), axis_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void ContactModelRollingTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModelRolling {frame="
     << state_->get_pinocchio()->frames[id_].name << ", radius=" << radius_
     << "}";
}

template <typename Scalar>
Scalar ContactModelRollingTpl<Scalar>::get_radius() const {
  return radius_;
}

template <typename Scalar>
const pinocchio::SE3Tpl<Scalar>& ContactModelRollingTpl<Scalar>::get_reference()
    const {
  return pref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s&
ContactModelRollingTpl<Scalar>::get_axis() const {
  return axis_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s&
ContactModelRollingTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModelRollingTpl<Scalar>::set_radius(const Scalar radius) {
  if (radius < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "radius has to be a positive value");
  }
  radius_ = radius;
}

template <typename Scalar>
void ContactModelRollingTpl<Scalar>::set_reference(const SE3& reference) {
  pref_ = reference;
  sRo_ = reference.rotation().transpose();
}

template <typename Scalar>
void ContactModelRollingTpl<Scalar>::set_axis(const Vector3s& axis) {
  axis_ = axis.normalized();
}

}  // namespace crocoddyl
