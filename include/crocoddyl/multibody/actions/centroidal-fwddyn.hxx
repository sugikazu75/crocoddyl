///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::
    DifferentialActionModelCentroidalFwdDynamicsTpl(
        std::shared_ptr<StateCentroidal> state, const std::size_t nc,
        std::shared_ptr<CostModelSum> costs,
        std::shared_ptr<ConstraintModelManager> constraints)
    : Base(state, 6 * nc, costs->get_nr(),
           constraints != nullptr ? constraints->get_ng() : 0,
           constraints != nullptr ? constraints->get_nh() : 0,
           constraints != nullptr ? constraints->get_ng_T() : 0,
           constraints != nullptr ? constraints->get_nh_T() : 0),
      state_centroidal_(state),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(state->get_pinocchio().get()),
      nc_(nc),
      mass_(state->get_mass()),
      inertia_(state->get_inertia()),
      inertia_inv_(state->get_inertia().inverse()),
      gravity_(Vector3s(Scalar(0.), Scalar(0.), Scalar(-9.81))),
      contact_placements_(nc, SE3::Identity()),
      contact_status_(nc, true),
      contact_references_(nc, pinocchio::LOCAL_WORLD_ALIGNED) {
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Costs doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  if (constraints_ != nullptr && constraints_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Constraints doesn't have the same control dimension "
                    "(it should be " +
                        std::to_string(nu_) + ")");
  }
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      vw = x.tail(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, 3> c =
      x.template head<3>();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, 3> v =
      x.template segment<3>(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, 3> w =
      x.template segment<3>(nq + 3);

  pinocchio::forwardKinematics(*pinocchio_, d->pinocchio, q, vw);
  pinocchio::updateFramePlacements(*pinocchio_, d->pinocchio);

  const Eigen::Quaternion<Scalar> quat(q[6], q[3], q[4], q[5]);
  d->R = quat.toRotationMatrix();
  d->Fc.setZero();
  d->Tw.setZero();
  for (std::size_t i = 0; i < nc_; ++i) {
    if (!contact_status_[i]) {
      continue;
    }
    if (contact_references_[i] == pinocchio::LOCAL) {
      d->S = contact_placements_[i].rotation();
    } else {
      d->S.setIdentity();
    }
    d->f_world.noalias() = d->S * u.template segment<3>(6 * i);
    d->tau_world.noalias() = d->S * u.template segment<3>(6 * i + 3);
    d->r_contact = contact_placements_[i].translation() - c;
    d->Fc += d->f_world;
    d->Tw += d->tau_world + d->r_contact.cross(d->f_world);
  }

  d->Fb.noalias() = d->R.transpose() * (d->Fc + mass_ * gravity_);
  d->Tb.noalias() = d->R.transpose() * d->Tw;
  d->xout.head(3).noalias() = d->Fb / mass_ - w.cross(v);
  d->Iw.noalias() = inertia_ * w;
  d->xout.tail(3).noalias() = inertia_inv_ * (d->Tb - w.cross(d->Iw));

  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      vw = x.tail(nv);

  pinocchio::forwardKinematics(*pinocchio_, d->pinocchio, q, vw);
  pinocchio::updateFramePlacements(*pinocchio_, d->pinocchio);
  d->xout.setZero();

  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d, false);
    constraints_->calc(d->constraints, x);
  }
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      vw = x.tail(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, 3> c =
      x.template head<3>();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, 3> v =
      x.template segment<3>(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, 3> w =
      x.template segment<3>(nq + 3);

  const Eigen::Quaternion<Scalar> quat(q[6], q[3], q[4], q[5]);
  d->R = quat.toRotationMatrix();
  d->Fc.setZero();
  d->Tw.setZero();
  for (std::size_t i = 0; i < nc_; ++i) {
    if (!contact_status_[i]) {
      continue;
    }
    if (contact_references_[i] == pinocchio::LOCAL) {
      d->S = contact_placements_[i].rotation();
    } else {
      d->S.setIdentity();
    }
    d->f_world.noalias() = d->S * u.template segment<3>(6 * i);
    d->tau_world.noalias() = d->S * u.template segment<3>(6 * i + 3);
    d->r_contact = contact_placements_[i].translation() - c;
    d->Fc += d->f_world;
    d->Tw += d->tau_world + d->r_contact.cross(d->f_world);
  }
  d->Fb.noalias() = d->R.transpose() * (d->Fc + mass_ * gravity_);
  d->Tb.noalias() = d->R.transpose() * d->Tw;

  d->Fx.setZero();
  d->Fu.setZero();
  pinocchio::skew(d->Fb, d->skew_tmp);
  d->Fx.template block<3, 3>(0, 3) = d->skew_tmp / mass_;
  pinocchio::skew(w, d->skew_tmp);
  d->Fx.template block<3, 3>(0, 6) = -d->skew_tmp;
  pinocchio::skew(v, d->skew_tmp);
  d->Fx.template block<3, 3>(0, 9) = d->skew_tmp;

  // c <- c + R dp only enters through (p_i - c) x f_i, giving
  // dT_b/ddp = R^T [F_c]_x R = [R^T F_c]_x. Gravity contributes to
  // F_b, but not to this torque derivative.
  d->Fcb.noalias() = d->R.transpose() * d->Fc;
  pinocchio::skew(d->Fcb, d->skew_tmp);
  d->Fx.template block<3, 3>(3, 0).noalias() = inertia_inv_ * d->skew_tmp;
  // R <- R exp(dth) gives d(R^T a)/ddth = [R^T a]_x for world-frame a.
  pinocchio::skew(d->Tb, d->skew_tmp);
  d->Fx.template block<3, 3>(3, 3).noalias() = inertia_inv_ * d->skew_tmp;
  d->Iw.noalias() = inertia_ * w;
  pinocchio::skew(d->Iw, d->skew_tmp);
  pinocchio::skew(w, d->skew_tmp2);
  d->Fx.template block<3, 3>(3, 9).noalias() =
      inertia_inv_ * (d->skew_tmp - d->skew_tmp2 * inertia_);

  for (std::size_t i = 0; i < nc_; ++i) {
    if (!contact_status_[i]) {
      continue;
    }
    if (contact_references_[i] == pinocchio::LOCAL) {
      d->S = contact_placements_[i].rotation();
    } else {
      d->S.setIdentity();
    }
    d->r_contact = contact_placements_[i].translation() - c;
    d->Fu.template block<3, 3>(0, 6 * i).noalias() =
        (d->R.transpose() * d->S) / mass_;
    pinocchio::skew(d->r_contact, d->skew_tmp);
    d->Fu.template block<3, 3>(3, 6 * i).noalias() =
        inertia_inv_ * d->R.transpose() * d->skew_tmp * d->S;
    d->Fu.template block<3, 3>(3, 6 * i + 3).noalias() =
        inertia_inv_ * d->R.transpose() * d->S;
  }

  costs_->calcDiff(d->costs, x, u);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  d->Fx.setZero();
  d->Fu.setZero();
  costs_->calcDiff(d->costs, x);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x);
  }
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelCentroidalFwdDynamicsTpl<NewScalar>
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::cast() const {
  typedef DifferentialActionModelCentroidalFwdDynamicsTpl<NewScalar> ReturnType;
  typedef StateCentroidalTpl<NewScalar> StateType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;

  ReturnType ret(
      std::make_shared<StateType>(
          state_centroidal_->template cast<NewScalar>()),
      nc_, std::make_shared<CostType>(costs_->template cast<NewScalar>()),
      constraints_ != nullptr ? std::make_shared<ConstraintType>(
                                    constraints_->template cast<NewScalar>())
                              : std::shared_ptr<ConstraintType>());
  std::vector<pinocchio::SE3Tpl<NewScalar> > placements;
  placements.reserve(nc_);
  for (std::size_t i = 0; i < nc_; ++i) {
    placements.push_back(contact_placements_[i].template cast<NewScalar>());
  }
  ret.set_contact_placements(placements);
  ret.set_contact_status(contact_status_);
  ret.set_contact_reference(contact_references_);
  ret.set_gravity(gravity_.template cast<NewScalar>());
  ret.set_u_lb(Base::u_lb_.template cast<NewScalar>());
  ret.set_u_ub(Base::u_ub_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
bool DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::quasiStatic(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
    const std::size_t, const Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, 3> c =
      x.template head<3>();
  u.setZero();
  d->quasi_A.setZero();
  d->quasi_b.head(3) = -mass_ * gravity_;
  d->quasi_b.tail(3).setZero();

  bool has_active_contact = false;
  for (std::size_t i = 0; i < nc_; ++i) {
    if (!contact_status_[i]) {
      continue;
    }
    has_active_contact = true;
    if (contact_references_[i] == pinocchio::LOCAL) {
      d->S = contact_placements_[i].rotation();
    } else {
      d->S.setIdentity();
    }
    d->r_contact = contact_placements_[i].translation() - c;
    d->quasi_A.template block<3, 3>(0, 6 * i) = d->S;
    pinocchio::skew(d->r_contact, d->skew_tmp);
    d->quasi_A.template block<3, 3>(3, 6 * i).noalias() = d->skew_tmp * d->S;
    d->quasi_A.template block<3, 3>(3, 6 * i + 3) = d->S;
  }
  if (!has_active_contact) {
    return;
  }
  u = d->quasi_A.completeOrthogonalDecomposition().solve(d->quasi_b);
}

template <typename Scalar>
std::size_t DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_ng()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng();
  } else {
    return Base::get_ng();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_nh()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh();
  } else {
    return Base::get_nh();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_ng_T()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng_T();
  } else {
    return Base::get_ng_T();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_nh_T()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh_T();
  } else {
    return Base::get_nh_T();
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_g_lb() const {
  if (constraints_ != nullptr) {
    return constraints_->get_lb();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_g_ub() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ub();
  } else {
    return g_ub_;
  }
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::set_contact_placements(const std::vector<SE3>& Ms) {
  if (Ms.size() != nc_) {
    throw_pretty("Invalid argument: "
                 << "contact placements has wrong size (it should be " +
                        std::to_string(nc_) + ")");
  }
  contact_placements_ = Ms;
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::set_contact_placement(const std::size_t i, const SE3& M) {
  if (i >= nc_) {
    throw_pretty("Invalid argument: " << "contact index is out of range");
  }
  contact_placements_[i] = M;
}

template <typename Scalar>
const std::vector<pinocchio::SE3Tpl<Scalar> >&
DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::get_contact_placements() const {
  return contact_placements_;
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::set_contact_status(const std::vector<bool>& active) {
  if (active.size() != nc_) {
    throw_pretty(
        "Invalid argument: " << "contact status has wrong size (it should be " +
                                    std::to_string(nc_) + ")");
  }
  contact_status_ = active;
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::set_contact_status(const std::size_t i, const bool active) {
  if (i >= nc_) {
    throw_pretty("Invalid argument: " << "contact index is out of range");
  }
  contact_status_[i] = active;
}

template <typename Scalar>
const std::vector<bool>& DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::get_contact_status() const {
  return contact_status_;
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::set_contact_reference(const std::vector<ReferenceFrame>& refs) {
  if (refs.size() != nc_) {
    throw_pretty("Invalid argument: "
                 << "contact references has wrong size (it should be " +
                        std::to_string(nc_) + ")");
  }
  for (std::size_t i = 0; i < nc_; ++i) {
    validate_reference(refs[i]);
  }
  contact_references_ = refs;
}

template <typename Scalar>
const std::vector<pinocchio::ReferenceFrame>&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_contact_reference()
    const {
  return contact_references_;
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::set_gravity(
    const Vector3s& g) {
  gravity_ = g;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_gravity() const {
  return gravity_;
}

template <typename Scalar>
std::size_t
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_ncontacts() const {
  return nc_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar> >&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_constraints()
    const {
  return constraints_;
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_pinocchio() const {
  return *pinocchio_;
}

template <typename Scalar>
const std::shared_ptr<StateCentroidalTpl<Scalar> >&
DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::get_state_centroidal()
    const {
  return state_centroidal_;
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<
    Scalar>::validate_reference(const ReferenceFrame ref) const {
  if (ref != pinocchio::LOCAL && ref != pinocchio::LOCAL_WORLD_ALIGNED) {
    throw_pretty("Invalid argument: " << "contact reference has to be LOCAL or "
                                         "LOCAL_WORLD_ALIGNED");
  }
}

template <typename Scalar>
void DifferentialActionModelCentroidalFwdDynamicsTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelCentroidalFwdDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_ << ", nc=" << nc_
     << "}";
}

}  // namespace crocoddyl
