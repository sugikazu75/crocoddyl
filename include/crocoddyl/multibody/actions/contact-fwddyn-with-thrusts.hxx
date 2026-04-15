///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::
    DifferentialActionModelContactFwdDynamicsWithThrustsTpl(
        std::shared_ptr<StateWithThrusts> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs, const Scalar JMinvJt_damping,
        const bool enable_force)
    : Base(state, actuation->get_nu(), costs->get_nr(), 0, 0),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(nullptr),
      pinocchio_(state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force),
      nf_(state->get_nthrusters()) {
  init();
}

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::
    DifferentialActionModelContactFwdDynamicsWithThrustsTpl(
        std::shared_ptr<StateWithThrusts> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs,
        std::shared_ptr<ConstraintModelManager> constraints,
        const Scalar JMinvJt_damping, const bool enable_force)
    : Base(state, actuation->get_nu(), costs->get_nr(), constraints->get_ng(),
           constraints->get_nh(), constraints->get_ng_T(),
           constraints->get_nh_T()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force),
      nf_(state->get_nthrusters()) {
  init();
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::init() {
  if (JMinvJt_damping_ < Scalar(0.)) {
    JMinvJt_damping_ = Scalar(0.);
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive, set to 0");
  }
  if (contacts_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Contacts doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Costs doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  Base::u_lb_ = actuation_->get_u_lb();
  Base::u_ub_ = actuation_->get_u_ub();
  robot_only_costs_ = (costs_->get_state()->get_ndx() != state_->get_ndx());
  thrust_reg_weight_ = VectorXs::Zero(nf_);
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::calc(
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

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.segment(nq, nv);

  // Computing the forward dynamics with the holonomic constraints defined by
  // the contact model
  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  actuation_->calc(d->multibody.actuation, x, u);
  // Contact model uses StateMultibody: pass only the robot-state slice
  contacts_->calc(d->multibody.contacts, x.head(nq + nv));

#ifndef NDEBUG
  Eigen::FullPivLU<MatrixXs> Jc_lu(d->multibody.contacts->Jc.topRows(nc));
  if (Jc_lu.rank() < static_cast<Eigen::Index>(nc) &&
      JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty(
        "A damping factor is needed as the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::forwardDynamics(
      *pinocchio_, d->pinocchio, d->multibody.actuation->tau,
      d->multibody.contacts->Jc.topRows(nc), d->multibody.contacts->a0.head(nc),
      JMinvJt_damping_);
  d->xout = d->pinocchio.ddq;
  contacts_->updateAcceleration(d->multibody.contacts, d->pinocchio.ddq);
  contacts_->updateForce(d->multibody.contacts, d->pinocchio.lambda_c);
  d->multibody.joint->a = d->pinocchio.ddq;
  d->multibody.joint->tau = u;
  // When costs use underlying StateMultibody, pass robot-only state slice
  costs_->calc(d->costs, robot_only_costs_ ? x.head(nq + nv) : x, u);
  d->cost = d->costs->cost;
  // Thrust regularization: 0.5 * sum_i(w_i * f_i^2)  (f = x.tail(nf_))
  if (thrust_reg_weight_.squaredNorm() > Scalar(0.)) {
    d->cost += Scalar(0.5) * thrust_reg_weight_.dot(x.tail(nf_).cwiseAbs2());
  }
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d);
    constraints_->calc(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::calc(
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
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.segment(nq, nv);

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);
  costs_->calc(d->costs, robot_only_costs_ ? x.head(nq + nv) : x);
  d->cost = d->costs->cost;
  if (thrust_reg_weight_.squaredNorm() > Scalar(0.)) {
    d->cost += Scalar(0.5) * thrust_reg_weight_.dot(x.tail(nf_).cwiseAbs2());
  }
  if (constraints_ != nullptr) {
    d->constraints->resize(this, d, false);
    constraints_->calc(d->constraints, x);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::calcDiff(
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

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.segment(nq, nv);

  Data* d = static_cast<Data*>(data.get());

  // Computing the dynamics derivatives
  // We resize the Kinv matrix because Eigen cannot call block operations
  // recursively: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=408. Therefore,
  // it is not possible to pass d->Kinv.topLeftCorner(nv + nc, nv + nc)
  d->Kinv.resize(nv + nc, nv + nc);
  pinocchio::computeRNEADerivatives(*pinocchio_, d->pinocchio, q, v, d->xout,
                                    d->multibody.contacts->fext);
  contacts_->updateRneaDiff(d->multibody.contacts, d->pinocchio);
  pinocchio::getKKTContactDynamicMatrixInverse(
      *pinocchio_, d->pinocchio, d->multibody.contacts->Jc.topRows(nc),
      d->Kinv);

  actuation_->calcDiff(d->multibody.actuation, x, u);
  // Contact model uses StateMultibody: pass only the robot-state slice
  contacts_->calcDiff(d->multibody.contacts, x.head(nq + nv));

  const Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  const Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  const Eigen::Block<MatrixXs> f_partial_dtau =
      d->Kinv.bottomLeftCorner(nc, nv);
  const Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  d->Fx.leftCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.block(0, nv, nv, nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.rightCols(nf_).setZero();
  // da0_dx has shape (nc, 2*nv): only affects the [q,v] columns of Fx
  d->Fx.leftCols(2 * nv).noalias() -=
      a_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
  d->Fx.noalias() += a_partial_dtau * d->multibody.actuation->dtau_dx;

  d->Fu.noalias() = a_partial_dtau * d->multibody.actuation->dtau_du;

  d->multibody.joint->da_dx = d->Fx;
  d->multibody.joint->da_du = d->Fu;

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dx.block(0, 0, nc, nv).noalias() =
        f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.block(0, nv, nc, nv).noalias() =
        f_partial_dtau * d->pinocchio.dtau_dv;
    d->df_dx.rightCols(nf_).setZero();
    // da0_dx has shape (nc, 2*nv): only affects the [q,v] columns
    d->df_dx.block(0, 0, nc, 2 * nv).noalias() +=
        f_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
    d->df_dx.topRows(nc).noalias() -=
        f_partial_dtau * d->multibody.actuation->dtau_dx;
    d->df_du.topRows(nc).noalias() =
        -f_partial_dtau * d->multibody.actuation->dtau_du;
    contacts_->updateAccelerationDiff(d->multibody.contacts,
                                      d->Fx.bottomRows(nv).leftCols(2 * nv));
    contacts_->updateForceDiff(d->multibody.contacts,
                               d->df_dx.topRows(nc).leftCols(2 * nv),
                               d->df_du.topRows(nc));
  }
  if (robot_only_costs_) {
    costs_->calcDiff(d->costs, x.head(nq + nv), u);
    // Zero-pad cost gradients into the full augmented-state action gradients
    const std::size_t robot_ndx = costs_->get_state()->get_ndx();
    d->Lx.head(robot_ndx) = d->costs->Lx;
    d->Lx.tail(nf_).setZero();
    d->Lu = d->costs->Lu;
    d->Lxx.topLeftCorner(robot_ndx, robot_ndx) = d->costs->Lxx;
    d->Lxx.topRightCorner(robot_ndx, nf_).setZero();
    d->Lxx.bottomRows(nf_).setZero();
    d->Lxu.topRows(robot_ndx) = d->costs->Lxu;
    d->Lxu.bottomRows(nf_).setZero();
    d->Luu = d->costs->Luu;
    // Thrust regularization gradient: dL/df_i = w_i*f_i, d^2L/df_i^2 = w_i
    if (thrust_reg_weight_.squaredNorm() > Scalar(0.)) {
      d->Lx.tail(nf_).array() +=
          thrust_reg_weight_.array() * x.tail(nf_).array();
      d->Lxx.bottomRightCorner(nf_, nf_).diagonal().array() +=
          thrust_reg_weight_.array();
    }
  } else {
    costs_->calcDiff(d->costs, x, u);
    // Thrust regularization gradient (non-robot-only path)
    if (thrust_reg_weight_.squaredNorm() > Scalar(0.)) {
      d->Lx.tail(nf_).array() +=
          thrust_reg_weight_.array() * x.tail(nf_).array();
      d->Lxx.bottomRightCorner(nf_, nf_).diagonal().array() +=
          thrust_reg_weight_.array();
    }
  }
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x, u);
  }
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  if (robot_only_costs_) {
    const std::size_t nq = state_->get_nq();
    const std::size_t nv = state_->get_nv();
    costs_->calcDiff(d->costs, x.head(nq + nv));
    const std::size_t robot_ndx = costs_->get_state()->get_ndx();
    d->Lx.head(robot_ndx) = d->costs->Lx;
    d->Lx.tail(nf_).setZero();
    d->Lxx.topLeftCorner(robot_ndx, robot_ndx) = d->costs->Lxx;
    d->Lxx.topRightCorner(robot_ndx, nf_).setZero();
    d->Lxx.bottomRows(nf_).setZero();
    if (thrust_reg_weight_.squaredNorm() > Scalar(0.)) {
      d->Lx.tail(nf_).array() +=
          thrust_reg_weight_.array() * x.tail(nf_).array();
      d->Lxx.bottomRightCorner(nf_, nf_).diagonal().array() +=
          thrust_reg_weight_.array();
    }
  } else {
    costs_->calcDiff(d->costs, x);
    if (thrust_reg_weight_.squaredNorm() > Scalar(0.)) {
      d->Lx.tail(nf_).array() +=
          thrust_reg_weight_.array() * x.tail(nf_).array();
      d->Lxx.bottomRightCorner(nf_, nf_).diagonal().array() +=
          thrust_reg_weight_.array();
    }
  }
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x);
  }
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  return d != nullptr;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::
    quasiStatic(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                std::size_t, Scalar) {
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

  // Static casting the data
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);

  // Build augmented static state: v = 0, preserve current thrust f
  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.segment(nq, nv).setZero();
  d->tmp_xstatic.tail(nf_) = x.tail(nf_);
  u.setZero();

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q,
                             d->tmp_xstatic.segment(nq, nv));
  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio, q);
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, d->tmp_xstatic.segment(nq, nv),
                  d->tmp_xstatic.segment(nq, nv));
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic.head(nq + nv));

  // Joints must balance: g - W*f = S*tau_joints + Jc^T*lambda
  // Subtract the thrust contribution already handled by the current state
  VectorXs tau_residual = d->pinocchio.tau - d->multibody.actuation->tau;

  // Allocates memory
  d->tmp_Jstatic.conservativeResize(nv, nu_ + nc);
  d->tmp_Jstatic.leftCols(nu_) = d->multibody.actuation->dtau_du;
  d->tmp_Jstatic.rightCols(nc) =
      d->multibody.contacts->Jc.topRows(nc).transpose();
  u.noalias() = (pseudoInverse(d->tmp_Jstatic) * tau_residual).head(nu_);
  u.head(nf_).setZero();
  d->pinocchio.tau.setZero();
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<NewScalar>
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::cast() const {
  typedef DifferentialActionModelContactFwdDynamicsWithThrustsTpl<NewScalar>
      ReturnType;
  typedef StateMultibodyWithThrustsTpl<NewScalar> StateType;
  typedef ContactModelMultipleTpl<NewScalar> ContactType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  if (constraints_) {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<ContactType>(contacts_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        std::make_shared<ConstraintType>(
            constraints_->template cast<NewScalar>()),
        scalar_cast<NewScalar>(JMinvJt_damping_), enable_force_);
    return ret;
  } else {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<ContactType>(contacts_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        scalar_cast<NewScalar>(JMinvJt_damping_), enable_force_);
    return ret;
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::get_ng() const {
  return constraints_ ? constraints_->get_ng() : Base::get_ng();
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::get_nh() const {
  return constraints_ ? constraints_->get_nh() : Base::get_nh();
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::get_ng_T() const {
  return constraints_ ? constraints_->get_ng_T() : Base::get_ng_T();
}

template <typename Scalar>
std::size_t DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::get_nh_T() const {
  return constraints_ ? constraints_->get_nh_T() : Base::get_nh_T();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::get_g_lb()
    const {
  return constraints_ ? constraints_->get_lb() : g_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::get_g_ub()
    const {
  return constraints_ ? constraints_->get_ub() : g_ub_;
}

template <typename Scalar>
const std::shared_ptr<ActuationModelAbstractTpl<Scalar>>&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::get_actuation()
    const {
  return actuation_;
}

template <typename Scalar>
const std::shared_ptr<ContactModelMultipleTpl<Scalar>>&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::get_contacts()
    const {
  return contacts_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar>>&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar>>&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::get_pinocchio()
    const {
  return *pinocchio_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::get_armature()
    const {
  return armature_;
}

template <typename Scalar>
const Scalar DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::get_damping_factor() const {
  return JMinvJt_damping_;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::set_armature(const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " +
                        std::to_string(state_->get_nv()) + ")");
  }
  armature_ = armature;
  with_armature_ = false;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::set_damping_factor(const Scalar damping) {
  if (damping < 0.) {
    throw_pretty(
        "Invalid argument: " << "The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

template <typename Scalar>
const typename DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::VectorXs&
DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::get_thrust_reg_weight() const {
  return thrust_reg_weight_;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<
    Scalar>::set_thrust_reg_weight(const VectorXs& weight) {
  if (static_cast<std::size_t>(weight.size()) != nf_) {
    throw_pretty("Invalid argument: " << "thrust_reg_weight must have size nf="
                                      << nf_);
  }
  if ((weight.array() < Scalar(0.)).any()) {
    throw_pretty("Invalid argument: "
                 << "thrust_reg_weight elements must be non-negative");
  }
  thrust_reg_weight_ = weight;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelContactFwdDynamicsWithThrusts {nx="
     << state_->get_nx() << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
     << ", nf=" << nf_ << ", nc=" << contacts_->get_nc() << "}";
}

template <typename Scalar>
IntegratedActionModelEulerWithThrustsTpl<Scalar>::
    IntegratedActionModelEulerWithThrustsTpl(
        std::shared_ptr<DifferentialModel> model, const Scalar time_step)
    : Base(model->get_state(), model->get_nu()),
      differential_(model),
      dt_(time_step),
      dt2_(time_step * time_step),
      nv_(model->get_state()->get_nv()),
      nf_(std::static_pointer_cast<StateMultibodyWithThrustsTpl<Scalar>>(
              model->get_state())
              ->get_nthrusters()) {
  Base::u_lb_ = model->get_u_lb();
  Base::u_ub_ = model->get_u_ub();
}

template <typename Scalar>
void IntegratedActionModelEulerWithThrustsTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
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
  differential_->calc(d->differential, x, u);

  const std::size_t nq = state_->get_nq();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.segment(nq, nv_);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      df = u.head(nf_);
  const VectorXs& vdot = d->differential->xout;

  d->dx.head(nv_).noalias() = v * dt_ + vdot * dt2_;
  d->dx.segment(nv_, nv_).noalias() = vdot * dt_;
  d->dx.tail(nf_).noalias() = df * dt_;

  state_->integrate(x, d->dx, d->xnext);
  d->cost = dt_ * d->differential->cost;
}

template <typename Scalar>
void IntegratedActionModelEulerWithThrustsTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  differential_->calc(d->differential, x);
  d->dx.setZero();
  d->xnext = x;
  d->cost = d->differential->cost;
}

template <typename Scalar>
void IntegratedActionModelEulerWithThrustsTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
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
  differential_->calcDiff(d->differential, x, u);

  const MatrixXs& da_dx = d->differential->Fx;
  const MatrixXs& da_du = d->differential->Fu;
  const std::size_t ndx = state_->get_ndx();

  d->Fx_tmp.setZero();
  d->Fx_tmp.topRows(nv_).noalias() = da_dx * dt2_;
  d->Fx_tmp.block(0, nv_, nv_, nv_).diagonal().array() += Scalar(dt_);
  d->Fx_tmp.block(nv_, 0, nv_, ndx).noalias() = da_dx * dt_;

  d->Fu_tmp.setZero();
  d->Fu_tmp.topRows(nv_).noalias() = da_du * dt2_;
  d->Fu_tmp.block(nv_, 0, nv_, nu_).noalias() = da_du * dt_;
  d->Fu_tmp.block(2 * nv_, 0, nf_, nf_).diagonal().array() = Scalar(dt_);

  state_->JintegrateTransport(x, d->dx, d->Fx_tmp, second);
  state_->Jintegrate(x, d->dx, d->Fx_tmp, d->Fx_tmp, first, addto);
  d->Fx = d->Fx_tmp;

  state_->JintegrateTransport(x, d->dx, d->Fu_tmp, second);
  d->Fu = d->Fu_tmp;

  d->Lx.noalias() = dt_ * d->differential->Lx;
  d->Lu.noalias() = dt_ * d->differential->Lu;
  d->Lxx.noalias() = dt_ * d->differential->Lxx;
  d->Lxu.noalias() = dt_ * d->differential->Lxu;
  d->Luu.noalias() = dt_ * d->differential->Luu;
}

template <typename Scalar>
void IntegratedActionModelEulerWithThrustsTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  differential_->calcDiff(d->differential, x);
  state_->Jintegrate(x, d->dx, d->Fx_tmp, d->Fx_tmp);
  d->Fx = d->Fx_tmp;
  d->Lx = d->differential->Lx;
  d->Lxx = d->differential->Lxx;
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
IntegratedActionModelEulerWithThrustsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool IntegratedActionModelEulerWithThrustsTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  return d != nullptr;
}

template <typename Scalar>
void IntegratedActionModelEulerWithThrustsTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t, const Scalar) {
  Data* d = static_cast<Data*>(data.get());
  differential_->quasiStatic(d->differential, u, x);
}

template <typename Scalar>
template <typename NewScalar>
IntegratedActionModelEulerWithThrustsTpl<NewScalar>
IntegratedActionModelEulerWithThrustsTpl<Scalar>::cast() const {
  typedef IntegratedActionModelEulerWithThrustsTpl<NewScalar> ReturnType;
  typedef DifferentialActionModelContactFwdDynamicsWithThrustsTpl<NewScalar>
      DiffType;
  ReturnType ret(
      std::make_shared<DiffType>(differential_->template cast<NewScalar>()),
      scalar_cast<NewScalar>(dt_));
  return ret;
}

template <typename Scalar>
const std::shared_ptr<
    DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>>&
IntegratedActionModelEulerWithThrustsTpl<Scalar>::get_differential() const {
  return differential_;
}

template <typename Scalar>
Scalar IntegratedActionModelEulerWithThrustsTpl<Scalar>::get_dt() const {
  return dt_;
}

template <typename Scalar>
void IntegratedActionModelEulerWithThrustsTpl<Scalar>::set_dt(const Scalar dt) {
  dt_ = dt;
  dt2_ = dt * dt;
}

template <typename Scalar>
void IntegratedActionModelEulerWithThrustsTpl<Scalar>::print(
    std::ostream& os) const {
  os << "IntegratedActionModelEulerWithThrusts {dt=" << dt_ << ", "
     << *differential_ << "}";
}

}  // namespace crocoddyl
