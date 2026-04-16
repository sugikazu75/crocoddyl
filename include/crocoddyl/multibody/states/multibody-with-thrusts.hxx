///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/joint-configuration.hpp>

namespace crocoddyl {

template <typename Scalar>
StateMultibodyWithThrustsTpl<Scalar>::StateMultibodyWithThrustsTpl(
    std::shared_ptr<StateMultibody> state, std::size_t n_thrusters)
    : Base(state->get_nq() + state->get_nv() + n_thrusters,
           2 * state->get_nv() + n_thrusters),
      state_(state),
      n_thrusters_(n_thrusters),
      x0_(VectorXs::Zero(state->get_nq() + state->get_nv() + n_thrusters)) {
  // The base constructor sets nv_ = ndx/2 and nq_ = nx - nv_, which doesn't
  // match the robot dimensions when nf > 0. Override them explicitly so that
  // get_nq() / get_nv() return the *robot* nq / nv, consistent with what
  // pinocchio-based actuation models and cost functions expect.
  nq_ = state->get_nq();
  nv_ = state->get_nv();

  // Zero state: neutral robot configuration, zero velocity, zero thrust.
  x0_.head(nq_) = pinocchio::neutral(*state->get_pinocchio());
  x0_.tail(nv_ + n_thrusters_).setZero();

  // Copy robot position / velocity limits from the underlying state.
  lb_.head(nq_ + nv_) = state->get_lb();
  ub_.head(nq_ + nv_) = state->get_ub();

  // Thrust limits default to unconstrained; set them via set_lb / set_ub.
  lb_.tail(n_thrusters_).fill(-std::numeric_limits<Scalar>::infinity());
  ub_.tail(n_thrusters_).fill(std::numeric_limits<Scalar>::infinity());

  Base::update_has_limits();
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
StateMultibodyWithThrustsTpl<Scalar>::zero() const {
  return x0_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
StateMultibodyWithThrustsTpl<Scalar>::rand() const {
  VectorXs xrand = VectorXs::Random(nx_);
  xrand.head(nq_) = pinocchio::randomConfiguration(*state_->get_pinocchio());
  return xrand;
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::diff(
    const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
    Eigen::Ref<VectorXs> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_)
    throw_pretty("Invalid argument: x0 has wrong dimension (should be " +
                 std::to_string(nx_) + ")");
  if (static_cast<std::size_t>(x1.size()) != nx_)
    throw_pretty("Invalid argument: x1 has wrong dimension (should be " +
                 std::to_string(nx_) + ")");
  if (static_cast<std::size_t>(dxout.size()) != ndx_)
    throw_pretty("Invalid argument: dxout has wrong dimension (should be " +
                 std::to_string(ndx_) + ")");

  // Robot part: pinocchio difference for q, Euclidean difference for v.
  state_->diff(x0.head(nq_ + nv_), x1.head(nq_ + nv_), dxout.head(2 * nv_));
  // Thrust part: Euclidean difference.
  dxout.tail(n_thrusters_) = x1.tail(n_thrusters_) - x0.tail(n_thrusters_);
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::integrate(
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
    Eigen::Ref<VectorXs> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_)
    throw_pretty("Invalid argument: x has wrong dimension (should be " +
                 std::to_string(nx_) + ")");
  if (static_cast<std::size_t>(dx.size()) != ndx_)
    throw_pretty("Invalid argument: dx has wrong dimension (should be " +
                 std::to_string(ndx_) + ")");
  if (static_cast<std::size_t>(xout.size()) != nx_)
    throw_pretty("Invalid argument: xout has wrong dimension (should be " +
                 std::to_string(nx_) + ")");

  // Robot part: pinocchio integrate for q, Euclidean for v.
  state_->integrate(x.head(nq_ + nv_), dx.head(2 * nv_), xout.head(nq_ + nv_));
  // Thrust part: Euclidean integration.
  xout.tail(n_thrusters_) = x.tail(n_thrusters_) + dx.tail(n_thrusters_);
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::safe_diff(
    const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
    Eigen::Ref<VectorXs> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_)
    throw_pretty("Invalid argument: x0 has wrong dimension (should be " +
                 std::to_string(nx_) + ")");
  if (static_cast<std::size_t>(x1.size()) != nx_)
    throw_pretty("Invalid argument: x1 has wrong dimension (should be " +
                 std::to_string(nx_) + ")");
  if (static_cast<std::size_t>(dxout.size()) != ndx_)
    throw_pretty("Invalid argument: dxout has wrong dimension (should be " +
                 std::to_string(ndx_) + ")");

  state_->safe_diff(x0.head(nq_ + nv_), x1.head(nq_ + nv_),
                    dxout.head(2 * nv_));

  // Thrust: propagate NaN for invalid components.
  for (std::size_t i = 0; i < n_thrusters_; ++i) {
    if (!isfinite(x0[nq_ + nv_ + i]) || !isfinite(x1[nq_ + nv_ + i])) {
      dxout[2 * nv_ + i] = std::numeric_limits<Scalar>::quiet_NaN();
    } else {
      dxout[2 * nv_ + i] = x1[nq_ + nv_ + i] - x0[nq_ + nv_ + i];
    }
  }
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::safe_integrate(
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
    Eigen::Ref<VectorXs> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_)
    throw_pretty("Invalid argument: x has wrong dimension (should be " +
                 std::to_string(nx_) + ")");
  if (static_cast<std::size_t>(dx.size()) != ndx_)
    throw_pretty("Invalid argument: dx has wrong dimension (should be " +
                 std::to_string(ndx_) + ")");
  if (static_cast<std::size_t>(xout.size()) != nx_)
    throw_pretty("Invalid argument: xout has wrong dimension (should be " +
                 std::to_string(nx_) + ")");

  state_->safe_integrate(x.head(nq_ + nv_), dx.head(2 * nv_),
                         xout.head(nq_ + nv_));

  // Thrust: propagate NaN for invalid components.
  for (std::size_t i = 0; i < n_thrusters_; ++i) {
    const Scalar xi = x[nq_ + nv_ + i];
    const Scalar dxi = dx[2 * nv_ + i];
    if (!isfinite(xi) || !isfinite(dxi)) {
      xout[nq_ + nv_ + i] = std::numeric_limits<Scalar>::quiet_NaN();
    } else {
      xout[nq_ + nv_ + i] = xi + dxi;
    }
  }
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::Jdiff(
    const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
    Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
    const Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond),
                "firstsecond must be one of {both, first, second}");
  if (static_cast<std::size_t>(x0.size()) != nx_)
    throw_pretty("Invalid argument: x0 has wrong dimension (should be " +
                 std::to_string(nx_) + ")");
  if (static_cast<std::size_t>(x1.size()) != nx_)
    throw_pretty("Invalid argument: x1 has wrong dimension (should be " +
                 std::to_string(nx_) + ")");

  if (firstsecond == first) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
        static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      throw_pretty("Invalid argument: Jfirst has wrong dimension (should be " +
                   std::to_string(ndx_) + "x" + std::to_string(ndx_) + ")");

    MatrixXs Jsecond_dummy = MatrixXs::Zero(2 * nv_, 2 * nv_);
    state_->Jdiff(x0.head(nq_ + nv_), x1.head(nq_ + nv_),
                  Jfirst.topLeftCorner(2 * nv_, 2 * nv_), Jsecond_dummy, first);
    Jfirst.bottomRightCorner(n_thrusters_, n_thrusters_).diagonal().array() =
        Scalar(-1.);

  } else if (firstsecond == second) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
        static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      throw_pretty("Invalid argument: Jsecond has wrong dimension (should be " +
                   std::to_string(ndx_) + "x" + std::to_string(ndx_) + ")");

    MatrixXs Jfirst_dummy = MatrixXs::Zero(2 * nv_, 2 * nv_);
    state_->Jdiff(x0.head(nq_ + nv_), x1.head(nq_ + nv_), Jfirst_dummy,
                  Jsecond.topLeftCorner(2 * nv_, 2 * nv_), second);
    Jsecond.bottomRightCorner(n_thrusters_, n_thrusters_).diagonal().array() =
        Scalar(1.);

  } else {  // both
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ ||
        static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      throw_pretty("Invalid argument: Jfirst has wrong dimension (should be " +
                   std::to_string(ndx_) + "x" + std::to_string(ndx_) + ")");
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ ||
        static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      throw_pretty("Invalid argument: Jsecond has wrong dimension (should be " +
                   std::to_string(ndx_) + "x" + std::to_string(ndx_) + ")");

    state_->Jdiff(x0.head(nq_ + nv_), x1.head(nq_ + nv_),
                  Jfirst.topLeftCorner(2 * nv_, 2 * nv_),
                  Jsecond.topLeftCorner(2 * nv_, 2 * nv_), both);
    Jfirst.bottomRightCorner(n_thrusters_, n_thrusters_).diagonal().array() =
        Scalar(-1.);
    Jsecond.bottomRightCorner(n_thrusters_, n_thrusters_).diagonal().array() =
        Scalar(1.);
  }
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::Jintegrate(
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
    Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
    const Jcomponent firstsecond, const AssignmentOp op) const {
  assert_pretty(is_a_Jcomponent(firstsecond),
                "firstsecond must be one of {both, first, second}");
  assert_pretty(is_a_AssignmentOp(op),
                "op must be one of {setto, addto, rmfrom}");

  auto apply_block = [&](Eigen::Ref<MatrixXs> J,
                         pinocchio::ArgumentPosition arg) {
    if (static_cast<std::size_t>(J.rows()) != ndx_ ||
        static_cast<std::size_t>(J.cols()) != ndx_)
      throw_pretty("Invalid argument: J has wrong dimension (should be " +
                   std::to_string(ndx_) + "x" + std::to_string(ndx_) + ")");

    switch (op) {
      case setto:
        pinocchio::dIntegrate(*state_->get_pinocchio(), x.head(nq_),
                              dx.head(nv_), J.topLeftCorner(nv_, nv_), arg,
                              pinocchio::SETTO);
        J.block(nv_, nv_, nv_, nv_).diagonal().array() = Scalar(1.);
        J.bottomRightCorner(n_thrusters_, n_thrusters_).diagonal().array() =
            Scalar(1.);
        break;
      case addto:
        pinocchio::dIntegrate(*state_->get_pinocchio(), x.head(nq_),
                              dx.head(nv_), J.topLeftCorner(nv_, nv_), arg,
                              pinocchio::ADDTO);
        J.block(nv_, nv_, nv_, nv_).diagonal().array() += Scalar(1.);
        J.bottomRightCorner(n_thrusters_, n_thrusters_).diagonal().array() +=
            Scalar(1.);
        break;
      case rmfrom:
        pinocchio::dIntegrate(*state_->get_pinocchio(), x.head(nq_),
                              dx.head(nv_), J.topLeftCorner(nv_, nv_), arg,
                              pinocchio::RMTO);
        J.block(nv_, nv_, nv_, nv_).diagonal().array() -= Scalar(1.);
        J.bottomRightCorner(n_thrusters_, n_thrusters_).diagonal().array() -=
            Scalar(1.);
        break;
      default:
        throw_pretty(
            "Invalid argument: allowed operators: setto, addto, rmfrom");
    }
  };

  if (firstsecond == first || firstsecond == both) {
    apply_block(Jfirst, pinocchio::ARG0);
  }
  if (firstsecond == second || firstsecond == both) {
    apply_block(Jsecond, pinocchio::ARG1);
  }
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::JintegrateTransport(
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
    Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond),
                "firstsecond must be either first or second");

  // Transport only the config-tangent rows (top nv_); v-rows and f-rows live
  // in Euclidean spaces and need no transport.
  switch (firstsecond) {
    case first:
      pinocchio::dIntegrateTransport(*state_->get_pinocchio(), x.head(nq_),
                                     dx.head(nv_), Jin.topRows(nv_),
                                     pinocchio::ARG0);
      break;
    case second:
      pinocchio::dIntegrateTransport(*state_->get_pinocchio(), x.head(nq_),
                                     dx.head(nv_), Jin.topRows(nv_),
                                     pinocchio::ARG1);
      break;
    default:
      throw_pretty(
          "Invalid argument: firstsecond must be either first or second. "
          "both is not supported for JintegrateTransport.");
  }
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
StateMultibodyWithThrustsTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const std::shared_ptr<pinocchio::ModelTpl<Scalar> >&
StateMultibodyWithThrustsTpl<Scalar>::get_pinocchio() const {
  return state_->get_pinocchio();
}

template <typename Scalar>
std::size_t StateMultibodyWithThrustsTpl<Scalar>::get_nthrusters() const {
  return n_thrusters_;
}

template <typename Scalar>
template <typename NewScalar>
StateMultibodyWithThrustsTpl<NewScalar>
StateMultibodyWithThrustsTpl<Scalar>::cast() const {
  typedef StateMultibodyWithThrustsTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()),
      n_thrusters_);
  return ret;
}

template <typename Scalar>
void StateMultibodyWithThrustsTpl<Scalar>::print(std::ostream& os) const {
  os << "StateMultibodyWithThrusts {nx=" << nx_ << ", ndx=" << ndx_
     << ", nthrusters=" << n_thrusters_
     << ", pinocchio=" << *state_->get_pinocchio() << "}";
}

}  // namespace crocoddyl
