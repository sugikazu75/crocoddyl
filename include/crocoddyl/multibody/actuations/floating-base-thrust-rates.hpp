///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_THRUST_RATES_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_THRUST_RATES_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/actuations/floating-base-distributed-thrusters.hpp"
#include "crocoddyl/multibody/states/multibody-with-thrusts.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"

namespace crocoddyl {

/**
 * @brief Actuation model for floating-base robots with thrust-rate inputs and
 *        link-attached thrusters (distributed / articulated configuration).
 *
 * Designed for use with `StateMultibodyWithThrustsTpl`.  The state stores the
 * *current* thruster forces f, and the control input u is split as
 *
 *   u = [ df (nf),  tau_joints (n_joints) ]
 *
 * where df is the rate of change of thrust (constrained via u_lb / u_ub)
 * and tau_joints are the non-floating-base joint torques.
 *
 * Each thruster is described by a `DistributedThrusterTpl`, which carries the
 * frame_id of the thruster frame so that the thrust-to-generalized-force
 * mapping  W_thrust(q)  is recomputed from frame Jacobians at every `calc`
 * call.
 *
 * The generalized torques are:
 *
 *   tau = W_thrust(q) * f_state + B_joints * tau_joints
 *
 * where f_state = x[ nq + nv : nq + nv + nf ]  (current thrust from state).
 *
 * Jacobians (computed in `calcDiff`):
 *   dtau_dx[ :, 0    : nv    ] = d(W_thrust(q)*f)/dq  via RNEA derivatives
 *   dtau_dx[ :, nv   : 2*nv  ] = 0               (tau does not depend on v)
 *   dtau_dx[ :, 2*nv : 2*nv+nf] = W_thrust(q)    (linear in f_state)
 *   dtau_du[ nv_floating:, nf: ] = I              (joint-torque block,
 * constant)
 *
 * \sa `StateMultibodyWithThrustsTpl`, `DistributedThrusterTpl`,
 *     `ActuationModelFloatingBaseDistributedThrustersTpl`
 */
template <typename _Scalar>
class ActuationModelFloatingBaseThrusterRatesTpl
    : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActuationModelBase,
                         ActuationModelFloatingBaseThrusterRatesTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef ActuationDataFloatingBaseThrusterRatesTpl<Scalar> Data;
  typedef StateMultibodyWithThrustsTpl<Scalar> StateWithThrusts;
  typedef DistributedThrusterTpl<Scalar> DistributedThruster;
  typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActuationModelFloatingBaseThrusterRatesTpl(
      std::shared_ptr<StateWithThrusts> state,
      const std::vector<DistributedThruster>& thrusters)
      : Base(state,
             state->get_nv() -
                 state->get_pinocchio()
                     ->joints[(
                         state->get_pinocchio()->existJointName("root_joint")
                             ? state->get_pinocchio()->getJointId("root_joint")
                             : 0)]
                     .nv() +
                 thrusters.size()),
        state_(state),
        thrusters_(thrusters),
        n_thrusters_(thrusters.size()) {
    // Update the joint actuation part
    const std::size_t root_id =
        state->get_pinocchio()->getJointId("root_joint");
    nv_floating_ = state->get_pinocchio()->joints[root_id].nv();
    n_joints_ = state->get_nv() - nv_floating_;

    // Zero-gravity model for RNEA derivatives of W_thrust(q)*f w.r.t. q
    zero_gravity_model_ =
        std::make_shared<PinocchioModel>(*(state_->get_pinocchio()));
    zero_gravity_model_->gravity.setZero();

    // Bounds on df (thrust rates)
    for (size_t i = 0; i < n_thrusters_; ++i) {
      Base::u_lb_(i) = Scalar(-1.) * thrusters_.at(i).delta_thrust_max_;
      Base::u_ub_(i) = thrusters_.at(i).delta_thrust_max_;
    }

    // Bounds on joint torques from pinocchio effort limits
    if (n_joints_ > 0) {
      Base::u_lb_.tail(n_joints_) =
          Scalar(-1.) *
          state->get_pinocchio()->effortLimit.segment(nv_floating_, n_joints_);
      Base::u_ub_.tail(n_joints_) =
          state->get_pinocchio()->effortLimit.segment(nv_floating_, n_joints_);
    }
  }

  virtual ~ActuationModelFloatingBaseThrusterRatesTpl() = default;

  virtual void calc(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const typename MathBase::VectorXs>& x,
      const Eigen::Ref<const typename MathBase::VectorXs>& u) override {
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        q = x.head(state_->get_nq());

    Data* d = static_cast<Data*>(data.get());

    updateWThrust(d, q);

    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        thrust = x.segment(state_->get_nq() + state_->get_nv(), n_thrusters_);

    data->tau.noalias() = d->W_thrust.leftCols(n_thrusters_) * thrust;
    data->tau.noalias() += d->W_thrust.rightCols(n_joints_) * u.tail(n_joints_);
  }

  virtual void calcDiff(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const typename MathBase::VectorXs>& x,
      const Eigen::Ref<const typename MathBase::VectorXs>& u) override {
    (void)u;
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        q = x.head(state_->get_nq());

    Data* d = static_cast<Data*>(data.get());

    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        thrust = x.segment(state_->get_nq() + state_->get_nv(), n_thrusters_);

    computeFExtByThrusts(thrust, d->fext);

    d->static_torque_partial_dq.setZero();

    pinocchio::computeStaticTorqueDerivatives(*(zero_gravity_model_),
                                              d->pinocchio, q, d->fext,
                                              d->static_torque_partial_dq);

    data->dtau_dx.leftCols(state_->get_nv()).noalias() =
        -d->static_torque_partial_dq;
    data->dtau_dx.block(0, state_->get_nv(), state_->get_nv(), state_->get_nv())
        .setZero();
    data->dtau_dx.rightCols(n_thrusters_) = d->W_thrust.leftCols(n_thrusters_);
  }

  virtual void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& tau) override {
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        q = x.head(state_->get_nq());

    Data* d = static_cast<Data*>(data.get());

    updateWThrust(d, q);

    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        thrust = x.segment(state_->get_nq() + state_->get_nv(), n_thrusters_);

    data->u.head(n_thrusters_).setZero();
    data->u.tail(n_joints_) =
        tau.tail(n_joints_) -
        d->W_thrust.bottomLeftCorner(n_joints_, n_thrusters_) * thrust;
  }

  virtual void torqueTransform(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const VectorXs>&,
      const Eigen::Ref<const VectorXs>&) override {}

  virtual std::shared_ptr<ActuationDataAbstractTpl<Scalar>> createData()
      override {
    std::shared_ptr<Data> data =
        std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);

    initData(data);
    return data;
  }

  template <typename NewScalar>
  ActuationModelFloatingBaseThrusterRatesTpl<NewScalar> cast() const {
    typedef ActuationModelFloatingBaseThrusterRatesTpl<NewScalar> ReturnType;
    typedef StateMultibodyWithThrustsTpl<NewScalar> StateType;
    typedef DistributedThrusterTpl<NewScalar> ThrusterType;
    std::vector<ThrusterType> thrusters_new =
        vector_cast<NewScalar>(thrusters_);
    ReturnType ret(
        std::make_shared<StateType>(state_->template cast<NewScalar>()),
        thrusters_new);
    return ret;
  }

  const std::vector<DistributedThruster>& get_thrusters() const {
    return thrusters_;
  }
  std::size_t get_nthrusters() const { return n_thrusters_; }
  std::size_t get_nv_floating() const { return nv_floating_; }
  std::size_t get_n_joints() const { return n_joints_; }

  virtual void print(std::ostream& os) const override {
    os << "ActuationModelFloatingBaseThrusterRates {nu=" << nu_
       << ", nthrusters=" << n_thrusters_ << ", n_joints=" << n_joints_ << "}";
  }

 protected:
  std::shared_ptr<PinocchioModel> zero_gravity_model_;
  std::shared_ptr<StateWithThrusts> state_;
  std::vector<DistributedThruster> thrusters_;  //!< List of thrusters
  std::size_t n_thrusters_;                     //!< Number of thrusters (= nf)
  std::size_t nv_floating_;  //!< DOF of the floating-base joint
  std::size_t n_joints_;     //!< nv - nv_floating (actuated joints)

  using Base::nu_;

 private:
  void updateWThrust(Data* d, const Eigen::Ref<const VectorXs>& q) {
    for (size_t i = 0; i < n_thrusters_; ++i) {
      d->tmp_thrust_jacobian.setZero();
      pinocchio::computeFrameJacobian(*(state_->get_pinocchio()), d->pinocchio,
                                      q, thrusters_.at(i).frame_id_,
                                      pinocchio::LOCAL, d->tmp_thrust_jacobian);

      d->W_thrust.col(i) = d->tmp_thrust_jacobian.transpose() *
                           thrusters_.at(i).thrust_wrench_unit_.toVector();
    }
  }

  void computeFExtByThrusts(const Eigen::Ref<const VectorXs>& u,
                            std::vector<Force>& fext) {
    for (auto& f : fext) f.setZero();

    for (size_t i = 0; i < n_thrusters_; i++) {
      pinocchio::JointIndex thruster_parent_joint_index =
          state_->get_pinocchio()->frames[thrusters_[i].frame_id_].parentJoint;

      fext.at(thruster_parent_joint_index) +=
          thrusters_[i].thrust_wrench_unit_parent_joint_ * u(i);
    }
  }

  void initData(const std::shared_ptr<Data>& data) {
    data->W_thrust.bottomRightCorner(n_joints_, n_joints_).diagonal().setOnes();

    data->dtau_du.bottomRightCorner(n_joints_, n_joints_).diagonal().setOnes();

    data->Mtau.block(n_thrusters_, nv_floating_, n_joints_, n_joints_)
        .diagonal()
        .setOnes();

    for (std::size_t k = 0; k < nv_floating_; ++k) data->tau_set[k] = false;
    for (std::size_t k = nv_floating_; k < state_->get_nv(); ++k)
      data->tau_set[k] = true;
  }
};

template <typename _Scalar>
struct ActuationDataFloatingBaseThrusterRatesTpl
    : public ActuationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationDataAbstractTpl<Scalar> Base;
  typedef StateMultibodyWithThrustsTpl<Scalar> StateWithThrusts;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  explicit ActuationDataFloatingBaseThrusterRatesTpl(Model<Scalar>* const model)
      : Base(model) {
    std::shared_ptr<StateWithThrusts> state =
        std::static_pointer_cast<StateWithThrusts>(model->get_state());
    pinocchio = pinocchio::DataTpl<Scalar>(*(state->get_pinocchio()));
    fext.resize(state->get_pinocchio()->njoints, Force::Zero());
    W_thrust = MatrixXs::Zero(state->get_nv(), model->get_nu());
    tmp_thrust_jacobian = MatrixXs::Zero(6, state->get_nv());
    static_torque_partial_dq = MatrixXs::Zero(state->get_nv(), state->get_nv());
    S = MatrixXs::Zero(state->get_nv(), state->get_nv());
  }

  virtual ~ActuationDataFloatingBaseThrusterRatesTpl() = default;

  pinocchio::DataTpl<Scalar> pinocchio;  //!< Pinocchio data
  std::vector<Force> fext;               //!< Per-joint ext. forces
  MatrixXs W_thrust;  //!< Thrust-to-torque mapping W_f(q) [nv x nf]
  MatrixXs tmp_thrust_jacobian;
  MatrixXs static_torque_partial_dq;
  MatrixXs S;

  using Base::dtau_du;
  using Base::dtau_dx;
  using Base::Mtau;
  using Base::tau;
  using Base::tau_set;
  using Base::u;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActuationModelFloatingBaseThrusterRatesTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActuationDataFloatingBaseThrusterRatesTpl)

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_THRUST_RATES_HPP_
