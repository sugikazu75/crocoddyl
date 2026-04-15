///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_WITH_THRUSTS_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_WITH_THRUSTS_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody-with-thrusts.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for contact forward dynamics with augmented
 *        thrust state.
 *
 * Counterpart of `DifferentialActionModelContactFwdDynamicsTpl` for systems
 * using `StateMultibodyWithThrustsTpl`.  The state is
 *   x = [q (nq), v (nv), f (nf)]
 * so the velocity block is extracted as  x.segment(nq, nv)  (not x.tail(nv)).
 *
 * The actuation model reads the current thrust f from x and joint torques from
 * u.  Consequently Fx has an extra block in its rightmost nf columns:
 *   d(vdot)/df = K^{-1}[:nv,:nv] * W_f(q)
 *
 * The contact model (`ContactModelMultipleTpl`) is built with the underlying
 * `StateMultibody` (not the augmented state) and receives the robot-only slice
 * x_robot = x.head(nq+nv) in every call.  The contact acceleration da0_dx
 * therefore has shape (nc, 2*nv) and contributes only to the [q,v] columns
 * of Fx; the f columns are filled solely by the actuation Jacobian W_f(q).
 *
 * \sa `DifferentialActionModelContactFwdDynamicsTpl`,
 *     `StateMultibodyWithThrustsTpl`,
 *     `ActuationModelFloatingBaseThrusterRatesTpl`
 */
template <typename _Scalar>
class DifferentialActionModelContactFwdDynamicsWithThrustsTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(
      DifferentialActionModelBase,
      DifferentialActionModelContactFwdDynamicsWithThrustsTpl)

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactFwdDynamicsWithThrustsTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyWithThrustsTpl<Scalar> StateWithThrusts;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @param[in] state            Augmented multibody+thrust state
   * @param[in] actuation        Actuation model (e.g.
   * FloatingBaseThrusterRates)
   * @param[in] contacts         Stack of rigid contacts (uses StateMultibody)
   * @param[in] costs            Cost-model sum
   * @param[in] JMinvJt_damping  Damping for operational-space inertia matrix
   * @param[in] enable_force     Enable contact-force derivative computation
   */
  DifferentialActionModelContactFwdDynamicsWithThrustsTpl(
      std::shared_ptr<StateWithThrusts> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ContactModelMultiple> contacts,
      std::shared_ptr<CostModelSum> costs,
      const Scalar JMinvJt_damping = Scalar(0.),
      const bool enable_force = false);

  /**
   * @param[in] state            Augmented multibody+thrust state
   * @param[in] actuation        Actuation model
   * @param[in] contacts         Stack of rigid contacts
   * @param[in] costs            Cost-model sum
   * @param[in] constraints      Constraint manager
   * @param[in] JMinvJt_damping  Damping for operational-space inertia matrix
   * @param[in] enable_force     Enable contact-force derivative computation
   */
  DifferentialActionModelContactFwdDynamicsWithThrustsTpl(
      std::shared_ptr<StateWithThrusts> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ContactModelMultiple> contacts,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints,
      const Scalar JMinvJt_damping = Scalar(0.),
      const bool enable_force = false);

  virtual ~DifferentialActionModelContactFwdDynamicsWithThrustsTpl() = default;

  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;

  virtual std::shared_ptr<DifferentialActionDataAbstract> createData() override;

  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data) override;

  virtual void quasiStatic(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
      const std::size_t maxiter = 100,
      const Scalar tol = Scalar(1e-9)) override;

  virtual std::size_t get_ng() const override;
  virtual std::size_t get_nh() const override;
  virtual std::size_t get_ng_T() const override;
  virtual std::size_t get_nh_T() const override;
  virtual const VectorXs& get_g_lb() const override;
  virtual const VectorXs& get_g_ub() const override;

  const std::shared_ptr<ActuationModelAbstract>& get_actuation() const;
  const std::shared_ptr<ContactModelMultiple>& get_contacts() const;
  const std::shared_ptr<CostModelSum>& get_costs() const;
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;
  const VectorXs& get_armature() const;
  const Scalar get_damping_factor() const;
  const VectorXs& get_thrust_reg_weight() const;
  void set_armature(const VectorXs& armature);
  void set_damping_factor(const Scalar damping);
  void set_thrust_reg_weight(const VectorXs& weight);

  template <typename NewScalar>
  DifferentialActionModelContactFwdDynamicsWithThrustsTpl<NewScalar> cast()
      const;

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::g_lb_;
  using Base::g_ub_;
  using Base::nu_;
  using Base::state_;

 private:
  void init();
  std::shared_ptr<ActuationModelAbstract> actuation_;
  std::shared_ptr<ContactModelMultiple> contacts_;
  std::shared_ptr<CostModelSum> costs_;
  std::shared_ptr<ConstraintModelManager> constraints_;
  pinocchio::ModelTpl<Scalar>* pinocchio_;
  bool with_armature_;
  VectorXs armature_;
  Scalar JMinvJt_damping_;
  bool enable_force_;
  std::size_t nf_;              //!< Number of thrusters
  bool robot_only_costs_;       //!< True when cost model uses underlying
                                //!< StateMultibody (ndx=2*nv)
  VectorXs thrust_reg_weight_;  //!< Per-thruster weight for 0.5*sum(w_i*f_i^2)
};

template <typename _Scalar>
struct DifferentialActionDataContactFwdDynamicsWithThrustsTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
  typedef DataCollectorJointActMultibodyInContactTpl<Scalar>
      DataCollectorJointActMultibodyInContact;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactFwdDynamicsWithThrustsTpl(
      Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            std::make_shared<JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu()),
            model->get_contacts()->createData(&pinocchio)),
        costs(model->get_costs()->createData(&multibody)),
        Kinv(model->get_state()->get_nv() +
                 model->get_contacts()->get_nc_total(),
             model->get_state()->get_nv() +
                 model->get_contacts()->get_nc_total()),
        df_dx(model->get_contacts()->get_nc_total(),
              model->get_state()->get_ndx()),
        df_du(model->get_contacts()->get_nc_total(), model->get_nu()),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_Jstatic(model->get_state()->get_nv(),
                    model->get_nu() + model->get_contacts()->get_nc_total()) {
    multibody.joint->dtau_du.diagonal().setOnes();
    // Share memory only when the cost model's ndx matches the action's ndx.
    // When costs use the underlying StateMultibody (robot-only state), their
    // gradients are copied with zero-padding in calcDiff instead.
    if (model->get_costs()->get_state()->get_ndx() ==
        model->get_state()->get_ndx()) {
      costs->shareMemory(this);
    }
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    Kinv.setZero();
    df_dx.setZero();
    df_du.setZero();
    tmp_xstatic.setZero();
    tmp_Jstatic.setZero();
    pinocchio.lambda_c.resize(model->get_contacts()->get_nc_total());
    pinocchio.lambda_c.setZero();
  }
  virtual ~DifferentialActionDataContactFwdDynamicsWithThrustsTpl() = default;

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorJointActMultibodyInContact multibody;
  std::shared_ptr<CostDataSumTpl<Scalar>> costs;
  std::shared_ptr<ConstraintDataManagerTpl<Scalar>> constraints;
  MatrixXs Kinv;         //!< KKT matrix inverse [(nv+nc) x (nv+nc)]
  MatrixXs df_dx;        //!< Contact force Jacobian w.r.t. state (nc x ndx)
  MatrixXs df_du;        //!< Contact force Jacobian w.r.t. control (nc x nu)
  VectorXs tmp_xstatic;  //!< Scratch: augmented static state [q, 0, f]
  MatrixXs tmp_Jstatic;  //!< Scratch: stacked [dtau_du | Jc^T]

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

/**
 * @brief Euler-integrated action model for systems with augmented thrust state.
 *
 * Wraps `DifferentialActionModelContactFwdDynamicsWithThrustsTpl` and
 * implements the following Euler step for state  x = [q, v, f]:
 *
 *   dx = [ v * dt + vdot * dt^2,   (config-tangent, nv)
 *          vdot * dt,               (velocity-tangent, nv)
 *          df * dt           ]      (thrust-tangent, nf)
 *   xnext = state->integrate(x, dx)
 *
 * where  vdot = DAM->xout  and  df = u.head(nf).
 *
 * Jacobians are assembled analogously to the standard Euler integrator,
 * accounting for the extra nf columns in the tangent space.
 */
template <typename _Scalar>
class IntegratedActionModelEulerWithThrustsTpl
    : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase,
                         IntegratedActionModelEulerWithThrustsTpl)

  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef IntegratedActionDataEulerWithThrustsTpl<Scalar> Data;
  typedef DifferentialActionModelContactFwdDynamicsWithThrustsTpl<Scalar>
      DifferentialModel;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialData;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @param[in] model      Differential action model
   * @param[in] time_step  Integration step (dt)
   */
  IntegratedActionModelEulerWithThrustsTpl(
      std::shared_ptr<DifferentialModel> model, const Scalar time_step = 1e-3);
  virtual ~IntegratedActionModelEulerWithThrustsTpl() = default;

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  virtual std::shared_ptr<ActionDataAbstract> createData() override;

  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  template <typename NewScalar>
  IntegratedActionModelEulerWithThrustsTpl<NewScalar> cast() const;

  const std::shared_ptr<DifferentialModel>& get_differential() const;
  Scalar get_dt() const;
  void set_dt(const Scalar dt);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::state_;

 private:
  std::shared_ptr<DifferentialModel> differential_;
  Scalar dt_;
  Scalar dt2_;  //!< dt^2
  std::size_t nv_;
  std::size_t nf_;
};

template <typename _Scalar>
struct IntegratedActionDataEulerWithThrustsTpl
    : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataEulerWithThrustsTpl(Model<Scalar>* const model)
      : Base(model),
        differential(model->get_differential()->createData()),
        dx(model->get_state()->get_ndx()),
        Fx_tmp(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Fu_tmp(model->get_state()->get_ndx(), model->get_nu()) {
    dx.setZero();
    Fx_tmp.setZero();
    Fu_tmp.setZero();
  }
  virtual ~IntegratedActionDataEulerWithThrustsTpl() = default;

  std::shared_ptr<DifferentialData> differential;
  VectorXs dx;      //!< Full tangent-space step (size ndx)
  MatrixXs Fx_tmp;  //!< Temporary before Jintegrate transport
  MatrixXs Fu_tmp;  //!< Temporary before Jintegrate transport

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/actions/contact-fwddyn-with-thrusts.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DifferentialActionModelContactFwdDynamicsWithThrustsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DifferentialActionDataContactFwdDynamicsWithThrustsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::IntegratedActionModelEulerWithThrustsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::IntegratedActionDataEulerWithThrustsTpl)

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_WITH_THRUSTS_HPP_
