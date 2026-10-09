///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_WITH_THRUSTS_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_WITH_THRUSTS_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody-with-thrusts.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for contact inverse dynamics with thrusters
 *
 * Inverse-dynamics counterpart of
 * `DifferentialActionModelContactFwdDynamicsWithThrustsTpl`, following the
 * formulation of "Inverse-Dynamics MPC via Nullspace Resolution" (Mastalli et
 * al., T-RO 2023). Unlike `DifferentialActionModelContactInvDynamicsTpl`,
 * the thrusts are kept as decision variables instead of being recovered from
 * the generalized torques through the pseudo-inverse of the configuration
 * dependent thrust map \f$\mathbf{W}(\mathbf{q})\f$. Two modes are selected by
 * the type of state:
 *  - `StateMultibody`: the thrusts \f$\boldsymbol{\lambda}\f$ are controls,
 *    \f$\mathbf{u} = (\boldsymbol{\lambda}, \dot{\mathbf{v}},
 *    \boldsymbol{\lambda}_c)\f$, and the actuation model must be
 *    `ActuationModelFloatingBaseDistributedThrustersTpl` or
 *    `ActuationModelFloatingBaseThrustersTpl`.
 *  - `StateMultibodyWithThrusts`: the thrusts are part of the state
 *    \f$\mathbf{x} = (\mathbf{q}, \mathbf{v}, \boldsymbol{\lambda})\f$ and the
 *    controls are their rates, \f$\mathbf{u} = (\dot{\boldsymbol{\lambda}},
 *    \dot{\mathbf{v}}, \boldsymbol{\lambda}_c)\f$, which keeps the thrusts
 *    continuous. The actuation model must read the thrusts from the state
 *    (e.g. `ActuationModelFloatingBaseThrusterRatesTpl`). Use it with
 *    `IntegratedActionModelEulerWithThrustsTpl`.
 *
 * In both cases the actuation input is \f$(\mathbf{w}, \boldsymbol{\tau}_j)\f$,
 * where \f$\mathbf{w}\f$ are the first `nf` controls, and its joint-torque
 * columns must map one-to-one onto the last \f$n_v - n_{vf}\f$ generalized
 * coordinates. The joint torques are not decision variables; they are
 * \f$\boldsymbol{\tau}_j = [\mathrm{RNEA}(\mathbf{q}, \mathbf{v},
 * \dot{\mathbf{v}}, \boldsymbol{\lambda}_c) - \mathbf{W}(\mathbf{q})
 * \boldsymbol{\lambda}]_j\f$ and are stored, together with \f$\mathbf{w}\f$, in
 * the joint data so that joint-effort residuals can bound them.
 *
 * The model adds the following equality constraints before those of the
 * user-defined constraint manager:
 *  - the \f$n_{vf}\f$ unactuated rows \f$[\mathrm{RNEA} -
 * \mathbf{W}\boldsymbol{\lambda}]_f = \mathbf{0}\f$, whose control Jacobian
 *    contains the full-rank floating-base block of the inertia matrix,
 *  - the contact acceleration of each active contact, or a zero force for each
 *    inactive one,
 * and, with a thrust state, the inequality constraints
 * \f$\boldsymbol{\lambda}_{min} \leq \boldsymbol{\lambda} \leq
 * \boldsymbol{\lambda}_{max}\f$ before the user-defined inequalities. With
 * thrust controls these limits are control bounds instead.
 *
 * Costs and constraints may be defined over the robot state
 * (`StateMultibody`) even when the thrusts are in the state; their derivatives
 * are zero-padded in the thrust columns.
 *
 * \sa `DifferentialActionModelContactInvDynamicsTpl`,
 * `DifferentialActionModelContactFwdDynamicsWithThrustsTpl`
 */
template <typename _Scalar>
class DifferentialActionModelContactInvDynamicsWithThrustsTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(
      DifferentialActionModelBase,
      DifferentialActionModelContactInvDynamicsWithThrustsTpl)

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactInvDynamicsWithThrustsTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef StateMultibodyWithThrustsTpl<Scalar> StateWithThrusts;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ContactItemTpl<Scalar> ContactItem;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact inverse-dynamics action model with thrusters
   *
   * @param[in] state      State of the multibody system, with
   * (`StateMultibodyWithThrusts`) or without (`StateMultibody`) thrusts
   * @param[in] actuation  Thruster actuation model
   * @param[in] contacts   Multiple contacts (nu = nf + nv + nc)
   * @param[in] costs      Stack of cost functions (nu = nf + nv + nc)
   */
  DifferentialActionModelContactInvDynamicsWithThrustsTpl(
      std::shared_ptr<StateAbstract> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ContactModelMultiple> contacts,
      std::shared_ptr<CostModelSum> costs);

  /**
   * @copybrief DifferentialActionModelContactInvDynamicsWithThrustsTpl
   *
   * @param[in] state        State of the multibody system, with
   * (`StateMultibodyWithThrusts`) or without (`StateMultibody`) thrusts
   * @param[in] actuation    Thruster actuation model
   * @param[in] contacts     Multiple contacts (nu = nf + nv + nc)
   * @param[in] costs        Stack of cost functions (nu = nf + nv + nc)
   * @param[in] constraints  Stack of constraint functions (nu = nf + nv + nc)
   */
  DifferentialActionModelContactInvDynamicsWithThrustsTpl(
      std::shared_ptr<StateAbstract> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ContactModelMultiple> contacts,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints);
  virtual ~DifferentialActionModelContactInvDynamicsWithThrustsTpl() = default;

  /**
   * @brief Compute the joint torques, the dynamics constraints and the costs
   *
   * @param[in] data  Inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$(\mathbf{w}, \dot{\mathbf{v}},
   * \boldsymbol{\lambda}_c)\in\mathbb{R}^{nf+nv+nc}\f$
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the terminal costs and constraints
   *
   * @param[in] data  Inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the dynamics, costs and constraints
   *
   * @param[in] data  Inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$(\mathbf{w}, \dot{\mathbf{v}},
   * \boldsymbol{\lambda}_c)\in\mathbb{R}^{nf+nv+nc}\f$
   */
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the derivatives of the terminal costs and constraints
   *
   * @param[in] data  Inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;

  virtual std::shared_ptr<DifferentialActionDataAbstract> createData() override;

  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data) override;

  /**
   * @brief Compute the thrusts and contact forces that hold the robot still
   *
   * It solves the floating-base rows of \f$\mathbf{g}(\mathbf{q}) =
   * \mathbf{W}(\mathbf{q})\boldsymbol{\lambda} + \mathbf{J}_c^T
   * \boldsymbol{\lambda}_c\f$ in the least-squares sense. With a thrust state,
   * the thrusts are taken from \f$\mathbf{x}\f$ and their rates are set to
   * zero. Thrust limits are not considered.
   */
  virtual void quasiStatic(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
      const std::size_t maxiter = 100,
      const Scalar tol = Scalar(1e-9)) override;

  /**
   * @brief Compute the thrusts that hold the robot still at a configuration
   *
   * It solves the bounded minimum-norm static equilibrium
   * \f$\mathbf{g}(\mathbf{q}) = \mathbf{W}(\mathbf{q})\boldsymbol{\lambda}
   * + \mathbf{S}\boldsymbol{\tau}_j + \mathbf{J}_c^T\boldsymbol{\lambda}_c\f$
   * over the active contacts, with the thrust limits and non-pulling contacts,
   * as `DifferentialActionModelContactFwdDynamicsWithThrustsTpl` does.
   *
   * @param[in] data  Inverse-dynamics data
   * @param[in] q     Robot configuration
   * @return the equilibrium thrusts (dim. nf)
   */
  VectorXs computeEquilibriumThrust(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& q);

  template <typename NewScalar>
  DifferentialActionModelContactInvDynamicsWithThrustsTpl<NewScalar> cast()
      const;

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

  /** @brief Return the number of thrusters */
  std::size_t get_nf() const;

  /** @brief Return the number of unactuated generalized coordinates */
  std::size_t get_nv_floating() const;

  /** @brief Return true if the thrusts are part of the state */
  bool get_thrust_state() const;

  /** @brief Return the weights of the thrust regularization
   * \f$\frac{1}{2}\sum_i w_i\lambda_i^2\f$ */
  const VectorXs& get_thrust_reg_weight() const;

  /** @brief Modify the weights of the thrust regularization */
  void set_thrust_reg_weight(const VectorXs& weight);

  /** @brief Return the weights of the thrust quadratic barrier */
  const VectorXs& get_thrust_barrier_weight() const;

  /** @brief Return the lower bounds of the thrust quadratic barrier */
  const VectorXs& get_thrust_barrier_lb() const;

  /** @brief Return the upper bounds of the thrust quadratic barrier */
  const VectorXs& get_thrust_barrier_ub() const;

  /**
   * @brief Modify the thrust quadratic barrier
   * \f$\frac{1}{2}\sum_i (w_i\min(\lambda_i - lb_i, 0))^2 + (w_i\max(\lambda_i
   * - ub_i, 0))^2\f$
   *
   * It is a soft counterpart of the thrust limits for solvers that ignore
   * inequality constraints and control bounds (e.g. SolverIntro). By default,
   * the weights are zero and the bounds are the thrust limits.
   */
  void set_thrust_barrier(const VectorXs& weight, const VectorXs& lb,
                          const VectorXs& ub);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::g_lb_;
  using Base::g_ub_;
  using Base::nu_;
  using Base::state_;

 private:
  static std::size_t getNumThrusters(
      const std::shared_ptr<StateAbstract>& state,
      const std::shared_ptr<ActuationModelAbstract>& actuation);
  void init();
  void resizeConstraintData(Data* d, const bool running_node) const;
  void calcUserConstraints(Data* d, const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>* u);
  void calcDiffUserConstraints(Data* d, const Eigen::Ref<const VectorXs>& x,
                               const Eigen::Ref<const VectorXs>* u);
  void calcDiffCosts(Data* d, const Eigen::Ref<const VectorXs>& x,
                     const Eigen::Ref<const VectorXs>* u);
  Scalar calcThrustCost(Data* d, const Eigen::Ref<const VectorXs>& thrust);
  void calcDiffThrustCost(Data* d, const Eigen::Ref<const VectorXs>& thrust,
                          Eigen::Ref<VectorXs> L, Eigen::Ref<MatrixXs> LL);

  std::shared_ptr<ActuationModelAbstract> actuation_;
  std::shared_ptr<ContactModelMultiple> contacts_;
  std::shared_ptr<CostModelSum> costs_;
  std::shared_ptr<ConstraintModelManager> constraints_;
  pinocchio::ModelTpl<Scalar>* pinocchio_;
  bool thrust_state_;      //!< True when the thrusts are part of the state
  std::size_t nf_;         //!< Number of thrusters
  std::size_t nvf_;        //!< Number of unactuated generalized coordinates
  std::size_t ng_thrust_;  //!< Number of thrust-limit inequalities
  VectorXs thrust_lb_;     //!< Lower thrust limits
  VectorXs thrust_ub_;     //!< Upper thrust limits
  VectorXs thrust_reg_weight_;      //!< Weights of the thrust regularization
  VectorXs thrust_barrier_weight_;  //!< Weights of the thrust barrier
  VectorXs thrust_barrier_lb_;      //!< Lower bounds of the thrust barrier
  VectorXs thrust_barrier_ub_;      //!< Upper bounds of the thrust barrier
};

template <typename _Scalar>
struct DifferentialActionDataContactInvDynamicsWithThrustsTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
  typedef DataCollectorJointActMultibodyInContactTpl<Scalar>
      DataCollectorJointActMultibodyInContact;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ContactItemTpl<Scalar> ContactItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::ArrayXs ArrayXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactInvDynamicsWithThrustsTpl(
      Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            std::make_shared<JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu()),
            model->get_contacts()->createData(&pinocchio)),
        u_act(model->get_actuation()->get_nu()),
        tau_thrust(model->get_state()->get_nv()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        dtau_du(model->get_state()->get_nv(), model->get_nu()),
        tmp_xstatic(model->get_state()->get_nx()),
        rlb_min(model->get_nf()),
        rub_max(model->get_nf()) {
    const std::size_t nf = model->get_nf();
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nc = model->get_contacts()->get_nc_total();
    // The accelerations and the thrust inputs are decision variables
    Fu.middleCols(nf, nv).diagonal().setOnes();
    multibody.joint->da_du.middleCols(nf, nv).diagonal().setOnes();
    multibody.joint->dtau_du.topLeftCorner(nf, nf).diagonal().setOnes();
    // The contact forces are decision variables too
    MatrixXs df_dx =
        MatrixXs::Zero(nc, model->get_contacts()->get_state()->get_ndx());
    MatrixXs df_du = MatrixXs::Zero(nc, model->get_nu());
    df_du.rightCols(nc).diagonal().setOnes();
    std::vector<bool> contact_status;
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator
             it = model->get_contacts()->get_contacts().begin();
         it != model->get_contacts()->get_contacts().end(); ++it) {
      contact_status.push_back(it->second->active);
      it->second->active = true;
    }
    model->get_contacts()->updateForceDiff(multibody.contacts, df_dx, df_du);
    std::size_t cid = 0;
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator
             it = model->get_contacts()->get_contacts().begin();
         it != model->get_contacts()->get_contacts().end(); ++it, ++cid) {
      it->second->active = contact_status[cid];
    }
    costs = model->get_costs()->createData(&multibody);
    if (model->get_costs()->get_state()->get_ndx() ==
        model->get_state()->get_ndx()) {
      costs->shareMemory(this);
    }
    constraints = model->get_constraints()->createData(&multibody);
    u_act.setZero();
    tau_thrust.setZero();
    dtau_dx.setZero();
    dtau_du.setZero();
    tmp_xstatic.setZero();
    rlb_min.setZero();
    rub_max.setZero();
  }
  virtual ~DifferentialActionDataContactInvDynamicsWithThrustsTpl() = default;

  pinocchio::DataTpl<Scalar> pinocchio;               //!< Pinocchio data
  DataCollectorJointActMultibodyInContact multibody;  //!< Multibody data
  std::shared_ptr<CostDataSum> costs;                 //!< Costs data
  std::shared_ptr<ConstraintDataManager>
      constraints;      //!< User-defined constraints data
  VectorXs u_act;       //!< Actuation input (thrust input, joint torques)
  VectorXs tau_thrust;  //!< Generalized forces produced by the thrusts
  MatrixXs dtau_dx;  //!< Jacobian of RNEA minus thrust forces w.r.t. the state
  MatrixXs
      dtau_du;  //!< Jacobian of RNEA minus thrust forces w.r.t. the control
  VectorXs tmp_xstatic;  //!< State used for computing the quasi-static input
  ArrayXs rlb_min;       //!< Weighted lower-barrier residual: w*(f-lb).min(0)
  ArrayXs rub_max;       //!< Weighted upper-barrier residual: w*(f-ub).max(0)

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::g;
  using Base::Gu;
  using Base::Gx;
  using Base::h;
  using Base::Hu;
  using Base::Hx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/actions/contact-invdyn-with-thrusts.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DifferentialActionModelContactInvDynamicsWithThrustsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DifferentialActionDataContactInvDynamicsWithThrustsTpl)

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_WITH_THRUSTS_HPP_
