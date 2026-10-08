///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_STATES_MULTIBODY_WITH_THRUSTS_HPP_
#define CROCODDYL_MULTIBODY_STATES_MULTIBODY_WITH_THRUSTS_HPP_

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Augmented multibody state that includes current thruster forces.
 *
 * The state vector is laid out as  x = [q (nq), v (nv), f (nf)]  where
 * - q  : robot configuration (pinocchio convention, size nq)
 * - v  : generalized velocity in the tangent space of q (size nv)
 * - f  : current thruster forces (Euclidean, size nf)
 *
 * The corresponding tangent vector is  dx = [dq (nv), dv (nv), df (nf)]
 * so  nx  = nq + nv + nf  and  ndx = 2*nv + nf.
 *
 * The q/v part is handled by an internal StateMultibody; the f part is
 * treated as a plain Euclidean variable.
 *
 * \sa `StateAbstractTpl`, `StateMultibodyTpl`
 */
template <typename _Scalar>
class StateMultibodyWithThrustsTpl : public StateAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(StateBase, StateMultibodyWithThrustsTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct the augmented state.
   *
   * @param[in] state       Underlying multibody state (provides q/v operators)
   * @param[in] n_thrusters Number of thrusters (dimension of f)
   */
  StateMultibodyWithThrustsTpl(std::shared_ptr<StateMultibody> state,
                               std::size_t n_thrusters);
  virtual ~StateMultibodyWithThrustsTpl() = default;

  /** @brief Return a zero state  [q_neutral, 0_v, 0_f]. */
  virtual VectorXs zero() const override;

  /** @brief Return a random state. */
  virtual VectorXs rand() const override;

  /**
   * @brief State-manifold difference  dx = x1 ⊖ x0.
   *
   * The q-part uses pinocchio::difference; v and f parts are Euclidean.
   */
  virtual void diff(const Eigen::Ref<const VectorXs>& x0,
                    const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const override;

  /**
   * @brief State-manifold integration  xout = x ⊕ dx.
   *
   * The q-part uses pinocchio::integrate; v and f parts are Euclidean.
   */
  virtual void integrate(const Eigen::Ref<const VectorXs>& x,
                         const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const override;

  virtual void safe_diff(const Eigen::Ref<const VectorXs>& x0,
                         const Eigen::Ref<const VectorXs>& x1,
                         Eigen::Ref<VectorXs> dxout) const override;
  virtual void safe_integrate(const Eigen::Ref<const VectorXs>& x,
                              const Eigen::Ref<const VectorXs>& dx,
                              Eigen::Ref<VectorXs> xout) const override;

  /**
   * @brief Jacobian of diff w.r.t. x0 (first) and/or x1 (second).
   *
   * The result is block-diagonal:
   *   [ J_mb_q   0    0  ]  nv rows
   *   [   0     ±I   0   ]  nv rows
   *   [   0      0   ±I  ]  nf rows
   * where J_mb_q is the pinocchio dDifference block and ± is -1 for first,
   * +1 for second.
   */
  virtual void Jdiff(const Eigen::Ref<const VectorXs>& x0,
                     const Eigen::Ref<const VectorXs>& x1,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const Jcomponent firstsecond = both) const override;

  /**
   * @brief Jacobian of integrate w.r.t. x (first) and/or dx (second).
   *
   * The result is block-diagonal:
   *   [ J_pin    0    0  ]  nv rows
   *   [   0      I    0  ]  nv rows
   *   [   0      0    I  ]  nf rows
   */
  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst,
                          Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both,
                          const AssignmentOp op = setto) const override;

  /**
   * @brief Parallel transport Jin from the tangent space at x⊕dx to that at x.
   *
   * Applies pinocchio's dIntegrateTransport to the top nv rows (the
   * configuration-tangent part); the remaining nv+nf rows are Euclidean and
   * are left unchanged.
   */
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& x,
                                   const Eigen::Ref<const VectorXs>& dx,
                                   Eigen::Ref<MatrixXs> Jin,
                                   const Jcomponent firstsecond) const override;

  /** @brief Return the underlying multibody state. */
  const std::shared_ptr<StateMultibody>& get_state() const;

  /** @brief Return the Pinocchio model. */
  const std::shared_ptr<PinocchioModel>& get_pinocchio() const;

  /** @brief Return the number of thrusters (nf). */
  std::size_t get_nthrusters() const;

  template <typename NewScalar>
  StateMultibodyWithThrustsTpl<NewScalar> cast() const;

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::has_limits_;
  using Base::lb_;
  using Base::ndx_;
  using Base::nq_;
  using Base::nv_;
  using Base::nx_;
  using Base::ub_;

 private:
  std::shared_ptr<StateMultibody> state_;  //!< Underlying multibody state
  std::size_t n_thrusters_;                //!< Number of thrusters (nf)
  VectorXs x0_;                            //!< Zero state [q_neutral, 0, 0]
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/states/multibody-with-thrusts.hxx"

extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::StateMultibodyWithThrustsTpl<double>;
extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::StateMultibodyWithThrustsTpl<float>;

#endif  // CROCODDYL_MULTIBODY_STATES_MULTIBODY_WITH_THRUSTS_HPP_
