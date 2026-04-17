///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_SQP_HPP_
#define CROCODDYL_CORE_SOLVERS_SQP_HPP_

#include <Eigen/Sparse>
#include <algorithm>
#include <limits>

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

/**
 * @brief Sequential Quadratic Programming (SQP) solver
 *
 * This implementation is backend independent (no odyn dependency). At each
 * iteration, it builds a sparse local QP from Crocoddyl derivatives and solves
 * it using a primal active-set strategy with sparse KKT systems.
 */
template <typename _Scalar>
class SolverSQPTpl : public SolverAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverSQPTpl)

  typedef _Scalar Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef CallbackAbstractTpl<Scalar> CallbackAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef Eigen::SparseMatrix<Scalar> SparseMatrixXs;
  using SolverAbstract::computeDynamicFeasibility;
  using SolverAbstract::computeEqualityFeasibility;
  using SolverAbstract::computeFeasibility;
  using SolverAbstract::computeInequalityFeasibility;
  using SolverAbstract::resizeData;

  explicit SolverSQPTpl(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverSQPTpl() = default;

  virtual void computeDirection(const bool recalc = true) override;
  virtual Scalar stoppingCriteria() override;
  virtual Vector3s expectedImprovement() override;
  virtual void computeMeritFunctionImprovement() override;
  virtual void computeExpectedMeritFunctionImprovement() override;
  virtual void updateMeritFunction() override;
  virtual bool checkAcceptance() override;
  virtual void computeCandidate(const Scalar step_length = Scalar(1.)) override;
  void updateCandidate() override;
  bool decreaseRegularizationCriteria() override;
  bool increaseRegularizationCriteria() override;
  void increaseRegularization() override;
  void decreaseRegularization() override;

  void computeQuadraticModel();
  virtual bool solveQuadraticModel();
  void extractQpDirection(const VectorXs& x);

  template <typename NewScalar>
  SolverSQPTpl<NewScalar> cast() const;

  std::size_t get_n() const noexcept;
  std::size_t get_m() const noexcept;
  std::size_t get_p() const noexcept;

  Scalar get_reg_incfactor() const;
  Scalar get_reg_decfactor() const;
  Scalar get_th_grad() const;
  Scalar get_th_stepdec() const;
  Scalar get_th_stepinc() const;
  Scalar get_th_minimprove() const;
  Scalar get_th_acceptnegstep() const;
  Scalar get_th_acceptminstep() const;
  Scalar get_rho() const;
  Scalar get_th_minfeas() const;
  Scalar get_upsilon() const;
  Scalar get_upsilon_decfactor() const;
  bool get_zero_upsilon() const;
  Scalar get_qp_feas_tol() const;
  std::size_t get_qp_maxiters() const;
  bool get_with_control_variation_bounds() const;
  const VectorXs& get_du_max() const;

  void set_reg_incfactor(const Scalar reg_factor);
  void set_reg_decfactor(const Scalar reg_factor);
  void set_th_grad(const Scalar th_grad);
  void set_th_noimprovement(const Scalar th_noimprovement);
  void set_th_stepdec(const Scalar th_step);
  void set_th_stepinc(const Scalar th_step);
  void set_th_minimprove(const Scalar th_step);
  void set_th_acceptnegstep(const Scalar th_acceptnegstep);
  void set_th_acceptminstep(const Scalar th_acceptminstep);
  void set_rho(const Scalar rho);
  void set_th_minfeas(const Scalar th_minfeas);
  void set_upsilon_decfactor(const Scalar th_step);
  void set_zero_upsilon(const bool zero_upsilon);
  void set_qp_feas_tol(const Scalar tol);
  void set_qp_maxiters(const std::size_t maxiters);
  void set_control_variation_bounds(const VectorXs& du_max);
  void disable_control_variation_bounds();

 protected:
  void allocateData();
  virtual void resizeRunningData() override;
  virtual void resizeTerminalData() override;
  void updateStateAndControlIndex();

  Scalar reg_incfactor_;
  Scalar reg_decfactor_;
  Scalar th_grad_;
  Scalar th_noimprovement_;
  Scalar th_stepdec_;
  Scalar th_stepinc_;
  Scalar th_minimprove_;
  Scalar th_acceptnegstep_;
  Scalar th_acceptminstep_;
  Scalar rho_;
  Scalar th_minfeas_;
  Scalar upsilon_;
  Scalar upsilon_decfactor_;
  bool zero_upsilon_;
  Scalar qp_feas_tol_;
  std::size_t qp_maxiters_;
  bool with_du_bounds_;
  VectorXs du_max_;

  std::size_t n_;
  std::size_t m_;
  std::size_t p_;
  VectorXs x_;
  SparseMatrixXs Q_;
  SparseMatrixXs A_;
  SparseMatrixXs G_;
  VectorXs c_;
  VectorXs b_;
  VectorXs h_;
  std::vector<std::size_t> xs_idx_;
  std::vector<std::size_t> us_idx_;
  std::vector<VectorXs> Lxx_dx_;
  std::vector<VectorXs> Luu_du_;
  std::vector<VectorXs> Lxu_du_;

  using SolverAbstract::acceptstep_;
  using SolverAbstract::alphas_;
  using SolverAbstract::callbacks_;
  using SolverAbstract::cost_;
  using SolverAbstract::cost_try_;
  using SolverAbstract::dfeas_;
  using SolverAbstract::dImpr_;
  using SolverAbstract::dPhi_;
  using SolverAbstract::dPhiexp_;
  using SolverAbstract::dreg_;
  using SolverAbstract::dus_;
  using SolverAbstract::DV_;
  using SolverAbstract::dV_;
  using SolverAbstract::dVexp_;
  using SolverAbstract::dVexp_full_;
  using SolverAbstract::dxs_;
  using SolverAbstract::feas_;
  using SolverAbstract::feasnorm_;
  using SolverAbstract::ffeas_;
  using SolverAbstract::ffeas_try_;
  using SolverAbstract::fs_;
  using SolverAbstract::fs_try_;
  using SolverAbstract::gfeas_;
  using SolverAbstract::gfeas_try_;
  using SolverAbstract::hfeas_;
  using SolverAbstract::hfeas_try_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::merit_;
  using SolverAbstract::ng_T_;
  using SolverAbstract::nh_T_;
  using SolverAbstract::preg_;
  using SolverAbstract::problem_;
  using SolverAbstract::reg_max_;
  using SolverAbstract::reg_min_;
  using SolverAbstract::steplength_;
  using SolverAbstract::stop_;
  using SolverAbstract::th_acceptstep_;
  using SolverAbstract::th_gaptol_;
  using SolverAbstract::th_stop_;
  using SolverAbstract::us_;
  using SolverAbstract::us_try_;
  using SolverAbstract::xs_;
  using SolverAbstract::xs_try_;
};

}  // namespace crocoddyl

#include "crocoddyl/core/solvers/sqp.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverSQPTpl)

#endif  // CROCODDYL_CORE_SOLVERS_SQP_HPP_
