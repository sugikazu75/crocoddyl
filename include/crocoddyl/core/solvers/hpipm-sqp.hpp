///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_HPIPM_SQP_HPP_
#define CROCODDYL_CORE_SOLVERS_HPIPM_SQP_HPP_

#include "crocoddyl/core/solver-base.hpp"

#ifdef CROCODDYL_WITH_HPIPM
#include <memory>
#include <vector>

#include "hpipm-cpp/hpipm-cpp.hpp"

namespace crocoddyl {

/**
 * @brief HPIPM-based Sequential Quadratic Programming (SQP) solver
 *
 * At each outer SQP iteration this solver builds a structured OCP-QP from
 * Crocoddyl derivatives and solves it with HPIPM's interior-point method.
 * The outer globalisation (merit function, line search, regularisation) is
 * identical to SolverSQPTpl.
 *
 * HPIPM works in double precision only; Scalar data are cast to double on
 * entry and back on exit.
 */
template <typename _Scalar>
class SolverHpipmSQPTpl : public SolverAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverHpipmSQPTpl)

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
  using SolverAbstract::computeDynamicFeasibility;
  using SolverAbstract::computeEqualityFeasibility;
  using SolverAbstract::computeFeasibility;
  using SolverAbstract::computeInequalityFeasibility;
  using SolverAbstract::resizeData;

  explicit SolverHpipmSQPTpl(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverHpipmSQPTpl() = default;

  // unique_ptr member suppresses implicit copy/move — define them explicitly
  SolverHpipmSQPTpl(const SolverHpipmSQPTpl& other)
      : SolverHpipmSQPTpl(std::move(other.cast<Scalar>())) {}
  SolverHpipmSQPTpl(SolverHpipmSQPTpl&&) noexcept = default;
  SolverHpipmSQPTpl& operator=(const SolverHpipmSQPTpl&) = delete;
  SolverHpipmSQPTpl& operator=(SolverHpipmSQPTpl&&) = default;

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
  bool solveQuadraticModel();
  void extractQpDirection();

  template <typename NewScalar>
  SolverHpipmSQPTpl<NewScalar> cast() const;

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
  void set_upsilon_decfactor(const Scalar factor);
  void set_zero_upsilon(const bool zero_upsilon);

  const hpipm::OcpQpIpmSolverSettings& get_hpipm_settings() const;
  const hpipm::OcpQpIpmSolverStatistics& get_hpipm_statistics() const;
  void set_hpipm_settings(const hpipm::OcpQpIpmSolverSettings& settings);

 protected:
  void allocateData();
  virtual void resizeRunningData() override;
  virtual void resizeTerminalData() override;

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

  hpipm::OcpQpIpmSolverSettings hpipm_settings_;
  std::unique_ptr<hpipm::OcpQpIpmSolver> hpipm_solver_;
  std::vector<hpipm::OcpQp> ocp_qp_;
  std::vector<hpipm::OcpQpSolution> qp_sol_;

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

#include "crocoddyl/core/solvers/hpipm-sqp.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverHpipmSQPTpl)

#endif  // CROCODDYL_WITH_HPIPM
#endif  // CROCODDYL_CORE_SOLVERS_HPIPM_SQP_HPP_
