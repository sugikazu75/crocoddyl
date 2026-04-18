///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_OSQP_SQP_HPP_
#define CROCODDYL_CORE_SOLVERS_OSQP_SQP_HPP_

#include <algorithm>
#include <limits>

#include "crocoddyl/core/solvers/sqp.hpp"

#ifdef CROCODDYL_WITH_OSQP
#include <OsqpEigen/OsqpEigen.h>

#include <Eigen/Sparse>
#endif

namespace crocoddyl {

#ifdef CROCODDYL_WITH_OSQP
template <typename _Scalar>
class SolverOsqpSQPTpl : public SolverSQPTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverOsqpSQPTpl)

  typedef _Scalar Scalar;
  typedef SolverSQPTpl<Scalar> Base;
  typedef typename Base::ShootingProblem ShootingProblem;
  typedef typename Base::ActionModelAbstract ActionModelAbstract;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;
  typedef typename Base::SparseMatrixXs SparseMatrixXs;

  explicit SolverOsqpSQPTpl(std::shared_ptr<ShootingProblem> problem);
  SolverOsqpSQPTpl(const SolverOsqpSQPTpl& other);
  virtual ~SolverOsqpSQPTpl() = default;

  template <typename NewScalar>
  SolverOsqpSQPTpl<NewScalar> cast() const;

  void computeQuadraticModel() override;
  virtual bool solveQuadraticModel() override;

  bool get_osqp_verbose() const;
  int get_osqp_max_iter() const;
  Scalar get_osqp_eps_abs() const;
  Scalar get_osqp_eps_rel() const;
  Scalar get_osqp_rho() const;
  Scalar get_osqp_sigma() const;
  bool get_osqp_warm_start() const;

  void set_osqp_verbose(const bool verbose);
  void set_osqp_max_iter(const int max_iter);
  void set_osqp_eps_abs(const Scalar eps_abs);
  void set_osqp_eps_rel(const Scalar eps_rel);
  void set_osqp_rho(const Scalar rho);
  void set_osqp_sigma(const Scalar sigma);
  void set_osqp_warm_start(const bool warm_start);

 protected:
  void resetQpSolver();

  bool osqp_verbose_;
  int osqp_max_iter_;
  Scalar osqp_eps_abs_;
  Scalar osqp_eps_rel_;
  Scalar osqp_rho_;
  Scalar osqp_sigma_;
  bool osqp_warm_start_;
  bool osqp_initialized_;
  std::size_t osqp_n_;
  std::size_t osqp_m_;

  Eigen::SparseMatrix<double> qp_hessian_sparse_;
  Eigen::SparseMatrix<double> qp_constraint_sparse_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> qp_gradient_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> qp_lower_bound_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> qp_upper_bound_;
  OsqpEigen::Solver qp_solver_;
  VectorXs h_lb_;

  using Base::A_;
  using Base::b_;
  using Base::c_;
  using Base::du_max_;
  using Base::extractQpDirection;
  using Base::G_;
  using Base::h_;
  using Base::iter_;
  using Base::Luu_du_;
  using Base::Lxu_du_;
  using Base::Lxx_dx_;
  using Base::m_;
  using Base::n_;
  using Base::p_;
  using Base::preg_;
  using Base::problem_;
  using Base::Q_;
  using Base::qp_feas_tol_;
  using Base::updateStateAndControlIndex;
  using Base::us_;
  using Base::us_idx_;
  using Base::with_du_bounds_;
  using Base::x_;
  using Base::xs_;
  using Base::xs_idx_;
};
#endif  // CROCODDYL_WITH_OSQP

}  // namespace crocoddyl

#include "crocoddyl/core/solvers/osqp-sqp.hxx"

#ifdef CROCODDYL_WITH_OSQP
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverOsqpSQPTpl)
#endif

#endif  // CROCODDYL_CORE_SOLVERS_OSQP_SQP_HPP_
