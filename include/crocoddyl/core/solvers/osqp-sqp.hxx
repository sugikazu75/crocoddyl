///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

#ifdef CROCODDYL_WITH_OSQP

template <typename Scalar>
SolverOsqpSQPTpl<Scalar>::SolverOsqpSQPTpl(
    std::shared_ptr<ShootingProblem> problem)
    : Base(problem),
      osqp_verbose_(false),
      osqp_max_iter_(4000),
      osqp_eps_abs_(ScaleNumerics<Scalar>(1e-6)),
      osqp_eps_rel_(ScaleNumerics<Scalar>(1e-6)),
      osqp_rho_(Scalar(0.1)),
      osqp_sigma_(Scalar(1e-6)),
      osqp_warm_start_(true),
      osqp_initialized_(false),
      osqp_n_(0),
      osqp_m_(0) {}

template <typename Scalar>
SolverOsqpSQPTpl<Scalar>::SolverOsqpSQPTpl(
    const SolverOsqpSQPTpl<Scalar>& other)
    : Base(other),
      osqp_verbose_(other.osqp_verbose_),
      osqp_max_iter_(other.osqp_max_iter_),
      osqp_eps_abs_(other.osqp_eps_abs_),
      osqp_eps_rel_(other.osqp_eps_rel_),
      osqp_rho_(other.osqp_rho_),
      osqp_sigma_(other.osqp_sigma_),
      osqp_warm_start_(other.osqp_warm_start_),
      osqp_initialized_(false),
      osqp_n_(0),
      osqp_m_(0) {}

template <typename Scalar>
template <typename NewScalar>
SolverOsqpSQPTpl<NewScalar> SolverOsqpSQPTpl<Scalar>::cast() const {
  typedef SolverOsqpSQPTpl<NewScalar> ReturnType;
  typedef ShootingProblemTpl<NewScalar> ProblemType;
  ReturnType ret(std::make_shared<ProblemType>(
      this->problem_->template cast<NewScalar>()));

  ret.setCallbacks(vector_cast<NewScalar>(this->callbacks_));
  ret.set_th_acceptstep(scalar_cast<NewScalar>(this->th_acceptstep_));
  ret.set_th_gaptol(scalar_cast<NewScalar>(this->th_gaptol_));
  ret.set_feasnorm(this->feasnorm_);
  ret.set_th_stop(std::sqrt(std::numeric_limits<NewScalar>::epsilon()) <
                          NewScalar(this->th_stop_)
                      ? scalar_cast<NewScalar>(this->th_stop_)
                      : std::sqrt(std::numeric_limits<NewScalar>::epsilon()));
  ret.set_alphas(vector_cast<NewScalar>(this->alphas_));
  ret.set_reg_incfactor(scalar_cast<NewScalar>(this->reg_incfactor_));
  ret.set_reg_decfactor(scalar_cast<NewScalar>(this->reg_decfactor_));
  ret.set_th_grad(scalar_cast<NewScalar>(this->th_grad_));
  ret.set_th_noimprovement(scalar_cast<NewScalar>(this->th_noimprovement_));
  ret.set_th_stepdec(scalar_cast<NewScalar>(this->th_stepdec_));
  ret.set_th_stepinc(scalar_cast<NewScalar>(this->th_stepinc_));
  ret.set_th_minimprove(scalar_cast<NewScalar>(this->th_minimprove_));
  ret.set_th_acceptnegstep(scalar_cast<NewScalar>(this->th_acceptnegstep_));
  ret.set_th_acceptminstep(scalar_cast<NewScalar>(this->th_acceptminstep_));
  ret.set_rho(scalar_cast<NewScalar>(this->rho_));
  ret.set_th_minfeas(scalar_cast<NewScalar>(this->th_minfeas_));
  ret.set_upsilon_decfactor(scalar_cast<NewScalar>(this->upsilon_decfactor_));
  ret.set_zero_upsilon(this->zero_upsilon_);
  ret.set_qp_feas_tol(scalar_cast<NewScalar>(this->qp_feas_tol_));
  ret.set_qp_maxiters(this->qp_maxiters_);
  if (this->with_du_bounds_) {
    ret.set_control_variation_bounds(this->du_max_.template cast<NewScalar>());
  } else {
    ret.disable_control_variation_bounds();
  }
  ret.setCandidate(vector_cast<NewScalar>(this->xs_),
                   vector_cast<NewScalar>(this->us_), this->is_feasible_);

  ret.set_osqp_verbose(osqp_verbose_);
  ret.set_osqp_max_iter(osqp_max_iter_);
  ret.set_osqp_eps_abs(scalar_cast<NewScalar>(osqp_eps_abs_));
  ret.set_osqp_eps_rel(scalar_cast<NewScalar>(osqp_eps_rel_));
  ret.set_osqp_rho(scalar_cast<NewScalar>(osqp_rho_));
  ret.set_osqp_sigma(scalar_cast<NewScalar>(osqp_sigma_));
  ret.set_osqp_warm_start(osqp_warm_start_);
  return ret;
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::computeQuadraticModel() {
  START_PROFILER("SolverOsqpSQP::computeQuadraticModel");
  auto addBlock = [=](std::vector<Eigen::Triplet<Scalar> >& T, std::size_t i0,
                      std::size_t j0, const MatrixXs& M,
                      Scalar eps = Scalar(0)) {
    const std::size_t r = static_cast<std::size_t>(M.rows());
    const std::size_t c = static_cast<std::size_t>(M.cols());
    for (std::size_t j = 0; j < c; ++j) {
      for (std::size_t i = 0; i < r; ++i) {
        const Scalar v = M(i, j);
        if (v != eps) {
          T.emplace_back(i0 + i, j0 + j, v);
        }
      }
    }
  };
  auto addIdentity = [=](std::vector<Eigen::Triplet<Scalar> >& T,
                         std::size_t i0, std::size_t j0, std::size_t n,
                         Scalar scale = Scalar(1.0)) {
    T.reserve(T.size() + n);
    for (std::size_t k = 0; k < n; ++k) {
      T.emplace_back(i0 + k, j0 + k, scale);
    }
  };

  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();

  // Count QP dimensions for the OSQP-native bounded-inequality form:
  //   h_lb <= G z <= h_ub
  // (different from SQP base class that duplicates inequalities).
  std::size_t n_count = ndx;
  std::size_t m_count = ndx;
  std::size_t p_count = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    n_count += ndx + nu;
    m_count += ndx + model->get_nh();
    p_count += model->get_ng();
    if (nu > 0) {
      p_count += nu;
    }
    if (t > 0 && t < T - 1) {
      p_count += ndx;
    }
  }
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  const std::size_t ng_T = model_T->get_ng_T();
  n_ = n_count;
  m_ = m_count + model_T->get_nh_T();
  p_ = p_count + ng_T;
  if (with_du_bounds_ && T > 1) {
    for (std::size_t t = 0; t + 1 < T; ++t) {
      const std::size_t nu = models[t]->get_nu();
      const std::size_t nu_next = models[t + 1]->get_nu();
      if (nu > 0 && nu_next > 0) {
        if (nu != nu_next || static_cast<std::size_t>(du_max_.size()) != nu) {
          throw_pretty("Invalid control variation bounds dimensions.");
        }
        p_ += nu;
      }
    }
  }

  // Keep the same decision-variable ordering as SQP base.
  if (iter_) {
    updateStateAndControlIndex();
  }

  std::vector<Eigen::Triplet<Scalar> > q_triplets;
  std::vector<Eigen::Triplet<Scalar> > a_triplets;
  std::vector<Eigen::Triplet<Scalar> > g_triplets;
  q_triplets.reserve(8 * n_);
  a_triplets.reserve(8 * n_);
  g_triplets.reserve(12 * n_);

  c_.setZero(n_);
  b_.setZero(m_);
  h_.setZero(p_);
  h_lb_.setZero(p_);

  std::size_t eq_idx = 0;
  std::size_t ineq_idx = 0;

  // Initial gap equality: dx0 = f0
  addIdentity(a_triplets, eq_idx, xs_idx_[0], ndx, Scalar(1.0));
  b_.segment(eq_idx, ndx) = this->fs_[0];
  eq_idx += ndx;

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data =
        problem_->get_runningDatas()[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t ng = model->get_ng();
    const std::size_t xp_idx = xs_idx_[t + 1];
    const std::size_t x_idx = xs_idx_[t];
    const std::size_t u_idx = us_idx_[t];

    // Quadratic cost.
    addBlock(q_triplets, x_idx, x_idx, data->Lxx);
    c_.segment(x_idx, ndx) = data->Lx;
    if (nu > 0) {
      addBlock(q_triplets, u_idx, u_idx, data->Luu);
      addBlock(q_triplets, x_idx, u_idx, data->Lxu);
      addBlock(q_triplets, u_idx, x_idx, data->Lxu.transpose());
      c_.segment(u_idx, nu) = data->Lu;
    }

    // Dynamics equalities: Fx dx + Fu du - dxp = -f
    addBlock(a_triplets, eq_idx, x_idx, data->Fx);
    if (nu > 0) {
      addBlock(a_triplets, eq_idx, u_idx, data->Fu);
    }
    addIdentity(a_triplets, eq_idx, xp_idx, ndx, Scalar(-1.0));
    b_.segment(eq_idx, ndx) = -this->fs_[t + 1];
    eq_idx += ndx;

    // Path equalities: Hx dx + Hu du = -h
    if (nh > 0) {
      addBlock(a_triplets, eq_idx, x_idx, data->Hx);
      if (nu > 0) {
        addBlock(a_triplets, eq_idx, u_idx, data->Hu);
      }
      b_.segment(eq_idx, nh) = -data->h;
      eq_idx += nh;
    }

    // Path inequalities in one shot:
    // g_lb - g <= Gx dx + Gu du <= g_ub - g
    if (ng > 0) {
      addBlock(g_triplets, ineq_idx, x_idx, data->Gx);
      if (nu > 0) {
        addBlock(g_triplets, ineq_idx, u_idx, data->Gu);
      }
      h_lb_.segment(ineq_idx, ng) = model->get_g_lb() - data->g;
      h_.segment(ineq_idx, ng) = model->get_g_ub() - data->g;

      // std::cout << "add path inequalities at stage " << t << ": " << ng << "
      // constraints" << std::endl; for(int i = 0; i < ng; ++i) {
      //   if(h_lb_[ineq_idx + i] > h_[ineq_idx + i]) {
      //     std::cout << "Warning: infeasible lower bound for inequality
      //     constraint " << ineq_idx + i
      //               << " at stage " << t << ": h_lb = " << h_lb_[ineq_idx +
      //               i]
      //               << ", h_ub = " << h_[ineq_idx + i] << std::endl;
      //   }
      // }
      ineq_idx += ng;
    }

    // State bounds (internal nodes): x_lb - x <= dx <= x_ub - x
    if (t > 0 && t < T - 1) {
      addIdentity(g_triplets, ineq_idx, x_idx, ndx, Scalar(1));
      model->get_state()->safe_diff(xs_[t], model->get_state()->get_lb(),
                                    h_lb_.segment(ineq_idx, ndx));
      model->get_state()->safe_diff(xs_[t], model->get_state()->get_ub(),
                                    h_.segment(ineq_idx, ndx));
      // std::cout << "add state bounds at stage " << t << ": " << ndx << "
      // constraints" << std::endl; for(int i = 0; i < ndx; ++i) {
      //   if(h_lb_[ineq_idx + i] > h_[ineq_idx + i]) {
      //     std::cout << "Warning: infeasible lower bound for state constraint
      //     " << i << " at stage " << t << "ineq_idx: " << ineq_idx + i << "\n"
      //               << "xs_: " << xs_[t](i)
      //               << ", x_lb: " << model->get_state()->get_lb()(i)
      //               << ", x_ub: " << model->get_state()->get_ub()(i)
      //               << ", h_lb: " << h_lb_[ineq_idx + i]
      //               << ", h_ub: " << h_[ineq_idx + i] << std::endl;
      //     std::cout << std::endl;
      //     std::cout << std::endl;
      //   }
      // }
      ineq_idx += ndx;
    }

    // Control bounds: u_lb - u <= du <= u_ub - u
    if (nu > 0) {
      addIdentity(g_triplets, ineq_idx, u_idx, nu, Scalar(1));
      h_lb_.segment(ineq_idx, nu) = model->get_u_lb() - us_[t];
      h_.segment(ineq_idx, nu) = model->get_u_ub() - us_[t];

      // std::cout << "add control bounds at stage " << t << ": " << nu << "
      // constraints" << std::endl; for(int i = 0; i < nu; ++i) {
      //   if(h_lb_[ineq_idx + i] > h_[ineq_idx + i]) {

      //     std::cout << "Warning: infeasible lower bound for control
      //     constraint " << ineq_idx + i
      //               << " at stage " << t << ": h_lb = " << h_lb_[ineq_idx +
      //               i]
      //               << ", h_ub = " << h_[ineq_idx + i] << std::endl;
      //   }
      // }
      ineq_idx += nu;
    }
  }

  const std::shared_ptr<ActionDataAbstract>& data_T =
      problem_->get_terminalData();
  const std::size_t nh_T = model_T->get_nh_T();
  const std::size_t x_idx = xs_idx_[T];

  // Terminal cost and primal regularization.
  addBlock(q_triplets, x_idx, x_idx, data_T->Lxx);
  c_.segment(x_idx, ndx) = data_T->Lx;
  if (preg_ != Scalar(0)) {
    for (std::size_t i = 0; i < n_; ++i) {
      q_triplets.emplace_back(i, i, preg_);
    }
  }

  // Terminal equalities: Hx_T dx_T = -h_T
  if (nh_T > 0) {
    addBlock(a_triplets, eq_idx, x_idx, data_T->Hx);
    b_.segment(eq_idx, nh_T) = -data_T->h.head(nh_T);
    eq_idx += nh_T;
  }
  // Terminal inequalities: g_lb - g <= Gx dx <= g_ub - g
  if (ng_T > 0) {
    addBlock(g_triplets, ineq_idx, x_idx, data_T->Gx);
    h_lb_.segment(ineq_idx, ng_T) = model_T->get_g_lb() - data_T->g;
    h_.segment(ineq_idx, ng_T) = model_T->get_g_ub() - data_T->g;
    ineq_idx += ng_T;
  }

  // Optional control-variation bounds:
  //  -du_max <= du_{t+1} - du_t <= du_max
  if (with_du_bounds_) {
    for (std::size_t t = 0; t + 1 < T; ++t) {
      const std::size_t nu = models[t]->get_nu();
      const std::size_t nu_next = models[t + 1]->get_nu();
      if (nu == 0 || nu_next == 0) {
        continue;
      }
      const std::size_t u_idx = us_idx_[t];
      const std::size_t up_idx = us_idx_[t + 1];
      const VectorXs du_nom = us_[t + 1] - us_[t];
      for (std::size_t i = 0; i < nu; ++i) {
        g_triplets.emplace_back(ineq_idx + i, u_idx + i, Scalar(-1));
        g_triplets.emplace_back(ineq_idx + i, up_idx + i, Scalar(1));
      }
      h_lb_.segment(ineq_idx, nu) = -du_max_ - du_nom;
      h_.segment(ineq_idx, nu) = du_max_ - du_nom;
      ineq_idx += nu;
    }
  }

  // Safety check: assembled rows must match allocated dimensions.
  if (eq_idx != m_ || ineq_idx != p_) {
    throw_pretty("OSQP SQP QP shape mismatch.");
  }

  Q_.resize(n_, n_);
  A_.resize(m_, n_);
  G_.resize(p_, n_);
  Q_.setFromTriplets(q_triplets.begin(), q_triplets.end(),
                     [](const Scalar a, const Scalar b) { return a + b; });
  A_.setFromTriplets(a_triplets.begin(), a_triplets.end(),
                     [](const Scalar a, const Scalar b) { return a + b; });
  G_.setFromTriplets(g_triplets.begin(), g_triplets.end(),
                     [](const Scalar a, const Scalar b) { return a + b; });
  Q_.makeCompressed();
  A_.makeCompressed();
  G_.makeCompressed();
  STOP_PROFILER("SolverOsqpSQP::computeQuadraticModel");
}

template <typename Scalar>
bool SolverOsqpSQPTpl<Scalar>::solveQuadraticModel() {
  if (n_ == 0) {
    x_.resize(0);
    return true;
  }

  auto to_osqp_upper = [](const Scalar v) -> double {
    const double dv = static_cast<double>(v);
    if (std::isnan(dv)) return OsqpEigen::INFTY;
    if (std::isinf(dv)) return (dv > 0.) ? OsqpEigen::INFTY : -OsqpEigen::INFTY;
    if (dv > static_cast<double>(OsqpEigen::INFTY)) return OsqpEigen::INFTY;
    if (dv < -static_cast<double>(OsqpEigen::INFTY)) return -OsqpEigen::INFTY;
    return static_cast<double>(dv);
  };

  // Build OSQP form: min 0.5 x'P x + q'x s.t. l <= A x <= u.
  // Use the symmetrized local Hessian directly.
  typename Base::SparseMatrixXs Qt(Q_.transpose());
  typename Base::SparseMatrixXs Hsym(Q_);
  Hsym += Qt;
  Hsym *= Scalar(0.5);
  Eigen::SparseMatrix<double> P = Hsym.template cast<double>();
  qp_hessian_sparse_.resize(P.rows(), P.cols());
  qp_hessian_sparse_ = P.template triangularView<Eigen::Upper>();
  qp_hessian_sparse_.makeCompressed();

  qp_gradient_.resize(static_cast<Eigen::Index>(n_));
  for (std::size_t i = 0; i < n_; ++i) {
    qp_gradient_[static_cast<Eigen::Index>(i)] = static_cast<double>(c_[i]);
  }

  // Stack equality and inequality matrices:
  // A_osqp = [ Aeq ; G ],  l = [ b ; h_lb ],  u = [ b ; h_ub ].
  std::vector<Eigen::Triplet<double> > Atrip;
  Atrip.reserve(static_cast<std::size_t>(A_.nonZeros() + G_.nonZeros()));
  for (int k = 0; k < A_.outerSize(); ++k) {
    for (typename Base::SparseMatrixXs::InnerIterator it(A_, k); it; ++it) {
      Atrip.emplace_back(it.row(), it.col(), static_cast<double>(it.value()));
    }
  }
  for (int k = 0; k < G_.outerSize(); ++k) {
    for (typename Base::SparseMatrixXs::InnerIterator it(G_, k); it; ++it) {
      Atrip.emplace_back(static_cast<int>(m_) + it.row(), it.col(),
                         static_cast<double>(it.value()));
    }
  }
  qp_constraint_sparse_.resize(static_cast<int>(m_ + p_), static_cast<int>(n_));
  qp_constraint_sparse_.setFromTriplets(Atrip.begin(), Atrip.end());
  qp_constraint_sparse_.makeCompressed();

  qp_lower_bound_.resize(static_cast<Eigen::Index>(m_ + p_));
  qp_upper_bound_.resize(static_cast<Eigen::Index>(m_ + p_));
  for (std::size_t i = 0; i < m_; ++i) {
    const Scalar bi_s = b_[i];
    if (!std::isfinite(static_cast<double>(bi_s))) {
      throw_pretty("Invalid equality bound b_ at row " << i);
    }
    const double bi = static_cast<double>(bi_s);
    qp_lower_bound_[static_cast<Eigen::Index>(i)] = bi;
    qp_upper_bound_[static_cast<Eigen::Index>(i)] = bi;
  }
  for (std::size_t i = 0; i < p_; ++i) {
    qp_lower_bound_[static_cast<Eigen::Index>(m_ + i)] =
        to_osqp_upper(h_lb_[i]);
    qp_upper_bound_[static_cast<Eigen::Index>(m_ + i)] = to_osqp_upper(h_[i]);
    if (qp_lower_bound_[static_cast<Eigen::Index>(m_ + i)] >
        qp_upper_bound_[static_cast<Eigen::Index>(m_ + i)]) {
      throw_pretty("Inconsistent inequality bounds at row "
                   << i << ": lower bound "
                   << qp_lower_bound_[static_cast<Eigen::Index>(m_ + i)]
                   << " > upper bound "
                   << qp_upper_bound_[static_cast<Eigen::Index>(m_ + i)]);
    }
  }

  const bool shape_changed =
      !osqp_initialized_ || osqp_n_ != n_ || osqp_m_ != (m_ + p_);
  auto initialize_solver = [&]() -> bool {
    resetQpSolver();
    qp_solver_.data()->clearHessianMatrix();
    qp_solver_.data()->clearLinearConstraintsMatrix();
    qp_solver_.settings()->setWarmStart(osqp_warm_start_);
    qp_solver_.settings()->setVerbosity(osqp_verbose_);
    qp_solver_.settings()->setMaxIteraction(std::max(1, osqp_max_iter_));
    qp_solver_.settings()->setAbsoluteTolerance(
        static_cast<double>(osqp_eps_abs_));
    qp_solver_.settings()->setRelativeTolerance(
        static_cast<double>(osqp_eps_rel_));
    qp_solver_.settings()->setRho(static_cast<double>(osqp_rho_));
    qp_solver_.settings()->setSigma(static_cast<double>(osqp_sigma_));
    qp_solver_.data()->setNumberOfVariables(static_cast<int>(n_));
    qp_solver_.data()->setNumberOfConstraints(static_cast<int>(m_ + p_));
    bool ok_init = true;
    ok_init &= qp_solver_.data()->setHessianMatrix(qp_hessian_sparse_);
    ok_init &= qp_solver_.data()->setGradient(qp_gradient_);
    ok_init &=
        qp_solver_.data()->setLinearConstraintsMatrix(qp_constraint_sparse_);
    ok_init &= qp_solver_.data()->setLowerBound(qp_lower_bound_);
    ok_init &= qp_solver_.data()->setUpperBound(qp_upper_bound_);
    ok_init &= qp_solver_.initSolver();
    std::cout << "OSQP solver initialization: "
              << (ok_init ? "success" : "failure") << std::endl;
    osqp_initialized_ = ok_init;
    osqp_n_ = ok_init ? n_ : 0;
    osqp_m_ = ok_init ? (m_ + p_) : 0;
    return ok_init;
  };

  bool ok = true;
  if (shape_changed) {
    ok = initialize_solver();
  } else {
    ok &= qp_solver_.updateHessianMatrix(qp_hessian_sparse_);
    ok &= qp_solver_.updateGradient(qp_gradient_);
    ok &= qp_solver_.updateLinearConstraintsMatrix(qp_constraint_sparse_);
    ok &= qp_solver_.updateBounds(qp_lower_bound_, qp_upper_bound_);
    if (!ok) {
      // Fallback: if incremental update fails, rebuild OSQP from scratch.
      ok = initialize_solver();
    }
  }

  ok &= qp_solver_.solve();
  if (!ok) {
    // Second fallback: rebuild and solve once again when OSQP solve fails.
    if (!initialize_solver()) {
      resetQpSolver();
      return false;
    }
    ok = qp_solver_.solve();
    if (!ok) {
      resetQpSolver();
      return false;
    }
  }

  const Eigen::VectorXd& sol = qp_solver_.getSolution();
  if (!sol.allFinite()) {
    resetQpSolver();
    return false;
  }

  x_.resize(n_);
  for (std::size_t i = 0; i < n_; ++i) {
    x_[i] = static_cast<Scalar>(sol[static_cast<Eigen::Index>(i)]);
  }
  return true;
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::resetQpSolver() {
  std::cout << "Resetting OSQP solver..." << std::endl;
  qp_solver_.clearSolver();
  qp_solver_.data()->clearHessianMatrix();
  qp_solver_.data()->clearLinearConstraintsMatrix();
  osqp_initialized_ = false;
  osqp_n_ = 0;
  osqp_m_ = 0;
}

template <typename Scalar>
bool SolverOsqpSQPTpl<Scalar>::get_osqp_verbose() const {
  return osqp_verbose_;
}

template <typename Scalar>
int SolverOsqpSQPTpl<Scalar>::get_osqp_max_iter() const {
  return osqp_max_iter_;
}

template <typename Scalar>
Scalar SolverOsqpSQPTpl<Scalar>::get_osqp_eps_abs() const {
  return osqp_eps_abs_;
}

template <typename Scalar>
Scalar SolverOsqpSQPTpl<Scalar>::get_osqp_eps_rel() const {
  return osqp_eps_rel_;
}

template <typename Scalar>
Scalar SolverOsqpSQPTpl<Scalar>::get_osqp_rho() const {
  return osqp_rho_;
}

template <typename Scalar>
Scalar SolverOsqpSQPTpl<Scalar>::get_osqp_sigma() const {
  return osqp_sigma_;
}

template <typename Scalar>
bool SolverOsqpSQPTpl<Scalar>::get_osqp_warm_start() const {
  return osqp_warm_start_;
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::set_osqp_verbose(const bool v) {
  osqp_verbose_ = v;
  resetQpSolver();
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::set_osqp_max_iter(const int v) {
  if (v <= 0) throw_pretty("osqp max_iter > 0.");
  osqp_max_iter_ = v;
  resetQpSolver();
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::set_osqp_eps_abs(const Scalar v) {
  if (v <= Scalar(0.)) throw_pretty("osqp eps_abs > 0.");
  osqp_eps_abs_ = v;
  resetQpSolver();
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::set_osqp_eps_rel(const Scalar v) {
  if (v <= Scalar(0.)) throw_pretty("osqp eps_rel > 0.");
  osqp_eps_rel_ = v;
  resetQpSolver();
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::set_osqp_rho(const Scalar v) {
  if (v <= Scalar(0.)) throw_pretty("osqp rho > 0.");
  osqp_rho_ = v;
  resetQpSolver();
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::set_osqp_sigma(const Scalar v) {
  if (v <= Scalar(0.)) throw_pretty("osqp sigma > 0.");
  osqp_sigma_ = v;
  resetQpSolver();
}

template <typename Scalar>
void SolverOsqpSQPTpl<Scalar>::set_osqp_warm_start(const bool v) {
  osqp_warm_start_ = v;
  resetQpSolver();
}

#endif  // CROCODDYL_WITH_OSQP

}  // namespace crocoddyl
