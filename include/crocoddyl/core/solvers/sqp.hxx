///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
SolverSQPTpl<Scalar>::SolverSQPTpl(std::shared_ptr<ShootingProblem> problem)
    : SolverAbstract(problem),
      reg_incfactor_(Scalar(10.)),
      reg_decfactor_(Scalar(5.)),
      th_grad_(ScaleNumerics<Scalar>(1e-12)),
      th_noimprovement_(
          std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(0.8))),
      th_stepdec_(Scalar(0.25)),
      th_stepinc_(Scalar(0.25)),
      th_minimprove_(Scalar(1e-2)),
      th_acceptnegstep_(Scalar(8)),
      th_acceptminstep_(Scalar(0.01)),
      rho_(Scalar(0.3)),
      th_minfeas_(std::sqrt(std::numeric_limits<Scalar>::epsilon() /
                            (Scalar(1.) - rho_))),
      upsilon_(Scalar(0.)),
      upsilon_decfactor_(Scalar(0.5)),
      zero_upsilon_(false),
      qp_feas_tol_(ScaleNumerics<Scalar>(1e-9)),
      qp_maxiters_(50),
      with_du_bounds_(false) {
  allocateData();
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::computeDirection(const bool recalc) {
  START_PROFILER("SolverSQP::computeDirection");
  if (recalc) {
    SolverAbstract::calcDir();
  }
  computeQuadraticModel();
  if (!solveQuadraticModel()) {
    x_.setZero();
  }
  extractQpDirection(x_);
  STOP_PROFILER("SolverSQP::computeDirection");
}

template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::stoppingCriteria() {
  feas_ = ffeas_ + gfeas_ + hfeas_;
  stop_ =
      std::max(feas_, std::abs(dVexp_full_) / (Scalar(1.) + std::abs(cost_)));
  return stop_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::Vector3s
SolverSQPTpl<Scalar>::expectedImprovement() {
  const std::size_t T = problem_->get_T();
  DV_.setZero();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    Lxx_dx_[t].noalias() = d->Lxx * dxs_[t];
    Luu_du_[t].noalias() = d->Luu * dus_[t];
    Lxu_du_[t].noalias() = d->Lxu * dus_[t];
    DV_[1] -= dxs_[t].dot(d->Lx);
    DV_[1] -= dus_[t].dot(d->Lu);
    DV_[2] -= dxs_[t].dot(Lxx_dx_[t]);
    DV_[2] -= dus_[t].dot(Luu_du_[t]);
    DV_[2] -= Scalar(2.) * dxs_[t].dot(Lxu_du_[t]);
  }
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  Lxx_dx_.back().noalias() = d->Lxx * dxs_.back();
  DV_[1] -= dxs_.back().dot(d->Lx);
  DV_[2] -= dxs_.back().dot(Lxx_dx_.back());
  return DV_;
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::computeMeritFunctionImprovement() {
  dPhi_ = dV_ + upsilon_ * dfeas_;
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::computeExpectedMeritFunctionImprovement() {
  dPhiexp_ = dVexp_ + steplength_ * upsilon_ * dfeas_;
}

template <typename Scalar>
bool SolverSQPTpl<Scalar>::checkAcceptance() {
  acceptstep_ = false;
  if ((std::abs(dPhi_) <= th_noimprovement_) &&
      (std::abs(dPhiexp_) <= th_noimprovement_)) {
    acceptstep_ = true;
  } else if (dPhiexp_ >= Scalar(0.)) {
    if (dPhi_ > Scalar(0.)) {
      if (dPhi_ > th_acceptstep_ * dPhiexp_ || std::abs(DV_[1]) < th_grad_) {
        acceptstep_ = true;
      }
    } else if (dV_ > th_acceptstep_ * dVexp_ || std::abs(DV_[1]) < th_grad_) {
      acceptstep_ = true;
    }
  } else {
    if (feas_ <= th_stop_) {
      if (dPhi_ > th_acceptnegstep_ * dPhiexp_) {
        acceptstep_ = true;
      }
    } else if (dV_ > th_acceptnegstep_ * dVexp_) {
      acceptstep_ = true;
    }
  }
  if (steplength_ <= th_acceptminstep_ && dImpr_ > Scalar(0.)) {
    acceptstep_ = true;
  }
  return acceptstep_;
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::updateMeritFunction() {
  if (iter_ == 0 && zero_upsilon_) {
    upsilon_ = 0.;
  }
  if (feas_ >= th_minfeas_) {
    upsilon_ = std::max(upsilon_ * upsilon_decfactor_,
                        dVexp_full_ / ((Scalar(1.) - rho_) * feas_));
  }
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::computeCandidate(const Scalar steplength) {
  START_PROFILER("SolverSQP::computeCandidate");
  if (steplength > Scalar(1.) || steplength < Scalar(0.)) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();

  models[0]->get_state()->integrate(xs_[0], steplength * dxs_[0], xs_try_[0]);
  fs_try_[0] = fs_[0] * (1 - steplength);
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    m->get_state()->integrate(xs_[t + 1], steplength * dxs_[t + 1],
                              xs_try_[t + 1]);
  }
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    if (m->get_nu() != 0) {
      us_try_[t] = us_[t] + steplength * dus_[t];
      m->calc(d, xs_try_[t], us_try_[t]);
    } else {
      m->calc(d, xs_try_[t]);
    }
    m->get_state()->diff(xs_try_[t + 1], d->xnext, fs_try_[t + 1]);
  }
  cost_try_ = Scalar(0.);
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    cost_try_ += d->cost;
    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverSQP::computeCandidate");
      throw_pretty("computeCandidate");
    }
  }
  const std::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;
  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverSQP::computeCandidate");
    throw_pretty("computeCandidate");
  }
  STOP_PROFILER("SolverSQP::computeCandidate");
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::computeQuadraticModel() {
  START_PROFILER("SolverSQP::computeQuadraticModel");
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
  if (iter_) {
    updateStateAndControlIndex();
  }
  std::vector<Eigen::Triplet<Scalar> > q_triplets;
  std::vector<Eigen::Triplet<Scalar> > a_triplets;
  std::vector<Eigen::Triplet<Scalar> > g_triplets;
  q_triplets.reserve(8 * n_);
  a_triplets.reserve(8 * n_);
  g_triplets.reserve(16 * n_);

  c_.setZero(n_);
  b_.setZero(m_);
  h_.setZero(p_);

  const std::size_t ndx = problem_->get_ndx();
  std::size_t eq_idx = 0;
  std::size_t ineq_idx = 0;
  // Initial gap equality: dx0 = f0
  addIdentity(a_triplets, eq_idx, xs_idx_[0], ndx, Scalar(1.0));
  b_.segment(eq_idx, ndx) = fs_[0];
  eq_idx += ndx;

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model =
        problem_->get_runningModels()[t];
    const std::shared_ptr<ActionDataAbstract>& data =
        problem_->get_runningDatas()[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t ng = model->get_ng();
    const std::size_t xp_idx = xs_idx_[t + 1];
    const std::size_t x_idx = xs_idx_[t];
    const std::size_t u_idx = us_idx_[t];

    // Quadratic cost terms
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
    b_.segment(eq_idx, ndx) = -fs_[t + 1];
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

    // Path inequalities: g_lb - g <= Gx dx + Gu du <= g_ub - g
    if (ng > 0) {
      // Upper side: Gx dx + Gu du <= g_ub - g
      addBlock(g_triplets, ineq_idx, x_idx, data->Gx);
      if (nu > 0) {
        addBlock(g_triplets, ineq_idx, u_idx, data->Gu);
      }
      h_.segment(ineq_idx, ng) = model->get_g_ub() - data->g;
      ineq_idx += ng;

      // Lower side: -Gx dx - Gu du <= g - g_lb
      addBlock(g_triplets, ineq_idx, x_idx, -data->Gx);
      if (nu > 0) {
        addBlock(g_triplets, ineq_idx, u_idx, -data->Gu);
      }
      h_.segment(ineq_idx, ng) = data->g - model->get_g_lb();
      ineq_idx += ng;
    }

    // State bounds: x_lb - x <= dx <= x_ub - x (internal nodes only)
    if (t > 0 && t < T - 1) {
      // Upper bound: dx <= x_ub - x
      addIdentity(g_triplets, ineq_idx, x_idx, ndx, Scalar(1));
      model->get_state()->safe_diff(xs_[t], model->get_state()->get_ub(),
                                    h_.segment(ineq_idx, ndx));
      ineq_idx += ndx;

      // Lower bound: -dx <= x - x_lb
      addIdentity(g_triplets, ineq_idx, x_idx, ndx, Scalar(-1));
      model->get_state()->safe_diff(model->get_state()->get_lb(), xs_[t],
                                    h_.segment(ineq_idx, ndx));
      ineq_idx += ndx;
    }

    // Control bounds: u_lb - u <= du <= u_ub - u
    if (nu > 0) {
      // Upper bound: du <= u_ub - u
      addIdentity(g_triplets, ineq_idx, u_idx, nu, Scalar(1));
      h_.segment(ineq_idx, nu) = model->get_u_ub() - us_[t];
      ineq_idx += nu;

      // Lower bound: -du <= u - u_lb
      addIdentity(g_triplets, ineq_idx, u_idx, nu, Scalar(-1));
      h_.segment(ineq_idx, nu) = us_[t] - model->get_u_lb();
      ineq_idx += nu;
    }
  }

  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& data_T =
      problem_->get_terminalData();
  const std::size_t nh_T = model_T->get_nh_T();
  const std::size_t ng_T = model_T->get_ng_T();
  const std::size_t x_idx = xs_idx_[T];

  // Terminal cost and regularization
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
    // Upper side: Gx dx <= g_ub - g
    addBlock(g_triplets, ineq_idx, x_idx, data_T->Gx);
    h_.segment(ineq_idx, ng_T) = model_T->get_g_ub() - data_T->g;
    ineq_idx += ng_T;

    // Lower side: -Gx dx <= g - g_lb
    addBlock(g_triplets, ineq_idx, x_idx, -data_T->Gx);
    h_.segment(ineq_idx, ng_T) = data_T->g - model_T->get_g_lb();
    ineq_idx += ng_T;
  }
  // Control-variation bounds: -du_max <= du_{t+1} - du_t <= du_max
  if (with_du_bounds_) {
    for (std::size_t t = 0; t + 1 < T; ++t) {
      const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
      const std::size_t nu_next =
          problem_->get_runningModels()[t + 1]->get_nu();
      if (nu == 0 || nu_next == 0) {
        continue;
      }
      if (nu != nu_next || static_cast<std::size_t>(du_max_.size()) != nu) {
        throw_pretty("Invalid control variation bounds dimensions.");
      }
      const std::size_t u_idx = us_idx_[t];
      const std::size_t up_idx = us_idx_[t + 1];
      const VectorXs du_nom = us_[t + 1] - us_[t];

      // Upper side: du_{t+1} - du_t <= du_max - (u_{t+1} - u_t)
      for (std::size_t i = 0; i < nu; ++i) {
        g_triplets.emplace_back(ineq_idx + i, u_idx + i, Scalar(-1));
        g_triplets.emplace_back(ineq_idx + i, up_idx + i, Scalar(1));
      }
      h_.segment(ineq_idx, nu) = du_max_ - du_nom;
      ineq_idx += nu;

      // Lower side: -(du_{t+1} - du_t) <= du_max + (u_{t+1} - u_t)
      for (std::size_t i = 0; i < nu; ++i) {
        g_triplets.emplace_back(ineq_idx + i, u_idx + i, Scalar(1));
        g_triplets.emplace_back(ineq_idx + i, up_idx + i, Scalar(-1));
      }
      h_.segment(ineq_idx, nu) = du_max_ + du_nom;
      ineq_idx += nu;
    }
  }

  // Finalize sparse matrices
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
  STOP_PROFILER("SolverSQP::computeQuadraticModel");
}

template <typename Scalar>
bool SolverSQPTpl<Scalar>::solveQuadraticModel() {
  typedef Eigen::Triplet<Scalar> Triplet;
  typedef Eigen::SparseMatrix<Scalar> SparseMatrixXs;
  x_.setZero(n_);
  if (n_ == 0) {
    return true;
  }
  std::vector<std::size_t> active;
  active.reserve(p_);
  const Scalar tol = qp_feas_tol_;
  const std::size_t maxit = std::max<std::size_t>(10, qp_maxiters_);

  SparseMatrixXs H = Q_;
  {
    SparseMatrixXs Hreg(n_, n_);
    std::vector<Triplet> hreg_triplets;
    const Scalar reg = std::max(preg_, ScaleNumerics<Scalar>(1e-9));
    hreg_triplets.reserve(n_);
    for (std::size_t i = 0; i < n_; ++i) {
      hreg_triplets.emplace_back(static_cast<Eigen::Index>(i),
                                 static_cast<Eigen::Index>(i), reg);
    }
    Hreg.setFromTriplets(hreg_triplets.begin(), hreg_triplets.end());
    H += Hreg;
  }

  for (std::size_t it = 0; it < maxit; ++it) {
    const std::size_t ma = m_ + active.size();
    SparseMatrixXs Aeq(ma, n_);
    std::vector<Triplet> aeq_triplets;
    aeq_triplets.reserve(A_.nonZeros() + active.size() * 16);
    VectorXs beq(ma);
    if (m_ > 0u) {
      for (int k = 0; k < A_.outerSize(); ++k) {
        for (typename SparseMatrixXs::InnerIterator itA(A_, k); itA; ++itA) {
          aeq_triplets.emplace_back(itA.row(), itA.col(), itA.value());
        }
      }
      beq.head(m_) = b_;
    }

    std::vector<int> row_to_active(p_, -1);
    for (std::size_t i = 0; i < active.size(); ++i) {
      row_to_active[active[i]] = static_cast<int>(i);
      beq[m_ + i] = h_[active[i]];
    }
    for (int k = 0; k < G_.outerSize(); ++k) {
      for (typename SparseMatrixXs::InnerIterator itG(G_, k); itG; ++itG) {
        const int active_row =
            row_to_active[static_cast<std::size_t>(itG.row())];
        if (active_row >= 0) {
          aeq_triplets.emplace_back(
              static_cast<Eigen::Index>(m_ +
                                        static_cast<std::size_t>(active_row)),
              itG.col(), itG.value());
        }
      }
    }
    Aeq.setFromTriplets(aeq_triplets.begin(), aeq_triplets.end(),
                        [](const Scalar a, const Scalar b) { return a + b; });

    SparseMatrixXs KKT(n_ + ma, n_ + ma);
    std::vector<Triplet> kkt_triplets;
    kkt_triplets.reserve(H.nonZeros() + 2 * Aeq.nonZeros());
    for (int k = 0; k < H.outerSize(); ++k) {
      for (typename SparseMatrixXs::InnerIterator itH(H, k); itH; ++itH) {
        kkt_triplets.emplace_back(itH.row(), itH.col(), itH.value());
      }
    }
    if (ma > 0u) {
      for (int k = 0; k < Aeq.outerSize(); ++k) {
        for (typename SparseMatrixXs::InnerIterator itAeq(Aeq, k); itAeq;
             ++itAeq) {
          const Eigen::Index r = itAeq.row();
          const Eigen::Index c = itAeq.col();
          const Scalar v = itAeq.value();
          kkt_triplets.emplace_back(c, static_cast<Eigen::Index>(n_) + r, v);
          kkt_triplets.emplace_back(static_cast<Eigen::Index>(n_) + r, c, v);
        }
      }
    }
    KKT.setFromTriplets(kkt_triplets.begin(), kkt_triplets.end(),
                        [](const Scalar a, const Scalar b) { return a + b; });
    VectorXs rhs(n_ + ma);
    rhs.head(n_) = -c_;
    if (ma > 0u) {
      rhs.tail(ma) = beq;
    }

    Eigen::SparseLU<SparseMatrixXs> kkt_solver;
    kkt_solver.analyzePattern(KKT);
    kkt_solver.factorize(KKT);
    if (kkt_solver.info() != Eigen::Success) {
      return false;
    }
    VectorXs sol = kkt_solver.solve(rhs);
    if (kkt_solver.info() != Eigen::Success) {
      return false;
    }
    if (!sol.allFinite()) {
      return false;
    }
    x_ = sol.head(n_);

    VectorXs viol = G_ * x_ - h_;
    Scalar max_viol = Scalar(0);
    std::size_t max_idx = 0;
    for (std::size_t i = 0; i < p_; ++i) {
      if (viol[i] > max_viol) {
        max_viol = viol[i];
        max_idx = i;
      }
    }
    if (max_viol > tol) {
      if (std::find(active.begin(), active.end(), max_idx) == active.end()) {
        active.push_back(max_idx);
      }
      continue;
    }

    bool removed = false;
    if (!active.empty()) {
      const VectorXs lambda = sol.tail(ma);
      Scalar min_lambda = Scalar(0);
      std::size_t remove_i = 0;
      for (std::size_t i = 0; i < active.size(); ++i) {
        const Scalar li = lambda[m_ + i];
        if (li < min_lambda) {
          min_lambda = li;
          remove_i = i;
        }
      }
      if (min_lambda < -tol) {
        active.erase(active.begin() + static_cast<long>(remove_i));
        removed = true;
      }
    }
    if (!removed) {
      return true;
    }
  }

  SparseMatrixXs M = H + A_.transpose() * A_ + G_.transpose() * G_;
  VectorXs r = -c_ + A_.transpose() * b_ + G_.transpose() * h_;
  Eigen::SimplicialLDLT<SparseMatrixXs> spd_solver;
  spd_solver.compute(M);
  if (spd_solver.info() != Eigen::Success) {
    return false;
  }
  x_ = spd_solver.solve(r);
  if (spd_solver.info() != Eigen::Success) {
    return false;
  }
  return x_.allFinite();
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::updateCandidate() {
  cost_ = cost_try_;
  ffeas_ = ffeas_try_;
  gfeas_ = gfeas_try_;
  hfeas_ = hfeas_try_;
  merit_ = cost_ + upsilon_ * (ffeas_ + gfeas_ + hfeas_);
}

template <typename Scalar>
bool SolverSQPTpl<Scalar>::decreaseRegularizationCriteria() {
  return (steplength_ >= th_stepdec_ && std::abs(dImpr_) > th_minimprove_);
}

template <typename Scalar>
bool SolverSQPTpl<Scalar>::increaseRegularizationCriteria() {
  return ((steplength_ >= th_stepinc_ && std::abs(dImpr_) <= th_minimprove_) ||
          !acceptstep_);
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::decreaseRegularization() {
  preg_ /= reg_decfactor_;
  if (preg_ < reg_min_) preg_ = reg_min_;
  dreg_ = preg_;
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::increaseRegularization() {
  preg_ *= reg_incfactor_;
  if (preg_ > reg_max_) preg_ = reg_max_;
  dreg_ = preg_;
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::extractQpDirection(const VectorXs& x) {
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model =
        problem_->get_runningModels()[t];
    const std::size_t nu = model->get_nu();
    dxs_[t] = x.segment(xs_idx_[t], ndx);
    if (nu > 0) dus_[t] = x.segment(us_idx_[t], nu);
  }
  dxs_.back() = x.segment(xs_idx_[T], ndx);
}

template <typename Scalar>
template <typename NewScalar>
SolverSQPTpl<NewScalar> SolverSQPTpl<Scalar>::cast() const {
  typedef SolverSQPTpl<NewScalar> ReturnType;
  typedef ShootingProblemTpl<NewScalar> ProblemType;
  ReturnType ret(
      std::make_shared<ProblemType>(problem_->template cast<NewScalar>()));
  ret.setCallbacks(vector_cast<NewScalar>(callbacks_));
  ret.set_th_acceptstep(scalar_cast<NewScalar>(th_acceptstep_));
  ret.set_th_gaptol(scalar_cast<NewScalar>(th_gaptol_));
  ret.set_feasnorm(feasnorm_);
  ret.set_th_stop(std::sqrt(std::numeric_limits<NewScalar>::epsilon()) <
                          NewScalar(th_stop_)
                      ? scalar_cast<NewScalar>(th_stop_)
                      : std::sqrt(std::numeric_limits<NewScalar>::epsilon()));
  ret.set_alphas(vector_cast<NewScalar>(alphas_));
  ret.set_reg_incfactor(scalar_cast<NewScalar>(reg_incfactor_));
  ret.set_reg_decfactor(scalar_cast<NewScalar>(reg_decfactor_));
  ret.set_th_grad(scalar_cast<NewScalar>(th_grad_));
  ret.set_th_noimprovement(scalar_cast<NewScalar>(th_noimprovement_));
  ret.set_th_stepdec(scalar_cast<NewScalar>(th_stepdec_));
  ret.set_th_stepinc(scalar_cast<NewScalar>(th_stepinc_));
  ret.set_th_minimprove(scalar_cast<NewScalar>(th_minimprove_));
  ret.set_th_acceptnegstep(scalar_cast<NewScalar>(th_acceptnegstep_));
  ret.set_th_acceptminstep(scalar_cast<NewScalar>(th_acceptminstep_));
  ret.set_rho(scalar_cast<NewScalar>(rho_));
  ret.set_th_minfeas(scalar_cast<NewScalar>(th_minfeas_));
  ret.set_upsilon_decfactor(scalar_cast<NewScalar>(upsilon_decfactor_));
  ret.set_zero_upsilon(zero_upsilon_);
  ret.set_qp_feas_tol(scalar_cast<NewScalar>(qp_feas_tol_));
  ret.set_qp_maxiters(qp_maxiters_);
  if (with_du_bounds_) {
    ret.set_control_variation_bounds(du_max_.template cast<NewScalar>());
  } else {
    ret.disable_control_variation_bounds();
  }
  ret.setCandidate(vector_cast<NewScalar>(xs_), vector_cast<NewScalar>(us_),
                   is_feasible_);
  return ret;
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::allocateData() {
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  Lxx_dx_.resize(T + 1);
  Luu_du_.resize(T);
  Lxu_du_.resize(T);
  xs_idx_.resize(T + 1);
  us_idx_.resize(T);

  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  n_ = 0;
  m_ = ndx;
  p_ = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t ng = model->get_ng();
    Lxx_dx_[t] = VectorXs::Zero(ndx);
    Luu_du_[t] = VectorXs::Zero(nu);
    Lxu_du_[t] = VectorXs::Zero(nu);
    n_ += ndx + nu;
    m_ += ndx + nh;
    p_ += ng + nu + ndx;
  }
  Lxx_dx_.back() = VectorXs::Zero(ndx);

  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  n_ += ndx;
  m_ += model_T->get_nh_T();
  p_ += model_T->get_ng_T() + ndx;
  if (with_du_bounds_ && T > 1) {
    for (std::size_t t = 0; t + 1 < T; ++t) {
      const std::size_t nu = models[t]->get_nu();
      const std::size_t nu_next = models[t + 1]->get_nu();
      if (nu > 0 && nu_next > 0) {
        if (nu != nu_next) {
          throw_pretty(
              "Invalid control dimensions across nodes for variation bounds.");
        }
        p_ += nu;
      }
    }
  }
  p_ *= 2;

  updateStateAndControlIndex();
  x_.setZero(n_);
  Q_.resize(n_, n_);
  A_.resize(m_, n_);
  G_.resize(p_, n_);
  Q_.setZero();
  A_.setZero();
  G_.setZero();
  c_.setZero(n_);
  b_.setZero(m_);
  h_.setZero(p_);
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverSQP::resizeRunningData");
  SolverAbstract::resizeRunningData();
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  Lxx_dx_.resize(T + 1);
  Luu_du_.resize(T);
  Lxu_du_.resize(T);
  xs_idx_.resize(T + 1);
  us_idx_.resize(T);
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();

  n_ = 0;
  m_ = ndx;
  p_ = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    n_ += ndx + nu;
    m_ += ndx + model->get_nh();
    p_ += model->get_ng() + nu + ndx;
    Lxx_dx_[t].conservativeResize(ndx);
    Luu_du_[t].conservativeResize(nu);
    Lxu_du_[t].conservativeResize(nu);
  }
  Lxx_dx_.back().conservativeResize(ndx);
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  n_ += ndx;
  m_ += model_T->get_nh_T();
  p_ += model_T->get_ng_T() + ndx;
  if (with_du_bounds_ && T > 1) {
    for (std::size_t t = 0; t + 1 < T; ++t) {
      const std::size_t nu = models[t]->get_nu();
      const std::size_t nu_next = models[t + 1]->get_nu();
      if (nu > 0 && nu_next > 0) {
        if (nu != nu_next) {
          throw_pretty(
              "Invalid control dimensions across nodes for variation bounds.");
        }
        p_ += nu;
      }
    }
  }
  p_ *= 2;

  updateStateAndControlIndex();
  x_.conservativeResize(n_);
  Q_.resize(n_, n_);
  A_.resize(m_, n_);
  G_.resize(p_, n_);
  Q_.setZero();
  A_.setZero();
  G_.setZero();
  c_.conservativeResize(n_);
  b_.conservativeResize(m_);
  h_.conservativeResize(p_);
  STOP_PROFILER("SolverSQP::resizeRunningData");
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::resizeTerminalData() {
  START_PROFILER("SolverSQP::resizeTerminalData");
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  m_ += model_T->get_nh_T() - nh_T_;
  const std::size_t p_half = p_ / 2;
  p_ = 2 * (p_half + model_T->get_ng_T() - ng_T_);
  A_.resize(m_, n_);
  G_.resize(p_, n_);
  A_.setZero();
  G_.setZero();
  b_.conservativeResize(m_);
  h_.conservativeResize(p_);
  STOP_PROFILER("SolverSQP::resizeTerminalData");
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::updateStateAndControlIndex() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  std::size_t nvar = 0;
  const std::size_t ndx = problem_->get_ndx();
  xs_idx_[0] = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nu = models[t]->get_nu();
    nvar += ndx;
    us_idx_[t] = nvar;
    if (nu > 0) {
      nvar += nu;
    }
    xs_idx_[t + 1] = nvar;
  }
}

template <typename Scalar>
std::size_t SolverSQPTpl<Scalar>::get_n() const noexcept {
  return n_;
}
template <typename Scalar>
std::size_t SolverSQPTpl<Scalar>::get_m() const noexcept {
  return m_;
}
template <typename Scalar>
std::size_t SolverSQPTpl<Scalar>::get_p() const noexcept {
  return p_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_reg_incfactor() const {
  return reg_incfactor_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_reg_decfactor() const {
  return reg_decfactor_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_th_grad() const {
  return th_grad_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_th_stepdec() const {
  return th_stepdec_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_th_stepinc() const {
  return th_stepinc_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_th_minimprove() const {
  return th_minimprove_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_th_acceptnegstep() const {
  return th_acceptnegstep_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_th_acceptminstep() const {
  return th_acceptminstep_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_rho() const {
  return rho_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_th_minfeas() const {
  return th_minfeas_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_upsilon() const {
  return upsilon_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_upsilon_decfactor() const {
  return upsilon_decfactor_;
}
template <typename Scalar>
bool SolverSQPTpl<Scalar>::get_zero_upsilon() const {
  return zero_upsilon_;
}
template <typename Scalar>
Scalar SolverSQPTpl<Scalar>::get_qp_feas_tol() const {
  return qp_feas_tol_;
}
template <typename Scalar>
std::size_t SolverSQPTpl<Scalar>::get_qp_maxiters() const {
  return qp_maxiters_;
}
template <typename Scalar>
bool SolverSQPTpl<Scalar>::get_with_control_variation_bounds() const {
  return with_du_bounds_;
}
template <typename Scalar>
const typename SolverSQPTpl<Scalar>::VectorXs&
SolverSQPTpl<Scalar>::get_du_max() const {
  return du_max_;
}

template <typename Scalar>
void SolverSQPTpl<Scalar>::set_reg_incfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty("Invalid argument: reg_incfactor value is higher than 1.");
  }
  reg_incfactor_ = regfactor;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_reg_decfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty("Invalid argument: reg_decfactor value is higher than 1.");
  }
  reg_decfactor_ = regfactor;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_grad(const Scalar th_grad) {
  if (Scalar(0.) > th_grad) {
    throw_pretty("Invalid argument: th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_noimprovement(const Scalar th_noimprovement) {
  if (Scalar(0.) > th_noimprovement) {
    throw_pretty(
        "Invalid argument: th_noimprovement value has to be positive.");
  }
  th_noimprovement_ = th_noimprovement;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_stepdec(const Scalar th_stepdec) {
  if (Scalar(0.) >= th_stepdec || th_stepdec > Scalar(1.)) {
    throw_pretty("Invalid argument: th_stepdec value should between 0 and 1.");
  }
  th_stepdec_ = th_stepdec;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_stepinc(const Scalar th_stepinc) {
  if (Scalar(0.) >= th_stepinc || th_stepinc > Scalar(1.)) {
    throw_pretty("Invalid argument: th_stepinc value should between 0 and 1.");
  }
  th_stepinc_ = th_stepinc;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_minimprove(const Scalar th_minimprove) {
  if (Scalar(0.) >= th_minimprove || th_minimprove > Scalar(100.)) {
    throw_pretty(
        "Invalid argument: th_minimprove value should between 0 and 100.");
  }
  th_minimprove_ = th_minimprove;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_acceptnegstep(const Scalar th_acceptnegstep) {
  if (Scalar(0.) > th_acceptnegstep) {
    throw_pretty(
        "Invalid argument: th_acceptnegstep value has to be positive.");
  }
  th_acceptnegstep_ = th_acceptnegstep;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_acceptminstep(const Scalar th_acceptminstep) {
  if (Scalar(0.) > th_acceptminstep || th_acceptminstep > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: th_acceptminstep value should be between 0 and 1.");
  }
  th_acceptminstep_ = th_acceptminstep;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_rho(const Scalar rho) {
  if (Scalar(0.) >= rho || rho > Scalar(1.)) {
    throw_pretty("Invalid argument: rho value should between 0 and 1.");
  }
  rho_ = rho;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_th_minfeas(const Scalar th_minfeas) {
  th_minfeas_ = th_minfeas;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_upsilon_decfactor(const Scalar factor) {
  if (Scalar(0.) >= factor || factor > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: upsilon_decfactor value should between 0 and 1.");
  }
  upsilon_decfactor_ = factor;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_zero_upsilon(const bool zero_upsilon) {
  zero_upsilon_ = zero_upsilon;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_qp_feas_tol(const Scalar tol) {
  if (tol <= Scalar(0.)) {
    throw_pretty("Invalid argument: qp_feas_tol value has to be positive.");
  }
  qp_feas_tol_ = tol;
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_qp_maxiters(const std::size_t maxiters) {
  qp_maxiters_ = std::max<std::size_t>(1, maxiters);
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::set_control_variation_bounds(
    const VectorXs& du_max) {
  if (du_max.size() == 0 || (du_max.array() < Scalar(0)).any()) {
    throw_pretty("Invalid argument: du_max needs nonnegative entries.");
  }
  with_du_bounds_ = true;
  du_max_ = du_max;
  resizeData();
}
template <typename Scalar>
void SolverSQPTpl<Scalar>::disable_control_variation_bounds() {
  with_du_bounds_ = false;
  du_max_.resize(0);
  resizeData();
}

}  // namespace crocoddyl
