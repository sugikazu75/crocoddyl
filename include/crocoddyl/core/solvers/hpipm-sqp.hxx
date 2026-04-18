///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_HPIPM

#include <numeric>

namespace crocoddyl {

template <typename Scalar>
SolverHpipmSQPTpl<Scalar>::SolverHpipmSQPTpl(
    std::shared_ptr<ShootingProblem> problem)
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
      zero_upsilon_(false) {
  allocateData();
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::computeDirection(const bool recalc) {
  START_PROFILER("SolverHpipmSQP::computeDirection");
  if (recalc) {
    SolverAbstract::calcDir();
  }
  computeQuadraticModel();
  if (!solveQuadraticModel()) {
    const std::size_t T = problem_->get_T();
    for (std::size_t t = 0; t <= T; ++t) dxs_[t].setZero();
    for (std::size_t t = 0; t < T; ++t) dus_[t].setZero();
  } else {
    extractQpDirection();
  }
  STOP_PROFILER("SolverHpipmSQP::computeDirection");
}

template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::stoppingCriteria() {
  feas_ = ffeas_ + gfeas_ + hfeas_;
  stop_ =
      std::max(feas_, std::abs(dVexp_full_) / (Scalar(1.) + std::abs(cost_)));
  return stop_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::Vector3s
SolverHpipmSQPTpl<Scalar>::expectedImprovement() {
  const std::size_t T = problem_->get_T();
  DV_.setZero();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
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
void SolverHpipmSQPTpl<Scalar>::computeMeritFunctionImprovement() {
  dPhi_ = dV_ + upsilon_ * dfeas_;
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::computeExpectedMeritFunctionImprovement() {
  dPhiexp_ = dVexp_ + steplength_ * upsilon_ * dfeas_;
}

template <typename Scalar>
bool SolverHpipmSQPTpl<Scalar>::checkAcceptance() {
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
void SolverHpipmSQPTpl<Scalar>::updateMeritFunction() {
  if (iter_ == 0 && zero_upsilon_) {
    upsilon_ = Scalar(0.);
  }
  if (feas_ >= th_minfeas_) {
    upsilon_ = std::max(upsilon_ * upsilon_decfactor_,
                        dVexp_full_ / ((Scalar(1.) - rho_) * feas_));
  }
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::computeCandidate(const Scalar steplength) {
  START_PROFILER("SolverHpipmSQP::computeCandidate");
  if (steplength > Scalar(1.) || steplength < Scalar(0.)) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  models[0]->get_state()->integrate(xs_[0], steplength * dxs_[0], xs_try_[0]);
  fs_try_[0] = fs_[0] * (1 - steplength);
  for (std::size_t t = 0; t < T; ++t) {
    models[t]->get_state()->integrate(xs_[t + 1], steplength * dxs_[t + 1],
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
      STOP_PROFILER("SolverHpipmSQP::computeCandidate");
      throw_pretty("computeCandidate");
    }
  }
  const std::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;
  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverHpipmSQP::computeCandidate");
    throw_pretty("computeCandidate");
  }
  STOP_PROFILER("SolverHpipmSQP::computeCandidate");
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::updateCandidate() {
  cost_ = cost_try_;
  ffeas_ = ffeas_try_;
  gfeas_ = gfeas_try_;
  hfeas_ = hfeas_try_;
  merit_ = cost_ + upsilon_ * (ffeas_ + gfeas_ + hfeas_);
}

template <typename Scalar>
bool SolverHpipmSQPTpl<Scalar>::decreaseRegularizationCriteria() {
  return (steplength_ >= th_stepdec_ && std::abs(dImpr_) > th_minimprove_);
}

template <typename Scalar>
bool SolverHpipmSQPTpl<Scalar>::increaseRegularizationCriteria() {
  return ((steplength_ >= th_stepinc_ && std::abs(dImpr_) <= th_minimprove_) ||
          !acceptstep_);
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::decreaseRegularization() {
  preg_ /= reg_decfactor_;
  if (preg_ < reg_min_) preg_ = reg_min_;
  dreg_ = preg_;
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::increaseRegularization() {
  preg_ *= reg_incfactor_;
  if (preg_ > reg_max_) preg_ = reg_max_;
  dreg_ = preg_;
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::computeQuadraticModel() {
  START_PROFILER("SolverHpipmSQP::computeQuadraticModel");
  const std::size_t T = problem_->get_T();
  const Scalar inf = std::numeric_limits<Scalar>::infinity();
  const double reg = static_cast<double>(preg_);

  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t ng = model->get_ng();
    hpipm::OcpQp& qp = ocp_qp_[t];

    // Cost
    qp.Q = data->Lxx.template cast<double>();
    qp.q = data->Lx.template cast<double>();
    if (reg != 0.) {
      qp.Q.diagonal().array() += reg;
    }

    // Dynamics: dx_{t+1} = Fx dx_t + Fu du_t + fs_{t+1}
    qp.A = data->Fx.template cast<double>();
    qp.b = fs_[t + 1].template cast<double>();

    if (nu > 0) {
      qp.R = data->Luu.template cast<double>();
      qp.S = data->Lxu.transpose().template cast<double>();
      qp.r = data->Lu.template cast<double>();
      qp.B = data->Fu.template cast<double>();
      if (reg != 0.) {
        qp.R.diagonal().array() += reg;
      }

      // Control bounds: lbu <= du <= ubu
      const VectorXs& u_lb = model->get_u_lb();
      const VectorXs& u_ub = model->get_u_ub();
      for (std::size_t i = 0; i < nu; ++i) {
        qp.lbu[i] =
            (u_lb[i] > -inf) ? static_cast<double>(u_lb[i] - us_[t][i]) : -1e30;
        qp.ubu[i] =
            (u_ub[i] < inf) ? static_cast<double>(u_ub[i] - us_[t][i]) : 1e30;
      }
    }

    // General constraints: path equalities (nh) + inequalities (ng)
    const std::size_t ng_total = nh + ng;
    if (ng_total > 0) {
      std::size_t r = 0;
      if (nh > 0) {
        qp.C.topRows(nh) = data->Hx.template cast<double>();
        if (nu > 0) qp.D.topRows(nh) = data->Hu.template cast<double>();
        qp.lg.head(nh) = (-data->h).template cast<double>();
        qp.ug.head(nh) = qp.lg.head(nh);
        r += nh;
      }
      if (ng > 0) {
        qp.C.middleRows(r, ng) = data->Gx.template cast<double>();
        if (nu > 0) qp.D.middleRows(r, ng) = data->Gu.template cast<double>();
        qp.lg.segment(r, ng) =
            (model->get_g_lb() - data->g).template cast<double>();
        qp.ug.segment(r, ng) =
            (model->get_g_ub() - data->g).template cast<double>();
      }
    }
  }

  // Terminal stage
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& data_T =
      problem_->get_terminalData();
  const std::size_t nh_T = model_T->get_nh_T();
  const std::size_t ng_T = model_T->get_ng_T();
  hpipm::OcpQp& qp_T = ocp_qp_[T];

  qp_T.Q = data_T->Lxx.template cast<double>();
  qp_T.q = data_T->Lx.template cast<double>();
  if (reg != 0.) {
    qp_T.Q.diagonal().array() += reg;
  }

  const std::size_t ng_T_total = nh_T + ng_T;
  if (ng_T_total > 0) {
    std::size_t r = 0;
    if (nh_T > 0) {
      qp_T.C.topRows(nh_T) = data_T->Hx.template cast<double>();
      qp_T.lg.head(nh_T) = (-data_T->h.head(nh_T)).template cast<double>();
      qp_T.ug.head(nh_T) = qp_T.lg.head(nh_T);
      r += nh_T;
    }
    if (ng_T > 0) {
      qp_T.C.middleRows(r, ng_T) = data_T->Gx.template cast<double>();
      qp_T.lg.segment(r, ng_T) =
          (model_T->get_g_lb() - data_T->g).template cast<double>();
      qp_T.ug.segment(r, ng_T) =
          (model_T->get_g_ub() - data_T->g).template cast<double>();
    }
  }
  STOP_PROFILER("SolverHpipmSQP::computeQuadraticModel");
}

template <typename Scalar>
bool SolverHpipmSQPTpl<Scalar>::solveQuadraticModel() {
  START_PROFILER("SolverHpipmSQP::solveQuadraticModel");
  const Eigen::VectorXd x0 = fs_[0].template cast<double>();
  const hpipm::HpipmStatus status = hpipm_solver_->solve(x0, ocp_qp_, qp_sol_);
  STOP_PROFILER("SolverHpipmSQP::solveQuadraticModel");
  return (status == hpipm::HpipmStatus::Success ||
          status == hpipm::HpipmStatus::MaxIterReached);
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::extractQpDirection() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    dxs_[t] = qp_sol_[t].x.cast<Scalar>();
    if (models[t]->get_nu() > 0) {
      dus_[t] = qp_sol_[t].u.cast<Scalar>();
    }
  }
  dxs_.back() = qp_sol_[T].x.cast<Scalar>();
}

template <typename Scalar>
template <typename NewScalar>
SolverHpipmSQPTpl<NewScalar> SolverHpipmSQPTpl<Scalar>::cast() const {
  typedef SolverHpipmSQPTpl<NewScalar> ReturnType;
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
  ret.set_hpipm_settings(hpipm_settings_);
  ret.setCandidate(vector_cast<NewScalar>(xs_), vector_cast<NewScalar>(us_),
                   is_feasible_);
  return ret;
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::allocateData() {
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();

  Lxx_dx_.resize(T + 1);
  Luu_du_.resize(T);
  Lxu_du_.resize(T);
  ocp_qp_.resize(T + 1);
  qp_sol_.resize(T + 1);

  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t ng = model->get_ng();
    const std::size_t ng_total = nh + ng;
    hpipm::OcpQp& qp = ocp_qp_[t];

    // State cost (sets nx[t] = ndx via q.size())
    qp.Q = Eigen::MatrixXd::Zero(ndx, ndx);
    qp.q = Eigen::VectorXd::Zero(ndx);

    // Dynamics
    qp.A = Eigen::MatrixXd::Zero(ndx, ndx);
    qp.b = Eigen::VectorXd::Zero(ndx);

    if (nu > 0) {
      qp.R = Eigen::MatrixXd::Zero(nu, nu);
      qp.S = Eigen::MatrixXd::Zero(nu, ndx);
      qp.r = Eigen::VectorXd::Zero(nu);  // sets nu[t] = nu via r.size()
      qp.B = Eigen::MatrixXd::Zero(ndx, nu);

      // Control bounds: always include all nu controls
      qp.idxbu.resize(nu);
      std::iota(qp.idxbu.begin(), qp.idxbu.end(), 0);
      qp.lbu = Eigen::VectorXd::Constant(nu, -1e30);
      qp.ubu = Eigen::VectorXd::Constant(nu, 1e30);
    } else {
      // nu = 0: sets nu[t] = 0 via r.size() = 0
      qp.R = Eigen::MatrixXd::Zero(0, 0);
      qp.S = Eigen::MatrixXd::Zero(0, ndx);
      qp.r = Eigen::VectorXd::Zero(0);
      qp.B = Eigen::MatrixXd::Zero(ndx, 0);
    }

    // General constraints (path equalities + inequalities)
    if (ng_total > 0) {
      qp.C = Eigen::MatrixXd::Zero(ng_total, ndx);
      qp.D = Eigen::MatrixXd::Zero(ng_total, static_cast<int>(nu));
      qp.lg = Eigen::VectorXd::Zero(ng_total);
      qp.ug = Eigen::VectorXd::Zero(ng_total);
    } else {
      qp.C = Eigen::MatrixXd::Zero(0, ndx);
      qp.D = Eigen::MatrixXd::Zero(0, static_cast<int>(nu));
      qp.lg = Eigen::VectorXd::Zero(0);
      qp.ug = Eigen::VectorXd::Zero(0);
    }

    Lxx_dx_[t] = VectorXs::Zero(ndx);
    Luu_du_[t] = VectorXs::Zero(nu);
    Lxu_du_[t] = VectorXs::Zero(ndx);
  }

  // Terminal stage (nu[T] = 0 is set automatically by OcpQpDim)
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  const std::size_t nh_T = model_T->get_nh_T();
  const std::size_t ng_T = model_T->get_ng_T();
  const std::size_t ng_T_total = nh_T + ng_T;
  hpipm::OcpQp& qp_T = ocp_qp_[T];

  qp_T.Q = Eigen::MatrixXd::Zero(ndx, ndx);
  qp_T.q = Eigen::VectorXd::Zero(ndx);

  if (ng_T_total > 0) {
    qp_T.C = Eigen::MatrixXd::Zero(ng_T_total, ndx);
    qp_T.lg = Eigen::VectorXd::Zero(ng_T_total);
    qp_T.ug = Eigen::VectorXd::Zero(ng_T_total);
  } else {
    qp_T.C = Eigen::MatrixXd::Zero(0, ndx);
    qp_T.lg = Eigen::VectorXd::Zero(0);
    qp_T.ug = Eigen::VectorXd::Zero(0);
  }

  Lxx_dx_.back() = VectorXs::Zero(ndx);

  hpipm_solver_ =
      std::make_unique<hpipm::OcpQpIpmSolver>(ocp_qp_, hpipm_settings_);
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverHpipmSQP::resizeRunningData");
  SolverAbstract::resizeRunningData();
  allocateData();
  STOP_PROFILER("SolverHpipmSQP::resizeRunningData");
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::resizeTerminalData() {
  START_PROFILER("SolverHpipmSQP::resizeTerminalData");
  allocateData();
  STOP_PROFILER("SolverHpipmSQP::resizeTerminalData");
}

template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_reg_incfactor() const {
  return reg_incfactor_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_reg_decfactor() const {
  return reg_decfactor_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_th_grad() const {
  return th_grad_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_th_stepdec() const {
  return th_stepdec_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_th_stepinc() const {
  return th_stepinc_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_th_minimprove() const {
  return th_minimprove_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_th_acceptnegstep() const {
  return th_acceptnegstep_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_th_acceptminstep() const {
  return th_acceptminstep_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_rho() const {
  return rho_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_th_minfeas() const {
  return th_minfeas_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_upsilon() const {
  return upsilon_;
}
template <typename Scalar>
Scalar SolverHpipmSQPTpl<Scalar>::get_upsilon_decfactor() const {
  return upsilon_decfactor_;
}
template <typename Scalar>
bool SolverHpipmSQPTpl<Scalar>::get_zero_upsilon() const {
  return zero_upsilon_;
}
template <typename Scalar>
const hpipm::OcpQpIpmSolverSettings&
SolverHpipmSQPTpl<Scalar>::get_hpipm_settings() const {
  return hpipm_settings_;
}
template <typename Scalar>
const hpipm::OcpQpIpmSolverStatistics&
SolverHpipmSQPTpl<Scalar>::get_hpipm_statistics() const {
  return hpipm_solver_->getSolverStatistics();
}

template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_reg_incfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty("Invalid argument: reg_incfactor value is higher than 1.");
  }
  reg_incfactor_ = regfactor;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_reg_decfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty("Invalid argument: reg_decfactor value is higher than 1.");
  }
  reg_decfactor_ = regfactor;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_grad(const Scalar th_grad) {
  if (Scalar(0.) > th_grad) {
    throw_pretty("Invalid argument: th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_noimprovement(
    const Scalar th_noimprovement) {
  if (Scalar(0.) > th_noimprovement) {
    throw_pretty(
        "Invalid argument: th_noimprovement value has to be positive.");
  }
  th_noimprovement_ = th_noimprovement;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_stepdec(const Scalar th_stepdec) {
  if (Scalar(0.) >= th_stepdec || th_stepdec > Scalar(1.)) {
    throw_pretty("Invalid argument: th_stepdec value should between 0 and 1.");
  }
  th_stepdec_ = th_stepdec;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_stepinc(const Scalar th_stepinc) {
  if (Scalar(0.) >= th_stepinc || th_stepinc > Scalar(1.)) {
    throw_pretty("Invalid argument: th_stepinc value should between 0 and 1.");
  }
  th_stepinc_ = th_stepinc;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_minimprove(const Scalar th_minimprove) {
  if (Scalar(0.) >= th_minimprove || th_minimprove > Scalar(100.)) {
    throw_pretty(
        "Invalid argument: th_minimprove value should between 0 and 100.");
  }
  th_minimprove_ = th_minimprove;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_acceptnegstep(
    const Scalar th_acceptnegstep) {
  if (Scalar(0.) > th_acceptnegstep) {
    throw_pretty(
        "Invalid argument: th_acceptnegstep value has to be positive.");
  }
  th_acceptnegstep_ = th_acceptnegstep;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_acceptminstep(
    const Scalar th_acceptminstep) {
  if (Scalar(0.) > th_acceptminstep || th_acceptminstep > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: th_acceptminstep value should be between 0 and 1.");
  }
  th_acceptminstep_ = th_acceptminstep;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_rho(const Scalar rho) {
  if (Scalar(0.) >= rho || rho > Scalar(1.)) {
    throw_pretty("Invalid argument: rho value should between 0 and 1.");
  }
  rho_ = rho;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_th_minfeas(const Scalar th_minfeas) {
  th_minfeas_ = th_minfeas;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_upsilon_decfactor(const Scalar factor) {
  if (Scalar(0.) >= factor || factor > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: upsilon_decfactor value should between 0 and 1.");
  }
  upsilon_decfactor_ = factor;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_zero_upsilon(const bool zero_upsilon) {
  zero_upsilon_ = zero_upsilon;
}
template <typename Scalar>
void SolverHpipmSQPTpl<Scalar>::set_hpipm_settings(
    const hpipm::OcpQpIpmSolverSettings& settings) {
  hpipm_settings_ = settings;
  if (hpipm_solver_) {
    hpipm_solver_->setSolverSettings(settings);
  }
}

}  // namespace crocoddyl

#endif  // CROCODDYL_WITH_HPIPM
