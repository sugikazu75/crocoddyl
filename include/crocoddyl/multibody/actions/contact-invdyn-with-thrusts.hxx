///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/utils/static-if.hpp>

#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/actuations/floating-base-distributed-thrusters.hpp"
#include "crocoddyl/multibody/actuations/floating-base-thrust-rates.hpp"
#include "crocoddyl/multibody/actuations/floating-base-thrusters.hpp"
#include "crocoddyl/multibody/utils/static-equilibrium.hpp"

namespace crocoddyl {

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::
    getNumThrusters(const std::shared_ptr<StateAbstract>& state,
                    const std::shared_ptr<ActuationModelAbstract>& actuation) {
  if (std::shared_ptr<StateWithThrusts> s =
          std::dynamic_pointer_cast<StateWithThrusts>(state)) {
    return s->get_nthrusters();
  }
  if (std::dynamic_pointer_cast<StateMultibody>(state) == nullptr) {
    throw_pretty("Invalid argument: "
                 << "the state should be StateMultibody or "
                    "StateMultibodyWithThrusts");
  }
  if (std::shared_ptr<ActuationModelFloatingBaseDistributedThrustersTpl<Scalar>>
          a = std::dynamic_pointer_cast<
              ActuationModelFloatingBaseDistributedThrustersTpl<Scalar>>(
              actuation)) {
    return a->get_nthrusters();
  }
  if (std::shared_ptr<ActuationModelFloatingBaseThrustersTpl<Scalar>> a =
          std::dynamic_pointer_cast<
              ActuationModelFloatingBaseThrustersTpl<Scalar>>(actuation)) {
    return a->get_nthrusters();
  }
  throw_pretty("Invalid argument: "
               << "with StateMultibody, the actuation should be "
                  "ActuationModelFloatingBaseDistributedThrusters or "
                  "ActuationModelFloatingBaseThrusters");
}

template <typename Scalar>
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::
    DifferentialActionModelContactInvDynamicsWithThrustsTpl(
        std::shared_ptr<StateAbstract> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs)
    : DifferentialActionModelContactInvDynamicsWithThrustsTpl(
          state, actuation, contacts, costs,
          std::make_shared<ConstraintModelManager>(
              contacts->get_state(), getNumThrusters(state, actuation) +
                                         state->get_nv() +
                                         contacts->get_nc_total())) {}

template <typename Scalar>
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::
    DifferentialActionModelContactInvDynamicsWithThrustsTpl(
        std::shared_ptr<StateAbstract> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs,
        std::shared_ptr<ConstraintModelManager> constraints)
    : Base(state,
           getNumThrusters(state, actuation) + state->get_nv() +
               contacts->get_nc_total(),
           costs->get_nr(),
           (std::dynamic_pointer_cast<StateWithThrusts>(state)
                ? getNumThrusters(state, actuation)
                : 0) +
               constraints->get_ng(),
           state->get_nv() + getNumThrusters(state, actuation) -
               actuation->get_nu() + contacts->get_nc_total() +
               constraints->get_nh(),
           (std::dynamic_pointer_cast<StateWithThrusts>(state)
                ? getNumThrusters(state, actuation)
                : 0) +
               constraints->get_ng_T(),
           constraints->get_nh_T()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(contacts->get_state()->get_pinocchio().get()),
      thrust_state_(std::dynamic_pointer_cast<StateWithThrusts>(state) !=
                    nullptr),
      nf_(getNumThrusters(state, actuation)),
      nvf_(state->get_nv() + nf_ - actuation->get_nu()),
      ng_thrust_(thrust_state_ ? nf_ : 0) {
  init();
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::init() {
  const std::size_t nv = state_->get_nv();
  if (actuation_->get_nu() < nf_ || actuation_->get_nu() - nf_ > nv) {
    throw_pretty("Invalid argument: "
                 << "the actuation should have nf thrust inputs followed by "
                    "at most nv joint torques");
  }
  if (contacts_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Contacts doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Costs doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  if (constraints_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Constraints doesn't have the same control dimension (it "
                    "should be " +
                        std::to_string(nu_) + ")");
  }
  if (contacts_->get_state()->get_nv() != nv) {
    throw_pretty("Invalid argument: "
                 << "Contacts should be defined over the robot state");
  }
  contacts_->setComputeAllContacts(true);

  // The actuation bounds the thrust input (thrusts or their rates); the
  // thrusts of a thrust state are bounded by inequality constraints instead
  VectorXs lb =
      VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity());
  VectorXs ub =
      VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity());
  thrust_lb_ =
      VectorXs::Constant(nf_, -std::numeric_limits<Scalar>::infinity());
  thrust_ub_ = VectorXs::Constant(nf_, std::numeric_limits<Scalar>::infinity());
  lb.head(nf_) = actuation_->get_u_lb().head(nf_);
  ub.head(nf_) = actuation_->get_u_ub().head(nf_);
  if (thrust_state_) {
    if (std::shared_ptr<ActuationModelFloatingBaseThrusterRatesTpl<Scalar>> a =
            std::dynamic_pointer_cast<
                ActuationModelFloatingBaseThrusterRatesTpl<Scalar>>(
                actuation_)) {
      for (std::size_t i = 0; i < nf_; ++i) {
        thrust_lb_(i) = a->get_thrusters()[i].min_thrust_;
        thrust_ub_(i) = a->get_thrusters()[i].max_thrust_;
      }
    }
  }
  Base::set_u_lb(lb);
  Base::set_u_ub(ub);
  thrust_reg_weight_ = VectorXs::Zero(nf_);
  thrust_barrier_weight_ = VectorXs::Zero(nf_);
  thrust_barrier_lb_ = thrust_lb_;
  thrust_barrier_ub_ = thrust_ub_;
  if (!thrust_state_) {
    thrust_barrier_lb_ = actuation_->get_u_lb().head(nf_);
    thrust_barrier_ub_ = actuation_->get_u_ub().head(nf_);
  }
  g_lb_.resize(ng_thrust_ + constraints_->get_ng());
  g_ub_.resize(ng_thrust_ + constraints_->get_ng());
  g_lb_.head(ng_thrust_) = thrust_lb_.head(ng_thrust_);
  g_ub_.head(ng_thrust_) = thrust_ub_.head(ng_thrust_);
  g_lb_.tail(constraints_->get_ng()) = constraints_->get_lb();
  g_ub_.tail(constraints_->get_ng()) = constraints_->get_ub();
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc_total();
  const std::size_t nj = nv - nvf_;
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.segment(nq, nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a =
      u.segment(nf_, nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      f_ext = u.tail(nc);

  d->xout = a;
  pinocchio::forwardKinematics(*pinocchio_, d->pinocchio, q, v, a);
  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio);
  contacts_->calc(d->multibody.contacts, x.head(nq + nv));
  contacts_->updateForce(d->multibody.contacts, f_ext);
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, v, a,
                  d->multibody.contacts->fext);
  pinocchio::updateGlobalPlacements(*pinocchio_, d->pinocchio);
  pinocchio::centerOfMass(*pinocchio_, d->pinocchio, q, v, a);

  // The thrusts generate W(q)*lambda, and the joint torques balance the rest
  d->u_act.head(nf_) = u.head(nf_);
  d->u_act.tail(nj).setZero();
  actuation_->calc(d->multibody.actuation, x, d->u_act);
  d->tau_thrust = d->multibody.actuation->tau;
  d->u_act.tail(nj) = d->pinocchio.tau.tail(nj) - d->tau_thrust.tail(nj);
  d->multibody.actuation->tau.tail(nj) += d->u_act.tail(nj);
  d->multibody.joint->a = a;
  d->multibody.actuation->u = d->u_act;
  d->multibody.joint->tau = d->u_act;

  costs_->calc(d->costs, x.head(costs_->get_state()->get_nx()), u);
  d->cost = d->costs->cost;
  d->cost += calcThrustCost(d, thrust_state_ ? x.tail(nf_) : u.head(nf_));

  resizeConstraintData(d, true);
  d->h.head(nvf_) = d->pinocchio.tau.head(nvf_) - d->tau_thrust.head(nvf_);
  std::size_t fid = 0;
  typename ContactModelMultiple::ContactDataContainer::const_iterator it_d =
      d->multibody.contacts->contacts.begin();
  for (typename ContactModelMultiple::ContactModelContainer::const_iterator
           it_m = contacts_->get_contacts().begin();
       it_m != contacts_->get_contacts().end(); ++it_m, ++it_d) {
    const std::size_t nc_i = it_m->second->contact->get_nc();
    if (it_m->second->active) {
      d->h.segment(nvf_ + fid, nc_i) = it_d->second->a0;
    } else {
      d->h.segment(nvf_ + fid, nc_i) = f_ext.segment(fid, nc_i);
    }
    fid += nc_i;
  }
  d->g.head(ng_thrust_) = x.tail(ng_thrust_);
  calcUserConstraints(d, x, &u);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.segment(nq, nv);

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);
  costs_->calc(d->costs, x.head(costs_->get_state()->get_nx()));
  d->cost = d->costs->cost;
  if (thrust_state_) {
    d->cost += calcThrustCost(d, x.tail(nf_));
  }

  resizeConstraintData(d, false);
  d->g.head(ng_thrust_) = x.tail(ng_thrust_);
  calcUserConstraints(d, x, nullptr);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc_total();
  const std::size_t nj = nv - nvf_;
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.segment(nq, nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a =
      u.segment(nf_, nv);

  pinocchio::computeRNEADerivatives(*pinocchio_, d->pinocchio, q, v, a,
                                    d->multibody.contacts->fext);
  contacts_->updateRneaDiff(d->multibody.contacts, d->pinocchio);
  d->pinocchio.M.template triangularView<Eigen::StrictlyLower>() =
      d->pinocchio.M.template triangularView<Eigen::StrictlyUpper>()
          .transpose();
  pinocchio::jacobianCenterOfMass(*pinocchio_, d->pinocchio, false);
  actuation_->calcDiff(d->multibody.actuation, x, d->u_act);
  contacts_->calcDiff(d->multibody.contacts, x.head(nq + nv));

  // Jacobians of the generalized forces left to the joints, RNEA - W*lambda
  d->dtau_dx.leftCols(nv) = d->pinocchio.dtau_dq;
  d->dtau_dx.middleCols(nv, nv) = d->pinocchio.dtau_dv;
  d->dtau_dx.rightCols(state_->get_ndx() - 2 * nv).setZero();
  d->dtau_dx -= d->multibody.actuation->dtau_dx;
  d->dtau_du.leftCols(nf_) = -d->multibody.actuation->dtau_du.leftCols(nf_);
  d->dtau_du.middleCols(nf_, nv) = d->pinocchio.M;
  d->dtau_du.rightCols(nc) = -d->multibody.contacts->Jc.topRows(nc).transpose();
  d->multibody.joint->dtau_dx.bottomRows(nj) = d->dtau_dx.bottomRows(nj);
  d->multibody.joint->dtau_du.bottomRows(nj) = d->dtau_du.bottomRows(nj);

  calcDiffCosts(d, x, &u);

  resizeConstraintData(d, true);
  d->Gx.setZero();
  d->Gu.setZero();
  d->Hx.setZero();
  d->Hu.setZero();
  d->Hx.topRows(nvf_) = d->dtau_dx.topRows(nvf_);
  d->Hu.topRows(nvf_) = d->dtau_du.topRows(nvf_);
  std::size_t fid = 0;
  typename ContactModelMultiple::ContactDataContainer::const_iterator it_d =
      d->multibody.contacts->contacts.begin();
  for (typename ContactModelMultiple::ContactModelContainer::const_iterator
           it_m = contacts_->get_contacts().begin();
       it_m != contacts_->get_contacts().end(); ++it_m, ++it_d) {
    const std::size_t nc_i = it_m->second->contact->get_nc();
    if (it_m->second->active) {
      d->Hx.block(nvf_ + fid, 0, nc_i, 2 * nv) = it_d->second->da0_dx;
      d->Hu.block(nvf_ + fid, nf_, nc_i, nv) = it_d->second->Jc;
    } else {
      d->Hu.block(nvf_ + fid, nf_ + nv + fid, nc_i, nc_i).diagonal().setOnes();
    }
    fid += nc_i;
  }
  d->Gx.block(0, 2 * nv, ng_thrust_, ng_thrust_).diagonal().setOnes();
  calcDiffUserConstraints(d, x, &u);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  calcDiffCosts(d, x, nullptr);
  resizeConstraintData(d, false);
  d->Gx.setZero();
  d->Gu.setZero();
  d->Hx.setZero();
  d->Hu.setZero();
  d->Gx.block(0, 2 * state_->get_nv(), ng_thrust_, ng_thrust_)
      .diagonal()
      .setOnes();
  calcDiffUserConstraints(d, x, nullptr);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::resizeConstraintData(Data* d, const bool running_node) const {
  const std::size_t ndx = state_->get_ndx();
  const std::size_t ng = running_node ? get_ng() : get_ng_T();
  const std::size_t nh = running_node ? get_nh() : get_nh_T();
  if (static_cast<std::size_t>(d->g.size()) != ng) {
    d->g.conservativeResize(ng);
    d->Gx.conservativeResize(ng, ndx);
    d->Gu.conservativeResize(ng, nu_);
  }
  if (static_cast<std::size_t>(d->h.size()) != nh) {
    d->h.conservativeResize(nh);
    d->Hx.conservativeResize(nh, ndx);
    d->Hu.conservativeResize(nh, nu_);
  }
  // The user may activate constraints after creating the data, so the
  // internal storage of the constraint data grows on demand
  ConstraintDataManagerTpl<Scalar>& cd = *d->constraints;
  const std::size_t ndx_c = constraints_->get_state()->get_ndx();
  const std::size_t ng_c =
      std::max(constraints_->get_ng(), constraints_->get_ng_T());
  const std::size_t nh_c =
      std::max(constraints_->get_nh(), constraints_->get_nh_T());
  if (static_cast<std::size_t>(cd.g_internal.size()) < ng_c) {
    cd.g_internal.resize(ng_c);
    cd.Gx_internal.resize(ng_c, ndx_c);
    cd.Gu_internal.resize(ng_c, nu_);
  }
  if (static_cast<std::size_t>(cd.h_internal.size()) < nh_c) {
    cd.h_internal.resize(nh_c);
    cd.Hx_internal.resize(nh_c, ndx_c);
    cd.Hu_internal.resize(nh_c, nu_);
  }
  cd.resize(constraints_.get(), running_node);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::calcUserConstraints(Data* d, const Eigen::Ref<const VectorXs>& x,
                                 const Eigen::Ref<const VectorXs>* u) {
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      xc = x.head(constraints_->get_state()->get_nx());
  if (u != nullptr) {
    constraints_->calc(d->constraints, xc, *u);
  } else {
    constraints_->calc(d->constraints, xc);
  }
  const std::size_t ng = d->constraints->g.size();
  const std::size_t nh = d->constraints->h.size();
  const std::size_t nh0 = u != nullptr ? nvf_ + contacts_->get_nc_total() : 0;
  d->g.segment(ng_thrust_, ng) = d->constraints->g;
  d->h.segment(nh0, nh) = d->constraints->h;
  const std::size_t ng_lb = constraints_->get_lb().size();
  g_lb_.conservativeResize(ng_thrust_ + ng_lb);
  g_ub_.conservativeResize(ng_thrust_ + ng_lb);
  g_lb_.tail(ng_lb) = constraints_->get_lb();
  g_ub_.tail(ng_lb) = constraints_->get_ub();
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::calcDiffUserConstraints(Data* d,
                                     const Eigen::Ref<const VectorXs>& x,
                                     const Eigen::Ref<const VectorXs>* u) {
  // Each constraint is handled separately because its state Jacobian spans
  // either the robot or the whole state (e.g. joint efforts depend on a
  // thrust state)
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      xc = x.head(constraints_->get_state()->get_nx());
  std::size_t ng_i = ng_thrust_;
  std::size_t nh_i = u != nullptr ? nvf_ + contacts_->get_nc_total() : 0;
  typename ConstraintModelManager::ConstraintDataContainer::const_iterator
      it_d = d->constraints->constraints.begin();
  for (typename ConstraintModelManager::ConstraintModelContainer::const_iterator
           it_m = constraints_->get_constraints().begin();
       it_m != constraints_->get_constraints().end(); ++it_m, ++it_d) {
    const std::shared_ptr<typename ConstraintModelManager::ConstraintItem>&
        m_i = it_m->second;
    if (!m_i->active ||
        (u == nullptr && !m_i->constraint->get_T_constraint())) {
      continue;
    }
    const std::shared_ptr<ConstraintDataAbstractTpl<Scalar>>& d_i =
        it_d->second;
    const std::size_t ng = m_i->constraint->get_ng();
    const std::size_t nh = m_i->constraint->get_nh();
    if (u != nullptr) {
      m_i->constraint->calcDiff(d_i, xc, *u);
      d->Gu.middleRows(ng_i, ng) = d_i->Gu;
      d->Hu.middleRows(nh_i, nh) = d_i->Hu;
    } else {
      m_i->constraint->calcDiff(d_i, xc);
    }
    d->Gx.block(ng_i, 0, ng, d_i->Gx.cols()) = d_i->Gx;
    d->Hx.block(nh_i, 0, nh, d_i->Hx.cols()) = d_i->Hx;
    ng_i += ng;
    nh_i += nh;
  }
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::calcDiffCosts(Data* d, const Eigen::Ref<const VectorXs>& x,
                           const Eigen::Ref<const VectorXs>* u) {
  if (costs_->get_state()->get_ndx() == state_->get_ndx()) {
    if (u != nullptr) {
      costs_->calcDiff(d->costs, x, *u);
    } else {
      costs_->calcDiff(d->costs, x);
    }
  } else {
    // Costs defined over the robot state, whose state Jacobian spans either
    // the robot or the whole state, are zero-padded in the thrust columns
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
        xc = x.head(costs_->get_state()->get_nx());
    d->Lx.setZero();
    d->Lu.setZero();
    d->Lxx.setZero();
    d->Lxu.setZero();
    d->Luu.setZero();
    typename CostModelSum::CostDataContainer::const_iterator it_d =
        d->costs->costs.begin();
    for (typename CostModelSum::CostModelContainer::const_iterator it_m =
             costs_->get_costs().begin();
         it_m != costs_->get_costs().end(); ++it_m, ++it_d) {
      const std::shared_ptr<typename CostModelSum::CostItem>& m_i =
          it_m->second;
      if (!m_i->active) {
        continue;
      }
      const std::shared_ptr<CostDataAbstractTpl<Scalar>>& d_i = it_d->second;
      if (u != nullptr) {
        m_i->cost->calcDiff(d_i, xc, *u);
      } else {
        m_i->cost->calcDiff(d_i, xc);
      }
      const std::size_t ndx_c = static_cast<std::size_t>(d_i->Lx.size());
      d->Lx.head(ndx_c).noalias() += m_i->weight * d_i->Lx;
      d->Lxx.topLeftCorner(ndx_c, ndx_c).noalias() += m_i->weight * d_i->Lxx;
      if (u != nullptr) {
        d->Lu.noalias() += m_i->weight * d_i->Lu;
        d->Lxu.topRows(ndx_c).noalias() += m_i->weight * d_i->Lxu;
        d->Luu.noalias() += m_i->weight * d_i->Luu;
      }
    }
  }
  if (thrust_state_) {
    calcDiffThrustCost(d, x.tail(nf_), d->Lx.tail(nf_),
                       d->Lxx.bottomRightCorner(nf_, nf_));
  } else if (u != nullptr) {
    calcDiffThrustCost(d, u->head(nf_), d->Lu.head(nf_),
                       d->Luu.topLeftCorner(nf_, nf_));
  }
}

template <typename Scalar>
Scalar
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::calcThrustCost(
    Data* d, const Eigen::Ref<const VectorXs>& thrust) {
  Scalar cost = Scalar(0.5) * thrust_reg_weight_.dot(thrust.cwiseAbs2());
  d->rlb_min = (thrust - thrust_barrier_lb_).array().min(Scalar(0.)) *
               thrust_barrier_weight_.array();
  d->rub_max = (thrust - thrust_barrier_ub_).array().max(Scalar(0.)) *
               thrust_barrier_weight_.array();
  cost += Scalar(0.5) * (d->rlb_min.matrix().squaredNorm() +
                         d->rub_max.matrix().squaredNorm());
  return cost;
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::calcDiffThrustCost(Data* d,
                                const Eigen::Ref<const VectorXs>& thrust,
                                Eigen::Ref<VectorXs> L,
                                Eigen::Ref<MatrixXs> LL) {
  // Regularization 0.5*w*f^2 and quadratic barrier 0.5*(w*(f-b))^2 outside
  // [lb, ub]
  L.array() += thrust_reg_weight_.array() * thrust.array() +
               (d->rlb_min + d->rub_max) * thrust_barrier_weight_.array();
  for (std::size_t i = 0; i < nf_; ++i) {
    const Scalar w2 = thrust_barrier_weight_(i) * thrust_barrier_weight_(i);
    LL(i, i) +=
        thrust_reg_weight_(i) +
        pinocchio::internal::if_then_else(
            pinocchio::internal::LE, thrust(i) - thrust_barrier_lb_(i),
            Scalar(0.), w2,
            pinocchio::internal::if_then_else(pinocchio::internal::GE,
                                              thrust(i) - thrust_barrier_ub_(i),
                                              Scalar(0.), w2, Scalar(0.)));
  }
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  return d != nullptr;
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::
    quasiStatic(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                std::size_t, Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  d->tmp_xstatic = x;
  d->tmp_xstatic.segment(nq, nv).setZero();
  u.setZero();

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q,
                             d->tmp_xstatic.segment(nq, nv));
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, d->tmp_xstatic.segment(nq, nv),
                  d->tmp_xstatic.segment(nq, nv));
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic.head(nq + nv));
  d->u_act.setZero();
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, d->u_act);

  // Solve g_f(q) = W_f(q)*lambda + Jc_f^T*f_ext for the thrust inputs (if they
  // are controls) and the forces of the active contacts
  const std::size_t nf = thrust_state_ ? 0 : nf_;
  const std::size_t nc_active = contacts_->get_nc();
  MatrixXs A(nvf_, nf + nc_active);
  A.leftCols(nf) = d->multibody.actuation->dtau_du.topLeftCorner(nvf_, nf);
  std::size_t col = nf;
  typename ContactModelMultiple::ContactDataContainer::const_iterator it_d =
      d->multibody.contacts->contacts.begin();
  for (typename ContactModelMultiple::ContactModelContainer::const_iterator
           it_m = contacts_->get_contacts().begin();
       it_m != contacts_->get_contacts().end(); ++it_m, ++it_d) {
    if (it_m->second->active) {
      const std::size_t nc_i = it_m->second->contact->get_nc();
      A.middleCols(col, nc_i) = it_d->second->Jc.leftCols(nvf_).transpose();
      col += nc_i;
    }
  }
  if (A.cols() == 0) {  // nothing to solve for (thrust state, no contacts)
    d->pinocchio.tau.setZero();
    return;
  }
  const VectorXs sol =
      pseudoInverse(A) *
      (d->pinocchio.tau.head(nvf_) - d->multibody.actuation->tau.head(nvf_));
  u.head(nf) = sol.head(nf);
  col = nf;
  std::size_t fid = 0;
  for (typename ContactModelMultiple::ContactModelContainer::const_iterator
           it_m = contacts_->get_contacts().begin();
       it_m != contacts_->get_contacts().end(); ++it_m) {
    const std::size_t nc_i = it_m->second->contact->get_nc();
    if (it_m->second->active) {
      u.segment(nf_ + nv + fid, nc_i) = sol.segment(col, nc_i);
      col += nc_i;
    }
    fid += nc_i;
  }
  d->pinocchio.tau.setZero();
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::
    computeEquilibriumThrust(
        const std::shared_ptr<DifferentialActionDataAbstract>& data,
        const Eigen::Ref<const VectorXs>& q) {
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  if (static_cast<std::size_t>(q.size()) != nq) {
    throw_pretty(
        "Invalid argument: " << "q has wrong dimension (it should be " +
                                    std::to_string(nq) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nj = nv - nvf_;
  const std::size_t nc = contacts_->get_nc();

  // Gravity at rest, with zero thrusts in the state
  d->tmp_xstatic.setZero();
  d->tmp_xstatic.head(nq) = q;
  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q,
                             d->tmp_xstatic.segment(nq, nv));
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, d->tmp_xstatic.segment(nq, nv),
                  d->tmp_xstatic.segment(nq, nv));
  const VectorXs g_tau = d->pinocchio.tau;
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic.head(nq + nv));
  d->u_act.setZero();
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, d->u_act);
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, d->u_act);

  // Solve g(q) = W(q)*lambda + S*tau_j + Jc^T*f_ext with bounded thrusts and
  // non-pulling contacts
  MatrixXs J = MatrixXs::Zero(nv, nf_ + nj + nc);
  if (thrust_state_) {
    J.leftCols(nf_) = d->multibody.actuation->dtau_dx.rightCols(nf_);
  } else {
    J.leftCols(nf_) = d->multibody.actuation->dtau_du.leftCols(nf_);
  }
  J.middleCols(nf_, nj) = d->multibody.actuation->dtau_du.rightCols(nj);
  VectorXs lb = VectorXs::Constant(nf_ + nj + nc,
                                   -std::numeric_limits<Scalar>::infinity());
  VectorXs ub = VectorXs::Constant(nf_ + nj + nc,
                                   std::numeric_limits<Scalar>::infinity());
  if (thrust_state_) {
    lb.head(nf_) = thrust_lb_.cwiseMax(state_->get_lb().tail(nf_));
    ub.head(nf_) = thrust_ub_.cwiseMin(state_->get_ub().tail(nf_));
  } else {
    lb.head(nf_) = Base::get_u_lb().head(nf_);
    ub.head(nf_) = Base::get_u_ub().head(nf_);
  }
  fillStaticEquilibriumContacts(*contacts_, *d->multibody.contacts,
                                d->pinocchio, true, nf_ + nj, J, lb);
  const VectorXs sol = solveStaticEquilibrium<Scalar>(J, g_tau, lb, ub);
  d->pinocchio.tau.setZero();
  return sol.head(nf_);
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelContactInvDynamicsWithThrustsTpl<NewScalar>
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::cast() const {
  typedef DifferentialActionModelContactInvDynamicsWithThrustsTpl<NewScalar>
      ReturnType;
  typedef ContactModelMultipleTpl<NewScalar> ContactType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  ReturnType ret(
      state_->template cast<NewScalar>(),
      actuation_->template cast<NewScalar>(),
      std::make_shared<ContactType>(contacts_->template cast<NewScalar>()),
      std::make_shared<CostType>(costs_->template cast<NewScalar>()),
      std::make_shared<ConstraintType>(
          constraints_->template cast<NewScalar>()));
  ret.set_thrust_reg_weight(thrust_reg_weight_.template cast<NewScalar>());
  ret.set_thrust_barrier(thrust_barrier_weight_.template cast<NewScalar>(),
                         thrust_barrier_lb_.template cast<NewScalar>(),
                         thrust_barrier_ub_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_ng() const {
  return ng_thrust_ + constraints_->get_ng();
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_nh() const {
  return nvf_ + contacts_->get_nc_total() + constraints_->get_nh();
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_ng_T() const {
  return ng_thrust_ + constraints_->get_ng_T();
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_nh_T() const {
  return constraints_->get_nh_T();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::get_g_lb()
    const {
  return g_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::get_g_ub()
    const {
  return g_ub_;
}

template <typename Scalar>
const std::shared_ptr<ActuationModelAbstractTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::get_actuation()
    const {
  return actuation_;
}

template <typename Scalar>
const std::shared_ptr<ContactModelMultipleTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::get_contacts()
    const {
  return contacts_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::get_pinocchio()
    const {
  return *pinocchio_;
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_nf() const {
  return nf_;
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_nv_floating() const {
  return nvf_;
}

template <typename Scalar>
bool DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_thrust_state() const {
  return thrust_state_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_thrust_reg_weight() const {
  return thrust_reg_weight_;
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::set_thrust_reg_weight(const VectorXs& weight) {
  if (static_cast<std::size_t>(weight.size()) != nf_) {
    throw_pretty(
        "Invalid argument: " << "weight has wrong dimension (it should be " +
                                    std::to_string(nf_) + ")");
  }
  thrust_reg_weight_ = weight;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_thrust_barrier_weight() const {
  return thrust_barrier_weight_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_thrust_barrier_lb() const {
  return thrust_barrier_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::get_thrust_barrier_ub() const {
  return thrust_barrier_ub_;
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<
    Scalar>::set_thrust_barrier(const VectorXs& weight, const VectorXs& lb,
                                const VectorXs& ub) {
  if (static_cast<std::size_t>(weight.size()) != nf_ ||
      static_cast<std::size_t>(lb.size()) != nf_ ||
      static_cast<std::size_t>(ub.size()) != nf_) {
    throw_pretty("Invalid argument: "
                 << "weight, lb and ub should have dimension nf=" << nf_);
  }
  if ((weight.array() < Scalar(0.)).any() || (lb.array() > ub.array()).any()) {
    throw_pretty("Invalid argument: "
                 << "the weights should be non-negative and lb <= ub");
  }
  thrust_barrier_weight_ = weight;
  thrust_barrier_lb_ = lb;
  thrust_barrier_ub_ = ub;
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsWithThrustsTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelContactInvDynamicsWithThrusts {nx="
     << state_->get_nx() << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
     << ", nf=" << nf_ << ", nc=" << contacts_->get_nc_total()
     << ", thrust_state=" << (thrust_state_ ? "true" : "false") << "}";
}

}  // namespace crocoddyl
