///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "actuation.hpp"

#include "crocoddyl/core/actuation/actuation-squashing.hpp"
#include "crocoddyl/core/actuation/squashing-base.hpp"
#include "crocoddyl/core/actuation/squashing/smooth-sat.hpp"
#include "crocoddyl/multibody/actuations/floating-base-distributed-thrusters.hpp"
#include "crocoddyl/multibody/actuations/floating-base-thrust-rates.hpp"
#include "crocoddyl/multibody/actuations/floating-base-thrusters.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ActuationModelTypes::Type> ActuationModelTypes::all(
    ActuationModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActuationModelTypes::Type type) {
  switch (type) {
    case ActuationModelTypes::ActuationModelFull:
      os << "ActuationModelFull";
      break;
    case ActuationModelTypes::ActuationModelFloatingBase:
      os << "ActuationModelFloatingBase";
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseThrusters:
      os << "ActuationModelFloatingBaseThrusters";
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseDistributedThrusters:
      os << "ActuationModelFloatingBaseDistributedThrusters";
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseThrusterRates:
      os << "ActuationModelFloatingBaseThrusterRates";
      break;
    case ActuationModelTypes::ActuationModelSquashingFull:
      os << "ActuationModelSquashingFull";
      break;
    case ActuationModelTypes::NbActuationModelTypes:
      os << "NbActuationModelTypes";
      break;
    default:
      break;
  }
  return os;
}

namespace {

/**
 * @brief Build a set of distributed thrusters anchored to body frames.
 *
 * The thrusters are spread over the deepest body frames of the tree so that
 * the thrust-to-torque mapping actually depends on the configuration.
 */
std::vector<crocoddyl::DistributedThruster> createDistributedThrusters(
    const pinocchio::Model& model, const double ctorque,
    const std::size_t n_thrusters = 4) {
  std::vector<pinocchio::FrameIndex> ids;
  for (std::size_t i = model.frames.size(); i-- > 0;) {
    if (model.frames[i].type == pinocchio::BODY &&
        model.frames[i].parentJoint > 0) {
      ids.push_back(i);
      if (ids.size() == n_thrusters) break;
    }
  }
  // Fall back to reusing the last frame when the model has fewer bodies than
  // the requested number of thrusters.
  while (ids.size() < n_thrusters) {
    ids.push_back(ids.empty() ? model.frames.size() - 1 : ids.back());
  }
  std::vector<crocoddyl::DistributedThruster> thrusters;
  for (std::size_t i = 0; i < n_thrusters; ++i) {
    thrusters.push_back(crocoddyl::DistributedThruster(
        static_cast<int>(ids[i]), model.frames[ids[i]].placement, ctorque,
        i % 2 == 0 ? crocoddyl::DT_CW : crocoddyl::DT_CCW, 0., 30., 5.));
  }
  return thrusters;
}

}  // namespace

bool requires_floating_base(ActuationModelTypes::Type actuation_type) {
  switch (actuation_type) {
    case ActuationModelTypes::ActuationModelFloatingBaseThrusters:
    case ActuationModelTypes::ActuationModelFloatingBaseDistributedThrusters:
    case ActuationModelTypes::ActuationModelFloatingBaseThrusterRates:
      return true;
    default:
      return false;
  }
}

bool requires_thrust_state(ActuationModelTypes::Type actuation_type) {
  return actuation_type ==
         ActuationModelTypes::ActuationModelFloatingBaseThrusterRates;
}

bool are_compatible(StateModelTypes::Type state_type,
                    ActuationModelTypes::Type actuation_type) {
  const bool thrust_state = get_nthrusters(state_type) != 0;
  if (requires_thrust_state(actuation_type)) {
    return thrust_state;
  }
  // Every other actuation model is built on a plain StateMultibody.
  if (thrust_state) {
    return false;
  }
  if (requires_floating_base(actuation_type)) {
    return state_type != StateModelTypes::StateMultibody_TalosArm &&
           state_type != StateModelTypes::StateMultibodyContact2D_TalosArm;
  }
  return true;
}

ActuationModelFactory::ActuationModelFactory() {}
ActuationModelFactory::~ActuationModelFactory() {}

std::shared_ptr<crocoddyl::ActuationModelAbstract>
ActuationModelFactory::create(ActuationModelTypes::Type actuation_type,
                              StateModelTypes::Type state_type) const {
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  StateModelFactory factory;
  std::shared_ptr<crocoddyl::StateAbstract> state = factory.create(state_type);
  if (!are_compatible(state_type, actuation_type)) {
    throw_pretty(__FILE__
                 ":\n Incompatible ActuationModelTypes::Type and "
                 "StateModelTypes::Type pairing");
  }
  std::shared_ptr<crocoddyl::StateMultibody> state_multibody;
  // Thruster objects
  std::vector<crocoddyl::Thruster> ps;
  const double d_cog = 0.1525;
  const double cf = 6.6e-5;
  const double cm = 1e-6;
  pinocchio::SE3 p1(Eigen::Matrix3d::Identity(), Eigen::Vector3d(d_cog, 0, 0));
  pinocchio::SE3 p2(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, d_cog, 0));
  pinocchio::SE3 p3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(-d_cog, 0, 0));
  pinocchio::SE3 p4(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, -d_cog, 0));
  ps.push_back(crocoddyl::Thruster(p1, cm / cf, crocoddyl::ThrusterType::CCW));
  ps.push_back(crocoddyl::Thruster(p2, cm / cf, crocoddyl::ThrusterType::CW));
  ps.push_back(crocoddyl::Thruster(p3, cm / cf, crocoddyl::ThrusterType::CW));
  ps.push_back(crocoddyl::Thruster(p4, cm / cf, crocoddyl::ThrusterType::CCW));
  // Distributed thruster objects. Unlike `Thruster`, they are attached to a
  // frame of the kinematic tree, so we pick the deepest body frames available.
  std::shared_ptr<crocoddyl::StateMultibodyWithThrusts> state_with_thrusts;
  std::vector<crocoddyl::DistributedThruster> dts;
  // Actuation Squashing objects
  std::shared_ptr<crocoddyl::ActuationModelAbstract> act;
  std::shared_ptr<crocoddyl::SquashingModelSmoothSat> squash;
  Eigen::VectorXd lb;
  Eigen::VectorXd ub;
  switch (actuation_type) {
    case ActuationModelTypes::ActuationModelFull:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation =
          std::make_shared<crocoddyl::ActuationModelFull>(state_multibody);
      break;
    case ActuationModelTypes::ActuationModelFloatingBase:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation = std::make_shared<crocoddyl::ActuationModelFloatingBase>(
          state_multibody);
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseThrusters:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation =
          std::make_shared<crocoddyl::ActuationModelFloatingBaseThrusters>(
              state_multibody, ps);
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseDistributedThrusters:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);
      dts = createDistributedThrusters(*state_multibody->get_pinocchio(),
                                       cm / cf);
      actuation = std::make_shared<
          crocoddyl::ActuationModelFloatingBaseDistributedThrusters>(
          state_multibody, dts);
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseThrusterRates:
      state_with_thrusts =
          std::dynamic_pointer_cast<crocoddyl::StateMultibodyWithThrusts>(
              state);
      if (state_with_thrusts == nullptr) {
        throw_pretty(__FILE__
                     ":\n ActuationModelFloatingBaseThrusterRates "
                     "requires a StateMultibodyWithThrusts");
      }
      dts = createDistributedThrusters(*state_with_thrusts->get_pinocchio(),
                                       cm / cf,
                                       state_with_thrusts->get_nthrusters());
      actuation =
          std::make_shared<crocoddyl::ActuationModelFloatingBaseThrusterRates>(
              state_with_thrusts, dts);
      break;
    case ActuationModelTypes::ActuationModelSquashingFull:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);

      act = std::make_shared<crocoddyl::ActuationModelFull>(state_multibody);

      lb = Eigen::VectorXd::Zero(state->get_nv());
      ub = Eigen::VectorXd::Zero(state->get_nv());
      lb.fill(-100.0);
      ub.fill(100.0);
      squash = std::make_shared<crocoddyl::SquashingModelSmoothSat>(
          lb, ub, state->get_nv());

      actuation = std::make_shared<crocoddyl::ActuationSquashingModel>(
          act, squash, state->get_nv());
      break;
    default:
      throw_pretty(__FILE__ ":\n Construct wrong ActuationModelTypes::Type");
      break;
  }
  return actuation;
}

}  // namespace unittest
}  // namespace crocoddyl
