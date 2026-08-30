///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ACTUATION_FACTORY_HPP_
#define CROCODDYL_ACTUATION_FACTORY_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/numdiff/actuation.hpp"
#include "state.hpp"

namespace crocoddyl {
namespace unittest {

struct ActuationModelTypes {
  enum Type {
    ActuationModelFull,
    ActuationModelFloatingBase,
    ActuationModelFloatingBaseThrusters,
    ActuationModelFloatingBaseDistributedThrusters,
    ActuationModelFloatingBaseThrusterRates,
    ActuationModelSquashingFull,
    NbActuationModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbActuationModelTypes);
    for (int i = 0; i < NbActuationModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ActuationModelTypes::Type type);

/**
 * @brief True for actuation models that need a floating-base robot.
 *
 * They compute their control dimension from the root joint, so a fixed-base
 * model (e.g. TalosArm) is not a valid pairing.
 */
bool requires_floating_base(ActuationModelTypes::Type actuation_type);

/**
 * @brief True for actuation models that need a `StateMultibodyWithThrusts`.
 *
 * These read the current thrust from the state, so a plain `StateMultibody`
 * is not a valid pairing.
 */
bool requires_thrust_state(ActuationModelTypes::Type actuation_type);

/**
 * @brief True when the actuation model can be built on top of the given state.
 */
bool are_compatible(StateModelTypes::Type state_type,
                    ActuationModelTypes::Type actuation_type);

class ActuationModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ActuationModelFactory();
  ~ActuationModelFactory();

  std::shared_ptr<crocoddyl::ActuationModelAbstract> create(
      ActuationModelTypes::Type actuation_type,
      StateModelTypes::Type state_type) const;
};

/**
 * @brief Update the actuation model needed for numerical differentiation.
 * We use the address of the object to avoid a copy from the
 * "boost::bind".
 *
 * @param model[in]  Pinocchio model
 * @param data[out]  Pinocchio data
 * @param x[in]      State vector
 * @param u[in]      Control vector
 */
template <typename Scalar>
void updateActuation(
    const std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<Scalar>>& model,
    const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<Scalar>>& data,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& u);

}  // namespace unittest
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "actuation.hxx"

#endif  // CROCODDYL_ACTUATION_FACTORY_HPP_
