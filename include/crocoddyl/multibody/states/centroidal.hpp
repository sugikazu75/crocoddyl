///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_STATES_CENTROIDAL_HPP_
#define CROCODDYL_MULTIBODY_STATES_CENTROIDAL_HPP_

#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Centroidal single-rigid-body state.
 *
 * The state is x = [c, quat_xyzw, v, w], where (c, quat_xyzw) is the SE(3)
 * placement of the CoM/body frame in the world and (v, w) is the body-frame
 * spatial velocity. The tangent follows Pinocchio's free-flyer LOCAL
 * convention, giving nx = 13 and ndx = 12.
 *
 * \sa `StateMultibodyTpl`
 */
template <typename _Scalar>
class StateCentroidalTpl : public StateMultibodyTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(StateBase, StateCentroidalTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> Base;
  typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Vector4s Vector4s;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Construct the centroidal state.
   *
   * @param[in] mass     Total mass of the robot
   * @param[in] inertia  Rotational inertia about the CoM, in body frame
   */
  StateCentroidalTpl(const Scalar mass, const Matrix3s& inertia);
  virtual ~StateCentroidalTpl() = default;

  /** @brief Return a random SE(3) pose and random body-frame twist. */
  virtual VectorXs rand() const override;

  /** @brief Return the total mass. */
  Scalar get_mass() const;

  /** @brief Return the body-frame rotational inertia about the CoM. */
  const Matrix3s& get_inertia() const;

  template <typename NewScalar>
  StateCentroidalTpl<NewScalar> cast() const;

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
  static std::shared_ptr<PinocchioModel> createModel(const Scalar mass,
                                                     const Matrix3s& inertia);

  Scalar mass_;
  Matrix3s inertia_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/states/centroidal.hxx"

extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::StateCentroidalTpl<double>;
extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::StateCentroidalTpl<float>;

#endif  // CROCODDYL_MULTIBODY_STATES_CENTROIDAL_HPP_
