///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <limits>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/spatial/inertia.hpp>

namespace crocoddyl {

template <typename Scalar>
StateCentroidalTpl<Scalar>::StateCentroidalTpl(const Scalar mass,
                                               const Matrix3s& inertia)
    : Base(createModel(mass, inertia)), mass_(mass), inertia_(inertia) {
  lb_.setConstant(-std::numeric_limits<Scalar>::infinity());
  ub_.setConstant(std::numeric_limits<Scalar>::infinity());
  Base::update_has_limits();
}

template <typename Scalar>
std::shared_ptr<pinocchio::ModelTpl<Scalar> >
StateCentroidalTpl<Scalar>::createModel(const Scalar mass,
                                        const Matrix3s& inertia) {
  typedef pinocchio::ModelTpl<Scalar> Model;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef pinocchio::InertiaTpl<Scalar> Inertia;

  if (mass <= Scalar(0.)) {
    throw_pretty("Invalid argument: " << "mass has to be positive");
  }
  if (!inertia.isApprox(inertia.transpose(),
                        Eigen::NumTraits<Scalar>::dummy_precision())) {
    throw_pretty("Invalid argument: " << "inertia has to be symmetric");
  }
  Eigen::LLT<Matrix3s> llt(inertia);
  if (llt.info() != Eigen::Success) {
    throw_pretty("Invalid argument: " << "inertia has to be positive definite");
  }

  std::shared_ptr<Model> model =
      std::allocate_shared<Model>(Eigen::aligned_allocator<Model>());
  const pinocchio::JointIndex jid =
      model->addJoint(0, pinocchio::JointModelFreeFlyerTpl<Scalar>(),
                      SE3::Identity(), "root_joint");
  model->appendBodyToJoint(jid, Inertia(mass, Vector3s::Zero(), inertia),
                           SE3::Identity());
  model->addFrame(pinocchio::FrameTpl<Scalar>(
      "centroidal", jid, jid, SE3::Identity(), pinocchio::BODY));
  return model;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateCentroidalTpl<Scalar>::rand()
    const {
  VectorXs xrand = VectorXs::Random(nx_);
  xrand.head(3) = Vector3s::Random();

  Vector4s qcoeffs = Vector4s::Random();
  if (qcoeffs.squaredNorm() <= Eigen::NumTraits<Scalar>::epsilon()) {
    qcoeffs << Scalar(0.), Scalar(0.), Scalar(0.), Scalar(1.);
  } else {
    qcoeffs.normalize();
  }
  xrand.segment(3, 4) = qcoeffs;
  xrand.tail(nv_) = VectorXs::Random(nv_);
  return xrand;
}

template <typename Scalar>
Scalar StateCentroidalTpl<Scalar>::get_mass() const {
  return mass_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s&
StateCentroidalTpl<Scalar>::get_inertia() const {
  return inertia_;
}

template <typename Scalar>
template <typename NewScalar>
StateCentroidalTpl<NewScalar> StateCentroidalTpl<Scalar>::cast() const {
  typedef StateCentroidalTpl<NewScalar> ReturnType;
  ReturnType ret(scalar_cast<NewScalar>(mass_),
                 inertia_.template cast<NewScalar>());
  ret.set_lb(lb_.template cast<NewScalar>());
  ret.set_ub(ub_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void StateCentroidalTpl<Scalar>::print(std::ostream& os) const {
  os << "StateCentroidal {nx=" << nx_ << ", ndx=" << ndx_ << ", mass=" << mass_
     << ", inertia=" << inertia_ << "}";
}

}  // namespace crocoddyl
