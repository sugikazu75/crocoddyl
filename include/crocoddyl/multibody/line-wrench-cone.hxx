///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
LineWrenchConeTpl<Scalar>::LineWrenchConeTpl(
    const Scalar mu, const Scalar length, const std::size_t nf,
    const bool inner_appr, const Scalar min_nforce, const Scalar max_nforce)
    : nf_(nf),
      length_(length),
      mu_(mu),
      inner_appr_(inner_appr),
      min_nforce_(min_nforce),
      max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1."
              << std::endl;
  }
  if (length < Scalar(0.)) {
    length_ = Scalar(0.);
    std::cerr << "Warning: length has to be a positive value, set to 0"
              << std::endl;
  }
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0"
              << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: max_nforce has to be a positive value, set to "
                 "infinity value"
              << std::endl;
  }
  A_ = MatrixX5s::Zero(nf_ + 7, 5);
  ub_ = VectorXs::Zero(nf_ + 7);
  lb_ = VectorXs::Zero(nf_ + 7);

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
LineWrenchConeTpl<Scalar>::LineWrenchConeTpl(const WrenchConeTpl<Scalar>& cone)
    : nf_(cone.get_nf()),
      length_(cone.get_box()[0]),
      mu_(cone.get_mu()),
      inner_appr_(cone.get_inner_appr()),
      min_nforce_(cone.get_min_nforce()),
      max_nforce_(cone.get_max_nforce()) {
  A_ = MatrixX5s::Zero(nf_ + 7, 5);
  ub_ = VectorXs::Zero(nf_ + 7);
  lb_ = VectorXs::Zero(nf_ + 7);
  update();
}

template <typename Scalar>
LineWrenchConeTpl<Scalar>::LineWrenchConeTpl(
    const LineWrenchConeTpl<Scalar>& cone)
    : nf_(cone.get_nf()),
      A_(cone.get_A()),
      ub_(cone.get_ub()),
      lb_(cone.get_lb()),
      length_(cone.get_length()),
      mu_(cone.get_mu()),
      inner_appr_(cone.get_inner_appr()),
      min_nforce_(cone.get_min_nforce()),
      max_nforce_(cone.get_max_nforce()) {}

template <typename Scalar>
LineWrenchConeTpl<Scalar>::LineWrenchConeTpl()
    : nf_(4),
      A_(nf_ + 7, 5),
      ub_(nf_ + 7),
      lb_(nf_ + 7),
      length_(std::numeric_limits<Scalar>::infinity()),
      mu_(Scalar(0.7)),
      inner_appr_(true),
      min_nforce_(Scalar(0.)),
      max_nforce_(std::numeric_limits<Scalar>::infinity()) {
  A_.setZero();
  ub_.setZero();
  lb_.setZero();

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
LineWrenchConeTpl<Scalar>::~LineWrenchConeTpl() {}

template <typename Scalar>
void LineWrenchConeTpl<Scalar>::update() {
  // Initialize the matrix and bounds
  A_.setZero();
  ub_.setZero();
  lb_.setOnes();
  lb_ *= -std::numeric_limits<Scalar>::infinity();

  // Compute the mu given the type of friction cone approximation
  Scalar mu = mu_;
  Scalar theta =
      static_cast<Scalar>(2.0) * pi<Scalar>() / static_cast<Scalar>(nf_);
  if (inner_appr_) {
    mu *= cos(theta * Scalar(0.5));
  }

  // Friction cone information
  // This segment of the matrix is defined as
  // [ 1  0 -mu  0  0;
  //  -1  0 -mu  0  0;
  //   0  1 -mu  0  0;
  //   0 -1 -mu  0  0;
  //   0  0   1  0  0]
  const Vector3s mu_nsurf = -mu * Vector3s::UnitZ();
  std::size_t row = 0;
  for (std::size_t i = 0; i < nf_ / 2; ++i) {
    const Scalar theta_i = theta * static_cast<Scalar>(i);
    const Vector3s tsurf_i(cos(theta_i), sin(theta_i), Scalar(0.));
    A_.row(row++).template head<3>() = (mu_nsurf + tsurf_i).transpose();
    A_.row(row++).template head<3>() = (mu_nsurf - tsurf_i).transpose();
  }
  A_(nf_, 2) = Scalar(1.);
  lb_(nf_) = min_nforce_;
  ub_(nf_) = max_nforce_;

  // CoP information. The center of pressure has to stay on the contact line,
  // i.e. |tau_b| <= L * f_n. The rows constraining the center of pressure
  // across the line degenerate into tau_h = 0, which a line contact satisfies
  // by construction, so they are dropped.
  const Scalar L = length_ * Scalar(0.5);
  // [0  0 -L  1  0;
  //  0  0 -L -1  0]
  A_.row(nf_ + 1) << Scalar(0.), Scalar(0.), -L, Scalar(1.), Scalar(0.);
  A_.row(nf_ + 2) << Scalar(0.), Scalar(0.), -L, Scalar(-1.), Scalar(0.);

  // Yaw-tau information. Of the eight rows of a surface wrench cone, those
  // differing only by the sign of the width collapse pairwise, leaving four.
  const Scalar mu_L = -mu * L;
  // The segment of the matrix that encodes the minimum torque is defined as
  // [0  L -mu*L -mu -1;
  //  0 -L -mu*L  mu -1]
  A_.row(nf_ + 3) << Scalar(0.), L, mu_L, -mu, Scalar(-1.);
  A_.row(nf_ + 4) << Scalar(0.), -L, mu_L, mu, Scalar(-1.);
  // The segment of the matrix that encodes the maximum torque is defined as
  // [0  L -mu*L  mu  1;
  //  0 -L -mu*L -mu  1]
  A_.row(nf_ + 5) << Scalar(0.), L, mu_L, mu, Scalar(1.);
  A_.row(nf_ + 6) << Scalar(0.), -L, mu_L, -mu, Scalar(1.);
}

template <typename Scalar>
template <typename NewScalar>
LineWrenchConeTpl<NewScalar> LineWrenchConeTpl<Scalar>::cast() const {
  typedef LineWrenchConeTpl<NewScalar> ReturnType;
  ReturnType ret(scalar_cast<NewScalar>(mu_), scalar_cast<NewScalar>(length_),
                 nf_, inner_appr_, scalar_cast<NewScalar>(min_nforce_),
                 scalar_cast<NewScalar>(max_nforce_));
  return ret;
}

template <typename Scalar>
const typename LineWrenchConeTpl<Scalar>::MatrixX5s&
LineWrenchConeTpl<Scalar>::get_A() const {
  return A_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
LineWrenchConeTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
LineWrenchConeTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
std::size_t LineWrenchConeTpl<Scalar>::get_nf() const {
  return nf_;
}

template <typename Scalar>
std::size_t LineWrenchConeTpl<Scalar>::get_nr() const {
  return nf_ + 7;
}

template <typename Scalar>
const Scalar LineWrenchConeTpl<Scalar>::get_length() const {
  return length_;
}

template <typename Scalar>
const Scalar LineWrenchConeTpl<Scalar>::get_mu() const {
  return mu_;
}

template <typename Scalar>
bool LineWrenchConeTpl<Scalar>::get_inner_appr() const {
  return inner_appr_;
}

template <typename Scalar>
const Scalar LineWrenchConeTpl<Scalar>::get_min_nforce() const {
  return min_nforce_;
}

template <typename Scalar>
const Scalar LineWrenchConeTpl<Scalar>::get_max_nforce() const {
  return max_nforce_;
}

template <typename Scalar>
void LineWrenchConeTpl<Scalar>::set_length(const Scalar length) {
  length_ = length;
  if (length < Scalar(0.)) {
    length_ = Scalar(0.);
    std::cerr << "Warning: length has to be a positive value, set to 0"
              << std::endl;
  }
}

template <typename Scalar>
void LineWrenchConeTpl<Scalar>::set_mu(const Scalar mu) {
  mu_ = mu;
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1."
              << std::endl;
  }
}

template <typename Scalar>
void LineWrenchConeTpl<Scalar>::set_inner_appr(const bool inner_appr) {
  inner_appr_ = inner_appr;
}

template <typename Scalar>
void LineWrenchConeTpl<Scalar>::set_min_nforce(const Scalar min_nforce) {
  min_nforce_ = min_nforce;
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0"
              << std::endl;
  }
}

template <typename Scalar>
void LineWrenchConeTpl<Scalar>::set_max_nforce(const Scalar max_nforce) {
  max_nforce_ = max_nforce;
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: max_nforce has to be a positive value, set to "
                 "infinity value"
              << std::endl;
  }
}

template <typename Scalar>
LineWrenchConeTpl<Scalar>& LineWrenchConeTpl<Scalar>::operator=(
    const LineWrenchConeTpl<Scalar>& other) {
  if (this != &other) {
    nf_ = other.get_nf();
    A_ = other.get_A();
    ub_ = other.get_ub();
    lb_ = other.get_lb();
    length_ = other.get_length();
    mu_ = other.get_mu();
    inner_appr_ = other.get_inner_appr();
    min_nforce_ = other.get_min_nforce();
    max_nforce_ = other.get_max_nforce();
  }
  return *this;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const LineWrenchConeTpl<Scalar>& X) {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  os << "        mu: " << scalar_cast<PrintableScalar>(X.get_mu()) << std::endl;
  os << "    length: " << scalar_cast<PrintableScalar>(X.get_length())
     << std::endl;
  os << "        nf: " << X.get_nf() << std::endl;
  os << "inner_appr: ";
  if (X.get_inner_appr()) {
    os << "true" << std::endl;
  } else {
    os << "false" << std::endl;
  }
  os << " min_force: " << scalar_cast<PrintableScalar>(X.get_min_nforce())
     << std::endl;
  os << " max_force: " << scalar_cast<PrintableScalar>(X.get_max_nforce())
     << std::endl;
  return os;
}

}  // namespace crocoddyl
