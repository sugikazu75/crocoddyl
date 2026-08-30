///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-line-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelContactLineWrenchConeTpl<Scalar>::
    ResidualModelContactLineWrenchConeTpl(std::shared_ptr<StateMultibody> state,
                                          const pinocchio::FrameIndex id,
                                          const LineWrenchCone& fref,
                                          const std::size_t nu,
                                          const bool fwddyn)
    : Base(state, fref.get_nr(), nu, fwddyn ? true : false,
           fwddyn ? true : false, true),
      fwddyn_(fwddyn),
      update_jacobians_(true),
      id_(id),
      fref_(fref) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelContactLineWrenchConeTpl<Scalar>::
    ResidualModelContactLineWrenchConeTpl(std::shared_ptr<StateMultibody> state,
                                          const pinocchio::FrameIndex id,
                                          const LineWrenchCone& fref)
    : Base(state, fref.get_nr()),
      fwddyn_(true),
      update_jacobians_(true),
      id_(id),
      fref_(fref) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual of the line wrench cone. The contact wrench is already
  // expressed at the middle of the contact line, in the contact frame; only the
  // torque about the contact line, which a line contact cannot carry, is
  // dropped.
  d->lambda.template head<3>() = d->contact->f.linear();
  d->lambda[3] = d->contact->f.angular()[1];
  d->lambda[4] = d->contact->f.angular()[2];
  data->r.noalias() = fref_.get_A() * d->lambda;
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_ || update_jacobians_) {
    updateJacobians(data);
  }
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->Rx.setZero();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelContactLineWrenchConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  if (!fwddyn_) {
    updateJacobians(d);
  }
  return d;
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::updateJacobians(
    const std::shared_ptr<ResidualDataAbstract>& data) {
  Data* d = static_cast<Data*>(data.get());

  // The contact force Jacobians are already 5-dimensional, so no selection
  // matrix is needed here.
  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const MatrixX5s& A = fref_.get_A();
  data->Rx.noalias() = A * df_dx;
  data->Ru.noalias() = A * df_du;
  update_jacobians_ = false;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelContactLineWrenchConeTpl<NewScalar>
ResidualModelContactLineWrenchConeTpl<Scalar>::cast() const {
  typedef ResidualModelContactLineWrenchConeTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, fref_.template cast<NewScalar>(), nu_, fwddyn_);
  return ret;
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::print(
    std::ostream& os) const {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  std::shared_ptr<StateMultibody> s =
      std::static_pointer_cast<StateMultibody>(state_);
  os << "ResidualModelContactLineWrenchCone {frame="
     << s->get_pinocchio()->frames[id_].name
     << ", mu=" << scalar_cast<PrintableScalar>(fref_.get_mu())
     << ", length=" << scalar_cast<PrintableScalar>(fref_.get_length()) << "}";
}

template <typename Scalar>
bool ResidualModelContactLineWrenchConeTpl<Scalar>::is_fwddyn() const {
  return fwddyn_;
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactLineWrenchConeTpl<Scalar>::get_id()
    const {
  return id_;
}

template <typename Scalar>
const LineWrenchConeTpl<Scalar>&
ResidualModelContactLineWrenchConeTpl<Scalar>::get_reference() const {
  return fref_;
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactLineWrenchConeTpl<Scalar>::set_reference(
    const LineWrenchCone& reference) {
  fref_ = reference;
  update_jacobians_ = true;
}

}  // namespace crocoddyl
