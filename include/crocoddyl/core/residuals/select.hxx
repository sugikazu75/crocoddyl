///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/residuals/select.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelSelectTpl<Scalar>::ResidualModelSelectTpl(
    std::shared_ptr<Base> residual, const std::vector<std::size_t>& rows)
    : Base(residual->get_state(), rows.size(), residual->get_nu(),
           residual->get_q_dependent(), residual->get_v_dependent(),
           residual->get_u_dependent()),
      residual_(residual),
      rows_(rows) {
  if (rows_.size() == 0) {
    throw_pretty("Invalid argument: " << "the list of rows cannot be empty");
  }
  const std::size_t nr = residual_->get_nr();
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (rows_[i] >= nr) {
      throw_pretty("Invalid argument: "
                   << "the row index " << rows_[i]
                   << " is out of range (the wrapped residual has " +
                          std::to_string(nr) + " rows)");
    }
  }
}

template <typename Scalar>
void ResidualModelSelectTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    data->r(i) = d->residual->r(rows_[i]);
  }
}

template <typename Scalar>
void ResidualModelSelectTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x);
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    data->r(i) = d->residual->r(rows_[i]);
  }
}

template <typename Scalar>
void ResidualModelSelectTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(d->residual, x, u);
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    data->Rx.row(i) = d->residual->Rx.row(rows_[i]);
    data->Ru.row(i) = d->residual->Ru.row(rows_[i]);
  }
}

template <typename Scalar>
void ResidualModelSelectTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(d->residual, x);
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    data->Rx.row(i) = d->residual->Rx.row(rows_[i]);
  }
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelSelectTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelSelectTpl<NewScalar> ResidualModelSelectTpl<Scalar>::cast() const {
  typedef ResidualModelSelectTpl<NewScalar> ReturnType;
  ReturnType ret(residual_->template cast<NewScalar>(), rows_);
  return ret;
}

template <typename Scalar>
const std::shared_ptr<ResidualModelAbstractTpl<Scalar> >&
ResidualModelSelectTpl<Scalar>::get_residual() const {
  return residual_;
}

template <typename Scalar>
const std::vector<std::size_t>& ResidualModelSelectTpl<Scalar>::get_rows()
    const {
  return rows_;
}

template <typename Scalar>
void ResidualModelSelectTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelSelect {rows=[";
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    os << rows_[i];
    if (i + 1 < rows_.size()) {
      os << ", ";
    }
  }
  os << "], residual=" << *residual_ << "}";
}

}  // namespace crocoddyl
