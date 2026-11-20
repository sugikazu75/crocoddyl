///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_SELECT_HPP_
#define CROCODDYL_CORE_RESIDUALS_SELECT_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Select a subset of the rows of another residual
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{S}\,\mathbf{\bar{r}}(\mathbf{x},\mathbf{u})\f$, where
 * \f$\mathbf{\bar{r}}(\cdot)\f$ is the wrapped residual and \f$\mathbf{S}\f$
 * is the selection matrix built from the given row indices. The Jacobians
 * follow the same selection, i.e.
 * \f$\mathbf{R_x}=\mathbf{S}\,\mathbf{\bar{R}_x}\f$ and
 * \f$\mathbf{R_u}=\mathbf{S}\,\mathbf{\bar{R}_u}\f$.
 *
 * The motivating case is a constraint on part of a residual. Costs can already
 * ignore rows through an activation model with zero weights, but
 * `ConstraintModelResidualTpl` takes no activation: every row of the residual
 * it is given becomes a constraint row. Wrapping the residual in this class is
 * how a constraint is placed on a subset of it -- for instance pinning the
 * position, pitch and yaw of a landing foot while leaving its roll free, by
 * selecting rows \f$(0,1,2,4,5)\f$ of a
 * `ResidualModelFramePlacementTpl`.
 *
 * The rows are taken in the order they are given, so this also reorders or
 * repeats rows if asked to. The state, `nu`, and the `q`/`v`/`u` dependency
 * flags are inherited from the wrapped residual.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `ConstraintModelResidualTpl`, `calc()`,
 * `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelSelectTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelSelectTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataSelectTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the selection residual model
   *
   * @param[in] residual  Residual model to take the rows from
   * @param[in] rows      Indices of the rows to keep, in the wanted order
   */
  ResidualModelSelectTpl(std::shared_ptr<Base> residual,
                         const std::vector<std::size_t>& rows);
  virtual ~ResidualModelSelectTpl() = default;

  /**
   * @brief Compute the selected rows of the wrapped residual
   *
   * @param[in] data  Selection residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the selected rows based on the state only
   *
   * @param[in] data  Selection residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the Jacobians of the selected rows
   *
   * @param[in] data  Selection residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the Jacobians of the selected rows based on the state only
   *
   * @param[in] data  Selection residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the selection residual data
   *
   * It also creates the data of the wrapped residual, sharing the same data
   * collector.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the selection residual model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelSelectTpl<NewScalar> A residual model with the new
   * scalar type.
   */
  template <typename NewScalar>
  ResidualModelSelectTpl<NewScalar> cast() const;

  /**
   * @brief Return the wrapped residual model
   */
  const std::shared_ptr<Base>& get_residual() const;

  /**
   * @brief Return the indices of the selected rows
   */
  const std::vector<std::size_t>& get_rows() const;

  /**
   * @brief Print relevant information of the selection residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::q_dependent_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::v_dependent_;

 private:
  std::shared_ptr<Base> residual_;  //!< Wrapped residual model
  std::vector<std::size_t> rows_;   //!< Indices of the kept rows
};

template <typename _Scalar>
struct ResidualDataSelectTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  ResidualDataSelectTpl(Model<Scalar>* const model,
                        DataCollectorAbstract* const data)
      : Base(model, data), residual(model->get_residual()->createData(data)) {}
  virtual ~ResidualDataSelectTpl() = default;

  std::shared_ptr<Base> residual;  //!< Data of the wrapped residual

  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residuals/select.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelSelectTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ResidualDataSelectTpl)

#endif  // CROCODDYL_CORE_RESIDUALS_SELECT_HPP_
