///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_LINE_WRENCH_CONE_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_LINE_WRENCH_CONE_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-rolling.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/line-wrench-cone.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Contact line-wrench cone residual function
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\f$, where \f$\mathbf{A}\f$ is
 * the inequality matrix of a line wrench cone and
 * \f$\boldsymbol{\lambda}=(f_h, f_b, f_n, \tau_b, \tau_n)\f$ is the 5d contact
 * wrench computed by `ContactModelRollingTpl` through
 * `DifferentialActionModelContactFwdDynamicsTpl`.
 *
 * Unlike `ResidualModelContactWrenchConeTpl`, no padding is needed: the contact
 * force Jacobians are already 5-dimensional, so both the residual and its
 * Jacobians are a direct product with \f$\mathbf{A}\f$.
 *
 * The residual vector and Jacobians are only computed once the contact dynamics
 * has solved for the contact wrench, hence this residual function cannot be
 * used with other action models.
 *
 * \sa `ResidualModelContactWrenchConeTpl`, `ContactModelRollingTpl`,
 * `LineWrenchConeTpl`
 */
template <typename _Scalar>
class ResidualModelContactLineWrenchConeTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase,
                         ResidualModelContactLineWrenchConeTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataContactLineWrenchConeTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef LineWrenchConeTpl<Scalar> LineWrenchCone;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename LineWrenchCone::MatrixX5s MatrixX5s;

  /**
   * @brief Initialize the contact line-wrench cone residual model
   *
   * Note that for the inverse-dynamics cases, the control vector contains the
   * generalized accelerations, torques, and all the contact forces.
   *
   * @param[in] state   Multibody state
   * @param[in] id      Reference frame id of the rolling contact
   * @param[in] fref    Reference line wrench cone
   * @param[in] nu      Dimension of control vector
   * @param[in] fwddyn  Indicates that we have a forward dynamics problem
   * (default true)
   */
  ResidualModelContactLineWrenchConeTpl(std::shared_ptr<StateMultibody> state,
                                        const pinocchio::FrameIndex id,
                                        const LineWrenchCone& fref,
                                        const std::size_t nu,
                                        const bool fwddyn = true);

  /**
   * @brief Initialize the contact line-wrench cone residual model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`. Note that
   * this constructor can be used for forward-dynamics cases only.
   *
   * @param[in] state  Multibody state
   * @param[in] id     Reference frame id of the rolling contact
   * @param[in] fref   Reference line wrench cone
   */
  ResidualModelContactLineWrenchConeTpl(std::shared_ptr<StateMultibody> state,
                                        const pinocchio::FrameIndex id,
                                        const LineWrenchCone& fref);
  virtual ~ResidualModelContactLineWrenchConeTpl() = default;

  /**
   * @brief Compute the contact line-wrench cone residual
   *
   * @param[in] data  Contact line-wrench cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the residual vector for nodes that depends only on the
   * state
   *
   * It updates the residual vector based on the state only. This function is
   * used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Contact line-wrench cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the contact line-wrench cone residual
   *
   * @param[in] data  Contact line-wrench cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the Jacobian of the residual functions with respect to the
   * state only
   *
   * @param[in] data  Contact line-wrench cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the contact line-wrench cone residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the contact line-wrench cone residual model to a different
   * scalar type.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelContactLineWrenchConeTpl<NewScalar> A residual model
   * with the new scalar type.
   */
  template <typename NewScalar>
  ResidualModelContactLineWrenchConeTpl<NewScalar> cast() const;

  /**
   * @brief Indicates if we are using the forward-dynamics (true) or
   * inverse-dynamics (false)
   */
  bool is_fwddyn() const;

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference line wrench cone
   */
  const LineWrenchCone& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  DEPRECATED("Do not use set_id, instead create a new model",
             void set_id(const pinocchio::FrameIndex id);)

  /**
   * @brief Modify the reference line wrench cone
   */
  void set_reference(const LineWrenchCone& reference);

  /**
   * @brief Print relevant information of the contact line-wrench cone residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::v_dependent_;

 private:
  void updateJacobians(const std::shared_ptr<ResidualDataAbstract>& data);

  bool fwddyn_;  //!< Indicates if we are using this function for forward
                 //!< dynamics
  bool update_jacobians_;     //!< Indicates if we need to update the Jacobians
  pinocchio::FrameIndex id_;  //!< Reference frame id
  LineWrenchCone fref_;       //!< Reference line wrench cone
};

template <typename _Scalar>
struct ResidualDataContactLineWrenchConeTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef Eigen::Matrix<Scalar, 5, 1> Vector5s;

  template <template <typename Scalar> class Model>
  ResidualDataContactLineWrenchConeTpl(Model<Scalar>* const model,
                                       DataCollectorAbstract* const data)
      : Base(model, data), lambda(Vector5s::Zero()) {
    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar>* d =
        dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorContact");
    }

    // Avoids data casting at runtime
    const pinocchio::FrameIndex id = model->get_id();
    const std::shared_ptr<StateMultibody>& state =
        std::static_pointer_cast<StateMultibody>(model->get_state());
    const std::string frame_name = state->get_pinocchio()->frames[id].name;
    bool found_contact = false;
    for (typename ContactModelMultiple::ContactDataContainer::iterator it =
             d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == id) {
        ContactDataRollingTpl<Scalar>* dr =
            dynamic_cast<ContactDataRollingTpl<Scalar>*>(it->second.get());
        if (dr == NULL) {
          throw_pretty(
              "Domain error: there isn't defined a rolling contact for " +
              frame_name);
        }
        found_contact = true;
        contact = it->second;
        break;
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " +
                   frame_name);
    }
  }
  virtual ~ResidualDataContactLineWrenchConeTpl() = default;

  std::shared_ptr<ForceDataAbstractTpl<Scalar> >
      contact;      //!< Contact force data
  Vector5s lambda;  //!< Contact wrench in the line-contact coordinates
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/contact-line-wrench-cone.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelContactLineWrenchConeTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataContactLineWrenchConeTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_LINE_WRENCH_CONE_HPP_
