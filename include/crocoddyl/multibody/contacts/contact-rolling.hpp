///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_ROLLING_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_ROLLING_HPP_

#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/**
 * @brief Rolling (line) contact model
 *
 * This contact model describes a cylindrical foot -- e.g. the "kamaboko"-shaped
 * sole used for humanoid walking -- that lies on a flat support and is free to
 * roll about its own axis. It sits between `ContactModel3DTpl` (which leaves
 * the pitch and yaw of the sole unconstrained) and `ContactModel6DTpl` (which
 * also locks the rolling degree of freedom).
 *
 * The reference frame `id` must be placed **on the cylinder axis**, not on the
 * sole surface. The rolling axis is the `axis` direction expressed in that
 * frame; it defaults to the frame's x-axis.
 *
 * The support is described by `pref`, an SE3 expressed in the world frame whose
 * columns carry the geometry of the terrain:
 *  - `pref.rotation().col(2)` is the surface normal \f$\mathbf{n}\f$,
 *  - `pref.rotation().col(0)` is the direction \f$\mathbf{h}\f$ the rolling
 *    axis should point to (i.e. the foot heading),
 *  - `pref.rotation().col(1)` is \f$\mathbf{b}=\mathbf{n}\times\mathbf{h}\f$,
 *    the direction the foot travels while rolling,
 *  - `pref.translation()` is the desired landing point on the surface.
 *
 * The roll component of `pref` is never used, since rolling is a free degree of
 * freedom. Its translation along \f$\mathbf{b}\f$ is not used either, because
 * the foot legitimately drifts along that direction as it rolls.
 *
 * The contact frame \f$C\f$ used internally is placed at the contact point
 * \f$\mathbf{p}_c = \mathbf{p}_f - r\,\mathbf{n}\f$ and oriented like the
 * support, i.e. its axes are \f$(\mathbf{h},\mathbf{b},\mathbf{n})\f$. The
 * holonomic constraint is that the *material* point of the foot located at
 * \f$\mathbf{p}_c\f$ has zero velocity and that the sole does not pitch nor
 * yaw:
 * \f[
 *   \mathbf{a}_0 = (a_h,\ a_b,\ a_n,\ \dot{\omega}_b,\ \dot{\omega}_n) =
 *   \mathbf{0},
 * \f]
 * hence `nc = 5`; only \f$\dot{\omega}_h\f$, the rolling acceleration, is left
 * free. Note that the no-slip condition applies to all three linear components,
 * including \f$\mathbf{b}\f$: rolling moves the frame origin, not the material
 * contact point.
 *
 * The Baumgarte position gain acts on four of these rows only. The
 * \f$\mathbf{b}\f$ translation is skipped, as it is not a holonomic quantity.
 *
 * The contact force is stored in `data->f` as a spatial force applied at the
 * contact point and expressed in the support axes, with a zero torque about
 * \f$\mathbf{h}\f$ -- which is exactly what a line contact can transmit. It can
 * therefore be fed to `WrenchConeTpl` built with an identity rotation and a
 * `box` of \f$(L, 0)\f$, with \f$L\f$ the length of the contact line.
 *
 * \sa `ContactModelAbstractTpl`, `calc()`, `calcDiff()`, `updateForce()`
 */
template <typename _Scalar>
class ContactModelRollingTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ContactModelBase, ContactModelRollingTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactDataRollingTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Initialize the rolling contact model
   *
   * @param[in] state   State of the multibody system
   * @param[in] id      Reference frame id, placed on the cylinder axis
   * @param[in] radius  Radius of the cylindrical sole
   * @param[in] pref    Support placement used for the Baumgarte stabilization
   * @param[in] nu      Dimension of the control vector
   * @param[in] gains   Baumgarte stabilization gains
   * @param[in] axis    Rolling axis expressed in the reference frame
   */
  ContactModelRollingTpl(std::shared_ptr<StateMultibody> state,
                         const pinocchio::FrameIndex id, const Scalar radius,
                         const SE3& pref, const std::size_t nu,
                         const Vector2s& gains = Vector2s::Zero(),
                         const Vector3s& axis = Vector3s::UnitX());

  /**
   * @brief Initialize the rolling contact model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state   State of the multibody system
   * @param[in] id      Reference frame id, placed on the cylinder axis
   * @param[in] radius  Radius of the cylindrical sole
   * @param[in] pref    Support placement used for the Baumgarte stabilization
   * @param[in] gains   Baumgarte stabilization gains
   * @param[in] axis    Rolling axis expressed in the reference frame
   */
  ContactModelRollingTpl(std::shared_ptr<StateMultibody> state,
                         const pinocchio::FrameIndex id, const Scalar radius,
                         const SE3& pref,
                         const Vector2s& gains = Vector2s::Zero(),
                         const Vector3s& axis = Vector3s::UnitX());
  virtual ~ContactModelRollingTpl() = default;

  /**
   * @brief Compute the rolling contact Jacobian and drift
   *
   * @param[in] data  Rolling contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ContactDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the rolling contact holonomic constraint
   *
   * @param[in] data  Rolling contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ContactDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   Rolling contact data
   * @param[in] force  5d force \f$(f_h, f_b, f_n, \tau_b, \tau_n)\f$
   */
  virtual void updateForce(const std::shared_ptr<ContactDataAbstract>& data,
                           const VectorXs& force) override;

  /**
   * @brief Create the rolling contact data
   */
  virtual std::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data) override;

  /**
   * @brief Cast the rolling contact model to a different scalar type.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ContactModelRollingTpl<NewScalar> A contact model with the new
   * scalar type.
   */
  template <typename NewScalar>
  ContactModelRollingTpl<NewScalar> cast() const;

  /**
   * @brief Return the radius of the cylindrical sole
   */
  Scalar get_radius() const;

  /**
   * @brief Return the support placement
   */
  const SE3& get_reference() const;

  /**
   * @brief Return the rolling axis expressed in the reference frame
   */
  const Vector3s& get_axis() const;

  /**
   * @brief Return the Baumgarte stabilization gains
   */
  const Vector2s& get_gains() const;

  /**
   * @brief Modify the radius of the cylindrical sole
   */
  void set_radius(const Scalar radius);

  /**
   * @brief Modify the support placement
   */
  void set_reference(const SE3& reference);

  /**
   * @brief Modify the rolling axis expressed in the reference frame
   */
  void set_axis(const Vector3s& axis);

  /**
   * @brief Print relevant information of the rolling contact model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::id_;
  using Base::nc_;
  using Base::nu_;
  using Base::state_;
  using Base::type_;

 private:
  Scalar radius_;   //!< Radius of the cylindrical sole
  SE3 pref_;        //!< Support placement
  Vector3s axis_;   //!< Rolling axis expressed in the reference frame
  Vector2s gains_;  //!< Baumgarte stabilization gains
  Matrix3s sRo_;    //!< Rotation from world to the support frame
};

template <typename _Scalar>
struct ContactDataRollingTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::MotionTpl<Scalar> Motion;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  ContactDataRollingTpl(Model<Scalar>* const model,
                        pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        cMf(SE3::Identity()),
        v(Motion::Zero()),
        v_c(Motion::Zero()),
        a0_local(Motion::Zero()),
        a0_full(Motion::Zero()),
        f_local(Force::Zero()),
        Jc_full(6, model->get_state()->get_nv()),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        fXjdv_dq(6, model->get_state()->get_nv()),
        fXjda_dq(6, model->get_state()->get_nv()),
        fXjda_dv(6, model->get_state()->get_nv()),
        da0_local_dx(6, model->get_state()->get_ndx()),
        dlin_dx(3, model->get_state()->get_ndx()),
        dang_dx(3, model->get_state()->get_ndx()),
        domega_c_dx(3, model->get_state()->get_ndx()),
        RcJv(3, model->get_state()->get_nv()),
        RcJw(3, model->get_state()->get_nv()),
        daxis_c_dq(3, model->get_state()->get_nv()),
        fJf_df(6, model->get_state()->get_nv()) {
    frame = model->get_id();
    jMf = model->get_state()->get_pinocchio()->frames[frame].placement;
    fXj = jMf.inverse().toActionMatrix();
    // The translation of cMf is constant: expressing -r*n in the support axes
    // always yields (0, 0, r).
    cMf.translation(Vector3s(Scalar(0.), Scalar(0.), model->get_radius()));
    pinocchio::skew(cMf.translation(), pT_skew);
    Rc.setIdentity();
    Jc_full.setZero();
    fJf.setZero();
    v_partial_dq.setZero();
    a_partial_dq.setZero();
    a_partial_dv.setZero();
    a_partial_da.setZero();
    fXjdv_dq.setZero();
    fXjda_dq.setZero();
    fXjda_dv.setZero();
    da0_local_dx.setZero();
    dlin_dx.setZero();
    dang_dx.setZero();
    domega_c_dx.setZero();
    RcJv.setZero();
    RcJw.setZero();
    daxis_c_dq.setZero();
    fJf_df.setZero();
    omega_c.setZero();
    dp.setZero();
    dp_c.setZero();
    axis_c.setZero();
    vr_lin.setZero();
    mr_lin.setZero();
    mr_ang.setZero();
    vv_skew.setZero();
    vw_skew.setZero();
    mr_lin_skew.setZero();
    mr_ang_skew.setZero();
    vr_lin_skew.setZero();
    omega_c_skew.setZero();
    axis_c_skew.setZero();
    fv_skew.setZero();
    fw_skew.setZero();
  }
  virtual ~ContactDataRollingTpl() = default;

  using Base::a0;
  using Base::da0_dx;
  using Base::df_du;
  using Base::df_dx;
  using Base::dtau_dq;
  using Base::f;
  using Base::fext;
  using Base::frame;
  using Base::fXj;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;

  SE3 cMf;      //!< Placement of the reference frame w.r.t. the contact frame
  Matrix3s Rc;  //!< Rotation from the reference frame to the support axes
  Motion v;     //!< Frame velocity in LOCAL
  Motion v_c;   //!< Contact-point velocity in the support axes
  Motion a0_local;  //!< Frame classical acceleration in LOCAL
  Motion a0_full;   //!< 6d contact drift in the support axes
  Force f_local;    //!< Contact wrench in LOCAL

  Vector3s omega_c;  //!< Angular velocity in the support axes
  Vector3s dp;       //!< Contact-point position error in world
  Vector3s dp_c;     //!< Contact-point position error in the support axes
  Vector3s axis_c;   //!< Rolling axis in the support axes
  Vector3s vr_lin;   //!< Rotated linear velocity of the frame origin
  Vector3s mr_lin;   //!< Rotated linear drift
  Vector3s mr_ang;   //!< Rotated angular drift

  Matrix6xs Jc_full;
  MatrixXs fJf;
  Matrix6xs v_partial_dq;
  Matrix6xs a_partial_dq;
  Matrix6xs a_partial_dv;
  Matrix6xs a_partial_da;
  Matrix6xs fXjdv_dq;
  Matrix6xs fXjda_dq;
  Matrix6xs fXjda_dv;
  Matrix6xs da0_local_dx;
  Matrix3xs dlin_dx;
  Matrix3xs dang_dx;
  Matrix3xs domega_c_dx;
  Matrix3xs RcJv;
  Matrix3xs RcJw;
  Matrix3xs daxis_c_dq;
  MatrixXs fJf_df;

  Matrix3s pT_skew;
  Matrix3s vv_skew;
  Matrix3s vw_skew;
  Matrix3s mr_lin_skew;
  Matrix3s mr_ang_skew;
  Matrix3s vr_lin_skew;
  Matrix3s omega_c_skew;
  Matrix3s axis_c_skew;
  Matrix3s fv_skew;
  Matrix3s fw_skew;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-rolling.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ContactModelRollingTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ContactDataRollingTpl)

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_ROLLING_HPP_
