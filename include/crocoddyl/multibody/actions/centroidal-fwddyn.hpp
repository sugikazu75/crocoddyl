///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CENTROIDAL_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CENTROIDAL_FWDDYN_HPP_

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/spatial/skew.hpp>

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/centroidal.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for centroidal SRBM forward dynamics.
 *
 * The state is the CoM/body SE(3) placement and LOCAL spatial velocity. The
 * control is one 6D wrench per contact, ordered as force then torque.
 *
 * \sa `DifferentialActionModelAbstractTpl`, `StateCentroidalTpl`
 */
template <typename _Scalar>
class DifferentialActionModelCentroidalFwdDynamicsTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DifferentialActionModelBase,
                         DifferentialActionModelCentroidalFwdDynamicsTpl)

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataCentroidalFwdDynamicsTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef StateCentroidalTpl<Scalar> StateCentroidal;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef pinocchio::ReferenceFrame ReferenceFrame;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Construct the centroidal forward-dynamics model.
   *
   * @param[in] state        Centroidal state
   * @param[in] nc           Number of contacts
   * @param[in] costs        Stack of costs
   * @param[in] constraints  Stack of constraints
   */
  DifferentialActionModelCentroidalFwdDynamicsTpl(
      std::shared_ptr<StateCentroidal> state, const std::size_t nc,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr);
  virtual ~DifferentialActionModelCentroidalFwdDynamicsTpl() = default;

  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;

  virtual std::shared_ptr<DifferentialActionDataAbstract> createData() override;

  template <typename NewScalar>
  DifferentialActionModelCentroidalFwdDynamicsTpl<NewScalar> cast() const;

  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data) override;

  virtual void quasiStatic(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
      const std::size_t maxiter = 100,
      const Scalar tol = Scalar(1e-9)) override;

  virtual std::size_t get_ng() const override;
  virtual std::size_t get_nh() const override;
  virtual std::size_t get_ng_T() const override;
  virtual std::size_t get_nh_T() const override;
  virtual const VectorXs& get_g_lb() const override;
  virtual const VectorXs& get_g_ub() const override;

  void set_contact_placements(const std::vector<SE3>& Ms);
  void set_contact_placement(const std::size_t i, const SE3& M);
  const std::vector<SE3>& get_contact_placements() const;

  void set_contact_status(const std::vector<bool>& active);
  void set_contact_status(const std::size_t i, const bool active);
  const std::vector<bool>& get_contact_status() const;

  void set_contact_reference(const std::vector<ReferenceFrame>& refs);
  const std::vector<ReferenceFrame>& get_contact_reference() const;

  void set_gravity(const Vector3s& g);
  const Vector3s& get_gravity() const;

  std::size_t get_ncontacts() const;
  const std::shared_ptr<CostModelSum>& get_costs() const;
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;
  const std::shared_ptr<StateCentroidal>& get_state_centroidal() const;

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::g_lb_;
  using Base::g_ub_;
  using Base::nu_;
  using Base::state_;

 private:
  void validate_reference(const ReferenceFrame ref) const;

  std::shared_ptr<StateCentroidal> state_centroidal_;
  std::shared_ptr<CostModelSum> costs_;
  std::shared_ptr<ConstraintModelManager> constraints_;
  pinocchio::ModelTpl<Scalar>* pinocchio_;
  std::size_t nc_;
  Scalar mass_;
  Matrix3s inertia_;
  Matrix3s inertia_inv_;
  Vector3s gravity_;
  std::vector<SE3> contact_placements_;
  std::vector<bool> contact_status_;
  std::vector<ReferenceFrame> contact_references_;
};

template <typename _Scalar>
struct DifferentialActionDataCentroidalFwdDynamicsTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef DataCollectorMultibodyTpl<Scalar> DataCollectorMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::Matrix3s Matrix3s;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataCentroidalFwdDynamicsTpl(
      Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio),
        costs(model->get_costs()->createData(&multibody)),
        R(Matrix3s::Identity()),
        S(Matrix3s::Identity()),
        skew_tmp(Matrix3s::Zero()),
        skew_tmp2(Matrix3s::Zero()),
        Fc(Vector3s::Zero()),
        Fb(Vector3s::Zero()),
        Fcb(Vector3s::Zero()),
        Tw(Vector3s::Zero()),
        Tb(Vector3s::Zero()),
        Iw(Vector3s::Zero()),
        f_world(Vector3s::Zero()),
        tau_world(Vector3s::Zero()),
        r_contact(Vector3s::Zero()),
        quasi_A(6, model->get_nu()),
        quasi_b(Vector6s::Zero()) {
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    quasi_A.setZero();
  }
  virtual ~DifferentialActionDataCentroidalFwdDynamicsTpl() = default;

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorMultibody multibody;
  std::shared_ptr<CostDataSumTpl<Scalar> > costs;
  std::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;
  Matrix3s R;
  Matrix3s S;
  Matrix3s skew_tmp;
  Matrix3s skew_tmp2;
  Vector3s Fc;
  Vector3s Fb;
  Vector3s Fcb;
  Vector3s Tw;
  Vector3s Tb;
  Vector3s Iw;
  Vector3s f_world;
  Vector3s tau_world;
  Vector3s r_contact;
  MatrixXs quasi_A;
  Vector6s quasi_b;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/centroidal-fwddyn.hxx>

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DifferentialActionModelCentroidalFwdDynamicsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DifferentialActionDataCentroidalFwdDynamicsTpl)

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CENTROIDAL_FWDDYN_HPP_
