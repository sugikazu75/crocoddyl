///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, The University of Tokyo
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_LINE_WRENCH_CONE_HPP_
#define CROCODDYL_MULTIBODY_LINE_WRENCH_CONE_HPP_

#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/**
 * @brief This class encapsulates the wrench cone of a line contact
 *
 * A line contact -- a cylindrical foot resting on a flat support, as modelled
 * by `ContactModelRollingTpl` -- transmits a 5d wrench
 * \f$(f_h, f_b, f_n, \tau_b, \tau_n)\f$ rather than a full 6d one: it carries
 * no torque about the contact line itself. This class is therefore the
 * degenerate case of `WrenchConeTpl` for a support box of zero width, written
 * directly in 5 dimensions.
 *
 * Compared to the \f$nf+13\f$ rows of a surface wrench cone, only \f$nf+7\f$
 * survive: the two center-of-pressure rows across the line collapse to the
 * identity \f$\tau_h=0\f$, and the eight yaw-torque rows collapse pairwise into
 * four. The rows are, in order, the \f$nf\f$ friction facets, the normal force,
 * the two center-of-pressure bounds along the line, and the four yaw-torque
 * bounds.
 *
 * Unlike `WrenchConeTpl` this class takes no rotation matrix. Rotating a 5d
 * line wrench is not meaningful in general -- any rotation but the identity
 * would tilt the contact line or the surface normal and reintroduce a torque
 * about \f$h\f$. The cone is always expressed in the contact frame produced by
 * `ContactModelRollingTpl`, whose x-axis is the contact line and whose z-axis
 * is the surface normal.
 *
 * /sa `WrenchConeTpl`, `FrictionConeTpl`, `ContactModelRollingTpl`
 */
template <typename _Scalar>
class LineWrenchConeTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 5, 1> Vector5s;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 5> MatrixX5s;

  template <typename>
  friend class LineWrenchConeTpl;

  /**
   * @brief Initialize the line wrench cone
   *
   * @param[in] mu          Friction coefficient
   * @param[in] length      Length of the contact line
   * @param[in] nf          Number of facets (default 4)
   * @param[in] inner_appr  Label that describes the type of friction cone
   * approximation (inner/outer)
   * @param[in] min_nforce  Minimum normal force (default 0.)
   * @param[in] max_nforce  Maximum normal force (default infinity)
   */
  LineWrenchConeTpl(
      const Scalar mu, const Scalar length, const std::size_t nf = 4,
      const bool inner_appr = true, const Scalar min_nforce = Scalar(0.),
      const Scalar max_nforce = std::numeric_limits<Scalar>::infinity());

  /**
   * @brief Initialize the line wrench cone from a surface one
   *
   * The width of `cone`'s support box is ignored: only its length, friction
   * coefficient, facets and normal-force bounds are kept.
   *
   * @param[in] cone  Surface wrench cone
   */
  explicit LineWrenchConeTpl(const WrenchConeTpl<Scalar>& cone);

  /**
   * @brief Initialize the line wrench cone
   *
   * @param[in] cone  Line wrench cone
   */
  LineWrenchConeTpl(const LineWrenchConeTpl<Scalar>& cone);

  /**
   * @brief Initialize the line wrench cone
   */
  explicit LineWrenchConeTpl();
  ~LineWrenchConeTpl();

  /**
   * @brief Update the matrix of line wrench cone inequalities
   *
   * This matrix-vector pair describes the feasible set as
   * \f$ lb \leq A \times \lambda \leq ub \f$, with the 5d wrench
   * \f$\lambda = (f_h, f_b, f_n, \tau_b, \tau_n)\f$ expressed at the middle of
   * the contact line, in the contact frame.
   */
  void update();

  /**
   * @brief Cast the line wrench cone to a different scalar type.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return LineWrenchConeTpl<NewScalar> A cone with the new scalar type.
   */
  template <typename NewScalar>
  LineWrenchConeTpl<NewScalar> cast() const;

  /**
   * @brief Return the matrix of the line wrench cone
   */
  const MatrixX5s& get_A() const;

  /**
   * @brief Return the lower bound of inequalities
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Return the upper bound of inequalities
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Return the number of facets
   */
  std::size_t get_nf() const;

  /**
   * @brief Return the number of inequalities, i.e. `nf + 7`
   */
  std::size_t get_nr() const;

  /**
   * @brief Return the length of the contact line
   */
  const Scalar get_length() const;

  /**
   * @brief Return the friction coefficient
   */
  const Scalar get_mu() const;

  /**
   * @brief Return the label that describes the type of friction cone
   * approximation (inner/outer)
   */
  bool get_inner_appr() const;

  /**
   * @brief Return the minimum normal force
   */
  const Scalar get_min_nforce() const;

  /**
   * @brief Return the maximum normal force
   */
  const Scalar get_max_nforce() const;

  /**
   * @brief Modify the length of the contact line
   *
   * Note that you need to run `update` for updating the inequality matrix and
   * bounds.
   */
  void set_length(const Scalar length);

  /**
   * @brief Modify the friction coefficient
   *
   * Note that you need to run `update` for updating the inequality matrix and
   * bounds.
   */
  void set_mu(const Scalar mu);

  /**
   * @brief Modify the label that describes the type of friction cone
   * approximation (inner/outer)
   *
   * Note that you need to run `update` for updating the inequality matrix and
   * bounds.
   */
  void set_inner_appr(const bool inner_appr);

  /**
   * @brief Modify the minimum normal force
   *
   * Note that you need to run `update` for updating the inequality matrix and
   * bounds.
   */
  void set_min_nforce(const Scalar min_nforce);

  /**
   * @brief Modify the maximum normal force
   *
   * Note that you need to run `update` for updating the inequality matrix and
   * bounds.
   */
  void set_max_nforce(const Scalar max_nforce);

  LineWrenchConeTpl<Scalar>& operator=(const LineWrenchConeTpl<Scalar>& other);

  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const LineWrenchConeTpl<Scalar>& X);

 private:
  std::size_t nf_;     //!< Number of facets
  MatrixX5s A_;        //!< Matrix of the line wrench cone
  VectorXs ub_;        //!< Upper bound of the line wrench cone
  VectorXs lb_;        //!< Lower bound of the line wrench cone
  Scalar length_;      //!< Length of the contact line
  Scalar mu_;          //!< Friction coefficient
  bool inner_appr_;    //!< Label that describes the type of friction cone
                       //!< approximation (inner/outer)
  Scalar min_nforce_;  //!< Minimum normal force
  Scalar max_nforce_;  //!< Maximum normal force
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/line-wrench-cone.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::LineWrenchConeTpl)

#endif  // CROCODDYL_MULTIBODY_LINE_WRENCH_CONE_HPP_
