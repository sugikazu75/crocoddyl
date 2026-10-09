///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_UTILS_STATIC_EQUILIBRIUM_HPP_
#define CROCODDYL_MULTIBODY_UTILS_STATIC_EQUILIBRIUM_HPP_

#include <algorithm>
#include <vector>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/core/utils/scalar.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {

/**
 * @brief Fill the contact columns of a static-equilibrium problem
 *
 * For each active contact, it writes the transposed contact Jacobian into
 * consecutive columns of `J`, starting at `col`, with the force expressed in
 * the contact frame (LOCAL), and bounds the normal force from below by zero.
 * The rows of `contacts.Jc` follow the active contacts, or all the contacts
 * when `all_rows` is true (i.e. when the contact model computes all
 * contacts).
 *
 * @param[in] model     Multiple contact model
 * @param[in] data      Multiple contact data
 * @param[in] pinocchio Pinocchio data with updated frame placements
 * @param[in] all_rows  True if `data.Jc` contains the inactive contacts too
 * @param[in] col       First column of the contact block in `J` and `lb`
 * @param[out] J        Equilibrium matrix (nv x nz)
 * @param[out] lb       Lower bounds of the unknowns (nz)
 */
template <typename Scalar>
void fillStaticEquilibriumContacts(const ContactModelMultipleTpl<Scalar>& model,
                                   const ContactDataMultipleTpl<Scalar>& data,
                                   const pinocchio::DataTpl<Scalar>& pinocchio,
                                   const bool all_rows, const std::size_t col,
                                   typename MathBaseTpl<Scalar>::MatrixXs& J,
                                   typename MathBaseTpl<Scalar>::VectorXs& lb) {
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const std::size_t nv = static_cast<std::size_t>(J.rows());
  std::size_t row = 0;
  std::size_t offset = col;
  for (typename ContactModelMultipleTpl<
           Scalar>::ContactModelContainer::const_iterator it =
           model.get_contacts().begin();
       it != model.get_contacts().end(); ++it) {
    const std::size_t nc_i = it->second->contact->get_nc();
    if (!it->second->active) {
      row += all_rows ? nc_i : 0;
      continue;
    }
    const Eigen::Block<const MatrixXs> Jc_i = data.Jc.block(row, 0, nc_i, nv);
    Eigen::Block<MatrixXs> J_i = J.block(0, offset, nv, nc_i);
    // The force bound applies along the normal of the contact frame, so the
    // Jacobians expressed in WORLD / LOCAL_WORLD_ALIGNED are rotated to LOCAL
    if (it->second->contact->get_type() == pinocchio::ReferenceFrame::LOCAL ||
        nc_i == 2) {
      J_i.noalias() = Jc_i.transpose();
    } else if (nc_i == 3 || nc_i == 6) {
      const Eigen::Ref<const typename MathBaseTpl<Scalar>::Matrix3s> oRf =
          pinocchio.oMf[it->second->contact->get_id()].rotation();
      MatrixXs Jc_local = MatrixXs::Zero(nc_i, nv);
      Jc_local.template topRows<3>().noalias() =
          oRf.transpose() * Jc_i.template topRows<3>();
      if (nc_i == 6) {
        Jc_local.template bottomRows<3>().noalias() =
            oRf.transpose() * Jc_i.template bottomRows<3>();
      }
      J_i.noalias() = Jc_local.transpose();
    } else {
      J_i.noalias() = Jc_i.transpose();
    }
    if (nc_i > 0) {
      const std::size_t normal_id = nc_i == 1 ? 0 : (nc_i == 2 ? 1 : 2);
      lb(offset + normal_id) = Scalar(0.);
    }
    row += nc_i;
    offset += nc_i;
  }
}

/**
 * @brief Solve a bounded minimum-norm static equilibrium
 *
 * It solves \f$\min_\mathbf{z} \frac{1}{2}\|\mathbf{z}\|^2\f$ subject to
 * \f$\mathbf{J}\mathbf{z} = \mathbf{g}\f$ and \f$\mathbf{lb} \leq \mathbf{z}
 * \leq \mathbf{ub}\f$ with a primal active-set method.
 *
 * @param[in] J   Equilibrium matrix (nv x nz)
 * @param[in] g   Generalized gravity forces (nv)
 * @param[in] lb  Lower bounds (nz)
 * @param[in] ub  Upper bounds (nz)
 * @return the solution \f$\mathbf{z}\f$
 */
template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs solveStaticEquilibrium(
    const typename MathBaseTpl<Scalar>::MatrixXs& J,
    const typename MathBaseTpl<Scalar>::VectorXs& g,
    const typename MathBaseTpl<Scalar>::VectorXs& lb,
    const typename MathBaseTpl<Scalar>::VectorXs& ub) {
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const std::size_t nv = static_cast<std::size_t>(J.rows());
  const std::size_t nz = static_cast<std::size_t>(J.cols());
  for (std::size_t i = 0; i < nz; ++i) {
    if (lb(i) > ub(i)) {
      throw_pretty(
          "Invalid argument: " << "The thrust bounds are inconsistent");
    }
  }

  // Weighted minimum-norm static solution.  Thrust is more expensive than joint
  // torques/contact forces, so ground support is used before thrust, while a
  // ceiling contact cannot pull because of the normal-force lower bound.
  VectorXs weights = VectorXs::Constant(nz, Scalar(1.));
  VectorXs inv_weights = weights.cwiseInverse();
  VectorXs sol = VectorXs::Zero(nz);
  VectorXs fixed = VectorXs::Zero(nz);
  VectorXs mu = VectorXs::Zero(nv);
  std::vector<int> active_bound(nz, 0);  // -1: lower, +1: upper, 0: free
  const Scalar bound_tol = ScaleNumerics<Scalar>(1e-9);
  const Scalar eq_tol = ScaleNumerics<Scalar>(1e-8);

  for (std::size_t iter = 0; iter <= 4 * nz + 10; ++iter) {
    std::vector<std::size_t> free_ids;
    free_ids.reserve(nz);
    VectorXs rhs = g;
    for (std::size_t i = 0; i < nz; ++i) {
      if (active_bound[i] != 0) {
        fixed(i) = active_bound[i] < 0 ? lb(i) : ub(i);
        rhs.noalias() -= J.col(i) * fixed(i);
        sol(i) = fixed(i);
      } else {
        free_ids.push_back(i);
      }
    }

    const std::size_t nfree = free_ids.size();
    if (nfree > 0) {
      MatrixXs AWinvAt = MatrixXs::Zero(nv, nv);
      for (std::size_t j = 0; j < nfree; ++j) {
        const std::size_t id = free_ids[j];
        AWinvAt.noalias() +=
            inv_weights(id) * J.col(id) * J.col(id).transpose();
      }
      mu.noalias() = -pseudoInverse(AWinvAt) * rhs;
      for (std::size_t j = 0; j < nfree; ++j) {
        const std::size_t id = free_ids[j];
        sol(id) = -inv_weights(id) * J.col(id).dot(mu);
      }
    } else {
      mu.setZero();
    }

    std::size_t worst_id = nz;
    Scalar worst_violation = Scalar(0.);
    int worst_bound = 0;
    for (std::size_t j = 0; j < nfree; ++j) {
      const std::size_t id = free_ids[j];
      if (sol(id) < lb(id) - bound_tol) {
        const Scalar violation = lb(id) - sol(id);
        if (violation > worst_violation) {
          worst_violation = violation;
          worst_id = id;
          worst_bound = -1;
        }
      } else if (sol(id) > ub(id) + bound_tol) {
        const Scalar violation = sol(id) - ub(id);
        if (violation > worst_violation) {
          worst_violation = violation;
          worst_id = id;
          worst_bound = 1;
        }
      }
    }
    if (worst_id != nz) {
      active_bound[worst_id] = worst_bound;
      continue;
    }

    if ((J * sol - g).norm() > eq_tol * std::max(Scalar(1.), g.norm())) {
      throw_pretty("Runtime error: "
                   << "No feasible bounded static equilibrium was found");
    }

    VectorXs gradient = weights.cwiseProduct(sol);
    gradient.noalias() += J.transpose() * mu;
    worst_id = nz;
    worst_violation = Scalar(0.);
    for (std::size_t i = 0; i < nz; ++i) {
      if (active_bound[i] == 0) {
        continue;
      }
      const Scalar violation = active_bound[i] < 0 ? -gradient(i) : gradient(i);
      if (violation > worst_violation + bound_tol) {
        worst_violation = violation;
        worst_id = i;
      }
    }
    if (worst_id != nz) {
      active_bound[worst_id] = 0;
    } else {
      return sol;
    }
  }
  throw_pretty("Runtime error: "
               << "The bounded static equilibrium QP did not converge");
}

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_UTILS_STATIC_EQUILIBRIUM_HPP_
