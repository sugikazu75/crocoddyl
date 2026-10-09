import unittest

import numpy as np

import crocoddyl

NX, NU, T = 4, 3, 10


def lqr(H, h):
    rng = np.random.default_rng(0)
    A = np.eye(NX) + 0.1 * rng.standard_normal((NX, NX))
    B = rng.standard_normal((NX, NU))
    return crocoddyl.ActionModelLQR(
        A,
        B,
        np.eye(NX),
        np.eye(NU),
        np.zeros((NX, NU)),
        np.zeros((0, NX + NU)),
        H,
        np.zeros(NX),
        np.ones(NX),
        np.ones(NU),
        np.zeros(0),
        h,
    )


def solve(model, eq_solver):
    x0 = np.ones(NX)
    problem = crocoddyl.ShootingProblem(
        x0, [model] * T, lqr(np.zeros((0, NX + NU)), np.zeros(0))
    )
    solver = crocoddyl.SolverIntro(problem, crocoddyl.FeasShoot, eq_solver)
    solver.solve([x0] * (T + 1), [np.zeros(NU)] * T, 20)
    return solver


class SolverIntroRankDeficiencyTest(unittest.TestCase):
    # A control-dependent constraint
    H_FULL = np.array([[0.3, -0.2, 0.0, 0.1, 1.0, 0.5, -0.4]])
    H_STATE = np.array([[1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]])

    def check_dependent(self, eq_solver):
        # The same constraint again (linearly dependent) and a zero row are
        # dropped, so the solution is the one of the full-rank problem
        ref = solve(lqr(self.H_FULL, np.array([0.2])), eq_solver)
        H = np.vstack([self.H_FULL, self.H_FULL, np.zeros((1, NX + NU))])
        sol = solve(lqr(H, np.array([0.2, 0.2, 0.0])), eq_solver)
        self.assertTrue(all(r == 1 for r in sol.Hu_rank), "Wrong rank.")
        for u_ref, u in zip(ref.us, sol.us):
            self.assertTrue(np.allclose(u_ref, u, atol=1e-8), "Wrong control.")

    def check_state_only(self, eq_solver):
        # A state-only constraint cannot be satisfied by the control of its node
        # and is ignored, while the control-dependent one is still satisfied
        H = np.vstack([self.H_FULL, self.H_STATE])
        sol = solve(lqr(H, np.array([0.2, 0.0])), eq_solver)
        self.assertTrue(all(r == 1 for r in sol.Hu_rank), "Wrong rank.")
        for d in sol.problem.runningDatas:
            self.assertLess(abs(d.h[0]), 1e-6, "Constraint not satisfied.")

    def test_lu_nullspace(self):
        self.check_dependent(crocoddyl.LuNull)
        self.check_state_only(crocoddyl.LuNull)

    def test_qr_nullspace(self):
        self.check_dependent(crocoddyl.QrNull)
        self.check_state_only(crocoddyl.QrNull)


if __name__ == "__main__":
    unittest.main()
