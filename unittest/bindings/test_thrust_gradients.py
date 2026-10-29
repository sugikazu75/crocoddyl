import unittest

import numpy as np
import pinocchio

import crocoddyl

EPS = 1e-6


def build_model(cost=None, constraint=None):
    model = pinocchio.buildSampleModelHumanoidRandom()
    robot = crocoddyl.StateMultibody(model)
    state = crocoddyl.StateMultibodyWithThrusts(robot, 4)
    thrusters = [
        crocoddyl.DistributedThruster(1, pinocchio.SE3(np.eye(3), np.array(p)), 0.02)
        for p in ([0.15, 0, 0], [0, 0.15, 0], [-0.15, 0, 0], [0, -0.15, 0])
    ]
    actuation = crocoddyl.ActuationModelFloatingBaseThrusterRates(state, thrusters)
    frame_id = model.getFrameId("rleg6_joint")
    contacts = crocoddyl.ContactModelMultiple(robot, actuation.nu)
    contacts.addContact(
        "c",
        crocoddyl.ContactModel6D(
            robot,
            frame_id,
            pinocchio.SE3.Identity(),
            pinocchio.LOCAL,
            actuation.nu,
            np.zeros(2),
        ),
    )
    costs = crocoddyl.CostModelSum(robot, actuation.nu)
    if cost is not None:
        costs.addCost("res", cost(robot, frame_id, actuation.nu), 1e-2)
    constraints = crocoddyl.ConstraintModelManager(robot, actuation.nu)
    if constraint is not None:
        constraints.addConstraint("res", constraint(robot, frame_id, actuation.nu))
    dam = crocoddyl.DifferentialActionModelContactFwdDynamicsWithThrusts(
        state, actuation, contacts, costs, constraints, 1e-9, True
    )
    return state, dam


def force_residual(robot, frame_id, nu):
    res = crocoddyl.ResidualModelContactForce(
        robot, frame_id, pinocchio.Force.Zero(), 6, nu
    )
    return crocoddyl.CostModelResidual(robot, res)


def cone_residual(robot, frame_id, nu):
    cone = crocoddyl.FrictionCone(np.eye(3), 0.7, 4, False)
    res = crocoddyl.ResidualModelContactFrictionCone(robot, frame_id, cone, nu)
    act = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(cone.lb, cone.ub)
    )
    return crocoddyl.CostModelResidual(robot, act, res)


def force_equality(robot, frame_id, nu):
    res = crocoddyl.ResidualModelContactForce(
        robot, frame_id, pinocchio.Force.Zero(), 6, nu
    )
    return crocoddyl.ConstraintModelResidual(robot, res)


def cone_inequality(robot, frame_id, nu):
    cone = crocoddyl.FrictionCone(np.eye(3), 0.7, 4, False)
    res = crocoddyl.ResidualModelContactFrictionCone(robot, frame_id, cone, nu)
    return crocoddyl.ConstraintModelResidual(robot, res, cone.lb, cone.ub)


class ThrustGradientTestCase(unittest.TestCase):
    COST = None
    CONSTRAINT = None

    def setUp(self):
        if self.COST is None and self.CONSTRAINT is None:
            self.skipTest("abstract test case")
        rng = np.random.default_rng(0)
        self.state, self.dam = build_model(self.COST, self.CONSTRAINT)
        self.data = self.dam.createData()
        self.x = self.state.rand()
        self.u = rng.random(self.dam.nu)
        self.ndx = self.state.ndx
        self.nf = self.state.ndx - 2 * self.state.nv

    def eval(self, x):
        self.dam.calc(self.data, x, self.u)
        return (
            self.data.cost,
            self.data.xout.copy(),
            self.data.g.copy(),
            self.data.h.copy(),
        )

    def num_diff(self):
        lx, fx = np.zeros(self.ndx), np.zeros((self.state.nv, self.ndx))
        gx = np.zeros((self.dam.ng, self.ndx))
        hx = np.zeros((self.dam.nh, self.ndx))
        for i in range(self.ndx):
            dx = np.zeros(self.ndx)
            dx[i] = EPS
            cp, ap, gp, hp = self.eval(self.state.integrate(self.x, dx))
            cm, am, gm, hm = self.eval(self.state.integrate(self.x, -dx))
            lx[i] = (cp - cm) / (2 * EPS)
            fx[:, i] = (ap - am) / (2 * EPS)
            gx[:, i] = (gp - gm) / (2 * EPS)
            hx[:, i] = (hp - hm) / (2 * EPS)
        return lx, fx, gx, hx

    def test_thrust_columns(self):
        self.dam.calc(self.data, self.x, self.u)
        self.dam.calcDiff(self.data, self.x, self.u)
        Lx, Fx = self.data.Lx.copy(), self.data.Fx.copy()
        Gx, Hx = self.data.Gx.copy(), self.data.Hx.copy()
        lx, fx, gx, hx = self.num_diff()
        thrust = np.hstack([lx[-self.nf :], gx[:, -self.nf :].ravel()])
        thrust = np.hstack([thrust, hx[:, -self.nf :].ravel()])
        self.assertGreater(np.linalg.norm(thrust), 1e-8, "Degenerate test.")
        self.assertTrue(np.allclose(Fx, fx, atol=1e-4), "Wrong Fx.")
        self.assertTrue(np.allclose(Lx, lx, atol=1e-4), "Wrong Lx.")
        self.assertTrue(np.allclose(Gx, gx, atol=1e-4), "Wrong Gx.")
        self.assertTrue(np.allclose(Hx, hx, atol=1e-4), "Wrong Hx.")


class ContactForceThrustGradientTest(ThrustGradientTestCase):
    COST = staticmethod(force_residual)


class FrictionConeThrustGradientTest(ThrustGradientTestCase):
    COST = staticmethod(cone_residual)


class ContactForceConstraintThrustGradientTest(ThrustGradientTestCase):
    CONSTRAINT = staticmethod(force_equality)


class FrictionConeConstraintThrustGradientTest(ThrustGradientTestCase):
    CONSTRAINT = staticmethod(cone_inequality)


class ThrustBarrierHessianTest(unittest.TestCase):
    def test_thrust_block(self):
        state, dam = build_model()
        nf = state.ndx - 2 * state.nv
        dam.set_thrust_barrier(
            np.array([2.0, 3.0, 4.0, 5.0]), np.full(nf, 0.5), np.full(nf, 0.6)
        )
        data = dam.createData()
        x = state.rand()
        u = np.random.default_rng(0).random(dam.nu)
        # Below, inside, above and below the bounds
        x[-nf:] = [0.0, 0.55, 1.0, 0.2]
        dam.calc(data, x, u)
        dam.calcDiff(data, x, u)
        Lxx = data.Lxx[-nf:, -nf:].copy()
        lxx = np.zeros((nf, nf))
        for i in range(nf):
            dx = np.zeros(state.ndx)
            dx[state.ndx - nf + i] = EPS
            grads = []
            for xi in (state.integrate(x, dx), state.integrate(x, -dx)):
                dam.calc(data, xi, u)
                dam.calcDiff(data, xi, u)
                grads.append(data.Lx[-nf:].copy())
            lxx[:, i] = (grads[0] - grads[1]) / (2 * EPS)
        self.assertGreater(np.linalg.norm(lxx), 1e-8, "Degenerate test.")
        self.assertTrue(np.allclose(Lxx, lxx, atol=1e-4), "Wrong thrust Lxx.")


if __name__ == "__main__":
    unittest.main()
