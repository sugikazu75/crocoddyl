import unittest

import numpy as np
import pinocchio

import crocoddyl

EPS = 1e-6


def build_model(residual):
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
    costs.addCost("res", residual(robot, frame_id, actuation.nu), 1e-2)
    constraints = crocoddyl.ConstraintModelManager(robot, actuation.nu)
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


class ThrustGradientTestCase(unittest.TestCase):
    RESIDUAL = None

    def setUp(self):
        if self.RESIDUAL is None:
            self.skipTest("abstract test case")
        rng = np.random.default_rng(0)
        self.state, self.dam = build_model(self.RESIDUAL)
        self.data = self.dam.createData()
        self.x = self.state.rand()
        self.u = rng.random(self.dam.nu)
        self.ndx = self.state.ndx

    def eval(self, x):
        self.dam.calc(self.data, x, self.u)
        return self.data.cost, self.data.xout.copy()

    def num_diff(self):
        lx, fx = np.zeros(self.ndx), np.zeros((self.state.nv, self.ndx))
        for i in range(self.ndx):
            dx = np.zeros(self.ndx)
            dx[i] = EPS
            cp, ap = self.eval(self.state.integrate(self.x, dx))
            cm, am = self.eval(self.state.integrate(self.x, -dx))
            lx[i] = (cp - cm) / (2 * EPS)
            fx[:, i] = (ap - am) / (2 * EPS)
        return lx, fx

    def test_thrust_columns(self):
        self.dam.calc(self.data, self.x, self.u)
        self.dam.calcDiff(self.data, self.x, self.u)
        lx, fx = self.num_diff()
        nf = self.state.ndx - 2 * self.state.nv
        self.assertGreater(np.linalg.norm(lx[-nf:]), 1e-8, "Degenerate test.")
        self.assertTrue(np.allclose(self.data.Fx, fx, atol=1e-4), "Wrong Fx.")
        self.assertTrue(np.allclose(self.data.Lx, lx, atol=1e-4), "Wrong Lx.")


class ContactForceThrustGradientTest(ThrustGradientTestCase):
    RESIDUAL = staticmethod(force_residual)


class FrictionConeThrustGradientTest(ThrustGradientTestCase):
    RESIDUAL = staticmethod(cone_residual)


if __name__ == "__main__":
    unittest.main()
