import unittest

import numpy as np
import pinocchio

import crocoddyl

EPS = 1e-6
TOL = 1e-4


def build_model(thrust_state):
    model = pinocchio.buildSampleModelHumanoidRandom()
    model.effortLimit[:] = 100.0
    # Thruster frames match the thruster poses so that W(q) and its derivative
    # describe the same wrenches
    thrusters = []
    for i, (joint, p) in enumerate(
        [
            ("root_joint", [0.15, 0, 0]),
            ("root_joint", [-0.15, 0, 0]),
            ("rarm5_joint", [0, 0.05, 0]),
            ("larm5_joint", [0, -0.05, 0]),
            ("lleg5_joint", [0.05, 0, 0]),
        ]
    ):
        jid = model.getJointId(joint)
        rot = pinocchio.utils.rotate("x", 0.3 * i) @ pinocchio.utils.rotate("y", 0.2)
        placement = pinocchio.SE3(rot, np.array(p))
        fid = model.addFrame(
            pinocchio.Frame(
                f"thruster{i}", jid, 0, placement, pinocchio.FrameType.OP_FRAME
            )
        )
        thrusters.append(
            crocoddyl.DistributedThruster(
                fid,
                placement,
                0.02,
                crocoddyl.DT_CW if i % 2 else crocoddyl.DT_CCW,
                0.0,
                30.0,
                5.0,
            )
        )
    robot = crocoddyl.StateMultibody(model)
    if thrust_state:
        state = crocoddyl.StateMultibodyWithThrusts(robot, len(thrusters))
        actuation = crocoddyl.ActuationModelFloatingBaseThrusterRates(state, thrusters)
    else:
        state = robot
        actuation = crocoddyl.ActuationModelFloatingBaseDistributedThrusters(
            robot, thrusters
        )
    nu = len(thrusters) + model.nv + 6 + 3
    contacts = crocoddyl.ContactModelMultiple(robot, nu)
    contacts.addContact(
        "rfoot",
        crocoddyl.ContactModel6D(
            robot,
            model.getFrameId("rleg6_joint"),
            pinocchio.SE3.Identity(),
            pinocchio.LOCAL,
            nu,
            np.array([0.0, 10.0]),
        ),
    )
    contacts.addContact(
        "lhand",
        crocoddyl.ContactModel3D(
            robot,
            model.getFrameId("larm6_body"),
            np.zeros(3),
            pinocchio.LOCAL_WORLD_ALIGNED,
            nu,
            np.array([0.0, 10.0]),
        ),
        False,
    )
    effort = crocoddyl.ResidualModelJointEffort(
        robot, actuation, np.zeros(actuation.nu), nu, False
    )
    costs = crocoddyl.CostModelSum(robot, nu)
    costs.addCost("effort", crocoddyl.CostModelResidual(robot, effort), 1e-2)
    costs.addCost(
        "force",
        crocoddyl.CostModelResidual(
            robot,
            crocoddyl.ResidualModelContactForce(
                robot,
                model.getFrameId("rleg6_joint"),
                pinocchio.Force.Zero(),
                6,
                nu,
            ),
        ),
        1e-3,
    )
    costs.addCost(
        "xreg",
        crocoddyl.CostModelResidual(
            robot, crocoddyl.ResidualModelState(robot, robot.zero(), nu)
        ),
        1e-1,
    )
    costs.addCost(
        "ureg",
        crocoddyl.CostModelResidual(robot, crocoddyl.ResidualModelControl(robot, nu)),
        1e-3,
    )
    constraints = crocoddyl.ConstraintModelManager(robot, nu)
    lim = model.effortLimit[6:]
    constraints.addConstraint(
        "effort",
        crocoddyl.ConstraintModelResidual(
            robot,
            crocoddyl.ResidualModelJointEffort(
                robot, actuation, np.zeros(actuation.nu), nu, False
            ),
            np.hstack([-np.inf * np.ones(len(thrusters)), -lim]),
            np.hstack([np.inf * np.ones(len(thrusters)), lim]),
        ),
    )
    # Inactive at creation, as the touchdown constraints of the gait planners
    constraints.addConstraint(
        "placement",
        crocoddyl.ConstraintModelResidual(
            robot,
            crocoddyl.ResidualModelFramePlacement(
                robot, model.getFrameId("lleg6_joint"), pinocchio.SE3.Identity(), nu
            ),
        ),
        False,
    )
    dam = crocoddyl.DifferentialActionModelContactInvDynamicsWithThrusts(
        state, actuation, contacts, costs, constraints
    )
    nf = len(thrusters)
    dam.thrust_reg_weight = 0.1 * np.ones(nf)
    # Narrow bounds so that both sides of the barrier are active
    dam.set_thrust_barrier(10.0 * np.ones(nf), 0.3 * np.ones(nf), 0.6 * np.ones(nf))
    return state, actuation, dam


class ThrustInvDynTestCase(unittest.TestCase):
    THRUST_STATE = None
    TOGGLE = False

    def setUp(self):
        if self.THRUST_STATE is None:
            self.skipTest("abstract test case")
        rng = np.random.default_rng(0)
        self.state, self.actuation, self.dam = build_model(self.THRUST_STATE)
        self.data = self.dam.createData()
        if self.TOGGLE:
            # Activated after creating the data
            self.dam.constraints.changeConstraintStatus("placement", True)
        self.x = self.state.rand()
        self.u = rng.random(self.dam.nu)

    def eval(self, x, u):
        self.dam.calc(self.data, x, u)
        return (
            self.data.cost,
            self.data.h.copy(),
            self.data.g.copy(),
            self.data.multibody.joint.tau.copy(),
        )

    def test_dimensions(self):
        nf, nv = self.dam.nf, self.state.nv
        self.assertEqual(self.dam.nv_floating, 6)
        self.assertEqual(self.dam.nh, 6 + 9 + (6 if self.TOGGLE else 0))
        self.assertEqual(self.dam.ng, (nf if self.THRUST_STATE else 0) + 5 + nv - 6)
        self.assertEqual(self.data.Hx.shape, (self.dam.nh, self.state.ndx))
        self.assertEqual(self.dam.g_lb.size, self.dam.ng)

    def test_derivatives(self):
        self.dam.calc(self.data, self.x, self.u)
        self.dam.calcDiff(self.data, self.x, self.u)
        ndx, nu = self.state.ndx, self.dam.nu
        joint = self.data.multibody.joint
        ana = [
            (self.data.Lx, self.data.Lu),
            (self.data.Hx, self.data.Hu),
            (self.data.Gx, self.data.Gu),
            (joint.dtau_dx, joint.dtau_du),
        ]
        ana = [(a.copy(), b.copy()) for a, b in ana]
        num_x = [
            np.zeros((np.atleast_1d(v).size, ndx)) for v in self.eval(self.x, self.u)
        ]
        num_u = [
            np.zeros((np.atleast_1d(v).size, nu)) for v in self.eval(self.x, self.u)
        ]
        for i in range(ndx):
            dx = np.zeros(ndx)
            dx[i] = EPS
            vp = self.eval(self.state.integrate(self.x, dx), self.u)
            vm = self.eval(self.state.integrate(self.x, -dx), self.u)
            for k in range(4):
                num_x[k][:, i] = (np.atleast_1d(vp[k]) - np.atleast_1d(vm[k])) / (
                    2 * EPS
                )
        for i in range(nu):
            du = np.zeros(nu)
            du[i] = EPS
            vp = self.eval(self.x, self.u + du)
            vm = self.eval(self.x, self.u - du)
            for k in range(4):
                num_u[k][:, i] = (np.atleast_1d(vp[k]) - np.atleast_1d(vm[k])) / (
                    2 * EPS
                )
        names = ["cost", "h", "g", "joint tau"]
        for k in range(4):
            dx_a = np.atleast_2d(ana[k][0])
            du_a = np.atleast_2d(ana[k][1])
            self.assertTrue(
                np.allclose(dx_a, num_x[k], atol=TOL),
                f"Wrong {names[k]} state derivative: {np.abs(dx_a - num_x[k]).max():g}",
            )
            self.assertTrue(
                np.allclose(du_a, num_u[k], atol=TOL),
                f"Wrong {names[k]} control derivative: "
                f"{np.abs(du_a - num_u[k]).max():g}",
            )
        if self.THRUST_STATE:
            nf = self.dam.nf
            self.assertGreater(
                np.linalg.norm(self.data.Hx[:6, -nf:]), 1e-8, "Degenerate test."
            )

    def test_consistency_with_forward_dynamics(self):
        # When the floating-base rows vanish, the actuation input reproduces the
        # accelerations through the forward dynamics
        nf, nv = self.dam.nf, self.state.nv
        self.dam.calc(self.data, self.x, self.u)
        self.dam.calcDiff(self.data, self.x, self.u)
        hu = self.data.Hu[:6]
        du = np.linalg.pinv(hu) @ -self.data.h[:6]
        u = self.u + du
        self.dam.calc(self.data, self.x, u)
        self.assertLess(np.abs(self.data.h[:6]).max(), 1e-6)
        pin_model = self.dam.pinocchio
        pin_data = pin_model.createData()
        q = self.x[: self.state.nq]
        v = self.x[self.state.nq : self.state.nq + nv]
        tau = self.data.multibody.actuation.tau.copy()
        fext = [f.copy() for f in self.data.multibody.contacts.fext]
        a = pinocchio.aba(pin_model, pin_data, q, v, tau, fext)
        self.assertTrue(np.allclose(a, u[nf : nf + nv], atol=1e-6))

    def test_quasi_static(self):
        x = self.x.copy()
        x[self.state.nq : self.state.nq + self.state.nv] = 0.0
        u = self.dam.quasiStatic(self.data, x)
        self.dam.calc(self.data, x, u)
        if not self.THRUST_STATE:
            self.assertLess(np.abs(self.data.h[:6]).max(), 1e-6)

    def test_equilibrium_thrust(self):
        nq, nv, nf = self.state.nq, self.state.nv, self.dam.nf
        q = self.x[:nq]
        thrust = self.dam.computeEquilibriumThrust(self.data, q)
        self.assertTrue(np.all(thrust >= -1e-9) and np.all(thrust <= 30.0 + 1e-9))
        # Thrusts and contact forces (or joint torques) balance gravity at rest
        x = self.x.copy()
        x[nq : nq + nv] = 0.0
        u = np.zeros(self.dam.nu)
        if self.THRUST_STATE:
            x[-nf:] = thrust
        else:
            u[:nf] = thrust
        self.dam.calc(self.data, x, u)
        jc = self.data.multibody.contacts.Jc[3:9]  # rfoot (contacts sorted by name)
        f, *_ = np.linalg.lstsq(jc[:, :6].T, self.data.h[:6], rcond=None)
        self.assertLess(np.abs(jc[:, :6].T @ f - self.data.h[:6]).max(), 1e-6)

    def test_terminal(self):
        self.dam.calc(self.data, self.x)
        self.dam.calcDiff(self.data, self.x)
        self.assertEqual(self.data.g.size, self.dam.ng_T)
        self.assertEqual(self.data.h.size, self.dam.nh_T)


class ThrustInputInvDynTest(ThrustInvDynTestCase):
    THRUST_STATE = False


class ThrustInputToggleInvDynTest(ThrustInvDynTestCase):
    THRUST_STATE = False
    TOGGLE = True


class ThrustStateToggleInvDynTest(ThrustInvDynTestCase):
    THRUST_STATE = True
    TOGGLE = True


class ThrustStateInvDynTest(ThrustInvDynTestCase):
    THRUST_STATE = True

    def test_integrator(self):
        iam = crocoddyl.IntegratedActionModelEulerWithThrusts(self.dam, 1e-2)
        data = iam.createData()
        iam.calc(data, self.x, self.u)
        iam.calcDiff(data, self.x, self.u)
        nf = self.dam.nf
        self.assertTrue(
            np.allclose(data.xnext[-nf:], self.x[-nf:] + 1e-2 * self.u[:nf], atol=1e-12)
        )
        self.assertEqual(iam.g_lb.size, self.dam.ng)
        self.assertTrue(np.allclose(iam.g_lb[:nf], 0.0))
        self.assertTrue(np.allclose(iam.g_ub[:nf], 30.0))


if __name__ == "__main__":
    unittest.main()
