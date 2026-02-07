import unittest
import numpy as np

from configs.vehicle_params import VehicleParams
from configs.scenarios import build_constant_scenario, build_circle_path
from controllers.longitudinal.pid_controller import PIDControllerAdaptive
from controllers.lateral.lqr_controller import LQRLateralController
from powertrain.throttle_brake import ThrottleBrakePowertrain
from sim.simulator import simulate


class TestSimulatorSmoke(unittest.TestCase):
    def test_smoke_run(self):
        p = VehicleParams()
        scenario = build_constant_scenario(
            name="smoke",
            v_ref=10.0,
            mu=p.mu0,
            grade=p.grade0,
            path=build_circle_path(radius=50.0, num_pts=200),
        )
        lon = PIDControllerAdaptive(Kp0=6.0, Ki0=0.1, beta=0.5, p=p)
        lat = LQRLateralController(p)
        powertrain = ThrottleBrakePowertrain(p)

        result = simulate(scenario, p, lon, lat, powertrain, dt=0.2, T_final=2.0)

        self.assertEqual(len(result.time), len(result.u))
        self.assertFalse(np.isnan(result.u).any())
        self.assertFalse(np.isinf(result.u).any())
        self.assertEqual(len(result.x), len(result.y))


if __name__ == "__main__":
    unittest.main()
