import numpy as np


class PurePursuitController:
    """
    纯追踪横向控制器。
    输入车辆状态与道路路径，输出转向角 delta。
    """
    def __init__(self, p, Ld: float = 5.0):
        self.Ld = Ld
        self.wheelbase = p.wheelbase

    def reset(self):
        pass

    def step(self, state, path, dt):
        _ = dt
        x_ref = path["x"]
        y_ref = path["y"]
        s_ref = path.get("s")
        if s_ref is None:
            s_ref = np.zeros_like(x_ref)
            s_ref[1:] = np.cumsum(np.hypot(np.diff(x_ref), np.diff(y_ref)))

        dx = x_ref - state.x
        dy = y_ref - state.y
        idx_near = int(np.argmin(dx * dx + dy * dy))
        s_target = s_ref[idx_near] + self.Ld
        idx_target = int(np.searchsorted(s_ref, s_target))
        idx_target = min(idx_target, len(x_ref) - 1)

        tx = x_ref[idx_target]
        ty = y_ref[idx_target]
        alpha = np.arctan2(ty - state.y, tx - state.x) - state.psi
        delta = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), self.Ld)
        return float(delta)
