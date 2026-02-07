from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import json
import os
import numpy as np


@dataclass
class Scenario:
    name: str
    grade_profile: Callable[[float], float]
    mu_profile: Callable[[float], float]
    v_ref_profile: Callable[[float], float]
    path: Dict[str, np.ndarray]
    meta: Optional[Dict[str, Any]] = None


def constant_profile(value: float) -> Callable[[float], float]:
    def _f(_: float) -> float:
        return float(value)
    return _f


def step_profile(step_time: float, v1: float, v2: float) -> Callable[[float], float]:
    def _f(t: float) -> float:
        return float(v1 if t < step_time else v2)
    return _f


def build_circle_path(radius: float, num_pts: int = 500) -> Dict[str, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, num_pts)
    x = radius * np.sin(angles)
    y = radius * (1 - np.cos(angles))
    s = np.zeros_like(x)
    s[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
    return {"x": x, "y": y, "s": s}


def build_straight_path(length: float, num_pts: int = 500) -> Dict[str, np.ndarray]:
    x = np.linspace(0, length, num_pts)
    y = np.zeros_like(x)
    s = np.copy(x)
    return {"x": x, "y": y, "s": s}


def build_constant_scenario(
    name: str,
    v_ref: float,
    mu: float,
    grade: float,
    path: Dict[str, np.ndarray],
) -> Scenario:
    return Scenario(
        name=name,
        grade_profile=constant_profile(grade),
        mu_profile=constant_profile(mu),
        v_ref_profile=constant_profile(v_ref),
        path=path,
    )


def build_default_scenarios(speeds: list[float], mu: float, grade: float) -> list[Scenario]:
    path = build_circle_path(radius=50.0, num_pts=800)
    scenarios = []
    for v in speeds:
        scenarios.append(build_constant_scenario(
            name=f"step_{v}ms",
            v_ref=v,
            mu=mu,
            grade=grade,
            path=path,
        ))
    return scenarios


def build_mu_step_scenario(v_ref: float, mu1: float, mu2: float, step_time: float = 10.0) -> Scenario:
    path = build_circle_path(radius=50.0, num_pts=800)
    return Scenario(
        name=f"mu_step_{mu1}_to_{mu2}",
        grade_profile=constant_profile(0.0),
        mu_profile=step_profile(step_time, mu1, mu2),
        v_ref_profile=constant_profile(v_ref),
        path=path,
    )


def build_grade_step_scenario(v_ref: float, g1: float, g2: float, step_time: float = 10.0) -> Scenario:
    path = build_circle_path(radius=50.0, num_pts=800)
    return Scenario(
        name=f"grade_step_{g1}_to_{g2}",
        grade_profile=step_profile(step_time, g1, g2),
        mu_profile=constant_profile(0.9),
        v_ref_profile=constant_profile(v_ref),
        path=path,
    )


def _build_path(spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    ptype = spec.get("type", "circle")
    if ptype == "circle":
        return build_circle_path(radius=float(spec.get("radius", 50.0)), num_pts=int(spec.get("num_pts", 800)))
    if ptype == "straight":
        return build_straight_path(length=float(spec.get("length", 200.0)), num_pts=int(spec.get("num_pts", 800)))
    raise ValueError(f"Unknown path type: {ptype}")


def load_scenarios(json_path: str, default_mu: float, default_grade: float) -> list[Scenario]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    path = _build_path(cfg.get("path", {}))
    scenarios = []
    for sc in cfg.get("scenarios", []):
        name = sc.get("name", "scenario")
        v_ref = float(sc.get("v_ref", 10.0))
        mu = float(sc.get("mu", default_mu))
        grade = float(sc.get("grade", default_grade))
        mode = sc.get("mode", "constant")
        if mode == "constant":
            scenarios.append(build_constant_scenario(name, v_ref, mu, grade, path))
        elif mode == "mu_step":
            mu2 = float(sc.get("mu2", mu))
            t = float(sc.get("step_time", 10.0))
            scenarios.append(build_mu_step_scenario(v_ref, mu, mu2, t))
        elif mode == "grade_step":
            g2 = float(sc.get("grade2", grade))
            t = float(sc.get("step_time", 10.0))
            scenarios.append(build_grade_step_scenario(v_ref, grade, g2, t))
        else:
            raise ValueError(f"Unknown scenario mode: {mode}")
    return scenarios
