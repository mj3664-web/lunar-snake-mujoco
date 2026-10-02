"""Lunar snake robot prototype built with MuJoCo.

The robot uses alternating yaw and pitch motion to create a sidewinding gait on
a procedurally generated lunar height field.  Run with ``--headless`` on a
machine without a display.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - friendly CLI error
    raise SystemExit(
        "MuJoCo is not installed. Run: python -m pip install -r requirements.txt"
    ) from exc


SEGMENT_COUNT = 14
SEGMENT_LENGTH = 0.38
SEGMENT_RADIUS = 0.07
TERRAIN_ROWS = 121
TERRAIN_COLS = 161


@dataclass
class Sample:
    time: float
    x: float
    y: float
    z: float
    speed: float
    energy: float


def _segment_xml(index: int) -> str:
    """Create one nested robot segment and its descendants."""
    color = "0.88 0.55 0.12 1" if index % 2 == 0 else "0.72 0.32 0.08 1"
    nose = ""
    if index == 0:
        nose = (
            '<geom name="head" type="ellipsoid" pos="-0.055 0 0.025" '
            'size="0.11 0.09 0.08" rgba="0.95 0.67 0.18 1"/>'
            '<site name="head_site" pos="-0.12 0 0.04" size="0.025" '
            'rgba="0.2 0.8 1 1"/>'
        )

    if index == 0:
        opening = '<body name="segment_00" pos="-4.2 0 0.48"><freejoint/>'
    else:
        opening = (
            f'<body name="segment_{index:02d}" pos="{SEGMENT_LENGTH} 0 0">'
            f'<joint name="yaw_{index:02d}" type="hinge" axis="0 0 1" '
            'range="-50 50" damping="0.11" armature="0.008"/>'
            f'<joint name="pitch_{index:02d}" type="hinge" axis="0 1 0" '
            'range="-28 28" damping="0.10" armature="0.008"/>'
        )

    geom = (
        f'<geom name="link_{index:02d}" type="capsule" '
        f'fromto="0 0 0 {SEGMENT_LENGTH} 0 0" size="{SEGMENT_RADIUS}" '
        f'rgba="{color}" friction="1.05 0.04 0.003" condim="4" '
        'density="1250"/>'
    )
    child = _segment_xml(index + 1) if index + 1 < SEGMENT_COUNT else ""
    return opening + nose + geom + child + "</body>"


def build_model_xml() -> str:
    actuators = []
    for index in range(1, SEGMENT_COUNT):
        actuators.append(
            f'<position name="yaw_motor_{index:02d}" joint="yaw_{index:02d}" '
            'kp="7.0" kv="0.35" ctrlrange="-0.87 0.87" forcerange="-3 3"/>'
        )
        actuators.append(
            f'<position name="pitch_motor_{index:02d}" joint="pitch_{index:02d}" '
            'kp="6.0" kv="0.30" ctrlrange="-0.49 0.49" forcerange="-2.5 2.5"/>'
        )

    return f"""
<mujoco model="lunar_snake">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.0025" gravity="0 0 -1.62" integrator="implicitfast"
          iterations="80" cone="elliptic"/>
  <size nconmax="500" njmax="2000"/>

  <visual>
    <headlight ambient="0.30 0.30 0.34" diffuse="0.75 0.72 0.65"
               specular="0.20 0.20 0.20"/>
    <rgba haze="0.04 0.05 0.08 1"/>
    <global azimuth="135" elevation="-24"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.005 0.008 0.018" rgb2="0.01 0.012 0.025" width="512" height="3072"/>
    <texture name="dust" type="2d" builtin="checker" rgb1="0.27 0.26 0.24"
             rgb2="0.34 0.33 0.30" width="512" height="512"/>
    <material name="regolith" texture="dust" texrepeat="9 6" reflectance="0.03"/>
    <hfield name="moon_surface" nrow="{TERRAIN_ROWS}" ncol="{TERRAIN_COLS}"
            size="8 5 0.55 0.25"/>
  </asset>

  <worldbody>
    <light pos="-3 -4 8" dir="0.35 0.25 -1" directional="true"
           diffuse="0.95 0.90 0.78" castshadow="true"/>
    <geom name="terrain" type="hfield" hfield="moon_surface" material="regolith"
          pos="0 0 -0.34" friction="1.10 0.06 0.004" condim="4"/>
    <geom name="rock_a" type="sphere" pos="1.3 1.1 0.02" size="0.25"
          rgba="0.28 0.27 0.25 1" friction="1.2 0.05 0.005"/>
    <geom name="rock_b" type="sphere" pos="3.0 -1.0 -0.04" size="0.34"
          rgba="0.31 0.30 0.27 1" friction="1.2 0.05 0.005"/>
    <geom name="rock_c" type="sphere" pos="-0.2 -1.7 -0.06" size="0.20"
          rgba="0.26 0.25 0.23 1" friction="1.2 0.05 0.005"/>
    {_segment_xml(0)}
  </worldbody>

  <actuator>
    {''.join(actuators)}
  </actuator>

  <sensor>
    <framepos name="head_position" objtype="site" objname="head_site"/>
    <framelinvel name="head_velocity" objtype="site" objname="head_site"/>
    <accelerometer name="head_accel" site="head_site"/>
  </sensor>
</mujoco>
"""


def make_terrain(seed: int = 7, flat: bool = False) -> np.ndarray:
    """Return normalized lunar terrain with low ridges and shallow craters."""
    if flat:
        return np.full((TERRAIN_ROWS, TERRAIN_COLS), 0.20, dtype=np.float64)

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(TERRAIN_ROWS), -1:1:complex(TERRAIN_COLS)]
    height = 0.18 + 0.035 * np.sin(2.3 * x + 0.8 * y)
    height += 0.025 * np.sin(5.2 * y - 1.1 * x)

    # Broad, smooth noise: summing sinusoids avoids extra SciPy dependencies.
    for frequency, amplitude in ((2.0, 0.030), (4.0, 0.018), (8.0, 0.009)):
        angle = rng.uniform(0, 2 * math.pi)
        phase = rng.uniform(0, 2 * math.pi)
        projection = math.cos(angle) * x + math.sin(angle) * y
        height += amplitude * np.sin(frequency * math.pi * projection + phase)

    # Crater bowls with raised rims.
    for cx, cy, radius, depth in (
        (0.38, 0.22, 0.20, 0.15),
        (-0.08, -0.48, 0.13, 0.10),
        (0.68, -0.46, 0.11, 0.08),
    ):
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        bowl = -depth * np.exp(-((distance / (radius * 0.72)) ** 2))
        rim = depth * 0.40 * np.exp(-(((distance - radius) / (radius * 0.18)) ** 2))
        height += bowl + rim

    height -= height.min()
    maximum = height.max()
    if maximum > 0:
        height /= maximum
    return height


class GaitController:
    def __init__(self, gait: str, speed: float) -> None:
        self.gait = gait
        self.omega = 2 * math.pi * speed

    def controls(self, t: float) -> np.ndarray:
        controls = np.zeros(2 * (SEGMENT_COUNT - 1), dtype=np.float64)
        spatial_phase = 0.72
        ramp = min(1.0, t / 1.5)

        for index in range(1, SEGMENT_COUNT):
            phase = self.omega * t - spatial_phase * index
            yaw = math.radians(34) * math.sin(phase)
            if self.gait == "sidewinding":
                # A quarter-cycle-offset vertical wave lifts alternating links.
                pitch = math.radians(12) * math.sin(phase + math.pi / 2)
            else:
                pitch = math.radians(3) * math.sin(phase + math.pi / 2)
            controls[2 * (index - 1)] = ramp * yaw
            controls[2 * (index - 1) + 1] = ramp * pitch
        return controls


def center_of_mass(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    masses = model.body_mass[1:]
    return np.average(data.xipos[1:], axis=0, weights=masses)


def simulate(args: argparse.Namespace) -> list[Sample]:
    model = mujoco.MjModel.from_xml_string(build_model_xml())
    model.hfield_data[:] = make_terrain(args.seed, args.flat_terrain).ravel()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    controller = GaitController(args.gait, args.speed)
    samples: list[Sample] = []
    start_com = center_of_mass(model, data).copy()
    prior_com = start_com.copy()
    prior_sample_time = 0.0
    energy = 0.0
    next_sample = 0.0

    def step() -> None:
        nonlocal prior_com, prior_sample_time, energy, next_sample
        data.ctrl[:] = controller.controls(data.time)
        power = float(np.sum(np.abs(data.actuator_force * data.actuator_velocity)))
        energy += power * model.opt.timestep
        mujoco.mj_step(model, data)

        if data.time >= next_sample:
            com = center_of_mass(model, data)
            elapsed = max(data.time - prior_sample_time, model.opt.timestep)
            speed = float(np.linalg.norm((com[:2] - prior_com[:2]) / elapsed))
            samples.append(Sample(data.time, *com, speed, energy))
            prior_com = com.copy()
            prior_sample_time = data.time
            next_sample += 0.1

    if args.headless:
        while data.time < args.duration:
            step()
    else:
        try:
            from mujoco import viewer as mujoco_viewer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("The MuJoCo viewer is unavailable; try --headless.") from exc

        with mujoco_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = np.array([-1.6, 0.0, 0.1])
            viewer.cam.distance = 6.8
            viewer.cam.azimuth = 128
            viewer.cam.elevation = -28
            while viewer.is_running() and data.time < args.duration:
                wall_start = time.perf_counter()
                target = data.time + 1 / 60
                while data.time < target:
                    step()
                viewer.sync()
                remaining = 1 / 60 - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)

    final_com = center_of_mass(model, data)
    displacement = final_com[:2] - start_com[:2]
    print(
        f"Finished {data.time:.2f} lunar seconds | "
        f"displacement=({displacement[0]:+.3f}, {displacement[1]:+.3f}) m | "
        f"distance={np.linalg.norm(displacement):.3f} m | energy={energy:.2f} J"
    )
    return samples


def write_csv(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "com_x_m", "com_y_m", "com_z_m", "speed_m_s", "energy_j"))
        for sample in samples:
            writer.writerow(
                (
                    f"{sample.time:.4f}",
                    f"{sample.x:.6f}",
                    f"{sample.y:.6f}",
                    f"{sample.z:.6f}",
                    f"{sample.speed:.6f}",
                    f"{sample.energy:.6f}",
                )
            )
    print(f"Telemetry written to {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0, help="simulation seconds")
    parser.add_argument("--gait", choices=("sidewinding", "serpentine"), default="sidewinding")
    parser.add_argument("--speed", type=float, default=0.65, help="gait cycles per second")
    parser.add_argument("--seed", type=int, default=7, help="terrain random seed")
    parser.add_argument("--headless", action="store_true", help="simulate without opening a viewer")
    parser.add_argument("--flat-terrain", action="store_true", help="replace craters with flat ground")
    parser.add_argument("--csv", type=Path, help="optional telemetry CSV path")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    return args


def main() -> None:
    args = parse_args()
    samples = simulate(args)
    if args.csv:
        write_csv(args.csv, samples)


if __name__ == "__main__":
    main()
