# Lunar Snake Robot — MuJoCo Prototype

This is a compact physics prototype of a 14-segment snake robot moving over a
cratered lunar height field. It includes:

- Lunar gravity: **1.62 m/s²**
- Coupled yaw/pitch joints on every segment
- Sidewinding and serpentine gait controllers
- Procedural ridges, craters, and static rocks
- Center-of-mass speed, displacement, and actuator-energy telemetry
- Interactive visualization or headless simulation

## Install

Use Python 3.10 or newer from this directory. On Windows, the `py` launcher is
recommended because `python` may point to an older installation:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

## Run

Open the interactive simulation:

```powershell
python lunar_snake.py
```

The simulation stops after 30 simulated seconds or when the viewer closes.
Use the mouse to orbit and zoom the camera.

Run a short headless test and save telemetry:

```powershell
python lunar_snake.py --headless --duration 10 --csv telemetry.csv
```

Try a mostly planar gait or a flat test surface:

```powershell
python lunar_snake.py --gait serpentine
python lunar_snake.py --flat-terrain
```

All options are shown by `python lunar_snake.py --help`.

## Model boundary

The terrain is a rigid height field with lunar-like geometry and contact
friction. It does **not** yet simulate deformable regolith, wheel sinkage, dust,
vacuum thermal effects, or radiation. Those require a particle/terramechanics
layer and calibrated lunar-soil parameters. The current model is intended for
controller prototyping and qualitative locomotion experiments.
