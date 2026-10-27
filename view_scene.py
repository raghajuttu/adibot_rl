"""Open the box-closing scene in the MuJoCo viewer, in the working pose.

    conda activate mujoco_rl
    python view_scene.py              # lid open
    python view_scene.py --closed     # lid closed, flaps tucked in

Backspace in the viewer resets to the model's zero pose (lift fully up, lid
closed); restart this script to get the working pose back.
"""
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent / "box_scene.xml"

LIFT_DOWN = 0.25      # m; lowers the shoulders to ~1.10 m for table work
LID_OPEN = 2.09       # rad (~120 deg), where the real lid stays open

# Flap angles (rad). Resting = where the creases hold them; tucked = inside the walls.
FLAPS_RESTING = {"flap_hinge": 0.8, "side_flap_left": 0.35, "side_flap_right": 0.35}
FLAPS_TUCKED = {"flap_hinge": 1.57, "side_flap_left": 0.0, "side_flap_right": 0.0}


def reset_home(m, d, lid_angle=LID_OPEN, flaps=FLAPS_RESTING):
    mujoco.mj_resetData(m, d)
    d.joint("P").qpos = LIFT_DOWN
    d.actuator("P").ctrl = LIFT_DOWN
    d.joint("lid_hinge").qpos = lid_angle
    for name, angle in flaps.items():
        d.joint(name).qpos = angle
    mujoco.mj_forward(m, d)


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    if "--closed" in sys.argv:
        reset_home(m, d, lid_angle=0.0, flaps=FLAPS_TUCKED)
    else:
        reset_home(m, d)
    mujoco.viewer.launch(m, d)
