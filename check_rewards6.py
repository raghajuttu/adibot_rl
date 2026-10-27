"""Proof that the v6 task (box_env6: hook and push with a closed fist) can be solved and that
every reward term can fire. Same IK controller as check_rewards5.py, both grippers closed.

    left arm:  to the front stop in front of the front wall (left half), wait there (pushing the
               lid at 120 deg drags the box towards the robot: 4.3 cm with box_env5's side stop,
               0.8 cm with the front stop, measured); as the lid goes from 55 to 45 deg, move
               out to the left and up, out of the front flap's way (env.left_target())
    right arm: up above the front flap -> over it and behind the lid -> onto the lid's outer
               face near the free edge -> push the lid along its arc (towards the robot, then
               down) until it is shut (< 2 deg) -> hand back and up

    python check_rewards6.py            # in the terminal only
    python check_rewards6.py --view     # also show it in the viewer
    python check_rewards6.py --save     # also save states along the way (demo_states6.npz)
    python check_rewards6.py --seed 3   # a randomised start (box pose, physics) instead of nominal

The hand orientation is left free (with the start orientation kept, the IK could not reach
behind the lid and the arm flailed, measured).
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from box_env1 import ARM_ACTUATORS, FRAME_SKIP, MAX_EPISODE_STEPS, MAX_STEP
from box_env4 import LID_SNAP_ANGLE
from box_env5 import LID_SPEED_FREE
from box_env6 import DEMO_FILE, PUSH_DOWN, PUSH_OUT, BoxCloseEnvV6
from view_scene import LID_OPEN

parser = argparse.ArgumentParser()
parser.add_argument("--view", action="store_true")
parser.add_argument("--save", action="store_true", help=f"save states along the run to {DEMO_FILE.name}")
parser.add_argument("--seed", type=int, default=None, help="randomised start with this seed (default: nominal)")
args = parser.parse_args()

env = BoxCloseEnvV6()
m, d = env.m, env.d
LEFT, RIGHT = ARM_ACTUATORS[:7], ARM_ACTUATORS[7:]
HAND_STEP = 0.003   # m per policy step = 15 cm/s
LEFT_STEP = 0.005   # left hand: 25 cm/s, at the stop before the right hand pushes
OVER = 0.08         # m: pass this high above the front flap's tip
LEAD = np.radians(0.12)      # push target: rotated this much further closed per step ...
LEAD_MAX = np.radians(6)     # ... up to this far ahead of the lid (a gentle push: ~1 rad/s)
PUSH_UNTIL = np.radians(2)   # push it all the way: with stiff hinge friction (x1.7-2) the lid
                             # stays where it is even at 25 deg (measured: 6 of 12 randomised
                             # runs failed when the push stopped at 28 deg)
BACK = np.array([0.03, 0.0, 0.06])   # hand back and up afterwards (other ways hit the torso)
SAVE_EVERY = 5

# IK on the *commanded* arm (see check_rewards5.py), with a slow gravity-sag correction.
ref = mujoco.MjData(m)
sag = {"left": np.zeros(3), "right": np.zeros(3)}


def ik_delta(joints, body, target, z_axis=None, y_axis=None):
    side = "left" if "left" in body else "right"
    miss = target - d.body(body).xpos
    if np.linalg.norm(miss) < 0.03:
        sag[side] = np.clip(sag[side] + 0.05 * miss, -0.03, 0.03)
    ref.qpos[:] = d.qpos
    for n in ARM_ACTUATORS:
        ref.joint(n).qpos = d.actuator(n).ctrl[0]
    mujoco.mj_kinematics(m, ref)
    mujoco.mj_comPos(m, ref)
    dofs = [m.joint(n).dofadr[0] for n in joints]
    jac_pos, jac_rot = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, ref, jac_pos, jac_rot, m.body(body).id)
    err = target + sag[side] - ref.body(body).xpos
    step = LEFT_STEP if side == "left" else HAND_STEP
    err = err if np.linalg.norm(err) < step else err / np.linalg.norm(err) * step
    J, e = jac_pos[:, dofs], err
    if z_axis is not None:
        R = ref.body(body).xmat.reshape(3, 3)
        rot_err = 0.5 * (np.cross(R[:, 2], z_axis) + np.cross(R[:, 1], y_axis))
        J, e = np.vstack([J, 0.3 * jac_rot[:, dofs]]), np.concatenate([e, 0.3 * rot_err])
    return J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)


if args.seed is None:
    env.reset(seed=0, options={"nominal": True})          # the real start, nothing randomised
else:
    env.reset(seed=args.seed, options={"lid_angle": LID_OPEN})
L0 = np.zeros(9)
mujoco.mju_quat2Mat(L0, env.left_quat_target)
L0 = L0.reshape(3, 3)
left_axes = (L0[:, 2].copy(), L0[:, 1].copy())


def left_target():
    """Straight to the front stop: the left hand starts in front of the box, 8 cm from it (a via
    point above it, towards the robot, ran the left wrist into the torso guard, measured)."""
    return env.left_target()


def lid_point(out, down, closed_by=0.0):
    """Point `down` m from the lid's free edge towards the hinge and `out` m off its outer
    face, turned `closed_by` rad further closed about the hinge."""
    hinge, edge = d.body("lid").xpos, d.site("lid_handle").xpos
    L = d.body("lid").xmat.reshape(3, 3)
    p = edge + down * (hinge - edge) / np.linalg.norm(hinge - edge) + out * L[:, 2]
    axis, v = -L[:, 1], p - hinge
    c, s = np.cos(closed_by), np.sin(closed_by)
    return hinge + v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


best, lowest = {}, {}
demo = {"qpos": [], "qvel": [], "ctrl": [], "pulled": [], "pinch_pulled": []}
total = {"return": 0.0, "steps": 0, "first_success": None, "clear_run": 0}
viewer = mujoco.viewer.launch_passive(m, d) if args.view else None


def run(steps, label, right_target, show=True):
    for _ in range(steps):
        a = np.zeros(14)
        a[:7] = np.clip(ik_delta(LEFT, "openarm_left_hand_tcp", left_target(), *left_axes) / MAX_STEP, -1, 1)
        a[7:] = np.clip(ik_delta(RIGHT, "openarm_right_hand_tcp", right_target()) / MAX_STEP, -1, 1)
        obs, reward, terminated, truncated, info = env.step(a)
        t = info["terms"]
        for k, v in t.items():
            best[k] = max(best.get(k, -np.inf), v)
            lowest[k] = min(lowest.get(k, np.inf), v)
        if d.joint("lid_hinge").qpos[0] > LID_SNAP_ANGLE:
            best["lid speed above 30 deg"] = max(best.get("lid speed above 30 deg", 0), abs(t["lid_vel"]))
        total["return"] += reward
        total["steps"] += 1
        if info["success"] and total["first_success"] is None:
            total["first_success"] = total["steps"]
        total["clear_run"] = total["clear_run"] + 1 if t["clear"] == 1 else 0
        if total["steps"] % SAVE_EVERY == 0 and not t["finished"] and total["clear_run"] >= SAVE_EVERY:
            for k in ("qpos", "qvel", "ctrl"):
                demo[k].append(getattr(d, k).copy())
            demo["pulled"].append(env.pulled)
            demo["pinch_pulled"].append(np.nan if env.pinch_pulled is None else env.pinch_pulled)
        if viewer:
            viewer.sync()
            time.sleep(FRAME_SKIP * m.opt.timestep)
        if terminated:
            raise SystemExit(f"!! episode ended at step {total['steps']}: box tipped or slid (nothing saved)")
    if show:
        moved = np.linalg.norm(d.body("box").xpos - env.box_start) * 100
        print(f"{label:30s} lid {np.degrees(d.joint('lid_hinge').qpos[0]):5.1f} deg  reward {reward:5.2f} | "
              f"push point {t['dist_right'] * 100:4.1f} cm  at_lid {t['at_lid']:.0f}  pulled {t['pulled']:.0f} | "
              f"stop {t['stop']:.2f} left force {t['left_force']:.1f} N | box moved {moved:.1f} cm still {t['still']:.2f}"
              f"{'  SUCCESS' if info['success'] else ''}")


tip0 = d.site("flap_edge").xpos.copy()
run(40, "1 up, above the front flap", lambda: tip0 + np.array([0, 0, OVER]))
behind = lid_point(0.05, -0.03)       # 5 cm off the outer face, 3 cm past the free edge
run(40, "2 over the flap", lambda: np.array([behind[0], behind[1], tip0[2] + OVER]))
run(40, "3 behind the lid", lambda: behind)
run(30, "4 onto the outer face", lambda: lid_point(PUSH_OUT + 0.03, PUSH_DOWN))
lead = {"angle": 0.0}
for k in range(400):
    lead["angle"] = min(lead["angle"] + LEAD, LEAD_MAX)
    run(1, f"5 push ({k + 1} steps)", lambda: lid_point(PUSH_OUT, PUSH_DOWN, lead["angle"]), show=k % 50 == 49)
    if d.joint("lid_hinge").qpos[0] < PUSH_UNTIL:
        break
up = d.body("openarm_right_hand_tcp").xpos.copy()
run(40, "6 hand back and up", lambda: up + BACK)
run(max(0, 470 - total["steps"]), "7 wait", lambda: up + BACK)

if args.save:
    if total["first_success"] is None or args.seed is not None:
        raise SystemExit("!! no success (or a randomised start): demo_states6.npz not written")
    np.savez(DEMO_FILE, **{k: np.array(v) for k, v in demo.items()})
    print(f"\nsaved {len(demo['qpos'])} states to {DEMO_FILE}")

env.reset(seed=0, options={"nominal": True})
idle = sum(env.step(np.zeros(14))[1] for _ in range(total["steps"]))
print(f"\nFirst success at step {total['first_success']} (an episode is {MAX_EPISODE_STEPS} steps)")
print(f"Return over {total['steps']} steps: scripted task {total['return']:.1f}  vs  arms standing still {idle:.1f}")
print("Best value each reward term reached (0 = never fired):")
for key in ("approach_raw", "approach_wide_raw", "lid", "stages", "stop", "stop_wide", "stop_orient",
            "success", "calm", "at_lid", "pulled", "push_progress"):
    print(f"  {key:17s} {best[key]:8.3f}")
print(f"  {'lid speed >30 deg':17s} {best.get('lid speed above 30 deg', 0):8.3f}  (rad/s, max; costs above {LID_SPEED_FREE})")
print(f"  {'slammed':17s} {best['slammed']:8.3f}   worst action_rate {best['action_rate']:.2f}, lowest still {lowest['still']:.2f}, "
      f"lowest clear {lowest['clear']:.2f}, max left force {best['left_force']:.1f} N")
