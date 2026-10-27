"""Box-closing env, version 6: hook and push with a closed fist (box_env5 = pinch the flap).

Why: PPO_v0.2 (box_env5, 10 M steps) learned to hover at the flap with an open gripper and
reached only 4-9 % success at the end; a 2 mm pinch of a 3 mm flap is also the hardest thing
to do on the real robot (the wrist D405 is blind below ~7 cm, exactly when the fingers close).
The lid doesn't need holding: its crease holds it wherever it is let go from 36 deg up, and
below that it falls shut by itself. So:

1. Both grippers stay closed (fists). The policy moves the 14 arm joints only (action 14).
   Reset closes the right gripper (box_env5 opened it for the pinch).
2. The right fist pushes the lid shut from its OUTER face, near the free edge: at 120 deg it
   goes over the front flap, behind the lid, and pushes it towards the robot; past vertical
   the same face is on top and the fist pushes it down. Push point (site-free, computed):
   PUSH_DOWN = 1 cm from the free edge towards the hinge, PUSH_OUT = 1.5 cm out from the
   outer face (the fist's half thickness). Measured (check_rewards6.py, nominal): the lid
   follows at <= 1.1 rad/s and shuts; the box slides 1.2 cm.
3. 'at_lid' (instead of the pinch): right TCP within NEAR = 5 cm of the push point and on the
   lid's outer side. Geometric, so the real robot can compute it from the lid pose (camera)
   and the arm joints (no contact sensing). Scripted push: 0.3-4.6 cm (median 1.1), always
   1.2-4 cm out from the outer face.
   'pulled' latches once ONE stretch of at_lid (gaps up to PUSH_GAP = 3 steps) has moved the
   lid down PULL_BY = 20 deg under control (steps at <= LID_SPEED_FREE = 2.5 rad/s), as in
   box_env5. A yank (fast steps don't count) or a slam earns nothing; a slam is now anything
   faster than LID_SLAM = 3 rad/s above 30 deg (box_env5: 8.5): no faster than the lid falls
   by itself (2.2), so only a gentle push pays. (The box_env5 names pinch_pulled / pinch_gap
   now track the current push.)
4. Reward terms (no gripper terms any more: around, approach_g, grasp, align are gone):
       approach        2.0   right TCP -> push point, 1 - tanh(10 d), doubled within 2 cm
       approach_wide   1.25  1 - tanh(2 d)
       lid            15.0   lid progress (120 deg -> 0) after 'pulled', or while at_lid with the lid
                             above 30 deg (resting on a lid it shut without control pays nothing)
       stages          1.0   0.5 past 5 % progress + 1 past 50 % (same condition as lid)
                             + 1 below 30 deg ('finished')
       success        15.0   shut + pulled + not slammed, x (0.5 + 0.5 still)
       left hand, lid_speed, calm, action_rate, joint_vel, still / clear gates: box_env5's.
   Reach pays first (approach), then pushing adds the lid terms. Once 'finished' the
   approach terms pay their maximum, so the fist can leave without losing reward.
5. Box mass x 0.8..12 (log-uniform, was 0.8..1.5): the real box will have 3 items in it
   (heavier, steadier); training on both the empty 45 g box and a heavy one covers that.

Observation (57), all measurable on the robot: arm joint angles (14) and speeds (14), lid
angle and speed (2), push point - right TCP (3), left target - left TCP (3), box position
change (3) and turn (1), last action (14), pulled, push progress (0..1 of PULL_BY), slammed.
(Dropped from box_env5: finger positions, front flap angle, the gripper switch state.)
Demo states: demo_states6.npz (check_rewards6.py --save), half the episodes (box_env5's DEMO_P).

    python box_env6.py     # check_env + reward terms at the start / with the lid closed
"""
import warnings
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from box_env1 import ARM_ACTUATORS, BoxCloseEnv, GRIPPER_ACTUATORS
from box_env4 import (BOX_LOST, BOX_TIPPED, LEFT_FORCE_MAX, LEFT_RETREAT_FROM, LEFT_RETREAT_TO, LEFT_YAW,
                      LID_DONE, LID_SNAP_ANGLE, REWARD_SCALE, SELF_CONTACT_GATE, STILL_FREE, TURN_TO_M, approach)
from box_env5 import (LEFT_AFTER, LID_SPEED_CAP, LID_SPEED_FREE, PULL_BY, RAND_FRICTION,
                      RAND_GRIP_FRICTION, RAND_HINGE, RAND_LID_MASS, STEP_TIME, BoxCloseEnvV5)
from view_scene import LID_OPEN

PUSH_DOWN = 0.01        # m: push point this far from the lid's free edge towards the hinge
PUSH_OUT = 0.015        # m: ... and this far out from the lid's outer face (fist half thickness)
LID_SLAM = 3.0          # rad/s above 30 deg = a slam: no credit this episode (box_env5: 8.5). A lid
                        # falling by itself reaches <= 2.2 there (worst randomisation corner), the
                        # scripted push 1.1. Measured: at 8.5 a 4x faster push (5.3 rad/s, front flap
                        # folded back, no success) returned 282 vs 285 for the gentle push; at 4.0 a
                        # 2.7x faster one (3.3 rad/s) returned 342, i.e. faster paid more.
NEAR = 0.05             # m: right TCP this close to the push point (and on the outer side) = at_lid
PUSH_GAP = 3            # steps away from the lid a push survives
# Left hand: a FRONT stop (box frame: 1.7 cm in front of the front wall, 4.5 cm left of the
# middle, 2.4 cm up). Pushing the lid at 120 deg drags the box towards the robot.
FRONT_STOP = np.array([-0.100, 0.045, 0.024])
LEFT_RETREAT_V6 = np.array([0.0, 0.06, 0.03])   # m, world: out to the left and up, as the lid goes 55 -> 45 deg
RAND_BOX_MASS = (0.8, 12.0)   # log-uniform: empty box (45 g) up to ~3 items inside
DEMO_FILE = Path(__file__).with_name("demo_states6.npz")
N_ARM = 14

W6 = {
    "approach": 2.0, "approach_wide": 1.25, "lid": 15.0, "stages": 1.0, "success": 15.0,
    "stop": 2.0, "stop_wide": 2.0, "stop_orient": 2.0, "calm": 3.0,
    "action_rate": -0.05, "joint_vel": -1e-4, "lid_speed": -8.0,
}


class BoxCloseEnvV6(BoxCloseEnvV5):

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ARM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(57,), dtype=np.float32)
        self.demo = dict(np.load(DEMO_FILE)) if DEMO_FILE.exists() else None
        if self.demo is None:
            warnings.warn(f"{DEMO_FILE} not found: no demo starts (run check_rewards6.py --save)")
        self.prev_action = np.zeros(N_ARM, dtype=np.float32)

    def _randomize_physics(self, on):
        """box_env5's, with the box mass log-uniform over RAND_BOX_MASS."""
        m, nom, rng = self.m, self.nominal, self.np_random
        m.body_mass[:], m.body_inertia[:] = nom["mass"], nom["inertia"]
        m.geom_friction[:, 0] = nom["friction"]
        m.dof_frictionloss[self.lid_dof] = nom["hinge"]
        if not on:
            return
        m.dof_frictionloss[self.lid_dof] *= np.exp(rng.uniform(*np.log(RAND_HINGE)))
        for ids, s in ((self.lid_bodies, rng.uniform(*RAND_LID_MASS)),
                       ([self.box_body], np.exp(rng.uniform(*np.log(RAND_BOX_MASS))))):
            m.body_mass[ids] *= s
            m.body_inertia[ids] *= s
        m.geom_friction[self.friction_geoms, 0] *= rng.uniform(*RAND_FRICTION)
        m.geom_friction[self.grip_geoms, 0] *= rng.uniform(*RAND_GRIP_FRICTION)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        d = self.d
        if not self.demo_start:     # box_env5 opened the right gripper for the pinch: close it
            for name in ("openarm_right_finger_joint1", "openarm_right_finger_joint2"):
                d.joint(name).qpos = 0.0
            d.actuator(GRIPPER_ACTUATORS[1]).ctrl = 0.0
            mujoco.mj_forward(self.m, d)
        # Front stop, fixed in the world where the box started (box_start / box_quat_start are
        # set before a demo state is restored, so a demo start gets the nominal point).
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.box_quat_start)
        self.stop_point = self.box_start + R.reshape(3, 3) @ FRONT_STOP
        # Stop orientation: the left hand's start orientation (box_env5 turned it 20 deg for the side stop).
        c, s = np.cos(-LEFT_YAW), np.sin(-LEFT_YAW)
        mujoco.mju_quat2Mat(R, self.left_quat_target)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R.reshape(3, 3)
        mujoco.mju_mat2Quat(self.left_quat_target, R.flatten())
        self.prev_action = np.zeros(N_ARM, dtype=np.float32)
        return self._get_obs(), {}

    def left_target(self):
        """Front stop while the lid is high, then (55 -> 45 deg) out to the left and up, so the
        front flap doesn't land on the hand."""
        lid = self.d.joint("lid_hinge").qpos[0]
        out = np.clip((LEFT_RETREAT_FROM - lid) / (LEFT_RETREAT_FROM - LEFT_RETREAT_TO), 0.0, 1.0)
        return self.stop_point + out * LEFT_RETREAT_V6

    def push_point(self):
        """Where the right fist pushes: near the free edge, just off the lid's outer face."""
        d = self.d
        edge, hinge = d.site("lid_handle").xpos, d.body("lid").xpos
        outer = d.body("lid").xmat.reshape(3, 3)[:, 2]
        return edge + PUSH_DOWN * (hinge - edge) / np.linalg.norm(hinge - edge) + PUSH_OUT * outer

    def _at_lid(self):
        """(at_lid, distance): right TCP within NEAR of the push point, on the lid's outer side."""
        d = self.d
        tcp = d.body("openarm_right_hand_tcp").xpos
        dist = float(np.linalg.norm(tcp - self.push_point()))
        outside = (tcp - d.site("lid_handle").xpos) @ d.body("lid").xmat.reshape(3, 3)[:, 2] > 0
        return bool(dist < NEAR and outside), dist

    def _get_obs(self):
        d = self.d
        return np.concatenate([
            [d.joint(n).qpos[0] for n in ARM_ACTUATORS],
            d.qvel[self.arm_qvel_adr],
            [d.joint("lid_hinge").qpos[0], d.qvel[self.lid_qvel_adr]],
            self.push_point() - d.body("openarm_right_hand_tcp").xpos,
            self.left_target() - d.body("openarm_left_hand_tcp").xpos,
            d.body("box").xpos - self.box_start,
            [self._box_turn()],
            self.prev_action,
            [self.pulled, self._pinch_progress(), self.slammed],
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        full = np.ones(16, dtype=np.float32)     # both grippers closed (+1)
        full[:N_ARM] = action
        _, _, _, truncated, _ = BoxCloseEnv.step(self, full)
        at_lid, dist = self._at_lid()
        lid = self.d.joint("lid_hinge").qpos[0]
        # 'pulled': ONE push (at_lid, gaps up to PUSH_GAP steps) moved the lid down PULL_BY,
        # counting only steps at <= LID_SPEED_FREE (box_env5's rule, with at_lid for the pinch).
        if at_lid:
            self.pinch_pulled, self.pinch_gap = self.pinch_pulled or 0.0, 0
            drop = self.prev_lid - lid
            if abs(drop) / STEP_TIME <= LID_SPEED_FREE:
                self.pinch_pulled = max(0.0, self.pinch_pulled + drop)
            if self.pinch_pulled >= PULL_BY:
                self.pulled = True
        elif self.pinch_pulled is not None:
            self.pinch_gap += 1
            if self.pinch_gap > PUSH_GAP:
                self.pinch_pulled = None
        self.prev_lid = lid
        if lid > LID_SNAP_ANGLE and abs(self.d.qvel[self.lid_qvel_adr]) > LID_SLAM:
            self.slammed = True
        reward, terms, success = self._reward_v6(action, at_lid, dist)
        self.prev_action = action
        tipped = self.d.body("box").xmat[8] < BOX_TIPPED
        lost = np.linalg.norm(self.d.body("box").xpos - self.box_start) > BOX_LOST
        success = success and not lost      # shut but shoved > 20 cm away: a failure
        info = {"is_success": None if self.demo_start else success, "success": success,
                "demo_start": self.demo_start, "terms": terms}
        return self._get_obs(), reward, bool(tipped or lost), truncated, info

    def _reward_v6(self, action, at_lid, dist):
        d = self.d
        lid_angle = d.joint("lid_hinge").qpos[0]
        lid_vel = d.qvel[self.lid_qvel_adr]
        progress = float(np.clip((LID_OPEN - lid_angle) / LID_OPEN, 0, 1))
        upright = self._box_upright()
        credit = self.pulled and not self.slammed
        finished = lid_angle < LID_SNAP_ANGLE and upright and credit
        shut = lid_angle < LID_DONE and upright and d.joint("flap_hinge").qpos[0] > 0
        success = shut and credit

        dist_reward, wide = approach(dist, 0.02), 1 - np.tanh(2 * dist)
        raw = {"approach_raw": dist_reward, "approach_wide_raw": wide}
        if finished:                                    # the fist may leave: reach terms at max
            dist_reward, wide = 2.0, 1.0
        # Lid pay only for the fist's own push: at the lid while the lid is still up, or after a
        # controlled push ('pulled'). Without that, a fist resting on a lid it shut fast (a fling,
        # 4x the scripted speed: no 'pulled') was paid the full lid term: return 282 vs 285 (measured).
        holding = float(credit or (at_lid and lid_angle > LID_SNAP_ANGLE))
        lid = progress * holding
        stages = (progress > 0.05) * 0.5 * holding + (progress > 0.5) * holding + finished * 1.0
        excess = min(max(0.0, abs(lid_vel) - LID_SPEED_FREE), LID_SPEED_CAP)
        lid_speed = excess ** 2 if lid_angle > LID_SNAP_ANGLE else 0.0

        dist_left = np.linalg.norm(d.body("openarm_left_hand_tcp").xpos - self.left_target())
        dot = abs(float(d.body("openarm_left_hand").xquat @ self.left_quat_target))
        left_twist = 2 * np.arccos(min(dot, 1.0))
        self_touch, left_on_lid, left_force = self._contacts()
        push_ok = 1 - np.tanh(max(0.0, left_force - LEFT_FORCE_MAX) / LEFT_FORCE_MAX)
        left_w = push_ok * (LEFT_AFTER if finished else 1.0)
        moved = np.linalg.norm(d.body("box").xpos - self.box_start) + TURN_TO_M * self._box_turn()
        still = 1 - np.tanh(10 * max(0.0, moved - STILL_FREE))

        action_rate = float(np.sum((action - self.prev_action) ** 2))
        arm_vel = d.qvel[self.arm_qvel_adr]
        joint_vel = float(np.sum(arm_vel ** 2))
        calm = float(success) * (1 - np.tanh(np.linalg.norm(arm_vel)))
        clear = SELF_CONTACT_GATE if (self_touch or left_on_lid) else 1.0

        terms = {
            "approach": dist_reward, "approach_wide": wide, "lid": lid, "stages": stages,
            "success": float(success), "stop": approach(dist_left, 0.02) * left_w,
            "stop_wide": (1 - np.tanh(2 * dist_left)) * left_w, "stop_orient": (1 - np.tanh(left_twist)) * left_w,
            "calm": calm, "action_rate": action_rate, "joint_vel": joint_vel, "lid_speed": lid_speed,
        }
        positive = sum(W6[k] * terms[k] for k in W6 if W6[k] > 0 and k != "success")
        negative = sum(W6[k] * terms[k] for k in W6 if W6[k] < 0)
        reward = REWARD_SCALE * ((positive * still + W6["success"] * terms["success"] * (0.5 + 0.5 * still))
                                 * clear + negative)
        terms.update(still=still, clear=clear, push_ok=push_ok, left_force=left_force,
                     finished=float(finished), upright=float(upright), at_lid=float(at_lid),
                     pulled=float(self.pulled), slammed=float(self.slammed),
                     push_progress=self._pinch_progress(), shut=float(shut), lid_vel=lid_vel, **raw,
                     dist_right=dist, dist_left=dist_left, left_twist=left_twist)
        return reward, terms, bool(success)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = BoxCloseEnvV6()
    check_env(env)
    for opts, label in (({"nominal": True}, "real start, nominal"),
                        ({"nominal": True, "lid_angle": 0.0}, "lid closed, no push")):
        env.reset(options=opts)
        _, r, *_, info = env.step(np.zeros(N_ARM))
        t = info["terms"]
        print(f"{label:22s} reward {r:.3f} | at_lid {t['at_lid']:.0f} dist {t['dist_right'] * 100:.1f} cm | "
              f"pulled {t['pulled']:.0f} shut {t['shut']:.0f} success {t['success']:.0f} lid {t['lid']:.2f}")
