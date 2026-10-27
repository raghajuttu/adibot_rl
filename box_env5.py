"""Box-closing env, version 5: box_env4 plus the fixes from the ppo_box_v5 run (2026-10-21).

What ppo_box_v5 (trained on box_env4) did: it never pinched the flap. The open right hand
hooked the lid's top edge, yanked it (5-21 N) past vertical, and the lid slammed shut on its
own at up to 14 rad/s, after 0.7 s. The left hand rose 20 cm above its stop and, after the lid
was shut, both arms wandered and the left one knocked the box 3.5 cm. It worked in the sim,
but with the hinge friction x2 plus 0.02 rad arm noise only 8 of 24 closed. Changes:

1. The pinch is required. A latch 'pulled' turns on once ONE pinch of the flap (box_env4's
   force/angle test; gaps up to PINCH_GAP = 3 steps allowed) has pulled the lid down
   PULL_BY = 20 deg UNDER CONTROL: a step counts only if the lid moved at <= 2.5 rad/s during it,
   and the lid going back up counts against it. Without 'pulled': no 'released' maximum, no
   'finished' stage, no success, and the lid term pays only while pinching. (In box_env4,
   below 55 deg every right-hand term paid its maximum however the lid got there, and the lid
   term paid progress even without a grip.) Code review (2026-10-22) found that the first
   version (any pinch while the lid is below 100 deg) let a lid pushed down open-handed plus
   one late pinch count as success; and closing the gripper on the flap at 75 deg jerked the
   lid down 32 deg in 3 steps (11 rad/s), which a 'moved 20 deg' rule also counted. Measured
   now, lid brought to 100 / 90 / 75 / 60 / 40 deg without a pinch, then pinch + push: no
   success, return 94 / 96 / 71 / 45 / 40; a physical closed-finger push to 95 deg + pinch:
   no success. 'pulled' and the pinch progress (0..1 of PULL_BY) are observed.
2. Lid speed: lid_speed = min(max(0, |lid vel| - 2.5 rad/s), 6)^2 while the lid is above
   30 deg, weight -8 (at most -5.76 per step). Below 30 deg the lid falls shut by itself,
   which isn't the policy's doing. And a SLAM, faster than 8.5 rad/s above 30 deg, sets the
   latch 'slammed' (observed): no released / finished / success for the rest of the episode.
   2.5: a lid simply let go at 51 deg falls at up to 2.2 rad/s above 30 deg (worst corner of
   the randomisation: hinge friction x0.5, lid mass x1.25; measured), a controlled pull
   ~1.4, the v5 fling 7-15. Why cap + latch: a pure square gave random exploration -18 per step
   at 13 rad/s and -26 at 17 (value-loss outliers); a cap alone made a 13-15 rad/s slam that
   shuts the lid cost less in total than a slower one, so with 'pulled' set a slam could pay
   for itself by closing sooner (second review). With the latch a slam never pays.
   Measured (return over the episode):
       scripted pinch-pull-push (check_rewards5.py)   229 over 470 steps, 270 over 500
                                                      (lid max 1.4 rad/s: costs 0)
       arms standing still                             51
       ppo_box_v5's fling replayed in box_env5         12 nominal, 5 randomised: no success (on this
                                                      box its fling no longer even shuts the lid, 1/12)
       random actions (40 training-style episodes)     46 (2.8 % of steps negative, worst step -5.7)
       scripted, hand/finger friction x0.6 / x1.2     216 / 234, success at step 437 / 435
3. After 'finished' the left-hand terms count LEFT_AFTER (3) times, so the left hand stays at
   its retreat point; once shut, 'calm' (weight 3, 1 - tanh(|arm joint speeds|)) pays for
   holding the arms still; and success is paid x (0.5 + 0.5 'still'), so knocking the box
   after closing costs (before: success was outside the 'still' gate).
4. Domain randomisation at every reset (Isaac Lab "events" style; real values unknown):
       lid hinge friction  x 0.5..2 (log-uniform)
       lid + flaps mass    x 0.8..1.25
                           together (measured, robot not touching): the lid stays put from
                           62-76 deg up (friction x0.5), 36 deg nominal, 52 deg (x1, heavy lid),
                           and never falls shut by itself (x1 light lid, x2): then the hand has
                           to push it all the way down.
       box mass            x 0.8..1.5
       sliding friction    x 0.7..1.3 (box, lid and table geoms together; MuJoCo uses the
                           larger of the two, so scaling the box alone did nothing on the table)
       hand / finger       x 0.6..1.2, its own factor (the robot geoms' friction 1.0 wins every
       friction            robot-cardboard contact: priority 1; without this the pinch and the
                           left-hand stop never saw any other friction)
       box position        +-1 cm in x and y, yaw +-3 deg
       arm start joints    +-0.02 rad (resampled, up to 3 tries, if it starts the left hand
                           in the torso guard: 2 % of starts did)
   Demo starts restore a saved state, so they get the physics changes but not the pose ones
   (their box_start and stop point are the nominal ones the demo was recorded with).
   reset(options={"nominal": True}) switches all of it off (for check_rewards5 and watching).
5. (PPO_v0.1 run, 2026-10-22: 0 % success at 7.5 M steps. The right gripper was closed in 100 %
   of real-start steps from 3 M on, so the hand came to the flap as a fist and never pinched;
   from demo states with the hand already at the flap it closed, pulled and shut the lid.)
   a. Reset opens the right gripper. mj_resetData sets every ctrl to 0, and 0 is CLOSED for the
      fingers (0.044 = open), so every real start (also in box_env4) began with a fist.
   b. Reaching the flap pays more than closing on it: 'around' (fingers open on both sides of
      the flap, within 6 cm) 0.125 -> 3, 'grasp' 5 -> 2.5. 'around' stays on while pinching
      (100 % of the scripted pinch steps), so reach pays first and closing adds to it.
      Measured, 500 steps standing at the start: hand open 83, fist 54 (were 55 / 54).
      Scripted solution 237 (was 230).

The box is assets/cardboard_box_v5.xml (scene box_scene_v5.xml), built like the real folded
box: the three flaps are bent extensions of the lid, starting at its outer edges, tilted
outwards, so a shut lid leaves them outside the walls, untucked (tucking is not part of
this task). They collide with the box. (cardboard_box.xml: flaps glued under the lid 3-5 mm
in from its edges, side flaps passing through the walls with box contact off, so a shut lid
had them sticking 8.6 mm out through the side walls.) Hinge friction 0.0163: the lid holds
from ~36 deg up. A shut lid lies flat on the walls (0 deg; was ~2 deg on the flap). Lid and
flap armature cut to 1e-4 / 2e-5 (was 5e-4 / 2e-4: the flaps moved like bodies 175x heavier);
the lid's 0 deg stop is stiffer (a 20 N press sinks it 3.5 deg, was 5.3). Reset puts the flaps
at their measured rest angles for the start lid angle (they used to swing for 1.5 s).

Observation (63) = box_env4's 60 + 'pulled' + pinch progress + 'slammed'. Demo states: demo_states5.npz
(check_rewards5.py --save; with 'pulled' and the current pinch's controlled pull).
Everything else (terms, weights, gates, gripper, termination) is box_env4's, see its docstring.

    python box_env5.py     # check_env + reward terms at the start / with the lid closed
"""
import warnings
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from box_env1 import ARM_ACTUATORS, FRAME_SKIP, GRIPPER_ACTUATORS, BoxCloseEnv
from box_env4 import (BOX_LOST, BOX_TIPPED, FINGER_OPEN, LEFT_GRIPPER, LID_DONE, LID_SNAP_ANGLE,
                      LEFT_FORCE_MAX, RELEASE_ANGLE, RIGHT_GRIPPER, LEFT_YAW, REWARD_SCALE,
                      SELF_CONTACT_GATE, START_LEFT_V4, START_RIGHT_V4, STILL_FREE, TURN_TO_M, W,
                      BoxCloseEnvV4, approach)
from view_scene import LID_OPEN

SCENE_V5 = Path(__file__).with_name("box_scene_v5.xml")
# Flap rest angles (rad, measured with the lid held at each angle: springs + weight), so the
# flaps don't swing at the start (a straight line between 0 and 120 deg was up to 8 deg off).
FLAP_REST_AT = np.radians([0, 17, 29, 40, 57, 75, 92, 109, 120])       # lid angle
FLAP_REST = [0.954, 0.987, 1.023, 1.048, 1.064, 1.058, 1.023, 0.957, 0.897]   # front flap
SIDE_FLAPS_REST = [0.309, 0.313, 0.315, 0.318, 0.327, 0.338, 0.351, 0.365, 0.374]   # outwards
PULL_BY = np.radians(20)         # 'pulled' = ONE pinch has pulled the lid down this far, counting
                                 # only steps at <= LID_SPEED_FREE (a jerk doesn't count)
PINCH_GAP = 3                    # steps without grasp a pinch survives (contact flicker)
LID_SPEED_FREE = 2.5             # rad/s: faster than this (above 30 deg) costs (a lid let go at
                                 # 51 deg falls at up to 2.2 rad/s there, the v5 fling 7-15)
LID_SPEED_CAP = 6.0              # rad/s over the threshold counted at most (per-step cost <= 5.76)
LID_SLAM = LID_SPEED_FREE + LID_SPEED_CAP   # 8.5 rad/s above 30 deg = a slam: no success this episode
LEFT_AFTER = 3.0                 # left-hand terms x this once the lid is 'finished'
DEMO_P = 0.5
DEMO_FILE = Path(__file__).with_name("demo_states5.npz")
# Domain randomisation, (low, high) of a scale factor unless noted.
RAND_HINGE = (0.5, 2.0)          # log-uniform
RAND_LID_MASS = (0.8, 1.25)
RAND_BOX_MASS = (0.8, 1.5)
RAND_FRICTION = (0.7, 1.3)
RAND_GRIP_FRICTION = (0.6, 1.2)  # robot hand / finger geoms (their friction 1.0 wins: priority 1)
RAND_BOX_XY = 0.01               # m, +-
RAND_BOX_YAW = np.radians(3)     # +-
RAND_ARM = 0.02                  # rad, +-

STEP_TIME = FRAME_SKIP * 0.002   # s per policy step (timestep 0.002, checked in __init__)

# around / grasp (box_env4: 0.125 / 5): reaching the flap with open fingers pays more than closing.
W5 = dict(W, around=3.0, grasp=2.5, lid_speed=-8.0, calm=3.0)


class BoxCloseEnvV5(BoxCloseEnvV4):

    def __init__(self):
        super().__init__(SCENE_V5)
        m = self.m
        assert np.isclose(FRAME_SKIP * m.opt.timestep, STEP_TIME), "STEP_TIME doesn't match the model"
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(63,), dtype=np.float32)
        self.demo = dict(np.load(DEMO_FILE)) if DEMO_FILE.exists() else None
        if self.demo is None and DEMO_P > 0:
            warnings.warn(f"{DEMO_FILE} not found: no demo starts (run check_rewards5.py --save)")
        self.pulled, self.pinch_pulled, self.pinch_gap, self.prev_lid = False, None, 0, LID_OPEN
        self.slammed = False
        # Nominal values, so every reset randomises from them (not from the last episode's).
        self.lid_dof = m.joint("lid_hinge").dofadr[0]
        self.lid_bodies = [m.body(b).id for b in ("lid", "flap", "side_flap_left", "side_flap_right")]
        self.box_body = m.body("box").id
        table = {i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name == "table"
                 or m.geom(i).name.startswith("table")}
        self.friction_geoms = sorted(self.box_geoms | self.lid_geoms | table)
        self.grip_geoms = [m.geom(f"openarm_{side}_{g}_collision").id for side in ("left", "right")
                           for g in ("hand", "left_finger", "right_finger")]
        self.nominal = {
            "hinge": m.dof_frictionloss[self.lid_dof],
            "mass": m.body_mass.copy(), "inertia": m.body_inertia.copy(),
            "friction": m.geom_friction[:, 0].copy(),
        }
        self.box_qpos_adr = m.jnt_qposadr[m.joint("box_free").id]

    def _randomize_physics(self, on):
        m, nom, rng = self.m, self.nominal, self.np_random
        m.body_mass[:], m.body_inertia[:] = nom["mass"], nom["inertia"]
        m.geom_friction[:, 0] = nom["friction"]
        m.dof_frictionloss[self.lid_dof] = nom["hinge"]
        if not on:
            return
        m.dof_frictionloss[self.lid_dof] *= np.exp(rng.uniform(*np.log(RAND_HINGE)))
        for ids, (lo, hi) in ((self.lid_bodies, RAND_LID_MASS), ([self.box_body], RAND_BOX_MASS)):
            s = rng.uniform(lo, hi)
            m.body_mass[ids] *= s
            m.body_inertia[ids] *= s
        m.geom_friction[self.friction_geoms, 0] *= rng.uniform(*RAND_FRICTION)
        m.geom_friction[self.grip_geoms, 0] *= rng.uniform(*RAND_GRIP_FRICTION)

    def reset(self, seed=None, options=None):
        BoxCloseEnv.reset(self, seed=seed)          # home pose, lid open, step_count 0
        options = options or {}
        nominal = options.get("nominal", False)
        rng, d = self.np_random, self.d
        self._randomize_physics(not nominal)
        # Demo starts only without options (watch / check scripts always get the real start).
        self.demo_start = not options and self.demo is not None and rng.random() < DEMO_P
        for name, angle in zip(ARM_ACTUATORS, START_LEFT_V4 + START_RIGHT_V4):
            d.joint(name).qpos = angle
            d.actuator(name).ctrl = angle
        # Right gripper open (mj_resetData left its ctrl at 0 = closed).
        for name in ("openarm_right_finger_joint1", "openarm_right_finger_joint2"):
            d.joint(name).qpos = FINGER_OPEN
        d.actuator(GRIPPER_ACTUATORS[1]).ctrl = FINGER_OPEN
        lid_start = options.get("lid_angle", LID_OPEN)
        d.joint("lid_hinge").qpos = lid_start
        d.joint("flap_hinge").qpos = np.interp(lid_start, FLAP_REST_AT, FLAP_REST)
        for name in ("side_flap_left", "side_flap_right"):
            d.joint(name).qpos = np.interp(lid_start, FLAP_REST_AT, SIDE_FLAPS_REST)
        mujoco.mj_forward(self.m, d)
        # Stop orientation from the nominal start pose (before the arm noise).
        c, s = np.cos(LEFT_YAW), np.sin(LEFT_YAW)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ d.body("openarm_left_hand").xmat.reshape(3, 3)
        mujoco.mju_mat2Quat(self.left_quat_target, R.flatten())
        # Pose noise on real starts only: a demo state brings its own poses, and its box_start /
        # stop point must be the nominal ones it was recorded with (check_rewards5: nominal start).
        if not (nominal or self.demo_start):
            base = [d.joint(n).qpos[0] for n in ARM_ACTUATORS]
            for _ in range(3):      # resample if the noise starts the left hand in the torso guard
                for name, angle in zip(ARM_ACTUATORS, base + rng.uniform(-RAND_ARM, RAND_ARM, len(base))):
                    d.joint(name).qpos = angle
                    d.actuator(name).ctrl = angle
                mujoco.mj_forward(self.m, d)
                if not self._contacts()[0]:
                    break
            a = self.box_qpos_adr
            d.qpos[a:a + 2] += rng.uniform(-RAND_BOX_XY, RAND_BOX_XY, 2)
            yaw = rng.uniform(-RAND_BOX_YAW, RAND_BOX_YAW)
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), d.qpos[a + 3:a + 7].copy())
            d.qpos[a + 3:a + 7] = q
            mujoco.mj_forward(self.m, d)
        self.box_start = d.body("box").xpos.copy()
        self.box_quat_start = d.body("box").xquat.copy()
        self.stop_point = d.site("box_side_stop").xpos.copy()   # fixed stop point (world)
        self.pulled, self.pinch_pulled, self.pinch_gap, self.slammed = False, None, 0, False
        if self.demo_start:
            i = rng.integers(len(self.demo["qpos"]))
            d.qpos[:], d.qvel[:], d.ctrl[:] = (self.demo[k][i] for k in ("qpos", "qvel", "ctrl"))
            self.pulled = bool(self.demo["pulled"][i])
            pinch_pulled = float(self.demo["pinch_pulled"][i])      # NaN = no pinch going on
            self.pinch_pulled = None if np.isnan(pinch_pulled) else pinch_pulled
            mujoco.mj_forward(self.m, d)
        self.prev_lid = d.joint("lid_hinge").qpos[0]
        self.right_closed = d.actuator(GRIPPER_ACTUATORS[1]).ctrl[0] < FINGER_OPEN / 2
        self.grip_count, self.grip_pending = 0, 0.0
        self.prev_action = np.zeros(16, dtype=np.float32)
        self.prev_action[LEFT_GRIPPER] = 1.0
        self.prev_action[RIGHT_GRIPPER] = 1.0 if self.right_closed else -1.0
        return self._get_obs(), {}

    def _pinch_progress(self):
        """How far the current pinch has pulled the lid (under control), 0..1 of PULL_BY."""
        return 0.0 if self.pinch_pulled is None else float(min(self.pinch_pulled / PULL_BY, 1.0))

    def _get_obs(self):
        return np.append(super()._get_obs(), np.float32([self.pulled, self._pinch_progress(), self.slammed]))

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action[LEFT_GRIPPER] = 1.0
        action[RIGHT_GRIPPER] = self._right_gripper(action[RIGHT_GRIPPER])
        _, _, _, truncated, _ = BoxCloseEnv.step(self, action)
        grasp = self._flap_grasped()
        lid = self.d.joint("lid_hinge").qpos[0]
        # 'pulled' latches once ONE pinch (gaps up to PINCH_GAP steps) has pulled the lid down
        # PULL_BY under control: a step counts only if the lid moved at <= LID_SPEED_FREE during
        # it (closing the gripper on the flap at 75 deg jerked the lid down 32 deg in 3 steps at
        # 11 rad/s, measured), and the lid going back up counts against it. Lid motion before
        # the pinch (hooked / pushed down) never counts.
        if grasp:
            self.pinch_pulled, self.pinch_gap = self.pinch_pulled or 0.0, 0
            drop = self.prev_lid - lid
            if abs(drop) / STEP_TIME <= LID_SPEED_FREE:
                self.pinch_pulled = max(0.0, self.pinch_pulled + drop)
            if self.pinch_pulled >= PULL_BY:
                self.pulled = True
        elif self.pinch_pulled is not None:
            self.pinch_gap += 1
            if self.pinch_gap > PINCH_GAP:
                self.pinch_pulled = None
        self.prev_lid = lid
        if lid > LID_SNAP_ANGLE and abs(self.d.qvel[self.lid_qvel_adr]) > LID_SLAM:
            self.slammed = True
        reward, terms, success = self._reward_v5(action, grasp)
        self.prev_action = action
        tipped = self.d.body("box").xmat[8] < BOX_TIPPED
        lost = np.linalg.norm(self.d.body("box").xpos - self.box_start) > BOX_LOST
        success = success and not lost      # shut but shoved > 20 cm away: a failure, not a success
        info = {"is_success": None if self.demo_start else success, "success": success,
                "demo_start": self.demo_start, "terms": terms}
        return self._get_obs(), reward, bool(tipped or lost), truncated, info

    def _reward_v5(self, action, grasp):
        """box_env4's _reward_v4 with changes 1-3 (see the docstring), marked 'v5'."""
        d = self.d
        tcp = d.body("openarm_right_hand_tcp").xpos
        tip = d.site("flap_edge").xpos
        F = d.body("flap").xmat.reshape(3, 3)
        flap_dir, flap_normal = -F[:, 0], F[:, 2]
        lf, rf, R = self._fingertips()
        dist = np.linalg.norm(tcp - tip)

        align_z = R[:, 2] @ -flap_dir
        align_y = R[:, 1] @ flap_normal
        align = 0.5 * (np.sign(align_z) * align_z ** 2 + align_y ** 2)

        sl, sr = (lf - tip) @ flap_normal, (rf - tip) @ flap_normal
        around = float(sl * sr < 0 and dist < 0.06)
        approach_g = around * ((0.04 - abs(sl)) + (0.04 - abs(sr)))
        grasp = float(grasp)

        lid_angle = d.joint("lid_hinge").qpos[0]
        lid_vel = d.qvel[self.lid_qvel_adr]
        progress = float(np.clip((LID_OPEN - lid_angle) / LID_OPEN, 0, 1))
        upright = self._box_upright()
        pulled = self.pulled
        # v5: all of these need the lid to have been pulled by the pinch, and never slammed.
        credit = pulled and not self.slammed
        released = lid_angle < RELEASE_ANGLE and upright and credit
        finished = lid_angle < LID_SNAP_ANGLE and upright and credit
        shut = lid_angle < LID_DONE and upright and d.joint("flap_hinge").qpos[0] > 0
        success = shut and credit
        dist_reward, wide = approach(dist, 0.02), 1 - np.tanh(2 * dist)
        raw = {"approach_raw": dist_reward, "align_raw": align, "around_raw": around,
               "approach_g_raw": approach_g, "grasp_raw": grasp, "approach_wide_raw": wide}
        if released:
            dist_reward, align, around, approach_g, grasp, wide = 2.0, 1.0, 1.0, 0.08, 1.0, 1.0
        holding = grasp > 0 or credit                  # v5: no lid pay for knocking / slamming it shut
        lid = (grasp + 1.0) * progress * holding
        stages = (progress > 0.05) * 0.5 * holding + (progress > 0.5) * grasp + finished * 1.0
        # v5: lid speed above 30 deg (below it the lid falls shut by itself).
        excess = min(max(0.0, abs(lid_vel) - LID_SPEED_FREE), LID_SPEED_CAP)
        lid_speed = excess ** 2 if lid_angle > LID_SNAP_ANGLE else 0.0

        dist_left = np.linalg.norm(d.body("openarm_left_hand_tcp").xpos - self.left_target())
        dot = abs(float(d.body("openarm_left_hand").xquat @ self.left_quat_target))
        left_twist = 2 * np.arccos(min(dot, 1.0))
        self_touch, left_on_lid, left_force = self._contacts()
        push_ok = 1 - np.tanh(max(0.0, left_force - LEFT_FORCE_MAX) / LEFT_FORCE_MAX)
        left_w = push_ok * (LEFT_AFTER if finished else 1.0)      # v5: x3 once finished
        moved = np.linalg.norm(d.body("box").xpos - self.box_start) + TURN_TO_M * self._box_turn()
        still = 1 - np.tanh(10 * max(0.0, moved - STILL_FREE))

        used = np.delete(action - self.prev_action, LEFT_GRIPPER)
        action_rate = float(np.sum(used ** 2))
        arm_vel = d.qvel[self.arm_qvel_adr]
        joint_vel = float(np.sum(arm_vel ** 2))
        calm = float(success) * (1 - np.tanh(np.linalg.norm(arm_vel)))   # v5: arms still once shut

        clear = SELF_CONTACT_GATE if (self_touch or left_on_lid) else 1.0

        terms = {
            "approach": dist_reward, "align": align, "around": around, "approach_g": approach_g,
            "grasp": grasp, "lid": lid, "stages": stages, "stop": approach(dist_left, 0.02) * left_w,
            "success": float(success), "action_rate": action_rate, "joint_vel": joint_vel,
            "stop_wide": (1 - np.tanh(2 * dist_left)) * left_w, "stop_orient": (1 - np.tanh(left_twist)) * left_w,
            "approach_wide": wide, "lid_speed": lid_speed, "calm": calm,
        }
        positive = sum(W5[k] * terms[k] for k in W5 if W5[k] > 0 and k != "success")
        negative = sum(W5[k] * terms[k] for k in W5 if W5[k] < 0)
        # v5: success x (0.5 + 0.5 still): knocking the box after closing costs.
        reward = REWARD_SCALE * ((positive * still + W5["success"] * terms["success"] * (0.5 + 0.5 * still))
                                 * clear + negative)
        terms.update(still=still, clear=clear, push_ok=push_ok, left_force=left_force,
                     released=float(released), finished=float(finished), upright=float(upright),
                     pulled=float(pulled), slammed=float(self.slammed),
                     pinch_progress=self._pinch_progress(), shut=float(shut),
                     lid_vel=lid_vel, **raw,
                     dist_right=dist, dist_left=dist_left, left_twist=left_twist)
        return reward, terms, bool(success)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = BoxCloseEnvV5()
    check_env(env)
    for opts, label in (({"nominal": True}, "real start, nominal"),
                        ({"nominal": True, "lid_angle": 0.0}, "lid closed, no pull")):
        env.reset(options=opts)
        _, r, *_ , info = env.step(np.zeros(16))
        t = info["terms"]
        print(f"{label:22s} reward {r:.3f} | pulled {t['pulled']:.0f} shut {t['shut']:.0f} "
              f"success {t['success']:.0f} lid {t['lid']:.2f} stages {t['stages']:.1f}")
