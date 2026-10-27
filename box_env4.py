"""Box-closing env, version 4: reward ported from Isaac Lab's OpenArm cabinet task.

Task: the right hand grabs the lid's front flap and pulls the lid shut while
the left hand stops the box at its left side. Box: the real 16 x 16 x 4 cm
pizza-style box (assets/cardboard_box.xml), 32 cm in front, lid open at 120 deg.
The lid stays where it is let go above ~37 deg and drops shut by itself below ~35 deg
(like the real box; stiff crease friction in the XML), so the pull has to bring it under
30 deg (or the hand lets go and pushes it there).
At 40 cm a top-down pinch of the flap was ~6-8 cm out of reach (IK check).

Isaac Lab's "open drawer" task (isaaclab_tasks/core/cabinet/mdp/rewards.py,
contrib/cabinet/config/openarm) is the same skill: grasp a handle, then pull.
Mapping used here:
    drawer handle     -> front flap tip (site flap_edge), flap plane = handle frame
    drawer opening    -> lid closing progress, 0 (open, 2.09 rad) .. 1 (closed)
Terms and weights follow Isaac Lab:
    approach   2.0    1 - tanh(10 d), doubled within 2 cm       (right TCP -> flap tip)
                      (Isaac's 1 / (1 + d^2) is nearly flat over our 8 cm start gap:
                      0.994 at 8 cm vs 1.0 at 0, so it never pulled the hand in.
                      tanh gives 0.34 at 8 cm, 0.62 at 4 cm, 0.80 at 2 cm.)
    align      0.5    gripper approach axis along the flap (towards the crease),
                      finger axis along the flap normal
    around     0.125  fingertips on opposite sides of the flap, close to the tip
    approach_g 5.0    fingertips close to the flap plane while "around"
    grasp      5.0    the flap is pinched: each right finger presses on it (>= 0.5 N) along the
                      closing direction, from opposite sides (ManiSkill is_grasping). "Both
                      fingers touch the flap" was not enough: the gripper starts closed, and
                      the closed fingertips pressed on the flap's face or edge also touch it
                      with both fingers.
                      (was Isaac's "fingers closing near the handle", 0.5: it paid 0.04 for a
                      full close, even on air, so the trained policy hovered half-open)
    lid        7.5    progress, x2 while grasped
    stages     1.0    0.5 at 5 % closed, +1 at 50 % while grasped, +1 below 30 deg
                      (Isaac's last stage needs the grasp; ours doesn't, the lid can be pushed)
    action_rate -0.05, joint_vel -1e-4                         (smoothness)
                      (Isaac: -0.01. With -0.01 a full-speed +-1 chatter on all joints cost
                      1 % of the reward, so jerky motion was free; -0.05 makes it ~5 %, and the
                      start reward under random actions stays positive (measured).)
Released (lid under 55 deg, box upright): the hand has to LET GO and push the lid shut
(it stays put above ~37 deg; pulling below ~50 deg jams the right wrist into the torso). Isaac's
drawer never needs a release (the hand keeps the handle, so its grasp terms keep paying
when the drawer is open). Ported 1:1, letting go cost grasp + the x2 on lid + a stage +
approach, and holding the flap at 35 deg paid 0.53/step against 0.28-0.36 for a closed
box (measured), so the best policy was to never finish. Now, below RELEASE_ANGLE every
right-hand term pays its maximum (ManiSkill: max dense reward on success), so letting go
costs nothing, and pushing the lid down still pays: lid progress, +1 stage below 30 deg
("finished"), success 15 outside the 'still' gate. (First set at 30 deg: then the release
at 45 deg dropped the reward 40 % for ~110 steps, and holding the flap was worth more to
PPO's discounted value than finishing; measured 49.8 vs 47.2. 55 deg = where the left hand
starts to back off too, and above the ~51 deg release of the scripted solution.)
Right gripper is binary (Isaac Lab's BinaryJointPositionAction), with a dead band and
a delay: it closes only after GRIP_SWITCH_STEPS steps in a row of action > GRIP_BAND, and
opens the same way with < -GRIP_BAND. Plain "> 0" flipped at every sign change of the
exploration noise and one open step moves the fingers 40 % of their stroke, so a grasp
never survived the pull (9 % of steps held, measured). The latched state (last action) and
the pending switch count are observed.
Added for the two-arm, free-standing box (Isaac's cabinet is fixed):
    stop       2.0    1 - tanh(10 d), doubled within 2 cm       (left TCP -> left target)
                      Left target: the side stop (site box_side_stop, closed fingers 5 mm from
                      the outside of the left wall) while the lid is above 55 deg, then moving
                      smoothly (55 -> 45 deg) to 6 cm further out and 4 cm up (LEFT_RETREAT),
                      so the side flap doesn't land on
                      the hand when the lid comes down (measured: it did, 21 N, and tipped the
                      box). The pinch shoves the box +y; with the stop the box slides 0.9 cm
                      instead of 4.0 (scripted test, 5 left-arm roles compared: front stop,
                      side stop, press down, grip a wall, back stop).
                      Both points are fixed in the world where they were at the start: the hand
                      is a stop, it should stay put when the box moves (with the point on the
                      box, the 2.8 cm grip shove dropped the term 1.97 -> 0.55, so closing the
                      gripper lowered the reward and the policy learned not to close).
                      grasp 5: closing must pay more than the 'still' gate costs for that shove.
Wide-range pull and left-hand orientation, from Isaac Lab's bimanual OpenArm reach
(there: -distance and -angle). Made positive and bounded (ManiSkill style), same slope
at the target: with negative terms 84 % of random-action steps had a negative reward,
which makes ending the episode early (knocking the box over) look attractive.
    approach_wide 1.25  1 - tanh(2 d)      (right TCP -> flap tip, pulls from far away)
    stop_wide     2.0   1 - tanh(2 d)      (left TCP -> left target)
    stop_orient   2.0   1 - tanh(angle)    (left hand vs its start orientation turned 20 deg
                                            about z: fingertips towards the wall)
    push_ok    gate   the three stop terms x (1 - tanh(max(0, F - 2 N) / 2 N)), F = left hand's
                      contact force on the box: a stop, not a pusher.
    still      gate   all positive terms x (1 - tanh(10 * max(0, moved - 1.5 cm))),
                      moved = box slide + 0.1 m/rad x box turn. Free up to 1.5 cm (was 3 cm,
                      when nothing held the box: the pinch alone moved it 6.5). With the side
                      stop the pinch moves it ~2 (0.9 cm slide + 8 deg turn), so holding the
                      box still now shows in the reward.
    failure           episode ends if the box tips over or slides more than 20 cm
                      (was 10 cm: with exploration noise, a hand near the flap bumps the box
                      past 10 cm in ~1 of 3 episodes; ending the episode threw away the rest
                      of its reward, so PPO learned to keep the right hand 5 cm back, measured.
                      The 'still' gate already pays almost nothing at 20 cm: 0.06.)
    success    +15    lid below 4 deg, box upright (< 5 deg tilt, not lifted > 5 mm) and the
                      flap not folded back over the lid (flap_hinge > 0). Paid outside the
                      'still' gate. (Before: lid angle only, so it counted with the box tipped
                      17 deg up on the flap. No termination on success.)
    clear      gate   all positive terms x 0.5 while the robot touches itself (arm-arm or
                      arm-torso) or the left hand touches the lid / flaps: the scripted pull
                      reached 30 deg by jamming the right wrist into the torso (up to 40 N),
                      which would hit the real robot's shell.
The total is multiplied by the step time 0.02 s, like Isaac Lab (reward x step_dt).
(With 0.1 the value-loss gradient was 40x the policy gradient, and PPO's shared
gradient clipping cut the policy's update to 7.5 %: measured. With 0.02: 47 %.)

Start: always the same (fixed arm poses, box in place, lid at 120 deg).
Demo starts (DEMO_P = 0.5): half the training episodes start from a state saved along the
scripted solution (check_rewards.py --save -> demo_states.npz): hand above the flap, around
it, holding it, mid-pull. From the normal start PPO found a real grip < 1 step per episode,
so it never learned that gripping pays. Starting there, it does, and then learns to get there
(Salimans & Chen 2018, "Learning Montezuma's Revenge from a single demonstration").
The rewards still measure from the normal start (box_start, stop point), so a demo state
where the box already moved is scored as moved. reset(options=...) never uses demo starts.
Demo episodes report info["is_success"] = None, so SB3's success_rate counts real starts
only (8 of the old 75 states closed with no action); info["success"] is always there.
Curriculum (off, CURRICULUM_P = 0): a share of episodes would start with the lid at 40-90 deg.

Observation (60), Isaac style: arm joint pos + vel, grippers, lid angle + vel,
flap angle, right TCP -> flap tip, left TCP -> left target, box offset and turn
from start, last action, pending right-gripper switch (-1..1).
info: is_success (None on demo starts), success, demo_start, terms (the paid values;
*_raw = the right-hand terms before the 'released' maximum, for printing).

    python box_env4.py     # check_env + reward terms at the start / with the lid closed
"""
import warnings
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from box_env1 import ARM_ACTUATORS, GRIPPER_ACTUATORS, BoxCloseEnv
from view_scene import LID_OPEN, SCENE

# Start poses for the 16 x 16 x 4 box at 32 cm (IK, touch nothing). Solved with
# "stay away from the joint limits": every joint keeps >= 0.27 rad (right) / 0.50 rad
# (left) to its limit. (The first poses had 4 right-arm joints pinned at their limits,
# so the arm could hardly move.)
#   left:  in front of the box (the old front-stop pose); it has to go round the
#          box's front-left corner to the side stop
#   right: 7 cm behind the flap pinch pose, pointing down along the flap (20 deg tilt:
#          the exact top-down pinch at the 120 deg lid is only reachable at the limits)
#          with the fingers across the flap. The pull keeps this hand orientation.
START_LEFT_V4 = [0.388, -0.85, 1.071, 1.935, -0.234, 0.224, -0.123]
START_RIGHT_V4 = [1.277, 0.405, -1.055, 1.519, 0.874, 0.518, -1.114]

LEFT_GRIPPER = 14            # action index, forced closed (flat "palm" for bracing)
RIGHT_GRIPPER = 15           # action index, binary with dead band and delay (see _right_gripper)
FINGER_OPEN = 0.044          # finger joint position when open
FINGERTIP = np.array([0.0, 0.0, 0.065])   # fingertip offset in the hand frame (TCP height)
GRASP_FORCE = 0.5            # N per finger, pressing on the flap ...
GRASP_COS = np.cos(np.radians(60))   # ... within 60 deg of the finger's closing direction
BOX_TIPPED = 0.8
BOX_LOST = 0.20             # m: box slid this far = failure (it can't slide off the table unnoticed)
# Real lid: stays wherever it is let go above ~37 deg (stiff crease friction, in the XML),
# and drops shut by itself below ~35 deg, resting at ~2 deg on the untucked flap.
# (A 0.03 Nm closing torque used to be added below 30 deg; with the side flaps excluded
# and the stiff friction it isn't needed, and it was 1.6x the lid's own weight torque.)
LID_SNAP_ANGLE = np.radians(30)   # below this the lid surely drops shut by itself = "finished"
RELEASE_ANGLE = np.radians(55)    # below this the right hand lets go: its terms pay their max
LID_DONE = np.radians(4)          # success: lid shut (rests at ~2 deg, so 2 deg of margin)
BOX_UPRIGHT = np.cos(np.radians(5))
BOX_LIFTED = 0.005                # m
GRIP_BAND = 0.3            # |action| below this keeps the gripper as it is
GRIP_SWITCH_STEPS = 3      # steps in a row needed to open / close (60 ms)
SELF_CONTACT_GATE = 0.5
# Start-angle curriculum: half the episodes start with the lid already at 40-90 deg,
# so the policy also practises the last part (pushing the lid under 30 deg) from the
# start instead of having to discover the whole pull-then-push sequence at once.
CURRICULUM_P = 0.0   # 0 = every episode starts the same (lid 120 deg); 0.5 = half start at 40-90 deg
LID_START_RANGE = (np.radians(40), np.radians(90))
LOW_LID = np.radians(65)        # below this, start the side flaps folded out (0.8 rad),
LOW_LID_SIDE_FLAPS = 0.8        # otherwise they overlap the walls and kick the lid up
TURN_TO_M = 0.1            # m per rad: a 9 deg turn counts like a 1.5 cm slide in the 'still' gate
STILL_FREE = 0.015         # m: box motion that costs nothing (with the side stop the pinch moves it ~2)
LEFT_YAW = np.radians(20)                 # stop orientation: start orientation turned about world z
LEFT_RETREAT = np.array([0.0, 0.06, 0.04])  # m, world: where the left hand goes from the stop ...
LEFT_RETREAT_FROM = np.radians(55)        # ... starting when the lid passes this angle ...
LEFT_RETREAT_TO = np.radians(45)          # ... and all the way out here (the right hand lets go at ~51)
LEFT_FORCE_MAX = 2.0                      # N: more than this on the box = pushing, not stopping
DEMO_P = 0.5               # share of episodes that start from a demo state
DEMO_FILE = Path(__file__).with_name("demo_states.npz")
REWARD_SCALE = 0.02         # = step time (s), Isaac Lab convention

W = {
    "approach": 2.0, "align": 0.5, "around": 0.125, "approach_g": 5.0, "grasp": 5.0,
    "lid": 7.5, "stages": 1.0, "stop": 2.0, "success": 15.0,
    "action_rate": -0.05, "joint_vel": -1e-4,
    # Isaac Lab bimanual OpenArm reach (contrib/reach/config/openarm/bimanual), per hand:
    # coarse distance -0.2/-0.25 and orientation error -0.1 next to the tanh term +0.1/+0.2.
    # Same ratios to our tanh weights, as positive 0..1 terms with the same slope at 0:
    "stop_wide": 2.0, "stop_orient": 2.0, "approach_wide": 1.25,
}


def approach(distance, threshold):
    """Reach reward 0..1 (ManiSkill style), doubled when close (Isaac Lab style)."""
    reward = 1.0 - np.tanh(10 * distance)
    return 2 * reward if distance <= threshold else reward


class BoxCloseEnvV4(BoxCloseEnv):

    def __init__(self, scene=SCENE):
        super().__init__(scene)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32)
        self.arm_qvel_adr = [self.m.joint(n).dofadr[0] for n in ARM_ACTUATORS]
        self.lid_qvel_adr = self.m.joint("lid_hinge").dofadr[0]
        self.flap_geom = self.m.geom("flap").id
        self.right_finger_geoms = {self.m.geom(f"openarm_right_{f}_finger_collision").id for f in ("left", "right")}
        self.finger_geom_left = self.m.geom("openarm_right_left_finger_collision").id
        self.prev_action = np.zeros(16, dtype=np.float32)
        self.box_quat_start = np.array([1.0, 0.0, 0.0, 0.0])   # set properly in reset()
        self.left_quat_target = np.array([1.0, 0.0, 0.0, 0.0])  # stop orientation, set in reset()
        self.stop_point = np.zeros(3)                            # set in reset()
        self.demo = dict(np.load(DEMO_FILE)) if DEMO_FILE.exists() else None
        if self.demo is None and DEMO_P > 0:
            warnings.warn(f"{DEMO_FILE} not found: no demo starts (run check_rewards.py --save)")
        self.demo_start = False
        self.right_closed, self.grip_count, self.grip_pending = False, 0, 0.0
        # Robot geoms: any contact between two of them is a self-collision (robot-robot
        # contacts exist only through the explicit pairs in assets/arm_collision.xml).
        names = [self.m.geom(i).name for i in range(self.m.ngeom)]
        self.robot_geoms = {i for i, n in enumerate(names) if n.startswith("openarm") or n == "torso_guard"}
        self.left_geoms = {i for i, n in enumerate(names) if n.startswith("openarm_left")}
        body_of = lambda i: self.m.body(self.m.geom_bodyid[i]).name
        self.box_geoms = {i for i in range(self.m.ngeom) if body_of(i) == "box"}
        self.lid_geoms = {i for i in range(self.m.ngeom)
                          if body_of(i) in ("lid", "flap", "side_flap_left", "side_flap_right")}
        self._force = np.zeros(6)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for name, angle in zip(ARM_ACTUATORS, START_LEFT_V4 + START_RIGHT_V4):
            self.d.joint(name).qpos = angle
            self.d.actuator(name).ctrl = angle
        # options={"lid_angle": rad} fixes the start angle (e.g. for watching); otherwise
        # the curriculum decides.
        lid_start = (options or {}).get("lid_angle")
        if lid_start is None and self.np_random.random() < CURRICULUM_P:
            lid_start = self.np_random.uniform(*LID_START_RANGE)
        if lid_start is not None:
            self.d.joint("lid_hinge").qpos = lid_start
            if lid_start < LOW_LID:
                for name in ("side_flap_left", "side_flap_right"):
                    self.d.joint(name).qpos = LOW_LID_SIDE_FLAPS
        mujoco.mj_forward(self.m, self.d)
        self.box_start = self.d.body("box").xpos.copy()
        self.box_quat_start = self.d.body("box").xquat.copy()
        c, s = np.cos(LEFT_YAW), np.sin(LEFT_YAW)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.d.body("openarm_left_hand").xmat.reshape(3, 3)
        mujoco.mju_mat2Quat(self.left_quat_target, R.flatten())
        self.stop_point = self.d.site("box_side_stop").xpos.copy()   # fixed stop point (world)
        self.demo_start = options is None and self.demo is not None and self.np_random.random() < DEMO_P
        if self.demo_start:
            i = self.np_random.integers(len(self.demo["qpos"]))
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = (self.demo[k][i] for k in ("qpos", "qvel", "ctrl"))
            mujoco.mj_forward(self.m, self.d)
        # Gripper states as step() would have sent them (left always closed, right as restored).
        self.right_closed = self.d.actuator(GRIPPER_ACTUATORS[1]).ctrl[0] < FINGER_OPEN / 2
        self.grip_count, self.grip_pending = 0, 0.0
        self.prev_action = np.zeros(16, dtype=np.float32)
        self.prev_action[LEFT_GRIPPER] = 1.0
        self.prev_action[RIGHT_GRIPPER] = 1.0 if self.right_closed else -1.0
        return self._get_obs(), {}

    def _get_obs(self):
        d = self.d
        obs = np.concatenate([
            [d.joint(n).qpos[0] for n in ARM_ACTUATORS],
            d.qvel[self.arm_qvel_adr],
            [d.joint(n).qpos[0] for n in GRIPPER_ACTUATORS],
            [d.joint("lid_hinge").qpos[0], d.qvel[self.lid_qvel_adr], d.joint("flap_hinge").qpos[0]],
            d.site("flap_edge").xpos - d.body("openarm_right_hand_tcp").xpos,
            self.left_target() - d.body("openarm_left_hand_tcp").xpos,
            d.body("box").xpos - self.box_start,
            [self._box_turn()],
            self.prev_action,
            [self.grip_pending],
        ])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action[LEFT_GRIPPER] = 1.0
        action[RIGHT_GRIPPER] = self._right_gripper(action[RIGHT_GRIPPER])
        _, _, _, truncated, _ = super().step(action)
        reward, terms, success = self._reward_v4(action)
        self.prev_action = action
        tipped = self.d.body("box").xmat[8] < BOX_TIPPED
        lost = np.linalg.norm(self.d.body("box").xpos - self.box_start) > BOX_LOST
        info = {"is_success": None if self.demo_start else success, "success": success,
                "demo_start": self.demo_start, "terms": terms}
        return self._get_obs(), reward, bool(tipped or lost), truncated, info

    def _right_gripper(self, a):
        """Binary gripper with a dead band and a delay: +1 closed, -1 open."""
        want = True if a > GRIP_BAND else False if a < -GRIP_BAND else None
        if want is None or want == self.right_closed:
            self.grip_count = 0
        else:
            self.grip_count += 1
            if self.grip_count >= GRIP_SWITCH_STEPS:
                self.right_closed, self.grip_count = want, 0
        # Observed: how far a switch has got (+ towards closing, - towards opening).
        self.grip_pending = (1 if want else -1) * self.grip_count / GRIP_SWITCH_STEPS
        return 1.0 if self.right_closed else -1.0

    def left_target(self):
        """Side stop while the lid is high, then (smoothly, 55 -> 45 deg) out of the lid's way."""
        lid = self.d.joint("lid_hinge").qpos[0]
        out = np.clip((LEFT_RETREAT_FROM - lid) / (LEFT_RETREAT_FROM - LEFT_RETREAT_TO), 0.0, 1.0)
        return self.stop_point + out * LEFT_RETREAT

    def _contacts(self):
        """(robot touches itself, left hand touches lid/flaps, left hand's force on the box in N)."""
        self_touch, left_on_lid, left_force = False, False, 0.0
        for i, c in enumerate(self.d.contact[:self.d.ncon]):
            g1, g2 = c.geom1, c.geom2
            if g1 in self.robot_geoms and g2 in self.robot_geoms:
                self_touch = True
            elif g1 in self.left_geoms or g2 in self.left_geoms:
                other = g2 if g1 in self.left_geoms else g1
                if other in self.lid_geoms:
                    left_on_lid = True
                elif other in self.box_geoms:
                    mujoco.mj_contactForce(self.m, self.d, i, self._force)
                    left_force += self._force[0]
        return self_touch, left_on_lid, left_force

    def _box_upright(self):
        return (self.d.body("box").xmat[8] > BOX_UPRIGHT
                and abs(self.d.body("box").xpos[2] - self.box_start[2]) < BOX_LIFTED)

    def _box_turn(self):
        """Angle (rad) the box has turned since the start, about any axis."""
        dot = abs(float(self.d.body("box").xquat @ self.box_quat_start))
        return 2 * np.arccos(min(dot, 1.0))

    def _flap_grasped(self):
        """True when the right fingers pinch the flap: each one presses on it with at least
        GRASP_FORCE, within 60 deg of its closing direction (towards the other finger)."""
        d = self.d
        closing = d.body("openarm_right_right_finger").xpos - d.body("openarm_right_left_finger").xpos
        closing /= np.linalg.norm(closing)          # left finger closes along +closing, right along -closing
        push = {True: np.zeros(3), False: np.zeros(3)}   # force each finger puts on the flap (world)
        for i, c in enumerate(d.contact[:d.ncon]):
            pair = (c.geom1, c.geom2)
            if self.flap_geom not in pair:
                continue
            finger = c.geom1 if c.geom2 == self.flap_geom else c.geom2
            if finger not in self.right_finger_geoms:
                continue
            mujoco.mj_contactForce(self.m, d, i, self._force)
            normal = c.frame[:3] if c.geom1 == finger else -c.frame[:3]   # finger -> flap
            push[finger == self.finger_geom_left] += self._force[0] * normal
        left, right = push[True], push[False]
        return all(np.linalg.norm(f) >= GRASP_FORCE and f @ direction >= GRASP_COS * np.linalg.norm(f)
                   for f, direction in ((left, closing), (right, -closing)))

    def _fingertips(self):
        R = self.d.body("openarm_right_hand").xmat.reshape(3, 3)
        lf = self.d.body("openarm_right_left_finger").xpos + R @ FINGERTIP
        rf = self.d.body("openarm_right_right_finger").xpos + R @ FINGERTIP
        return lf, rf, R

    def _reward_v4(self, action):
        d = self.d
        tcp = d.body("openarm_right_hand_tcp").xpos
        tip = d.site("flap_edge").xpos
        F = d.body("flap").xmat.reshape(3, 3)
        flap_dir, flap_normal = -F[:, 0], F[:, 2]       # crease -> tip, plate normal
        lf, rf, R = self._fingertips()
        dist = np.linalg.norm(tcp - tip)

        # 1. approach + align (Isaac: approach_ee_handle, align_ee_handle)
        align_z = R[:, 2] @ -flap_dir                   # fingers point from the tip towards the crease
        align_y = R[:, 1] @ flap_normal                 # fingers close across the plate
        align = 0.5 * (np.sign(align_z) * align_z ** 2 + align_y ** 2)

        # 2. grasp shaping (Isaac: align_grasp_around_handle, approach_gripper_handle, grasp_handle)
        sl, sr = (lf - tip) @ flap_normal, (rf - tip) @ flap_normal
        around = float(sl * sr < 0 and dist < 0.06)
        approach_g = around * ((0.04 - abs(sl)) + (0.04 - abs(sr)))
        grasp = float(self._flap_grasped())

        # 3. close the lid (Isaac: open_drawer_bonus, multi_stage_open_drawer)
        lid_angle = d.joint("lid_hinge").qpos[0]
        progress = float(np.clip((LID_OPEN - lid_angle) / LID_OPEN, 0, 1))
        upright = self._box_upright()
        # Released: below 55 deg the hand lets go and pushes the lid shut. From here on the
        # right hand's terms pay their maximum, so letting go costs nothing. Finished: under
        # 30 deg the lid drops shut by itself.
        released = lid_angle < RELEASE_ANGLE and upright
        finished = lid_angle < LID_SNAP_ANGLE and upright
        dist_reward, wide = approach(dist, 0.02), 1 - np.tanh(2 * dist)
        raw = {"approach_raw": dist_reward, "align_raw": align, "around_raw": around,
               "approach_g_raw": approach_g, "grasp_raw": grasp, "approach_wide_raw": wide}
        if released:
            dist_reward, align, around, approach_g, grasp, wide = 2.0, 1.0, 1.0, 0.08, 1.0, 1.0
        lid = (grasp + 1.0) * progress
        # Isaac's last stage needs the grasp; here the key point is getting under 30 deg
        # (the lid then drops shut by itself), and that may be done by pushing, not gripping.
        stages = (progress > 0.05) * 0.5 + (progress > 0.5) * grasp + finished * 1.0
        # Shut, box standing as it was, flap not folded back over the lid (not tucked: optional).
        success = lid_angle < LID_DONE and upright and d.joint("flap_hinge").qpos[0] > 0

        # 4. two arms + free box
        dist_left = np.linalg.norm(d.body("openarm_left_hand_tcp").xpos - self.left_target())
        # Left hand: start orientation turned 20 deg about z (fingertips towards the left wall).
        # Angle between two orientations = 2 acos(|q1 . q2|)  (Isaac: quat_error_magnitude).
        dot = abs(float(d.body("openarm_left_hand").xquat @ self.left_quat_target))
        left_twist = 2 * np.arccos(min(dot, 1.0))
        self_touch, left_on_lid, left_force = self._contacts()
        push_ok = 1 - np.tanh(max(0.0, left_force - LEFT_FORCE_MAX) / LEFT_FORCE_MAX)
        moved = np.linalg.norm(d.body("box").xpos - self.box_start) + TURN_TO_M * self._box_turn()
        still = 1 - np.tanh(10 * max(0.0, moved - STILL_FREE))   # <=1.5 cm -> 1, 3.5 cm -> 0.80, 6.5 cm -> 0.54

        # 5. smoothness (Isaac: action_rate_l2, joint_vel_l2); the forced left gripper is excluded
        used = np.delete(action - self.prev_action, LEFT_GRIPPER)
        action_rate = float(np.sum(used ** 2))
        joint_vel = float(np.sum(d.qvel[self.arm_qvel_adr] ** 2))

        clear = SELF_CONTACT_GATE if (self_touch or left_on_lid) else 1.0

        terms = {
            "approach": dist_reward, "align": align, "around": around, "approach_g": approach_g,
            "grasp": grasp, "lid": lid, "stages": stages, "stop": approach(dist_left, 0.02) * push_ok, "success": float(success),
            "action_rate": action_rate, "joint_vel": joint_vel,
            "stop_wide": (1 - np.tanh(2 * dist_left)) * push_ok, "stop_orient": (1 - np.tanh(left_twist)) * push_ok,
            "approach_wide": wide,
        }
        # success is paid outside the 'still' gate: the lid falling shut shoves the box a bit.
        positive = sum(W[k] * terms[k] for k in W if W[k] > 0 and k != "success")
        negative = sum(W[k] * terms[k] for k in W if W[k] < 0)
        reward = REWARD_SCALE * ((positive * still + W["success"] * terms["success"]) * clear + negative)
        # For printing only (not rewards): the gates and plain distances / angle.
        terms.update(still=still, clear=clear, push_ok=push_ok, left_force=left_force,
                     released=float(released), finished=float(finished), upright=float(upright), **raw,
                     dist_right=dist, dist_left=dist_left, left_twist=left_twist)
        return reward, terms, bool(success)


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    import box_env1
    from view_scene import FLAPS_TUCKED, reset_home

    env = BoxCloseEnvV4()
    check_env(env)
    print("check_env: OK")

    def show(label):
        obs, reward, terminated, truncated, info = env.step(np.zeros(16))
        parts = "  ".join(f"{k} {v:.2f}" for k, v in info["terms"].items())
        print(f"{label}: obs {obs.shape}  reward {reward:.3f}  success {info['is_success']}\n    {parts}")

    env.reset(options={"lid_angle": LID_OPEN})     # fixed lid: no curriculum
    show("start pose, lid open")
    box_env1.reset_home = lambda m, d: reset_home(m, d, lid_angle=0.0, flaps=FLAPS_TUCKED)
    env.reset(options={"lid_angle": 0.0})
    show("lid closed (cheat)  ")
