"""Box-closing RL environment, version 1 (base class for v2 and v3).

Robot setup, start pose, actions, observation, and the first reward
(negative distances + lid angle, +10 bonus and termination when closed).
v2/v3 inherit from BoxCloseEnv and replace the reward.

    conda activate mujoco_rl
    python box_env1.py         # run 10 s with zero actions in the viewer
"""
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# Reuse the scene path and home pose from view_scene.py (same folder).
from view_scene import SCENE, reset_home

# ---------------------------------------------------------------------------
# Settings. Constants are written in CAPITALS so they are easy to spot.
# ---------------------------------------------------------------------------
# The 14 arm motors the policy controls. The list is built with a
# "list comprehension": a short way to write a for loop that makes a list.
ARM_ACTUATORS = [f"openarm_{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
GRIPPER_ACTUATORS = ["openarm_left_finger_joint1", "openarm_right_finger_joint1"]

MAX_STEP = 0.03      # rad: biggest change per arm joint in one step (action = +-1)
LID_CLOSED = 0.05    # rad: below this the lid counts as closed (a dropped lid rests at ~0.06-0.09)
FRAME_SKIP = 10      # physics steps per policy step: 10 x 0.002 s = 0.02 s (50 Hz)
MAX_EPISODE_STEPS = 500  # 500 x 0.02 s = 10 s per episode
# Start pose (joint angles, rad), found in the viewer for the full-size box at 47 cm.
START_LEFT = [-0.583, -0.367, 0.754, 0.611, -0.408, -0.275, -0.518]
START_RIGHT = [0.974, 0.681, -0.833, 1.04, -0.0471, 0.00785, 0.22]


class BoxCloseEnv(gym.Env):
    """Two OpenArm arms, one cardboard box with a hinged lid."""

    def __init__(self, scene=SCENE):
        super().__init__()  # let gym.Env set itself up first

        # MuJoCo model (fixed) and data (changes every step). box_env5 passes its own scene.
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)

        # Look up the limits of each arm motor once, so step() is fast.
        # Shape (14, 2): one [low, high] row per motor.
        self.arm_ranges = np.array([self.m.actuator(n).ctrlrange for n in ARM_ACTUATORS])

        # The policy outputs 16 numbers, each between -1 and 1:
        #   [0:14]  arm joints  -> delta (small change)
        #   [14:16] grippers    -> absolute (-1 = open, +1 = closed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        # The policy sees 23 numbers (see _get_obs). They have no fixed
        # limits, so we use -inf..inf.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        self.step_count = 0

    # -----------------------------------------------------------------------
    def _get_obs(self):
        """Build the observation vector: 14 + 2 + 1 + 3 + 3 = 23 numbers."""
        # Arm joint angles. The arm motor and its joint have the same name.
        arm_angles = [self.d.joint(n).qpos[0] for n in ARM_ACTUATORS]
        # Gripper finger positions: 0 = closed, 0.044 = open.
        grippers = [self.d.joint(n).qpos[0] for n in GRIPPER_ACTUATORS]
        lid_angle = [self.d.joint("lid_hinge").qpos[0]]
        box_pos = self.d.body("box").xpos  # already an array [x, y, z]
        lid_handle = self.d.site("lid_handle").xpos

        obs = np.concatenate([arm_angles, grippers, lid_angle, box_pos, lid_handle])
        return obs.astype(np.float32)

    def _compute_reward(self):
        right_hand = self.d.body("openarm_right_hand_tcp").xpos
        left_hand = self.d.body("openarm_left_hand_tcp").xpos
        lid_handle = self.d.site("lid_handle").xpos
        box_left = self.d.site("box_left_side").xpos
        box_moved = np.linalg.norm(self.d.body("box").xpos - self.box_start)
        dist_right = np.linalg.norm(right_hand - lid_handle)
        dist_left = np.linalg.norm(box_left - left_hand)
        lid_angle = self.d.joint("lid_hinge").qpos[0]
        bonus = 10 * (lid_angle < LID_CLOSED)

        return -dist_right - dist_left - 5 * box_moved - lid_angle + bonus

    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # sets up self.np_random for later randomness
        base_pose = np.concatenate([START_LEFT, START_RIGHT])
        reset_home(self.m, self.d)
        for i, name in enumerate(ARM_ACTUATORS):
            self.d.actuator(name).ctrl = base_pose[i]   # motor target (else it pulls back)
            self.d.joint(name).qpos = base_pose[i]      # teleport
        mujoco.mj_forward(self.m, self.d)
        self.box_start = self.d.body("box").xpos.copy()
        self.step_count = 0
        return self._get_obs(), {}

    # -----------------------------------------------------------------------
    def step(self, action):
        # --- 1. Arms: delta control ------------------------------------
        for i, name in enumerate(ARM_ACTUATORS):
            old_target = self.d.actuator(name).ctrl[0]
            delta = action[i] * MAX_STEP                         # scale
            new_target = old_target + delta                      # add
            low, high = self.arm_ranges[i]
            self.d.actuator(name).ctrl = np.clip(new_target, low, high)  # clip + send

        # --- 2. Grippers: absolute control -----------------------------
        # Map action -1..+1 to finger position 0.044 (open)..0 (closed).
        for j, name in enumerate(GRIPPER_ACTUATORS):
            a = action[14 + j]
            self.d.actuator(name).ctrl = (1 - a) / 2 * 0.044

        # --- 3. Physics --------------------------------------------------
        # The policy decides 50 times per second; physics runs 500 times
        # per second. So we hold the same command for FRAME_SKIP steps.
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.m, self.d)

        # --- 4. Result ---------------------------------------------------
        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.d.joint("lid_hinge").qpos[0] < LID_CLOSED
        truncated = self.step_count >= MAX_EPISODE_STEPS  # out of time
        info = {}
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Runs only when you start this file directly (not when another file imports it).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = BoxCloseEnv()
    obs,info = env.reset()
    with mujoco.viewer.launch_passive(env.m,env.d) as viewer:
        for i in range(MAX_EPISODE_STEPS):
            action = np.zeros(16)
            obs, reward, terminated, truncated, info = env.step(action)
            if env.step_count % 50 == 0:
                print(f"step {env.step_count:3d}  lid {obs[16]:.2f} rad  box {np.round(obs[17:20], 3)} reward {reward:.3f}")
            viewer.sync()
            time.sleep(FRAME_SKIP * env.m.opt.timestep)
            if terminated or truncated:
                break

        print (f"step ended at {env.step_count:3d} terminated {terminated} reward {reward}")

      