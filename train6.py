"""Train PPO on the box-closing task, box_env6: hook and push with a closed fist
(train5.py: box_env5, the pinch; train.py: box_env4).

    conda activate mujoco_rl
    python train6.py                  # ~1.5 h for 10 M steps on 12 cores -> PPO_v0.3.zip
    python train6.py --resume         # continue PPO_v0.3.zip for TOTAL_STEPS more (only if box_env6 is unchanged)
    python train6.py --resume checkpoints/PPO_v0.3_<steps>_steps   # branch from a checkpoint:
                                      # saved as PPO_v0.3_from<steps>.zip, the main run stays as it is
    Ctrl+C stops training and still saves the model.
    ls checkpoints/                   # a copy every 499,992 steps (12 envs): PPO_v0.3_1999968_steps.zip, ...
A fresh run refuses to start if PPO_v0.3.zip already exists (it would be overwritten):
--resume it, rename it, or add --overwrite. A second branch from the same checkpoint
gets _b2, _b3, ... (PPO_v0.3_from1999968_b2.zip). On --resume the PPO settings below (HP) are
applied again, so editing e.g. learning_rate here and resuming takes effect.
    tensorboard --logdir runs        # watch rollout/success_rate and rollout/real_start_return

Half the episodes start from demo states (demo_states6.npz), which pay much more than the
real start, so rollout/ep_rew_mean mixes the two. success_rate counts real starts only,
and rollout/real_start_return / real_start_len are the real-start episodes alone.

PPO settings follow Isaac Lab's OpenArm cabinet task (rsl_rl_ppo_cfg.py):
networks 256-128-64 with ELU, lr 5e-4, entropy 1e-3, 5 epochs, 4 minibatches,
gamma 0.99, lambda 0.95, clip 0.2, target KL 0.02, initial action std 1.0.
(Not identical: Isaac's desired_kl 0.02 drives an adaptive learning rate; SB3's target_kl
stops an update early at KL > 0.03 with a fixed lr. If the log says "Early stopping at
step 0/1" in most updates, lower learning_rate.)
"""
import argparse
import re
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from box_env6 import BoxCloseEnvV6

N_ENVS = 12              # parallel env copies (16 cores, leave some free)
N_STEPS = 1024           # steps per env per update -> 12 x 1024 = 12288 per update
TOTAL_STEPS = 10_000_000   # ppo_box_v5 first closed at ~3 M; this task is harder (pinch required, randomised)
MODEL_NAME = "PPO_v0.3"     # = box_env6, fist push (PPO_v0.2 = box_env5 pinch, PPO_v0.1: fist start bug)
MODEL_FILE = f"{MODEL_NAME}.zip"   # explicit: SB3 adds .zip only to names without a dot
# PPO settings, used for a fresh run and again on --resume (else the zip's saved values win).
HP = dict(n_steps=N_STEPS, batch_size=N_ENVS * N_STEPS // 4, n_epochs=5,
          learning_rate=5e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
          ent_coef=1e-3, vf_coef=1.0, max_grad_norm=1.0, target_kl=0.02)
POLICY_KWARGS = dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                     activation_fn=nn.ELU, log_std_init=0.0)


def taken(name):
    """True if a run called `name` already left a model zip or checkpoints."""
    pattern = re.compile(rf"{re.escape(name)}_\d+_steps\.zip")
    return Path(f"{name}.zip").exists() or any(pattern.fullmatch(p.name) for p in Path("checkpoints").glob("*.zip"))


class RealStartStats(BaseCallback):
    """Logs return and length of the episodes that began at the real start (no demo state)."""

    def __init__(self):
        super().__init__()
        self.returns, self.lengths = [], []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info and not info.get("demo_start", False):
                self.returns.append(info["episode"]["r"])
                self.lengths.append(info["episode"]["l"])
        if self.returns:
            self.logger.record("rollout/real_start_return", np.mean(self.returns[-100:]))
            self.logger.record("rollout/real_start_len", np.mean(self.lengths[-100:]))
        return True

if __name__ == "__main__":   # required: each subprocess imports this file
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs="?", const=MODEL_FILE, default=None, metavar="ZIP",
                        help=f"continue training {MODEL_FILE}, or branch from the given checkpoint")
    parser.add_argument("--overwrite", action="store_true", help=f"start fresh even if {MODEL_FILE} exists")
    args = parser.parse_args()
    if args.resume is None and taken(MODEL_NAME) and not args.overwrite:
        raise SystemExit(f"{MODEL_FILE} or its checkpoints already exist. Use --resume to continue it, "
                         f"rename / move them, or --overwrite to start a new run over them.")

    env = make_vec_env(BoxCloseEnvV6, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    name = MODEL_NAME
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log="runs", **HP)
        if Path(args.resume).resolve() not in (Path(MODEL_FILE).resolve(), Path(MODEL_NAME).resolve()):
            name = base = f"{MODEL_NAME}_from{model.num_timesteps}"   # a branch: don't overwrite the main run
            k = 1
            while taken(name):          # nor an earlier branch from the same checkpoint
                k += 1
                name = f"{base}_b{k}"
        print(f"resumed {args.resume} at {model.num_timesteps} steps, saving as {name}.zip, "
              f"lr {model.learning_rate}, target_kl {model.target_kl}")
    else:
        model = PPO("MlpPolicy", env, device="cpu", verbose=1, tensorboard_log="runs",
                    policy_kwargs=POLICY_KWARGS, **HP)
    # A copy every 500 k steps in checkpoints/, so you can watch it while it trains.
    save = CheckpointCallback(save_freq=500_000 // N_ENVS, save_path="checkpoints", name_prefix=name)
    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=[save, RealStartStats()],
                    reset_num_timesteps=args.resume is None, tb_log_name=name)
    finally:   # also on Ctrl+C or a crash, so --resume always has a file to continue from
        model.save(f"{name}.zip")   # explicit .zip: SB3 adds it only to names without a dot
        print(f"saved {name}.zip at {model.num_timesteps} steps")
