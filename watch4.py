"""Watch a trained box-closing policy (box_env6, hook and push; watch3.py is for box_env5, watch2.py for
box_env4): the left fist waits in front of the box as a stop (and moves out to the left as the lid goes from
55 to 45 deg), the right fist goes over the front flap, behind the lid, and pushes it shut gently.

Plays episodes back to back until the viewer window is closed. Every second it
prints what each hand is doing; after each episode one result line; at the end a
summary. --nominal: no randomisation (box exactly in place, nominal physics); otherwise every
episode gets box_env6's random box pose, arm start, friction, masses and hinge friction.

Keys in the viewer window (every letter key is already a viewer shortcut, so these are not letters):
    Enter    replay the current episode from the start (same seed: same box, physics and actions);
             right after an episode ends (2 s, or while paused) it replays that one
    Delete   skip to a new episode
    Space    pause / continue (also freezes the end pose of an episode)
(Backspace is the viewer's own reset: it zeroes the whole simulation, don't use it here.)
Each episode prints its seed; python watch4.py --seed N starts from that episode again.

    conda activate mujoco_rl
    python watch4.py                    # best-guess actions, lid starts at 120 deg (the real task)
    python watch4.py --model PPO_v0.3 --nominal   # no randomisation
    python watch4.py --random           # sample actions like during training
    python watch4.py --model checkpoints/PPO_v0.3_4999920_steps --slow 2   # ls checkpoints/ for the names
    python watch4.py --seed 7           # episodes 7, 8, 9, ... (same as the 'seed' they printed)
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from box_env1 import FRAME_SKIP, MAX_EPISODE_STEPS
from box_env4 import BOX_LOST, LID_DONE
from box_env6 import LID_SLAM, BoxCloseEnvV6
from view_scene import LID_OPEN

PRINT_EVERY = 50     # steps (1 s)
KEY_ENTER, KEY_DELETE, KEY_SPACE = 257, 261, 32   # GLFW key codes

keys = {"replay": False, "skip": False, "paused": False}


def on_key(keycode):
    """Runs in the viewer's thread: only set flags, the main loop acts on them."""
    if keycode == KEY_ENTER:
        keys["replay"] = True
    elif keycode == KEY_DELETE:
        keys["skip"] = True
    elif keycode == KEY_SPACE:
        keys["paused"] = not keys["paused"]
        print("paused (Space to continue)" if keys["paused"] else "continue")


def result_of(env, info, terminated):
    if terminated:     # first: a box shoved away after closing is a failure, not a success
        slid = np.linalg.norm(env.d.body("box").xpos - env.box_start)
        return "FAIL: box slid > 20 cm" if slid > BOX_LOST else "FAIL: box tipped over"
    if info["is_success"]:
        return "SUCCESS (lid closed)"
    if info["terms"]["shut"]:
        if info["terms"]["slammed"]:
            return f"lid shut, but SLAMMED (> {LID_SLAM:.0f} rad/s: no success)"
        return "lid shut WITHOUT a gentle push (no success)"
    if env.d.joint("lid_hinge").qpos[0] < LID_DONE:
        if not info["terms"]["upright"]:
            return "lid down, but box tilted / lifted (no success)"
        return "lid down, but front flap folded back (no success)"
    return "time out (lid not closed)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PPO_v0.3")
    parser.add_argument("--random", action="store_true", help="sample actions instead of the best guess")
    parser.add_argument("--nominal", action="store_true", help="no randomisation")
    parser.add_argument("--slow", type=float, default=1.0, help="playback slow-down factor")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = BoxCloseEnvV6()
    results = {}   # seed -> result (a replay overwrites, so it isn't counted twice)
    print("Keys in the viewer: Enter = replay this episode, Delete = new episode, Space = pause")

    with mujoco.viewer.launch_passive(env.m, env.d, key_callback=on_key) as viewer:
        episode, seed = 0, args.seed
        while viewer.is_running():
            # Same seed = same random box pose / physics; seeding the policy too makes --random repeat.
            obs, info = env.reset(seed=seed, options={"lid_angle": LID_OPEN, "nominal": args.nominal})
            model.set_random_seed(seed)
            episode += 1
            print(f"-- ep {episode}, seed {seed}")
            keys["replay"] = keys["skip"] = keys["paused"] = False   # Enter / Delete also un-pause
            total, at_lid, max_moved, first_success = 0.0, 0, 0.0, None
            terminated = truncated = False
            for _ in range(MAX_EPISODE_STEPS):
                while keys["paused"] and viewer.is_running() and not (keys["replay"] or keys["skip"]):
                    viewer.sync()
                    time.sleep(0.05)
                if keys["replay"] or keys["skip"] or not viewer.is_running():
                    break
                action, _ = model.predict(obs, deterministic=not args.random)
                obs, reward, terminated, truncated, info = env.step(action)
                t = info["terms"]
                total += reward
                at_lid += t["at_lid"]
                max_moved = max(max_moved, np.linalg.norm(env.d.body("box").xpos - env.box_start))
                if info["is_success"] and first_success is None:
                    first_success = env.step_count
                if env.step_count % PRINT_EVERY == 0:
                    lid_deg = np.degrees(env.d.joint("lid_hinge").qpos[0])
                    print(f"ep {episode} {env.step_count // 50:2d} s | lid {lid_deg:5.1f} deg  reward {reward:5.2f} | "
                          f"RIGHT to push point {t['dist_right'] * 100:4.1f} cm  at lid {t['at_lid']:.0f}  "
                          f"push {t['push_progress'] * 100:3.0f} %"
                          f"{' (pulled)' if t['pulled'] else ''}{' (SLAMMED)' if t['slammed'] else ''}"
                          f"{' (finished)' if t['finished'] else ''} | "
                          f"LEFT to target {t['dist_left'] * 100:4.1f} cm  twist {np.degrees(t['left_twist']):3.0f} deg  "
                          f"force {t['left_force']:3.1f} N | "
                          f"still {t['still']:.2f} | lid speed {abs(t['lid_vel']):.1f} rad/s")
                viewer.sync()
                time.sleep(FRAME_SKIP * env.m.opt.timestep * args.slow)
                if terminated or truncated:
                    break
            if keys["replay"]:
                print(f"== ep {episode}: replaying (seed {seed})\n")
                continue                        # same seed again
            if keys["skip"]:
                print(f"== ep {episode}: skipped\n")
                seed += 1
                continue
            if not (terminated or truncated):
                break   # window closed mid-episode: don't count it
            result = result_of(env, info, terminated)
            results[seed] = result
            closed_at = f"closed at {first_success * 0.02:.1f} s" if first_success else "never closed"
            print(f"== ep {episode} (seed {seed}): {result} | lid {np.degrees(env.d.joint('lid_hinge').qpos[0]):.1f} deg, "
                  f"{closed_at} | fist at the lid {at_lid * 0.02:.1f} s | box moved max {max_moved * 100:.1f} cm | "
                  f"return {total:.1f}   (Enter: watch it again)\n")
            # Hold the end pose ~2 s (or while paused) so Enter can still replay this episode.
            hold = 0
            while (hold < 40 or keys["paused"]) and viewer.is_running() and not (keys["replay"] or keys["skip"]):
                viewer.sync()
                time.sleep(0.05)
                hold += 1
            if not keys["replay"]:
                seed += 1

    if results:
        n, kinds = len(results), list(results.values())
        print(f"Summary over {n} episodes ({'sampled' if args.random else 'best-guess'} actions):")
        for kind in sorted(set(kinds)):
            print(f"  {kind:40s} {kinds.count(kind)} / {n}")
