# tracklog

Oct 27, 2026. adibot box-closing RL: every version, reward change and problem from Oct 15 to PPO_v0.3.
Online copy: https://claude.ai/artifact/1zKXQKTLMbCombDX9iiAqe

## Task and setup

The goal is for adibot (two OpenArm 7-DOF arms) to close the lid of a 45 g clamp-type cardboard box in MuJoCo, then move the policy to the real robot. One arm has to hold the box still while the other closes it.

| Item | Value |
| --- | --- |
| Simulator | MuJoCo, Gymnasium env, Stable-Baselines3 PPO |
| Project folder | `~/Downloads/adibot_mujoco`, conda env `mujoco_rl` |
| Box | 45 g empty. The real box will hold 3 items, so it will be heavier and slide less |
| Lid start | 120 deg open |
| Control | Delta joint targets, 0.03 rad per policy step, 50 steps/s, 500 steps per episode |
| Real cameras | Wrist: D405i. Head: Gemini 336L. Both at 640x480 |
| Training | 12 parallel envs, 10 M steps, about 1.5 h |

Rules set during the project:

- The same start pose every run, with official OpenArm values and collisions on.
- Make minimal changes ("dont keep adding things").
- Measure instead of guessing, and use proven open-source patterns.
- Never overwrite an older watch file; write a new one.
- Explain the why in plain English, with tables and real numbers.

## Version timeline

Ten versions in 8 days. The task finally worked when the right hand stopped trying to pinch and pushed with a closed fist: PPO_v0.3 reached 91% success. Newest first.

| Date | Version | What changed | Why |
| --- | --- | --- | --- |
| Oct 22 | box_env6, PPO_v0.3 | Right hand always a fist (14 actions). Hooks over the flap and pushes the lid's outer face. Left fist stops the box in front. Lid limit 3 rad/s. Box mass 36-540 g. 57 measurable observations | Pinching never worked in training, and a 2 mm pinch is hard to see with 640x480 cameras |
| Oct 22 | box_env5, PPO_v0.2 | Reset opens the right gripper. around 0.125 to 3.0, grasp 5.0 to 2.5 | v0.1 learned to keep a fist |
| Oct 22 | box_env5, PPO_v0.1 | `pulled` latch (pinch must pull the lid at least 20 deg at a controlled speed). Slam latch. Lid speed penalty. Calm term. Domain randomization. Box model rebuilt to match the real flaps | ppo_box_v5 flung the lid shut at 14 rad/s |
| Oct 21 | box_env4, ppo_box_v5 | 2 review rounds, side stop for the left hand, demo starts, ManiSkill is_grasping | ppo_box_v4 never closed the gripper |
| Oct 20 | box_env4, ppo_box_v4 | Real box (16x16x4 cm, 45 g) at 32 cm. Isaac Lab OpenArm cabinet reward and PPO settings. Arm collision pairs. Official motor values | Pull with v3 was out of reach and the arms passed through each other |
| Oct 20 | box_env3 | Pinch the front flap and pull. Left hand braces the corner. Box at 70% | v2's right hand sat under the lid, 10 cm out of reach |
| Oct 19 | box_env2 | ManiSkill/robosuite style: bounded tanh terms, still gate, everything positive | v1 slapped the lid and shoved the box 10-15 cm |
| Oct 19 | box_env1, PPO_3 | First PPO run (1 M steps). User-written reward: distances, box moved, lid angle, +10 on close | Learning stage |
| Oct 18 | Box model | Airplane box with front tuck flap and trapezoid side flaps. Start pose near the box | Random arms never touched the box |
| Oct 15 | box_env.py | Skeleton: 16 actions, 20 observations, reward 0 | Learning stage |

## How the reward design changed

The reward started as one line of negative distances and ended as bounded positive terms. Two latches now decide whether a close counts: `pulled` (the lid was actually moved by the hand) and `slammed` (it fell too fast). Nearly every change below was a fix for a shortcut the policy, or a test, found.

### box_env1 (user-written)

`reward = -dist_right - dist_left - 5*box_moved - lid_angle`, plus +10 when lid < 0.05 rad.

- **Box weight 5:** 3 cm of sliding costs as much as 15 cm of distance. Without it, dragging the box towards the hand is free.
- **Threshold 0.05 rad, not 0.1:** a lid that just drops onto the flaps reads 0.09. A threshold of 0.1 would pay for letting it fall.
- **Problem:** a 12 cm shove cost only 0.6, against +1.9 for the lid. PPO learned to slap the lid and shove the box.

### box_env2 (open-source pattern)

`reward = (reach + hold + 2*lid) * still`, or `5*still` once closed, minus `0.01*sum(a^2)`, all divided by 5.

- reach = 1-tanh(5d), and lid = 1-angle/2.0.
- **Still gate:** still = 1-tanh(10*box_moved). It multiplies everything, so moving the box loses all pay.
- **All positive:** ending the episode when the box tips is now a loss, not an escape.
- **Scores:** standing still 84.5, the slap 4-20, a closed lid about 490.

### box_env3

- **New grasp term:** +1 when both right fingers touch the front flap.
- **Targets:** the right target became the flap tip. The left target became a brace at the front-left corner.

### box_env4 (Isaac Lab OpenArm cabinet port)

Starting weights:

| Term | Weight |
| --- | --- |
| lid (x2 while grasped) | 7.5 |
| success | 5 |
| approach_g | 5 |
| approach | 2 |
| brace | 2 |
| stages | 1 |
| align | 0.5 |
| grasp | 0.5 |
| around | 0.125 |
| action_rate | -0.01 |
| joint_vel | -0.0001 |

Changes in order, each with its reason:

1. **Still gate also counts box rotation** (17 deg counts as much as 3 cm). Episode ends if the box tips or slides more than 10 cm.
2. **Approach 1/(1+d^2) changed to 1-tanh(10d).** The old form paid 0.994 at 8 cm against 1.0 touching, so there was no reason to get close.
3. **Reward scale x0.1 changed to x0.02** (the step time, as in Isaac). The policy's share of the gradient went from 7.5% to 34%.
4. **Negative distance and twist terms made positive, 0 to 1.** 84% of random steps had been negative, which made ending the episode early attractive. After: 0%.
5. **3 cm free before the still gate starts** (later 1.5 cm). Pinching the flap turns the box about 9 deg.
6. **Grasp fix.**
   - The gripper action became binary (above 0 = closed). Grasp weight 5. The lid x2 bonus needs a real grip.
   - Closing the gripper had lowered the reward (0.27 to 0.32 after the fix).
7. **Demo starts** (Salimans and Chen 2018). Half the episodes start from states saved along the scripted solution.
8. **Review fixes.**
   - Success worth 15, paid outside the still gate.
   - Success also needs tilt < 5 deg, lift < 5 mm and the flap not folded back.
   - Robot self-contact halves the reward.
9. **Side stop for the left hand.** Terms stop, stop_wide and stop_orient. A `clear` gate halves the reward if the left hand touches the lid.
10. **RELEASE_ANGLE 55 deg.** Below it every right-hand term pays its maximum. action_rate went from -0.01 to -0.05.
11. **Grasp = ManiSkill is_grasping:** each finger at least 0.5 N, from opposite sides. A closed fist could earn the old grasp by poking.

### box_env5

1. **`pulled` latch:** no success and no lid pay unless one continuous pinch pulled the lid at least 20 deg at no more than 2.5 rad/s.
2. **Slam latch:** a lid faster than 8.5 rad/s means no success for that episode.
3. **Lid speed penalty:** capped and quadratic, above 2.5 rad/s, while the lid is above 30 deg. The worst step went from -26 to -5.7.
4. **Calm term:** pays for holding both arms still after the close. Left-arm terms x3 after closing.
5. **v0.2 reweight:** around 0.125 to 3.0, grasp 5.0 to 2.5. Reaching the flap must pay more than closing it. A proposed "no empty fist" penalty was rejected ("dont keep adding things").
6. **Scores:** scripted 237, standing still 78 (open hand), the old fling -16, random +20.

### box_env6 (hook and push)

1. **Grasp terms removed:** around, grasp, approach_g.
2. **`pulled` changed:** it now means the fist at the lid's outer face moved the lid at least 20 deg at a controlled speed. `at_lid` is geometric, not contact-based, so the real robot can measure it.
3. **Slam limit 8.5 to 3 rad/s:**

| Slam limit | Gentle push | Fast push | Problem |
| --- | --- | --- | --- |
| 8.5 rad/s | 285 | 282 (5.3 rad/s, flap folded) | Nearly as good to fling |
| 4 rad/s | 285 | 342 (3.3 rad/s) | Fling pays more |
| 3 rad/s (kept) | 285 | 85 and 71 | Gentle wins |

A lid falling on its own never exceeds 2.2 rad/s, so 3 rad/s never punishes a natural fall.

4. **Scores:** scripted 285, standing still 59, random-action worst step -0.02.

## Every problem we hit

Most failures came from the reward paying for a shortcut, or from the simulated box and robot not matching the real ones. Grouped by kind, oldest first within each group.

### Policy shortcuts (reward loopholes)

| Version | What the policy did | Cause | Fix |
| --- | --- | --- | --- |
| v1 PPO_3 | Slapped the lid, shoved the box 10-15 cm, tipped it once. 0 closes without action noise | Box penalty too weak (12 cm cost 0.6 vs +1.9 for the lid) | box_env2: still gate multiplies everything |
| v2 | Right hand sat under the lid, 6.5 cm away. Gripper never moved. Success 0 in 5 M steps | No term needed the gripper. The target behind the lid was 10 cm out of reach (66 cm vs a 62 cm arm) | box_env3: pinch the flap and pull, box moved closer |
| v4 | Gripper never closed (command -0.8 to +0.5, fingers 22-27 mm). Success 0 | Grasp paid 0.04 at most. `around` was easy to fool. Closing lowered the reward | Binary gripper, grasp weight 5, demo starts |
| v4 | Holding the flap at 35-45 deg paid 0.52 per step vs 0.28-0.36 for closed | Pinch terms outpaid finishing | Max right-hand pay below 30 deg, later 55 deg. Success 15 |
| v4 | Success counted with the box tilted 14-18 deg and lifted 2.4 cm | Success checked only the lid angle | Success needs tilt < 5 deg, lift < 5 mm, flap not folded |
| v4 | Hand stayed further back (3.7 to 5.0 cm) | Action noise near the flap ended episodes (box moved > 10 cm) | Box-lost limit 20 cm |
| v4/v5 | A closed fist earned the grasp by poking the flap | Contact on both fingers was enough | ManiSkill is_grasping (0.5 N each, opposite sides) |
| ppo_box_v5 | Hooked the lid edge with an open hand and flung it shut: 14 rad/s, 0.7 s, box dragged 3.4 cm. Never pinched. 65% "success" | Paid for the result, not for how it was done. No speed limit | box_env5: `pulled` and `slammed` latches, speed penalty |
| v0.1 review | Push without pinch then a brief pinch counted. A 13-15 rad/s slam was paid. A box shoved > 20 cm counted as success | Latch gaps | One continuous pinch must pull at least 20 deg. Lost box = fail |
| PPO_v0.1 | Kept the right gripper closed as a fist from 3 M steps. 0/18 real starts | `mj_resetData` sets ctrl = 0, which is closed, so every real start began as a fist. Demo states paid closing right away | Reset opens the gripper (158/158) |
| PPO_v0.1 | Hand drifted 10-25 cm from the flap | Reach far away paid almost the same (1.48 at 10 cm, 0.85 at 20 cm) | v0.2: around 3.0, grasp 2.5 |
| PPO_v0.2 | Hovered at the flap with an open gripper, both hands drifting 3-4 cm/s early on | Pinching too hard to discover. A constant action offset integrates under delta control | box_env6: hook and push |
| v6 test | A 4x faster push scored 282 vs 285. A 2.7x push scored 342 with a 4 rad/s limit | Slam limit too high | Limit 3 rad/s: gentle 285, fast 85 and 71 |

### Physics and robot model

| Problem | Cause | Fix |
| --- | --- | --- |
| Random arms never touched the box | Hands started 0.87-0.9 m away, below the table | Start pose near the box, dense reach reward |
| Side flaps landed on the walls; lid stopped 6 deg short | Flap tilt and crease position | Tuck-in flaps, later rebuilt to match the real box |
| 70% box: flaps held the lid at 21 deg | Crease springs too strong for the smaller box | Springs x0.34 |
| Top-down pinch out of reach at 40 and 34 cm | Arm length | Box at 32 cm |
| Box half off the table | Table edge at x = 32 cm | Table extended to x = 12 cm |
| Arms passed through each other | Robot self-collision was off; base convex hull swallows the arms | `assets/arm_collision.xml`, 137 contact pairs |
| Lid could never close below 30 deg | Side flaps rested on the wall tops | Flaps pass inside the walls |
| Guessed motor values | Not taken from OpenArm | Official armature, damping, friction, gripper; gravity feed-forward 80% |
| Start joints pinned at their limits (4 right joints at 0.00 margin) | Hand-posed start | IK start poses with at least 0.27 rad margin |
| Pinching turned the box 9 deg; unbraced pull slid it 3-8 cm | 45 g box | Left-hand brace, then side stop, then front stop |
| Left wall pinch moved the box 8 cm and turned it 70 deg | Square grip unreachable (forearm hits torso) | Dropped; left hand is a stop, not a gripper |
| Lid crept shut from 40 deg in 10 s | Soft hinge friction | Stiffer crease; snap torque removed |
| Left wrist jammed the torso, up to 39.5 N | Retreat path | Retreat sideways and up; self-contact halves reward |
| Left target jumped 7 cm and flickered 8 times | Hard switch at 50 deg | Smooth blend from 55 to 45 deg |
| Side flaps stuck 8.6 mm through the walls | Box contact off, wrong tilt | Contact on, flaps at the lid's outer edge, 18 deg outward |
| Side flap hit the left hand below 28 deg | Hand stayed in place | Left retreat; minimum gap 32 mm |
| Over 30 N on the shut lid tips the box | Soft contacts, 45 g box | Known limit; the scripted push uses about 12 N |
| v6: pushing dragged the box 4.3 cm; front flap jammed on the robot base | Side stop gives no force in that direction | Left fist in front of the box: 0.5-1.6 cm |
| v6: left wrist hit the torso guard (460 steps in one start) | Route via a point above the box | Straight route: 0 contacts in 17 starts |
| v6: lid stayed at 13-25 deg after the push | Hinge friction x1.7-2 | Push until below 2 deg: 20/20 succeed |
| v6: hook IK flailed behind the lid | Start orientation kept fixed | Free right-hand orientation |

### Training setup

| Problem | Cause | Fix |
| --- | --- | --- |
| PPO_1/2: episodes 1 step long, reward 10 | Cheat start (closed lid) left in reset | Removed |
| Reward x0.1 barely moved the policy (std stuck) | Value loss dominated | Scale x0.02 |
| Gripper flipped 179 times per episode, 0 grips | Never found the reward | Demo starts, dead band +-0.3, 3-step delay |
| success_rate inflated | 8 of 75 demo states close by themselves | Count real starts only, log real_start_return |
| Policy saw hidden targets (0.4-0.5 rad gap) | Delta targets not in the observation | Observation fixed |
| Lid speed from the simulator not measurable | 1.2 rad/s noise vs a 0.1-0.75 signal on hardware | Observations limited to what cameras and joints give |
| x1.5 friction or x2 box mass broke the script | No randomization | Domain randomization (Isaac Lab events style) |
| "Early stopping at step 3, max kl 0.03" | 1.5 x target_kl 0.02 | Normal, no fix |
| Isaac is not the same setup | Isaac: 1 arm, 8 actions, 4096 envs. Here: 16 actions, 12 envs, about 30x less data | Kept the Isaac settings; scripted left arm rejected (must be bimanual) |

### Code and tool errors

| Error | Cause | Fix |
| --- | --- | --- |
| Observation stuck at (20,) | 4 copies of the env; VS Code and the terminal used different ones | One file in one folder |
| Reward was a 3-vector / `'numpy.float64' object is not callable` / NameError | Subtracted vectors instead of distances; missing `*`; local vs `self.` | Fixed in box_env1 |
| Sign flip gave +1.684 for an open lid | Brackets around negative terms | One flat line, a sign on each term |
| `ImportError make_venc_env` | Typo | `make_vec_env` |
| `ModuleNotFoundError: torch` | Ran `/usr/bin/python3` | `conda activate mujoco_rl` |
| watch.py IndentationError, missing imports, `MAX_STEPS` | User code | Fixed; constant is MAX_EPISODE_STEPS |
| Viewer closed at the end of the `with` | No loop | `while viewer.is_running()` |
| SubprocVecEnv failed from stdin | Needs a real `__main__` file | Script file |
| Background run killed | Low memory | Run training in your own terminal |
| check_rewards drifted 24 cm, left hand slammed the box | Fed measured joints back as targets; 50 cm/s hand speed; gravity-sag windup | IK on the commanded arm, 15-25 cm/s, slow sag correction |
| `--save` silently saved nothing | Relative path | Absolute path plus a warning |
| train.py lost the model on Ctrl+C | Saved only on a normal finish | try/finally save |
| train.py run instead of train5.py | Would overwrite ppo_box_v5 | train5 refuses to overwrite without --overwrite |
| watch3 FileNotFoundError for PPO_v0.2_2999952 | Claude gave a checkpoint name that did not exist yet | `ls checkpoints/` first |
| "GLXBadDrawable", "NV-GLX missing", segfault on viewer close | Linux viewer | Harmless |
| Stray file `ppo_box_v4_no grasp.zip` | Unknown | Now in the archive |

### Sim-to-real

- **Pinch:** it needs about 2 mm accuracy, but the D405 is blind below about 7 cm.
- **Camera resolution:** at 640x480 the wrist camera gives about 0.3-0.6 mm/px. The head camera gives about 2 mm/px and about 5 mm depth error.
- **Result:** this was the main reason for switching to the fist push, which needs only the lid pose.

## Training results

PPO_v0.3 is the first run that really solves the task: 91% success from the real start at 10 M steps. ppo_box_v5's 65% was a lid fling, not a real close. Newest first.

| Run | Env | Steps | Success | Real-start return | What it did |
| --- | --- | --- | --- | --- | --- |
| PPO_v0.3 | box_env6 | 10 M | 0 until 3 M, 9% at 4 M, 38% at 5 M, 76% at 7 M, **91% at 10 M** | 20 to 372 (script 285) | Fist push. 3/5 randomised starts closed with deterministic actions |
| PPO_v0.2 | box_env5 | 10 M | 0 until 7 M, 3% at 8 M, 4-9% at 10 M | 14 to 87-99 | Hovers at the flap with an open gripper. Still improving when stopped |
| PPO_v0.1 | box_env5 | about 7 M | 0 | 27 (standing still 54) | Fist from 3 M on (reset bug) |
| ppo_box_v5 | box_env4 | 5 M | 0.16 at 3 M, 0.65 at 5 M | 15 to 245 (script 203) | Flung the lid at 14 rad/s; never pinched. 1/12 on the v5 box |
| ppo_box_v4 (PPO_3) | box_env4 | 5 M + 5 M | 0 | about 90 to 124 | Gripper half open, never gripped |
| ppo_box_v4 (PPO_2) | box_env4 | 5 M | 2-3% (curriculum starts only) | 380-410 | Arms never touched the box |
| ppo_box_v2 | box_env2 | 5 M | 0 | | Hands 6.5 / 2.5 cm from targets |
| PPO_3 | box_env1 | 1 M | 5/10 with noise, 0 without | -1566 to -1036 | Slap and shove |

PPO_v0.3's action noise (std) fell from 1.00 to 0.84. It is still learning, so more steps could raise success further.

## Open items

- [ ] Watch PPO_v0.3 with `python watch4.py --model PPO_v0.3` and count its successes over about 50 randomised starts.
- [ ] Weigh the real box with its 3 items, then set `RAND_BOX_MASS` to match. It is now x0.8-12, i.e. 36-540 g.
- [ ] After success, stop or move both arms back to a home pose.
- [ ] Lid pose and box pose from the D405i wrist camera and the Gemini 336L head camera, at 640x480.
- [ ] Match the real controller: 100 Hz, MIT-mode kp/kd, gravity feed-forward 0.8, 10.5 Nm cap.
- [ ] Rebuild `demo_states6.npz` with `python check_rewards6.py --save` before training box_env6 again.

## Files

The project folder holds only what PPO_v0.3 needs to run, watch and retrain.

| File | Role |
| --- | --- |
| `tracklog.md` | This log |
| `box_env6.py` | Current task: fist push, left hand as front stop |
| `box_env5.py`, `box_env4.py`, `box_env1.py` | Parent classes that box_env6 builds on (still imported) |
| `view_scene.py`, `box_scene.xml`, `box_scene_v5.xml` | Scene, home pose, lid start angle |
| `adibot.xml`, `assets/`, `meshes/` | Robot and box models |
| `check_rewards6.py` | Scripted solution and reward check |
| `train6.py`, `watch4.py` | Train and watch PPO_v0.3 |
| `PPO_v0.3.zip` | Trained model (91%) |

Moved to `~/Downloads/adibot_mujoco_old` (78 MB):

- **Old envs and backups:** box_env, box_env2, box_env3, box_env4_before_*.
- **Old scripts:** check_rewards, check_rewards5, train, train5, watch, watch2, watch3.
- **Demo states:** all 4 `.npz` files.
- **Old models:** ppo_box_v4, ppo_box_v5, PPO_v0.1, PPO_v0.2.
- **Checkpoints:** all checkpoints, including PPO_v0.3's 20.
- **Old TensorBoard runs.**
- **URDF tools:** convert_urdf.py, adibot.urdf, source/, scene.xml. Only needed to rebuild adibot.xml.
- **Leftovers:** explore.py, view_scene_orig.py, convert_urdf_orig.py, adibot_before_v1params.xml.

`snapshots/` (the v0.2 pinch copy) and `runs/` (v0.2 and v0.3 TensorBoard logs) are in the system Trash (removed Oct 27, 17:12).
