# adibot_rl

A bimanual robot (adibot, two OpenArm 7-DOF arms) learns to close a cardboard box lid in MuJoCo with PPO (Stable-Baselines3). The left fist holds the box steady and the right fist pushes the lid shut.

Current model `PPO_v0.3.zip`: 91% success after 10 M steps.

## Run

```
conda activate mujoco_rl
python check_rewards6.py --view          # scripted solution in the viewer
python watch4.py --model PPO_v0.3        # watch the trained policy
python check_rewards6.py --save          # make demo_states6.npz (needed before training)
python train6.py                         # train (~1.5 h on 12 cores)
```

## Files

- `box_env6.py`: the task. It builds on `box_env5.py`, `box_env4.py` and `box_env1.py`.
- `train6.py`, `watch4.py`, `check_rewards6.py`: train, watch, scripted check.
- `adibot.xml`, `box_scene_v5.xml`, `assets/`, `meshes/`: robot and box models.
- `tracklog.md`: every version, reward change and problem so far.
