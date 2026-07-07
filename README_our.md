# Our MimicKit Setup Notes

Last updated: 2026-07-01

This file records the local setup needed to run the ADD + heightmap SMPL generalsit motion-tracking training in this checkout.

## Paths

- Repo: `/home/ubuntu/FAR/SGM/main_train/MimicKit`
- Isaac Gym env: `/home/ubuntu/.holosoma_deps/miniconda3/envs/hsgym`
- Isaac Gym package: `/home/ubuntu/.holosoma_deps/isaacgym`
- Recommended env setup script: `/home/ubuntu/FAR/holosoma/scripts/source_isaacgym_setup.sh`
- Download cache: `.cache/downloads/MimicKit_Data.zip`
- Generated training dataset: `data/datasets/dataset_smpl_bedlam2_clean_7177.yaml`
- Generated training motions: `data/motions/smpl_bedlam2_clean_7177/`
- Body-support inspection subset: `data/datasets/dataset_smpl_generalsit.yaml`

## Environment Setup

Use the existing Isaac Gym conda environment. The default shell was in `hssim`, but Isaac Gym is installed in `hsgym`.

```bash
source /home/ubuntu/FAR/holosoma/scripts/source_isaacgym_setup.sh
cd /home/ubuntu/FAR/SGM/main_train/MimicKit
```

The source script activates:

```text
/home/ubuntu/.holosoma_deps/miniconda3/envs/hsgym/bin/python
```

and appends the conda lib path to `LD_LIBRARY_PATH`, which is required for `libpython3.8.so.1.0`.

Verified imports:

```bash
python - <<'PY'
import isaacgym
import torch
print("isaacgym ok")
print(torch.__version__, torch.cuda.is_available())
PY
```

Expected local result:

```text
isaacgym ok
2.4.1+cu121 True
```

## Installed Python Dependencies

Installed MimicKit requirements into `hsgym`:

```bash
/home/ubuntu/.holosoma_deps/miniconda3/envs/hsgym/bin/pip install -r requirements.txt
```

This installed the missing packages needed by this repo, including:

```text
gymnasium
diffusers
moviepy
pyglet
tensorboardX
```

Torch was already present in `hsgym` and was not replaced:

```text
torch 2.4.1+cu121
```

## MimicKit Data Assets

Downloaded the official MimicKit data zip from the README SharePoint link. The public page embeds a temporary `.downloadUrl`; that URL was extracted and downloaded as:

```text
.cache/downloads/MimicKit_Data.zip
```

The zip was extracted and synced into `data/`:

```bash
unzip -q .cache/downloads/MimicKit_Data.zip -d .cache/downloads/MimicKit_Data_extracted
rsync -a --ignore-existing .cache/downloads/MimicKit_Data_extracted/MimicKit_Data/ data/
rm -rf .cache/downloads/MimicKit_Data_extracted
```

Important verified files:

```text
data/assets/smpl/smpl.xml
data/assets/smpl/smpl.usd
data/motions/smpl/smpl_walk.pkl
```

The download cache is ignored by git via `.gitignore`:

```text
.cache/
```

## BEDLAM2 Clean 7177 Dataset

The training input is the filtered BEDLAM2 clean subset, not only the 125 body-support inspection subset. The manifest has 7177 motions after applying:

```text
min_y_min >= -0.02 m
drop one additional worst candidate: nl_1213_3XL_2303.npz
```

Source manifest:

```text
/home/ubuntu/FAR/_sgm/bedlam2/motions_clean_ge_minus2cm_drop_worst1_manifest.csv
```

The 7177 z-up npz files were converted to MimicKit pkl format:

```bash
python tools/smpl_to_mimickit/build_generalsit_dataset.py \
  --manifest_csv /home/ubuntu/FAR/_sgm/bedlam2/motions_clean_ge_minus2cm_drop_worst1_manifest.csv \
  --manifest_input_column zup_source \
  --output_dir data/motions/smpl_bedlam2_clean_7177 \
  --dataset_file data/datasets/dataset_smpl_bedlam2_clean_7177.yaml \
  --input_coordinate_system z-up \
  --z_correction none \
  --loop clamp
```

Verification:

```text
motions: 7177
total frames: 2114142
frame dim: 75
```

The 125 body-support / ground-contact candidate set remains available for inspection:

```text
data/datasets/dataset_smpl_generalsit.yaml
data/motions/smpl_generalsit/
```

## Added Training Entry

The heightmap ADD environment appends a root-centered local heightmap observation to the normal ADD observation.

Main files:

```text
mimickit/envs/heightmap_add_env.py
data/envs/add_smpl_generalsit_heightmap_env.yaml
data/agents/add_smpl_generalsit_heightmap_agent.yaml
args/add_smpl_generalsit_heightmap_args.txt
```

The env config uses flat ground height by default:

```yaml
heightmap_obs: True
heightmap_num_rows: 15
heightmap_num_cols: 15
heightmap_ground_height: 0.0
heightmap_relative_to_root: True
```

Optional real heightmap observation can be enabled later with:

```yaml
heightmap_file: "data/heightmaps/example.npy"
heightmap_origin: [-5.0, -5.0]
heightmap_resolution: 0.05
```

Current collision ground is still Isaac Gym's flat plane. The heightmap is an observation channel, not terrain collision.

## Smoke Test

A small headless smoke test was run with 2 envs:

```bash
source /home/ubuntu/FAR/holosoma/scripts/source_isaacgym_setup.sh
cd /home/ubuntu/FAR/SGM/main_train/MimicKit

python mimickit/run.py \
  --arg_file args/add_smpl_generalsit_heightmap_args.txt \
  --mode train \
  --num_envs 2 \
  --visualize false \
  --max_samples 1 \
  --out_dir output/smoke_bedlam2_clean_7177_heightmap \
  --logger txt
```

Result:

```text
Isaac Gym GPU PhysX started.
Loaded 7177 motions with total length 70232.164s.
SMPL character built with 69 DoFs and mass 54.992 kg.
ADD agent built with 3,852,359 parameters.
One training iteration completed.
```

Smoke test log:

```text
output/smoke_bedlam2_clean_7177_heightmap/log.txt
```

## Full Training Command

Use:

```bash
source /home/ubuntu/FAR/holosoma/scripts/source_isaacgym_setup.sh
cd /home/ubuntu/FAR/SGM/main_train/MimicKit

python mimickit/run.py \
  --arg_file args/add_smpl_generalsit_heightmap_args.txt \
  --mode train \
  --visualize false
```

The arg file currently sets:

```text
--num_envs 4096
--engine_config data/engines/isaac_gym_engine.yaml
--env_config data/envs/add_smpl_generalsit_heightmap_env.yaml
--agent_config data/agents/add_smpl_generalsit_heightmap_agent.yaml
--out_dir output/add_smpl_bedlam2_clean_7177_heightmap/
```

For debugging, override `--num_envs` and `--max_samples` from the command line.

## ADD Tracking Policy Notes

This setup trains a SMPL motion-tracking policy with ADD on the filtered 7177-motion BEDLAM2 clean subset.

The tracking data path is controlled by:

```yaml
# data/envs/add_smpl_generalsit_heightmap_env.yaml
motion_file: "data/datasets/dataset_smpl_bedlam2_clean_7177.yaml"
```

The policy gets future target motion observations through:

```yaml
enable_tar_obs: True
tar_obs_steps: [1, 2, 3]
rand_reset: True
```

`rand_reset` samples random motions and times from the 7177-motion dataset, so the policy is trained as a generalist tracker instead of overfitting one clip.

The heightmap entry is:

```yaml
env_name: "heightmap_add"
heightmap_obs: True
heightmap_num_rows: 15
heightmap_num_cols: 15
heightmap_ground_height: 0.0
```

The current default heightmap is a flat z=0 observation. It is appended to the normal ADD observation. The Isaac Gym collision ground is still a flat plane.

The ADD agent uses differential discriminator reward by default:

```yaml
# data/agents/add_smpl_generalsit_heightmap_agent.yaml
agent_name: "ADD"
task_reward_weight: 0.0
disc_reward_weight: 1.0
```

This is the standard ADD-style objective in this codebase: the discriminator sees the difference between reference/demo motion features and policy motion features. The environment still computes the tracking reward terms and tracking diagnostics. To mix in explicit DeepMimic-style tracking reward, increase `task_reward_weight` and reduce or keep `disc_reward_weight` after a baseline ADD run.

Recommended long run:

```bash
source /home/ubuntu/FAR/holosoma/scripts/source_isaacgym_setup.sh
cd /home/ubuntu/FAR/SGM/main_train/MimicKit

python mimickit/run.py \
  --arg_file args/add_smpl_generalsit_heightmap_args.txt \
  --mode train \
  --visualize false \
  --logger txt \
  --save_int_models true
```

With the current arg/agent config, each iteration collects:

```text
4096 envs * 32 steps_per_iter = 131072 samples
```

`iters_per_output: 100` writes `model.pt`, `log.txt`, and optional `int_models/model_*.pt` every 100 iterations, which is about 13.1M samples. For a short debug run:

```bash
python mimickit/run.py \
  --arg_file args/add_smpl_generalsit_heightmap_args.txt \
  --mode train \
  --visualize false \
  --num_envs 512 \
  --max_samples 10000000 \
  --out_dir output/debug_add_smpl_bedlam2_clean_7177_heightmap \
  --logger txt \
  --save_int_models true
```

Test or visualize a checkpoint with:

```bash
python mimickit/run.py \
  --arg_file args/add_smpl_generalsit_heightmap_args.txt \
  --mode test \
  --num_envs 4 \
  --visualize true \
  --model_file output/add_smpl_bedlam2_clean_7177_heightmap/model.pt
```

## W&B and 8-GPU Training

W&B is logged in locally as entity `zihanw22`. The SGM project has been created here:

```text
https://wandb.ai/zihanw22/sgm
```

The W&B logger now reads these environment variables:

```bash
export WANDB_ENTITY=zihanw22
export WANDB_PROJECT=sgm
export WANDB_NAME=add-sgm-8gpu-debug-train
```

Do not use the built-in internal multi-device command for Isaac Gym on this machine:

```bash
python mimickit/run.py ... --devices cuda:0 cuda:1 ...
```

That path reproduced `GymPhysXCuda` illegal memory access during multi-process Isaac Gym tensor initialization. Use the external-rank launcher instead. It starts one OS process per GPU with `CUDA_VISIBLE_DEVICES=<rank>` and passes `--devices cuda:0 --proc_rank <rank> --num_procs <world_size>` into MimicKit. This preserves MimicKit's distributed rank/world-size logic and motion sharding while keeping each Isaac Gym process scoped to one visible GPU.

Launcher:

```text
tools/launch_add_sgm_8gpu_external.sh
```

Useful debug command:

```bash
OUT_DIR=output/debug_8gpu_add_sgm_external_20iter \
WANDB_NAME=debug-8gpu-external-20iter \
NUM_ENVS=64 \
MAX_SAMPLES=327680 \
MASTER_PORT=6132 \
SAVE_INT_MODELS=false \
tools/launch_add_sgm_8gpu_external.sh
```

Full 8-GPU debug training command:

```bash
OUT_DIR=output/add_sgm_8gpu_debug_train \
WANDB_NAME=add-sgm-8gpu-debug-train \
NUM_ENVS=512 \
MAX_SAMPLES=100000000 \
MASTER_PORT=6134 \
SAVE_INT_MODELS=true \
tools/launch_add_sgm_8gpu_external.sh
```

This uses:

```text
8 GPUs * 512 envs/GPU = 4096 envs total
4096 envs * 32 steps_per_iter = 131072 samples/iteration
```

Verified 8-GPU runs:

```text
debug-8gpu-external-smoke: 2048 samples, W&B run https://wandb.ai/zihanw22/sgm/runs/wjknq5zq
debug-8gpu-external-20iter: 327680 samples, W&B run https://wandb.ai/zihanw22/sgm/runs/tak4zjj9
debug-8gpu-external-512env-smoke: 131072 samples, W&B run https://wandb.ai/zihanw22/sgm/runs/0k1aqo7z
```

Current long debug training was started in tmux:

```bash
tmux attach -t sgm_add_8gpu_train
tail -f output/add_sgm_8gpu_debug_train/rank_0.log
```

Current W&B run:

```text
https://wandb.ai/zihanw22/sgm/runs/aqcs86g8
```

That run was capped at `MAX_SAMPLES=100000000` and completed normally at
`100,007,936` samples. It was useful as an 8-GPU stability/debug run, but it
used pure ADD reward:

```text
task_reward_weight: 0.0
disc_reward_weight: 1.0
```

For long tracking training, use the tracking-dominant ADD config:

```text
args/add_smpl_generalsit_heightmap_track_args.txt
data/agents/add_smpl_generalsit_heightmap_track_agent.yaml
```

The long run is warm-started from the 1e8-sample debug checkpoint and targets
about five days at the observed 8-GPU throughput:

```bash
tmux new -d -s sgm_add_8gpu_5day
tmux attach -t sgm_add_8gpu_5day

ARG_FILE=args/add_smpl_generalsit_heightmap_track_args.txt \
OUT_DIR=output/add_sgm_8gpu_track_5day_from_1e8 \
MODEL_FILE=output/add_sgm_8gpu_debug_train/model.pt \
WANDB_NAME=add-sgm-8gpu-track-5day-from-1e8 \
NUM_ENVS=512 \
MAX_SAMPLES=14000000000 \
MASTER_PORT=6136 \
SAVE_INT_MODELS=true \
tools/launch_add_sgm_8gpu_external.sh
```

At the previous measured speed of about `31.6k samples/s`, `14B` samples is
roughly `5.1` days. With `8 * 512` envs and `steps_per_iter=32`, that is about
`106,812` iterations.

## 2-Hour Training Monitor

A monitor script checks the long 8-GPU run every two hours:

```text
tools/monitor_add_sgm_training.py
```

It records checks in:

```text
output/add_sgm_monitor/monitor_events.log
output/add_sgm_monitor/state.json
```

Policy:

```text
healthy: no action
hard failure: restart from latest checkpoint
sample stagnation across two checks: restart from latest checkpoint
root/body tracking drift after >=250M samples: switch to strict tracking config
```

Strict tracking config:

```text
args/add_smpl_generalsit_heightmap_track_strict_args.txt
data/agents/add_smpl_generalsit_heightmap_track_strict_agent.yaml
```

The strict config keeps ADD but shifts the reward mix to:

```text
task_reward_weight: 0.85
disc_reward_weight: 0.15
```

Start or inspect the monitor:

```bash
tmux new -d -s sgm_add_8gpu_monitor \
  'cd /home/ubuntu/FAR/SGM/main_train/MimicKit && python tools/monitor_add_sgm_training.py --interval-sec 7200 >> output/add_sgm_monitor/monitor_daemon.log 2>&1'

tmux attach -t sgm_add_8gpu_monitor
tail -f output/add_sgm_monitor/monitor_events.log
```

As of 2026-07-02 03:42 UTC, the active monitored run is:

```text
output/add_sgm_8gpu_track_auto_restart_same_20260702_034101
https://wandb.ai/zihanw22/sgm/runs/ssy88bu2
```

It resumed from:

```text
output/add_sgm_8gpu_track_5day_from_1e8/int_models/model_0000002400.pt
```

The monitor state carries forward `314,703,872` samples from the previous run
and sets the remaining active run budget to `13,685,296,128` samples.

## Viser Policy Rollout Viewer

Use this to inspect the current tracking policy on the filtered 7177-motion
dataset. It starts a separate one-env IsaacGym rollout and does not touch the
8-GPU training job.

```bash
cd /home/ubuntu/FAR/SGM/main_train/MimicKit
source /home/ubuntu/FAR/holosoma/scripts/source_isaacgym_setup.sh

python tools/viser_policy_rollout.py \
  --model-file latest \
  --arg-file args/add_smpl_generalsit_heightmap_track_args.txt \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5050 \
  --max-steps 240
```

Open:

```text
http://127.0.0.1:5050
```

Useful smoke test:

```bash
python tools/viser_policy_rollout.py --dry-run --max-steps 2 --motion-index 0 --device cuda:0
```

Viewer notes:

```text
blue skeleton: policy rollout
orange skeleton: reference motion
red points: ground contacts from simulator contact forces
grid: z=0 ground plane
```

The GUI can filter/select motions, load a specific motion, randomize/reroll,
change rollout start time, change max rollout steps, scrub frames, and toggle
policy/reference/contacts/trajectories. This viewer shows policy rollout from
the simulator; it is not the standalone raw motion viewer. `--model-file latest`
resolves to the newest stable checkpoint under:

```text
output/add_sgm_8gpu_track_reward_fixed_20260706_222327
```

## 2026-07-06 Fixed Tracking Reward Restart

The previous long run:

```text
output/add_sgm_8gpu_track_auto_restart_same_20260702_034101
```

was stopped after visual rollout showed the policy losing root height and
collapsing even though the z-up reference motion was correct. The root cause was
not the z-up conversion or the Viser viewer. `heightmap_add` inherits ADD/AMP,
and AMP's `_update_reward()` intentionally leaves the task reward at zero. That
made `task_reward_weight` ineffective, so the policy mainly learned the ADD
discriminator signal and did not get a strong root/height tracking constraint.

Fixes applied:

```text
mimickit/envs/heightmap_add_env.py now calls DeepMimicEnv._update_reward()
data/envs/add_smpl_generalsit_heightmap_env.yaml pose_termination_dist: 0.5
tools/monitor_add_sgm_training.py flags root_pos_err > 0.45 as unhealthy
tools/viser_policy_rollout.py latest default points at the fixed run
```

The fixed strict run was launched from the old latest checkpoint:

```text
source checkpoint:
output/add_sgm_8gpu_track_auto_restart_same_20260702_034101/int_models/model_0000102000.pt

active fixed run:
output/add_sgm_8gpu_track_reward_fixed_20260706_222327

wandb:
https://wandb.ai/zihanw22/sgm/runs/6cfm1c92
```

Launch command:

```bash
cd /home/ubuntu/FAR/SGM/main_train/MimicKit
ARG_FILE=args/add_smpl_generalsit_heightmap_track_strict_args.txt \
OUT_DIR=output/add_sgm_8gpu_track_reward_fixed_20260706_222327 \
MODEL_FILE=output/add_sgm_8gpu_track_auto_restart_same_20260702_034101/int_models/model_0000102000.pt \
WANDB_NAME=add-sgm-8gpu-track-reward-fixed-20260706_222327 \
NUM_ENVS=512 \
MAX_SAMPLES=14000000000 \
MASTER_PORT=6877 \
SAVE_INT_MODELS=true \
tools/launch_add_sgm_8gpu_external.sh
```

Initial fixed-run sanity check:

```text
Train_Return/Test_Return are nonzero immediately.
Initial strict episode length is short because the old checkpoint fails root
tracking quickly; the key metric is whether episode length rises over time.
```

## 2026-07-07 Curriculum Tracking Restart

The strict fixed run was useful as a reward sanity check, but it was too harsh
as the next long training job from the old collapsed checkpoint:

```text
output/add_sgm_8gpu_track_reward_fixed_20260706_222327
latest checked checkpoint: int_models/model_0000002600.pt
iteration 2600: Test_Episode_Length ~= 8.9 steps, Root_Pos_Err ~= 0.16m
```

It improved only slowly because `pose_termination_dist: 0.5` terminates before
the policy has learned reliable root/support tracking. The active run is now a
root-support curriculum:

```text
active run:
output/add_sgm_8gpu_track_curriculum_20260707_012908

source checkpoint:
output/add_sgm_8gpu_track_reward_fixed_20260706_222327/int_models/model_0000002600.pt

wandb:
https://wandb.ai/zihanw22/sgm/runs/4ii12sx9
```

Launch command:

```bash
cd /home/ubuntu/FAR/SGM/main_train/MimicKit
ARG_FILE=args/add_smpl_generalsit_heightmap_track_curriculum_args.txt \
OUT_DIR=output/add_sgm_8gpu_track_curriculum_20260707_012908 \
MODEL_FILE=output/add_sgm_8gpu_track_reward_fixed_20260706_222327/int_models/model_0000002600.pt \
WANDB_NAME=add-sgm-8gpu-track-curriculum-20260707_012908 \
NUM_ENVS=512 \
MAX_SAMPLES=14000000000 \
MASTER_PORT=6887 \
SAVE_INT_MODELS=true \
tools/launch_add_sgm_8gpu_external.sh
```

Curriculum differences:

```text
pose_termination_dist: 0.8
task_reward_weight: 0.9
disc_reward_weight: 0.1
reward_root_pose_w: 0.35
reward_root_vel_w: 0.15
reward_key_pos_w: 0.15
all SMPL bodies are allowed as contact_bodies for ground/body-support motions
```

Initial curriculum check:

```text
Episode length is logged in control steps, not seconds.
control_freq=30Hz, so the 10s episode cap is about 300 steps.

Iteration 0
Test_Episode_Length: 14.61 steps (~0.49s)
Train_Episode_Length: 12.86 steps (~0.43s)
Root_Pos_Err: 0.326m
Body_Pos_Err: 0.071m

Iteration 200 / 26.3M samples
Test_Return: 10.86
Test_Episode_Length: 22.46 steps (~0.75s)
Train_Return: 8.97
Train_Episode_Length: 19.45 steps (~0.65s)
Root_Pos_Err: 0.441m
Body_Pos_Err: 0.076m
Root_Vel_Err: 0.386

Iteration 400 / 52.6M samples
Test_Return: 13.06
Test_Episode_Length: 28.17 steps (~0.94s)
Train_Return: 12.55
Train_Episode_Length: 29.75 steps (~0.99s)
Root_Pos_Err: 0.486m
Body_Pos_Err: 0.080m
Root_Vel_Err: 0.317
```

This is materially better than the strict restart at the same 26.3M samples
(`Test_Episode_Length` 22.46 steps vs 8.45 steps), but it is not yet a
finished policy. The next check should confirm that episode length continues
rising toward the 300-step cap while root position error falls or at least stays
bounded.

Z-up/root-height data check:

```text
dataset: data/datasets/dataset_smpl_bedlam2_clean_7177.yaml
motions: 7177
root z min quantiles:
  min=0.147m, p01=0.394m, p05=0.630m, median=0.867m
root z min < 0.3m: 13 motions
root z min < 0.05m: 0 motions
```

The current collapse/grounding issue is therefore not that the 7177 training
motions are globally y-up or pasted to z=0. The active failure mode to watch is
policy root/support tracking.

Monitor/viewer updates:

```text
tools/monitor_add_sgm_training.py profile: track_curriculum
tools/monitor_add_sgm_training.py startup grace: 600s before treating a new run
  with no log rows as dead
tools/monitor_add_sgm_training.py root-tight fallback:
  after >=750M samples, if episode length is >90 steps but Root_Pos_Err remains >0.55
  without improvement, switch from track_curriculum to track_root_tight
tools/monitor_add_sgm_training.py strict switch:
  only when Root_Pos_Err <0.30 and Test_Episode_Length >180 steps
tools/viser_policy_rollout.py --model-file latest now resolves under:
output/add_sgm_8gpu_track_curriculum_20260707_012908
```

Current Viser rollout:

```text
session: sgm_policy_rollout_5050
url: http://127.0.0.1:5050
checkpoint: output/add_sgm_8gpu_track_curriculum_20260707_012908/int_models/model_0000000400.pt

Initial motion index 0:
Frame 0 root z policy/ref: 0.950/0.950
Frame 0 body min z policy/ref: 0.598/0.598
Final done: FAIL after 13 frames (~0.40s)
Policy root z range: 0.188..0.950
Ref root z range: 0.949..0.950
```

The viewer confirms frame 0 is z-up and aligned, but the current policy still
collapses quickly. This is an early-checkpoint policy failure, not a raw motion
z-up failure.
