# DSRL
Implementation of DSRL algorithm. You should first run diffusion policy training algorithm in `examples/baselines/diffusion_policy` and save the checkpoint under `examples/baselines/dsrl/diffusion_policy` to run dsrl code.

## How to run the code
I noticed the algorithm is sensitive to hyperparameters so if you change the task, please tune the parameter to suite your setting.
```
# PickCube-v1
python dsrl.py --env_id="PickCube-v1" \
  --num_envs=32 --buffer_size=500_000 \
  --total_timesteps=2_000_000 --eval_freq=50_000 \
  --control-mode="pd_ee_delta_pos" --bootstrap_at_done=never \
  --reward_bias -1.0 --noise_action_scale=1.5 --clip_entropy \
  --utd=0.2 --clip_grad --track

# PegInsertionSide-v1
CUDA_VISIBLE_DEVICES=4 python dsrl.py --env_id="PegInsertionSide-v1" --max_episode_steps=200 --target_entropy=-112 --track
```