import gymnasium as gym
import torch

from collections import defaultdict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from dataclasses import dataclass
import tyro
import numpy as np
import random

from diffusion_policy.make_env import make_eval_envs
from train import Agent
from train import Args
from diffusion_policy.evaluate import evaluate

def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs=dict(obs_horizon=args.obs_horizon),
        video_dir=f'runs/{args.exp_name}/eval_videos' if args.capture_video else None
    )

    obs, _ = eval_envs.reset(seed=args.seed)
    eval_metrics = defaultdict(list)
    agent = Agent(eval_envs, args).to(device)
    agent.load_state_dict(torch.load(f'runs/{args.exp_name}/checkpoints/best_eval_success_once.pt')['ema_agent'])

    eval_metrics = evaluate(
        args.num_eval_episodes,
        agent, 
        eval_envs, 
        device, 
        args.sim_backend,
        progress_bar=True,
    )
    for k in eval_metrics.keys():
        print(f"{k}_mean: {np.mean(eval_metrics[k])}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)