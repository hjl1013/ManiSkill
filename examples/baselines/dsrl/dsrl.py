import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional, List
import tyro
import os
import time
import random
import numpy as np
from collections import defaultdict
import tqdm
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium import spaces
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack, FlattenActionSpaceWrapper, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Constants for log std bounds (same as SAC)
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ActionChunkWrapper(gym.Wrapper):
    def __init__(self, env, act_steps, max_episode_steps=None):
        super().__init__(env)
        self.act_steps = act_steps
        self.max_episode_steps = max_episode_steps or getattr(env, '_max_episode_steps', 300)
        
        # Modify action space to accept action chunks
        self.action_space = env.action_space
        
        # Observation space remains the same
        self.observation_space = env.observation_space
        
        # Handle vectorized environments
        self.is_vector_env = hasattr(env, 'num_envs')
        if self.is_vector_env:
            self.num_envs = env.num_envs
        else:
            self.num_envs = 1
    
    def step(self, action):
        # Reshape action to (num_envs, act_steps, action_dim)
        if len(action.shape) <= 2:
            action = action.reshape(self.num_envs, self.act_steps, -1)
        
        obs_list = []
        reward_list = []
        final_terminated = False
        final_truncated = False
        final_info = dict()
        
        for i in range(self.act_steps):
            obs_i, reward_i, terminated_i, truncated_i, info_i = self.env.step(action[:, i])
            
            obs_list.append(obs_i)
            reward_list.append(reward_i)
            final_terminated = final_terminated | terminated_i
            final_truncated = final_truncated | truncated_i
            final_info.update(info_i)
        
        # Return final observation, sum of rewards, and final termination status
        final_obs = obs_list[-1]
        total_reward = sum(reward_list)

        return final_obs, total_reward, final_terminated, final_truncated, final_info

    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0, reward_bias=0.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward * self.reward_scale + self.reward_bias
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "DSRL"
    """the group of the run for wandb"""
    
    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    
    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    
    # Diffusion Policy specific arguments
    obs_horizon: int = 2
    """observation horizon"""
    act_horizon: int = 8
    """action horizon for execution"""
    pred_horizon: int = 16
    """prediction horizon for diffusion model"""
    diffusion_step_embed_dim: int = 64
    """diffusion step embedding dimension"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net dimensions"""
    n_groups: int = 8
    """number of groups for group normalization"""
    
    # DSRL specific arguments
    pretrained_diffusion_path: Optional[str] = "/home/hyunjun/projects/p-self-improvement/ManiSkill/examples/baselines/dsrl/diffusion_policy/diffusion_policy.pt"
    """path to pretrained diffusion policy model"""
    noise_action_scale: float = 1
    """scale for noise action"""
    noise_actor_lr: float = 3e-4
    """learning rate for noise actor"""
    critic_lr: float = 3e-4
    """learning rate for critics"""
    noise_critic_lr: float = 3e-4
    """learning rate for noise critic"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """target network update rate"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""
    
    # Training specific arguments
    learning_starts: int = 4_000
    """timestep to start learning"""
    training_freq: int = 512
    """training frequency (in steps)"""
    utd: float = 0.05
    """update to data ratio"""
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""
    noise_critic_frequency: int = 2
    """the frequency of training noise critic"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
    # Environment specific arguments
    max_episode_steps: int = 100
    """the number of steps to run in each evaluation environment during evaluation"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    reward_mode: str = "sparse"
    """the reward mode to use for the environment"""
    reward_scale: float = 1.0
    """the scale of the reward"""
    reward_bias: float = 0.0
    """the bias of the reward"""
    no_entropy: bool = False
    """if toggled, do not use entropy regularization"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

class ReplayBuffer:
    def __init__(self, env, args, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        
        # Store observations (already stacked by FrameStack wrapper)
        # env.single_observation_space.shape is (obs_horizon, obs_dim)
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        
        # Store action chunks - now the environment handles chunking, so we store the full chunk
        # env.single_action_space.shape is (act_horizon * act_dim,) after ActionChunkWrapper
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs, env.single_action_space.shape[0] * args.act_horizon)).to(storage_device)
        
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
            
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),  # (B, act_horizon * act_dim)
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )


class DiffusionPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        
        assert len(env.single_observation_space.shape) == 2  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        
        self.act_dim = env.single_action_space.shape[0]
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=np.prod(env.single_observation_space.shape),  # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        
        self.num_diffusion_iters = 10
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        
        # Freeze all parameters since this is pretrained
        for param in self.parameters():
            param.requires_grad = False
    
    def get_action(self, obs_seq, initial_noise=None):
        """
        Get action from diffusion policy, optionally starting from provided initial noise
        obs_seq: (B, obs_horizon, obs_dim)
        initial_noise: (B, pred_horizon, act_dim) - if None, sample from Gaussian
        """
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
            
            # Initialize action from provided noise or Gaussian noise
            if initial_noise is not None:
                noisy_action_seq = initial_noise.clone()
            else:
                noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)
            
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                
                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(  # type: ignore
                    model_output=noise_pred,
                    timestep=k,  # type: ignore
                    sample=noisy_action_seq,
                ).prev_sample  # type: ignore
        
        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

class NoiseActor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.pred_horizon = args.pred_horizon
        self.act_dim = env.single_action_space.shape[0]
        
        # Input: observation sequence (obs_horizon * obs_dim)
        # Output: initial noise for diffusion (pred_horizon * act_dim)
        obs_dim = np.prod(env.single_observation_space.shape)
        noise_dim = self.pred_horizon * self.act_dim
        
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.fc_mean = nn.Linear(256, noise_dim)
        self.fc_logstd = nn.Linear(256, noise_dim)
        
        # Action rescaling for tanh-wrapped distribution
        # Since this is for noise generation, we'll use a standard range [-1, 1] scaled
        self.register_buffer("action_scale", torch.ones(noise_dim, dtype=torch.float32) * args.noise_action_scale)
        self.register_buffer("action_bias", torch.zeros(noise_dim, dtype=torch.float32))
        
        # Initialize with small values to start close to Gaussian noise
        nn.init.xavier_uniform_(self.fc_mean.weight, gain=0.01)
        nn.init.constant_(self.fc_mean.bias, 0)
        nn.init.xavier_uniform_(self.fc_logstd.weight, gain=0.01)
        nn.init.constant_(self.fc_logstd.bias, -1)  # Start with small std
    
    def forward(self, obs_seq):
        """
        obs_seq: (B, obs_horizon, obs_dim)
        returns: mean, log_std both (B, pred_horizon * act_dim)
        """
        x = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std
    
    def get_action(self, obs_seq):
        """
        Sample noise from the actor with tanh-wrapped distribution
        obs_seq: (B, obs_horizon, obs_dim)
        returns: noise (B, pred_horizon, act_dim), log_prob (B, 1)
        """
        mean, log_std = self(obs_seq)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        
        # Apply action scaling and bias
        noise_flat = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (tanh correction)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Reshape to (B, pred_horizon, act_dim)
        noise = noise_flat.view(-1, self.pred_horizon, self.act_dim)
        
        return noise, log_prob
    
    def get_eval_action(self, obs_seq):
        """
        Get deterministic noise (mean) for evaluation with tanh transformation
        obs_seq: (B, obs_horizon, obs_dim)
        returns: noise (B, pred_horizon, act_dim)
        """
        mean, _ = self(obs_seq)
        # Apply tanh transformation and scaling for deterministic evaluation
        noise_flat = torch.tanh(mean) * self.action_scale + self.action_bias
        noise = noise_flat.view(-1, self.pred_horizon, self.act_dim)
        return noise

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.action_scale = self.action_scale.to(device)
            self.action_bias = self.action_bias.to(device)
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

class NoiseCritic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.pred_horizon = args.pred_horizon
        self.act_dim = env.single_action_space.shape[0]
        
        # Input: observation sequence (obs_horizon * obs_dim) + noise (pred_horizon * act_dim)
        obs_dim = np.prod(env.single_observation_space.shape)
        noise_dim = self.pred_horizon * self.act_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, obs_seq, noise):
        """
        obs_seq: (B, obs_horizon, obs_dim)
        noise: (B, pred_horizon, act_dim)
        returns: Q-value (B, 1)
        """
        obs_flat = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        noise_flat = noise.flatten(start_dim=1)  # (B, pred_horizon * act_dim)
        x = torch.cat([obs_flat, noise_flat], dim=1)
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.action_dim = env.single_action_space.shape[0]
        
        # Input: observation sequence (obs_horizon * obs_dim) + flattened action chunk
        obs_dim = np.prod(env.single_observation_space.shape)
        action_chunck_dim = self.act_horizon * self.action_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_chunck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, obs_seq, action_chunk):
        """
        obs_seq: (B, obs_horizon, obs_dim)
        action_chunk: (B, act_horizon * act_dim) - flattened action chunk
        returns: Q-value (B, 1)
        """
        obs_flat = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        x = torch.cat([obs_flat, action_chunk], dim=1)
        return self.net(x)

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        
        # Initialize all components
        self.diffusion_policy = DiffusionPolicy(env, args)
        self.noise_actor = NoiseActor(env, args)
        self.noise_critic1 = NoiseCritic(env, args)
        self.noise_critic2 = NoiseCritic(env, args)
        self.critic1 = Critic(env, args)
        self.critic2 = Critic(env, args)
        
        # Target networks for critics
        self.critic1_target = Critic(env, args)
        self.critic2_target = Critic(env, args)
        
        # Initialize target networks with same weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Load pretrained diffusion policy if path is provided
        if args.pretrained_diffusion_path is not None:
            self.load_diffusion_policy(args.pretrained_diffusion_path)
    
    def load_diffusion_policy(self, path):
        """Load pretrained diffusion policy weights"""
        checkpoint = torch.load(path, map_location='cpu')
        if 'ema_agent' in checkpoint:
            self.diffusion_policy.load_state_dict(checkpoint['ema_agent'])
        elif 'agent' in checkpoint:
            self.diffusion_policy.load_state_dict(checkpoint['agent'])
        else:
            self.diffusion_policy.load_state_dict(checkpoint)
        print(f"Loaded pretrained diffusion policy from {path}")
    
    def get_action(self, obs_seq, deterministic=False):
        """
        Get action for environment interaction
        obs_seq: (B, obs_horizon, obs_dim)
        returns: action_chunk (B, act_horizon * act_dim) - flattened for ActionChunkWrapper
        """
        if deterministic:
            # Use deterministic noise for evaluation
            noise = self.noise_actor.get_eval_action(obs_seq)
        else:
            # Sample noise from actor
            noise, _ = self.noise_actor.get_action(obs_seq)

        # Use diffusion policy to generate action chunk from noise
        action_chunk = self.diffusion_policy.get_action(obs_seq, initial_noise=noise)  # (B, act_horizon, act_dim)
        
        # Flatten action chunk for ActionChunkWrapper
        return action_chunk.flatten(start_dim=1)  # (B, act_horizon * act_dim)
    
    def get_action_and_log_prob(self, obs_seq):
        """
        Get action and log probability for training
        obs_seq: (B, obs_horizon, obs_dim)
        returns: action_chunk (B, act_horizon * act_dim), noise_log_prob (B, 1), noise
        """
        # Sample noise from actor with log probability
        noise, noise_log_prob = self.noise_actor.get_action(obs_seq)
        
        # Use diffusion policy to generate action chunk from noise
        action_chunk = self.diffusion_policy.get_action(obs_seq, initial_noise=noise)  # (B, act_horizon, act_dim)
        
        # Flatten action chunk for consistency with replay buffer
        action_chunk_flat = action_chunk.flatten(start_dim=1)  # (B, act_horizon * act_dim)
        
        return action_chunk_flat, noise_log_prob, noise
    
    def update_target_networks(self, tau):
        """Update target networks with soft update"""
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: Optional[SummaryWriter] = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
        
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    print("Initializing the environment...")
    env_kwargs = dict(
        obs_mode="state", 
        render_mode="rgb_array", 
        sim_backend="gpu",
        max_episode_steps=args.max_episode_steps,
        reconfiguration_freq=1,
        reward_mode=args.reward_mode,
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    
    # Create training environments
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    # envs = ActionChunkWrapper(envs, act_steps=args.act_horizon)  # Wrap with action chunking
    envs = FrameStack(envs, num_stack=args.obs_horizon)  # Stack observations for obs_horizon
    envs = RewardWrapper(envs, reward_scale=args.reward_scale, reward_bias=args.reward_bias)
    envs = ManiSkillVectorEnv(envs, ignore_terminations=True, record_metrics=True)
    
    # Create evaluation environments
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = FrameStack(eval_envs, num_stack=args.obs_horizon)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos" if args.checkpoint else "test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, 
                                save_video=args.capture_video, trajectory_name="trajectory", 
                                max_steps_per_video=args.max_episode_steps, video_fps=30)
    eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=True, record_metrics=True)  # type: ignore
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Initialize logger
    from mani_skill.utils import gym_utils
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", 
                                    num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", 
                                    env_horizon=max_episode_steps)
            config["eval_env_cfg"] = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", 
                                         num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", 
                                         env_horizon=max_episode_steps)
            if args.control_mode is not None:
                config["env_cfg"]["control_mode"] = args.control_mode
                config["eval_env_cfg"]["control_mode"] = args.control_mode
                
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["dsrl", "diffusion_policy", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # Initialize the agent
    print("Initializing the agent...")
    agent = Agent(envs, args).to(device)
    
    # Load checkpoint if provided
    ckpt = None
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        agent.noise_actor.load_state_dict(ckpt['noise_actor'])
        agent.critic1.load_state_dict(ckpt['critic1'])
        agent.critic2.load_state_dict(ckpt['critic2'])
        agent.noise_critic1.load_state_dict(ckpt['noise_critic1'])
        agent.noise_critic2.load_state_dict(ckpt['noise_critic2'])
        # Update target networks
        agent.critic1_target.load_state_dict(agent.critic1.state_dict())
        agent.critic2_target.load_state_dict(agent.critic2.state_dict())
    
    # Initialize optimizers
    noise_actor_optimizer = optim.Adam(agent.noise_actor.parameters(), lr=args.noise_actor_lr)
    critic_optimizer = optim.Adam(list(agent.critic1.parameters()) + list(agent.critic2.parameters()), lr=args.critic_lr)
    noise_critic_optimizer = optim.Adam(list(agent.noise_critic1.parameters()) + list(agent.noise_critic2.parameters()), lr=args.noise_critic_lr)
    
    # Automatic entropy tuning
    if args.autotune:
        # Target entropy for noise space (pred_horizon * act_dim)
        target_entropy = -torch.prod(torch.Tensor([args.pred_horizon * envs.single_action_space.shape[0]]).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # Load log_alpha from checkpoint if available
        if args.checkpoint is not None and ckpt is not None and 'log_alpha' in ckpt and ckpt['log_alpha'] is not None:
            log_alpha.data.copy_(ckpt['log_alpha'])
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.noise_actor_lr)
    else:
        alpha = args.alpha
        log_alpha = None
        alpha_optimizer = None
        target_entropy = None

    # Initialize the replay buffer
    print("Initializing the replay buffer...")
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
        args=args
    )

    # Training loop
    print("Starting training...")
    obs, info = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * args.steps_per_env
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    # history for rollouts
    action_queue = []
    action_history = []
    obs_history = []
    reward_history = []
    done_history = []
    info_history = []

    while global_step < args.total_timesteps:
        # Evaluation
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            print("Evaluating...")
            agent.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            
            eval_action_queue = []
            for _ in range(args.max_episode_steps):
                with torch.no_grad():
                    # Generate action chunk directly from agent
                    if len(eval_action_queue) == 0:
                        action_chunk = agent.get_action(eval_obs, deterministic=True)
                        action_chunk = action_chunk.reshape(args.num_eval_envs, args.act_horizon, -1)
                        eval_action_queue = [action_chunk[:, i] for i in range(args.act_horizon)]
                    actions = eval_action_queue.pop(0)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actions)
                    
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                if v:  # Check if list is not empty
                    mean = torch.stack(v).float().mean()
                    eval_metrics_mean[k] = mean
                    if logger is not None:
                        logger.add_scalar(f"eval/{k}", mean, global_step)
            
            if eval_metrics_mean:
                pbar.set_description(
                    f"success_once: {eval_metrics_mean.get('success_once', 0):.2f}, "
                    f"return: {eval_metrics_mean.get('return', 0):.2f}"
                )
            
            eval_time = time.perf_counter() - stime
            cumulative_times["eval_time"] += eval_time
            if logger is not None:
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            agent.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'noise_actor': agent.noise_actor.state_dict(),
                    'critic1': agent.critic1_target.state_dict(),
                    'critic2': agent.critic2_target.state_dict(),
                    'noise_critic1': agent.noise_critic1.state_dict(),
                    'noise_critic2': agent.noise_critic2.state_dict(),
                    'log_alpha': log_alpha if args.autotune else None,
                }, model_path)
                print(f"Model saved to {model_path}")

        # Collect samples from environments
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += args.num_envs # one environment takes act_horizon steps

            # Generate action chunk from agent
            if len(action_queue) == 0:
                if not learning_has_started:
                    noise_action_space = spaces.Box(
                        low=np.tile(np.expand_dims(envs.action_space.low, axis=1), (1, args.pred_horizon, 1)) * args.noise_action_scale,
                        high=np.tile(np.expand_dims(envs.action_space.high, axis=1), (1, args.pred_horizon, 1)) * args.noise_action_scale,
                        dtype=envs.action_space.dtype
                    )
                    noise = torch.tensor(noise_action_space.sample(), dtype=torch.float32, device=device)
                    action_chunk = agent.diffusion_policy.get_action(obs, initial_noise=noise)
                else:
                    action_chunk = agent.get_action(obs, deterministic=False)
                    action_chunk = action_chunk.reshape(args.num_envs, args.act_horizon, -1)
                action_queue = [action_chunk[:, i] for i in range(args.act_horizon)]
            actions = action_queue.pop(0)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # update histories
            if len(action_history) == args.act_horizon:
                action_history.pop(0)
                obs_history.pop(0)
                reward_history.pop(0)
                done_history.pop(0)
                info_history.pop(0)
            action_history.append(actions)
            obs_history.append(obs)
            reward_history.append(rewards)
            done_history.append(terminations)
            info_history.append(infos)

            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
            if "final_info" in infos: # this means environment resetted
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            if len(action_history) == args.act_horizon:
                real_action_chunk = torch.stack(action_history, dim=1).flatten(start_dim=1).clone()
                real_obs = obs_history[0].clone()
                real_rewards = torch.stack(reward_history, dim=1).sum(dim=1).clone()
                rb.add(real_obs, real_next_obs, real_action_chunk, real_rewards, stop_bootstrap)

            # reset histories
            if "final_info" in infos:
                action_queue = []
                action_history = []
                obs_history = []
                reward_history = []
                done_history = []
                info_history = []

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # Training
        if global_step < args.learning_starts or args.evaluate:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        
        # Initialize loss variables
        critic1_loss = torch.tensor(0.0)
        critic2_loss = torch.tensor(0.0) 
        critic_loss = torch.tensor(0.0)
        noise_critic1_loss = torch.tensor(0.0)
        noise_critic2_loss = torch.tensor(0.0)
        noise_critic_loss = torch.tensor(0.0)
        noise_actor_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)
        mean_log_probs = torch.tensor(0.0)
        
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # Train Critics (SAC-style)
            with torch.no_grad():
                # Sample next actions using current noise actor
                next_action_chunks, next_log_probs, next_noise = agent.get_action_and_log_prob(data.next_obs)
                
                # Compute target Q values
                target_q1 = agent.critic1_target(data.next_obs, next_action_chunks)
                target_q2 = agent.critic2_target(data.next_obs, next_action_chunks)
                if args.no_entropy:
                    min_target_q = torch.min(target_q1, target_q2)
                else:
                    min_target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
                target_q_value = data.rewards.unsqueeze(-1) + (1 - data.dones.unsqueeze(-1)) * args.gamma * min_target_q

            # Update Critics
            current_q1 = agent.critic1(data.obs, data.actions)
            current_q2 = agent.critic2(data.obs, data.actions)
            critic1_loss = F.mse_loss(current_q1, target_q_value)
            critic2_loss = F.mse_loss(current_q2, target_q_value)
            critic_loss = critic1_loss + critic2_loss

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Train Noise Critics (mapping between noise and action spaces)
            if global_update % args.noise_critic_frequency == 0:
                with torch.no_grad():
                    # Sample noise from Gaussian for current observations
                    original_act_dim = envs.single_action_space.shape[0]  # Get original action dim
                    gaussian_noise = torch.randn((data.obs.shape[0], args.pred_horizon, original_act_dim), device=device)
                    # Get action chunks from diffusion policy using this noise
                    diffusion_actions_unflat = agent.diffusion_policy.get_action(data.obs, initial_noise=gaussian_noise)  # (B, act_horizon, act_dim)
                    diffusion_actions = diffusion_actions_unflat.flatten(start_dim=1)  # Flatten for critic
                    # Get critic values for these actions
                    target_critic1_value = agent.critic1(data.obs, diffusion_actions)
                    target_critic2_value = agent.critic2(data.obs, diffusion_actions)

                # Predict noise critic values
                noise_critic1_pred = agent.noise_critic1(data.obs, gaussian_noise)
                noise_critic2_pred = agent.noise_critic2(data.obs, gaussian_noise)
                
                # Loss: MSE between noise critic predictions and actual critic values
                noise_critic1_loss = F.mse_loss(noise_critic1_pred, target_critic1_value)
                noise_critic2_loss = F.mse_loss(noise_critic2_pred, target_critic2_value)
                noise_critic_loss = noise_critic1_loss + noise_critic2_loss

                noise_critic_optimizer.zero_grad()
                noise_critic_loss.backward()
                noise_critic_optimizer.step()

            # Train Noise Actor (SAC-style using Noise Critics)
            if global_update % args.policy_frequency == 0:
                # Sample noise from actor
                sampled_noise, log_probs = agent.noise_actor.get_action(data.obs)
                
                # Get Q values from noise critics
                q1_noise = agent.noise_critic1(data.obs, sampled_noise)
                q2_noise = agent.noise_critic2(data.obs, sampled_noise)
                min_q_noise = torch.min(q1_noise, q2_noise)
                
                # Actor loss: maximize Q value, minimize entropy
                if args.no_entropy:
                    noise_actor_loss = -min_q_noise.mean()
                else:
                    noise_actor_loss = (alpha * log_probs - min_q_noise).mean()

                noise_actor_optimizer.zero_grad()
                noise_actor_loss.backward()
                noise_actor_optimizer.step()

                # Update entropy coefficient
                if args.autotune and log_alpha is not None and alpha_optimizer is not None and target_entropy is not None and not args.no_entropy:
                    with torch.no_grad():
                        _, log_probs_detached = agent.noise_actor.get_action(data.obs)
                    alpha_loss = (-log_alpha.exp() * (log_probs_detached + target_entropy)).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp().item()
                    mean_log_probs = log_probs_detached.mean().item()

            # Update target networks
            if global_update % args.target_network_frequency == 0:
                agent.update_target_networks(args.tau)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training metrics
        if logger is not None and (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/critic1_loss", critic1_loss.item(), global_step)
            logger.add_scalar("losses/critic2_loss", critic2_loss.item(), global_step)
            logger.add_scalar("losses/critic_loss", critic_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/noise_critic1_loss", noise_critic1_loss.item(), global_step)
            logger.add_scalar("losses/noise_critic2_loss", noise_critic2_loss.item(), global_step)
            logger.add_scalar("losses/noise_critic_loss", noise_critic_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/noise_actor_loss", noise_actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("losses/mean_log_probs", mean_log_probs, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'noise_actor': agent.noise_actor.state_dict(),
            'critic1': agent.critic1_target.state_dict(),
            'critic2': agent.critic2_target.state_dict(),
            'noise_critic1': agent.noise_critic1.state_dict(),
            'noise_critic2': agent.noise_critic2.state_dict(),
            'log_alpha': log_alpha if args.autotune else None,
        }, model_path)
        print(f"Final model saved to {model_path}")

    envs.close()
    eval_envs.close()
    if logger is not None:
        logger.close()