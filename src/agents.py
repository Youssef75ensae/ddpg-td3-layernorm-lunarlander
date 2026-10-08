import copy

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rl_mind.env import VecEnv
from rl_mind.data import ReplayBuffer
from rl_mind.collectors import TransitionCollector
from rl_mind.evaluation import Evaluator
from rl_mind.nn import soft_update
from rl_mind.notebook import run_directory

from src.config import DDPGConfig, TD3Config
from src.networks import ContinuousQNetwork, ContinuousDeterministicActor
from src.exploration import GaussianNoise, CPUWrapper
from src.bias import measure_bias


# Select device: CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIAS_INTERVAL = 25_000


def compute_actor_loss(q_values):
    return -q_values.mean()


def log_bias(evaluator, metrics, step):
    for key in ("bias", "bias_valid", "bias_norm", "q_mean", "mc_mean", "frac_truncated"):
        evaluator.writer.add_scalar(f"bias/{key}", metrics[key], step)


def run_ddpg(cfg: DDPGConfig, run_name: str) -> Evaluator:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    actor = ContinuousDeterministicActor(
        env.observation_dim, cfg.actor_hidden, env.action_dim, cfg.use_layernorm
    ).to(device)
    critic = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.use_layernorm
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    collector = TransitionCollector(env, CPUWrapper(GaussianNoise(actor, cfg.action_noise)))
    buffer = ReplayBuffer(cfg.buffer_size)

    run_dir = run_directory(run_name)
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )
    bias_history = []
    next_bias_step = BIAS_INTERVAL

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        # Move batch tensors to device
        obs = batch.obs.to(device)
        actions = batch.action.value.to(device)
        next_obs = batch.next_obs.to(device)
        rewards = batch.reward.to(device)
        terminated = batch.terminated.to(device)

        q_values = critic(obs, actions)
        with torch.no_grad():
            next_actions = actor(next_obs).value
            next_q_values = target_critic(next_obs, next_actions)

        # Compute target on device
        target = rewards + cfg.gamma * (~terminated) * next_q_values
        critic_loss = F.mse_loss(q_values, target)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        actor_loss = compute_actor_loss(critic(obs, actor(obs).value))
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        soft_update(critic, target_critic, cfg.tau)

        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
        evaluator.writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)

        # Periodic bias measurement (steps can stall on resets, hence a threshold)
        if collector.steps >= next_bias_step:
            metrics = measure_bias(
                actor, critic, cfg.env_name, cfg.seed, gamma=cfg.gamma
            )
            bias_history.append({"step": collector.steps, **metrics})
            log_bias(evaluator, metrics, collector.steps)
            pbar.set_postfix(bias=f"{metrics['bias_valid']:.1f}")
            next_bias_step = (collector.steps // BIAS_INTERVAL + 1) * BIAS_INTERVAL

        if result := evaluator.run_if_needed(collector.steps, actor):
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    evaluator.writer.flush()
    evaluator.bias_history = bias_history
    return evaluator


def run_td3(cfg: TD3Config, run_name: str) -> Evaluator:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    actor = ContinuousDeterministicActor(
        env.observation_dim, cfg.actor_hidden, env.action_dim, cfg.use_layernorm
    ).to(device)
    critic_1 = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.use_layernorm
    ).to(device)
    critic_2 = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.use_layernorm
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    target_critic_1 = copy.deepcopy(critic_1).to(device)
    target_critic_2 = copy.deepcopy(critic_2).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_1_opt = torch.optim.Adam(critic_1.parameters(), lr=cfg.lr_critic)
    critic_2_opt = torch.optim.Adam(critic_2.parameters(), lr=cfg.lr_critic)

    collector = TransitionCollector(env, CPUWrapper(GaussianNoise(actor, cfg.action_noise)))
    buffer = ReplayBuffer(cfg.buffer_size)

    run_dir = run_directory(run_name)
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )
    bias_history = []
    next_bias_step = BIAS_INTERVAL

    updates = 0
    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        # Move batch tensors to device
        obs = batch.obs.to(device)
        actions = batch.action.value.to(device)
        next_obs = batch.next_obs.to(device)
        rewards = batch.reward.to(device)
        terminated = batch.terminated.to(device)

        q_1 = critic_1(obs, actions)
        q_2 = critic_2(obs, actions)

        with torch.no_grad():
            next_actions = target_actor(next_obs).value
            noise = (torch.randn_like(next_actions) * cfg.target_noise).clamp(
                -cfg.target_noise_clip, cfg.target_noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            next_q_1 = target_critic_1(next_obs, next_actions)
            next_q_2 = target_critic_2(next_obs, next_actions)
            next_q = torch.minimum(next_q_1, next_q_2)

        target = rewards + cfg.gamma * (~terminated) * next_q

        critic_1_loss = F.mse_loss(q_1, target)
        critic_1_opt.zero_grad()
        critic_1_loss.backward()
        critic_1_opt.step()

        critic_2_loss = F.mse_loss(q_2, target)
        critic_2_opt.zero_grad()
        critic_2_loss.backward()
        critic_2_opt.step()

        evaluator.writer.add_scalar("loss/critic_1", critic_1_loss.item(), collector.steps)
        evaluator.writer.add_scalar("loss/critic_2", critic_2_loss.item(), collector.steps)

        updates += 1

        if updates % cfg.policy_delay == 0:
            actor_loss = compute_actor_loss(critic_1(obs, actor(obs).value))
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            soft_update(actor, target_actor, cfg.tau)
            soft_update(critic_1, target_critic_1, cfg.tau)
            soft_update(critic_2, target_critic_2, cfg.tau)

            evaluator.writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)

        # Periodic bias measurement (critic_1)
        if collector.steps >= next_bias_step:
            metrics = measure_bias(
                actor, critic_1, cfg.env_name, cfg.seed, gamma=cfg.gamma
            )
            bias_history.append({"step": collector.steps, **metrics})
            log_bias(evaluator, metrics, collector.steps)
            pbar.set_postfix(bias=f"{metrics['bias_valid']:.1f}")
            next_bias_step = (collector.steps // BIAS_INTERVAL + 1) * BIAS_INTERVAL

        if result := evaluator.run_if_needed(collector.steps, actor):
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    evaluator.writer.flush()
    evaluator.bias_history = bias_history
    return evaluator