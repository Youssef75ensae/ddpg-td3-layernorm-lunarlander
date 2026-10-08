import math

import gymnasium as gym
import numpy as np
import torch


def measure_bias(actor, critic, env_name, seed, n_episodes=10, gamma=0.98, trunc_tol=0.02):
    """Overestimation bias: mean Q(s, pi(s)) minus the Monte-Carlo return."""
    device = next(critic.parameters()).device
    margin = math.ceil(math.log(trunc_tol) / math.log(gamma))

    env = gym.make(env_name)
    max_len = env.spec.max_episode_steps or 1000
    q_values, mc_returns, valid = [], [], []
    n_truncated = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        trajectory = []
        terminated, truncated = False, False

        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_tensor = actor.act(obs_tensor)
                q = critic(obs_tensor, a_tensor).item()
            obs, r, terminated, truncated, _ = env.step(a_tensor.squeeze(0).cpu().numpy())
            trajectory.append((q, float(r)))
            truncated = truncated or len(trajectory) >= max_len  # safety net

        cut = truncated and not terminated
        n_truncated += int(cut)

        # Discounted returns, computed backwards
        T = len(trajectory)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][1] + gamma * G
            returns[t] = G

        for t, (q, _) in enumerate(trajectory):
            q_values.append(q)
            mc_returns.append(returns[t])
            valid.append(not cut or T - t >= margin)

    env.close()

    q_all = np.asarray(q_values)
    g_all = np.asarray(mc_returns)
    v = np.asarray(valid)

    q_mean = float(q_all.mean())
    mc_mean = float(g_all.mean())
    if v.any():
        bias_valid = float(np.mean(q_all[v] - g_all[v]))
        bias_norm = bias_valid / (float(np.mean(np.abs(g_all[v]))) + 1e-8)
    else:
        bias_valid = bias_norm = float("nan")

    return {
        "q_mean": q_mean,
        "mc_mean": mc_mean,
        "bias": q_mean - mc_mean,
        "bias_valid": bias_valid,
        "bias_norm": bias_norm,
        "frac_truncated": n_truncated / n_episodes,
    }