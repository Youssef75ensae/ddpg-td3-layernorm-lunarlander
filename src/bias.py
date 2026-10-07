import gymnasium as gym
import numpy as np
import torch


# Select device: CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_bias(actor, critic, env_name, seed, n_states=100, n_trajectories=50, horizon=500, gamma=0.98):
    """Estimate overestimation bias by comparing Q-values to Monte Carlo returns."""
    env = gym.make(env_name)
    env.reset(seed=seed)

    # 1. Collect n_states states visited by the current policy
    states = []
    obs, _ = env.reset()
    for _ in range(n_states):
        states.append(obs.copy())
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = actor.act(obs_tensor).squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # 2. For each state, compute Q-value and MC return
    q_values = []
    mc_returns = []
    for s in states:
        with torch.no_grad():
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            a = actor.act(s_tensor).squeeze(0)
            q = critic(s_tensor, a.unsqueeze(0)).item()
        q_values.append(q)

        returns = []
        for _ in range(n_trajectories):
            env.reset()
            env.unwrapped.state = s
            total = 0.0
            disc = 1.0
            obs = s
            for _ in range(horizon):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    act = actor.act(obs_tensor).squeeze(0).cpu().numpy()
                obs, r, terminated, truncated, _ = env.step(act)
                total += disc * r
                disc *= gamma
                if terminated or truncated:
                    break
            returns.append(total)
        mc_returns.append(np.mean(returns))

    q_mean = np.mean(q_values)
    mc_mean = np.mean(mc_returns)
    bias = q_mean - mc_mean
    env.close()
    return {"q_mean": float(q_mean), "mc_mean": float(mc_mean), "bias": float(bias)}