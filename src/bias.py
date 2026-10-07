def measure_bias(actor, critic, env_name, seed, n_episodes=10, gamma=0.98):
    """Estimate overestimation bias using actual trajectories.

    Runs n_episodes with the deterministic policy. At each step t, records
    Q(s_t, a_t) and computes the Monte Carlo return G_t from t onwards.
    Bias = mean(Q) - mean(G).
    """
    env = gym.make(env_name)
    q_values = []
    mc_returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        trajectory = []
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated) and steps < 1000:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                a_tensor = actor.act(obs_tensor).squeeze(0)
                q = critic(obs_tensor, a_tensor.unsqueeze(0)).item()
            a_np = a_tensor.cpu().numpy()
            next_obs, r, terminated, truncated, _ = env.step(a_np)
            trajectory.append((q, r))
            obs = next_obs
            steps += 1

        # Compute discounted returns backward
        G = 0.0
        for q, r in reversed(trajectory):
            G = r + gamma * G
            q_values.append(q)
            mc_returns.append(G)

    env.close()
    q_mean = float(np.mean(q_values))
    mc_mean = float(np.mean(mc_returns))
    return {"q_mean": q_mean, "mc_mean": mc_mean, "bias": q_mean - mc_mean}