from dataclasses import dataclass

@dataclass(frozen=True)
class DDPGConfig:
    env_name: str = "LunarLanderContinuous-v3"
    seed: int = 0

    max_steps: int = 300_000
    n_envs: int = 1
    steps_per_update: int = 1
    learning_starts: int = 5_000

    buffer_size: int = 200_000
    batch_size: int = 64

    gamma: float = 0.98
    tau: float = 0.005
    action_noise: float = 0.1

    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64)
    use_layernorm: bool = False
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3

    eval_interval: int = 5_000
    n_eval_envs: int = 10


@dataclass(frozen=True)
class TD3Config(DDPGConfig):
    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5