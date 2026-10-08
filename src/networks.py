import torch
import torch.nn as nn
from torch import Tensor

from rl_mind.core import Action, Actor


def build_mlp_with_ln(
    sizes: list[int],
    use_layernorm: bool = False,
    output_activation: nn.Module | None = None,
) -> nn.Module:
    """Build an MLP, optionally inserting a LayerNorm after each hidden layer."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_hidden = i < len(sizes) - 2
        if is_hidden:
            if use_layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ContinuousQNetwork(nn.Module):
    """Q(s, a) for continuous actions, with an optional LayerNorm."""

    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.model = build_mlp_with_ln(
            [obs_dim + action_dim, *hidden, 1],
            use_layernorm=use_layernorm,
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


class ContinuousDeterministicActor(Actor[Action]):
    """Deterministic actor for continuous actions, with an optional LayerNorm."""

    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.model = build_mlp_with_ln(
            [obs_dim, *hidden, action_dim],
            use_layernorm=use_layernorm,
            output_activation=nn.Tanh(),
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, obs: Tensor) -> Action:
        obs = obs.to(self.device)
        return Action(value=self.model(obs))

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        obs = obs.to(self.device)
        return self.model(obs)