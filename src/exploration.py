import torch
from torch import Tensor

from rl_mind.core import Action, Actor


class GaussianNoise(Actor[Action]):
    """Adds Gaussian noise to the actions of another actor (training only)."""

    def __init__(self, actor: Actor[Action], sigma: float):
        super().__init__()
        self.actor = actor
        self.sigma = sigma

    def forward(self, obs: Tensor) -> Action:
        action = self.actor(obs).value
        return Action(value=action + self.sigma * torch.randn_like(action))

    def act(self, obs: Tensor) -> Tensor:
        return self.actor.act(obs)