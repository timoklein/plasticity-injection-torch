import torch
import torch.nn as nn

from .utils import _freeze_module


class PlasticityInjectionHead(nn.Module):
    """Network module for applying plasticity injection to the head of a DQN agent.

    Plasticity injection algorithm:
        1. Freeze the original layer.
        2. Initialize a fresh layer and a frozen copy of it.
        3. Forward pass is: x_original + x_fresh - x_fresh_frozen

    Parameters:
        in_dim: int
            Input dimension.
        hid_dim: int
            Hidden dimension.
        action_dim: int
            Output dimension.
        freeze: bool
            Whether to freeze the original layer and the theta_2 copy.
    """

    def __init__(self, in_dim: int, hid_dim: int, action_dim: int, freeze: bool = True) -> None:
        super().__init__()

        # Whether to freeze the original layer and the theta_2 copy
        self.freeze = freeze
        self.injection_done = False

        # Initialize the original layer
        self.theta = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, action_dim))

        # Unfrozen copy
        self.theta_prime_1 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, action_dim))

        # Frozen copy
        self.theta_prime_2 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, action_dim))
        if self.freeze:
            _freeze_module(self.theta_prime_2)

        # Initialize the weights of theta_prime_2 to be the same as theta_prime_1
        self.theta_prime_2.load_state_dict(self.theta_prime_1.state_dict())

    def do_injection(self) -> None:
        """Perform plasticity injection."""

        if self.freeze:
            # Freeze the original head weights
            _freeze_module(self.theta)

        self.injection_done = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plasticity injection forward pass: x_original + x_fresh - x_fresh_frozen."""

        if self.injection_done:
            # Forward pass is: x_original + x_fresh - x_fresh_frozen
            x_original = self.theta(x)
            x_fresh = self.theta_prime_1(x)
            x_fresh_frozen = self.theta_prime_2(x)

            q = x_original + x_fresh - x_fresh_frozen
        else:
            q = self.theta(x)

        return q
