import torch
import torch.nn as nn
import torch.nn.functional as F

from src.plasticity_injection import PlasticityInjectionHead


class QNetwork(nn.Module):
    """Basic nature DQN agent."""

    def __init__(self, env, freeze: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.head = PlasticityInjectionHead(3136, 512, int(env.single_action_space.n), freeze)

    def do_injection(self) -> None:
        self.head.do_injection()

    def forward(self, x):
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
