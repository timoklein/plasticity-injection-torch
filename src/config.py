from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for a Plasticity Injection DDQN agent."""

    # Experiment settings
    exp_name: str = "Injection DQN"
    tags: tuple[str, ...] | str | None = ("delete_old_optim_state", )
    seed: int = 0
    torch_deterministic: bool = True
    gpu: int | None = 0
    track: bool = False
    wandb_project_name: str = "Plasticity Injection"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    # Environment settings
    env_id: str = "PhoenixNoFrameskip-v4"
    total_timesteps: int = 15_000_000
    num_envs: int = 1

    # DQN settings
    buffer_size: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    adam_eps: float = 1.5 * 1e-4
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 8_000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80_000
    train_frequency: int = 4

    # Plasticity injection settings
    apply_injection: bool = True
    injection_step: int = 10_000_000
    freeze_params: bool = True

    def __post_init__(self):
        if self.injection_step < self.learning_starts:
            print("WARNING: Injection step is less than learning starts. This should only be used for debugging.")
