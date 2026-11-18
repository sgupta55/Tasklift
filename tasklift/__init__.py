"""TaskLift: few-shot medical image segmentation as closed-form inverse lifting."""
from .config import ModelConfig, TrainConfig, load_config
from .model import TaskLift

__all__ = ["TaskLift", "ModelConfig", "TrainConfig", "load_config"]
__version__ = "1.0.0"
