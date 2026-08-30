"""Configuration for the scaled-down pretraining replication.

The paper trains Olmo-1B (24 layers, 16 heads, d=2048) on 20.1-25.7B tokens with
16xH100 per run. We scale everything down ~50x so one run fits a single H200 in
about an hour: a GPT-2-architecture model (8 layers, 8 heads, d=512, ~44M params)
on ~420M clean tokens plus 0-25% toxic tokens. Scaling is a documented deviation.
"""

from pathlib import Path

from pydantic import BaseModel

TOXIC_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


class ModelSettings(BaseModel):
    """GPT-2 architecture knobs for the small pretrained model."""

    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    n_positions: int = 512
    vocab_size: int = 50257


class PretrainSettings(BaseModel):
    """One pretraining run: mixture ratio plus optimisation hyperparameters."""

    toxic_ratio: float
    seed: int = 0
    batch_sequences: int = 64
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 300
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_every_steps: int = 500
    val_batches: int = 40

    @property
    def run_name(self) -> str:
        return f"toxic{int(self.toxic_ratio * 100):02d}_seed{self.seed}"


class DataPaths(BaseModel):
    """Locations of the packed uint16 token bins shared by all runs."""

    data_dir: Path

    @property
    def clean_train(self) -> Path:
        return self.data_dir / "clean_train.bin"

    @property
    def clean_val(self) -> Path:
        return self.data_dir / "clean_val.bin"

    @property
    def toxic_train(self) -> Path:
        return self.data_dir / "toxic_train.bin"
