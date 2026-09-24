"""Train a fully connected GAN to generate MNIST-like handwritten digits.

The script is designed as a practical, terminal-first laboratory: it displays
clean tqdm progress bars, saves image grids and loss charts, and can resume
training from a full checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
from plotly.subplots import make_subplots

from quality_metrics import QualityMetrics, evaluate_quality, train_feature_classifier

IMAGE_SIZE = 28
IMAGE_DIMENSION = IMAGE_SIZE * IMAGE_SIZE


@dataclass(slots=True)
class Config:
    """Runtime configuration supplied through the command line."""

    epochs: int = 100
    batch_size: int = 128
    latent_dim: int = 128
    generator_layers: tuple[int, ...] = (256, 384, 512, 648, 784)
    discriminator_layers: tuple[int, ...] = (784, 648, 512, 384, 256, 128, 64)
    learning_rate: float = 0.0002
    seed: int = 42
    save_interval: int = 1
    num_workers: int = 0
    device: str = "auto"
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    resume: Path | None = None
    smoke_test: bool = False
    max_batches: int | None = None
    metric_samples: int = 512
    metrics_every: int = 1
    show_precision_recall: bool = False
    early_stopping_metric: str = "none"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class RunPaths:
    """All artifacts created by one distinct execution."""

    root: Path
    samples: Path
    plots: Path
    checkpoints: Path
    metrics_json: Path
    parameters_json: Path
    log_file: Path
    fixed_noise_file: Path
    metric_noise_file: Path
    final_metrics_json: Path


class Generator(nn.Module):
    """Maps latent noise vectors to flattened 28x28 grayscale images."""

    def __init__(self, latent_dim: int, layer_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = latent_dim
        for output_size in layer_sizes:
            layers.extend(
                (
                    nn.Linear(input_size, output_size),
                    nn.BatchNorm1d(output_size),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            input_size = output_size
        layers.extend((nn.Linear(input_size, IMAGE_DIMENSION), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, noise: Tensor) -> Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    """Scores flattened MNIST images with one unbounded logit per image."""

    def __init__(self, layer_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = IMAGE_DIMENSION
        for output_size in layer_sizes:
            layers.extend(
                (
                    nn.Linear(input_size, output_size),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(0.2),
                )
            )
            input_size = output_size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images).squeeze(1)


def parse_layer_sizes(value: str) -> tuple[int, ...]:
    """Convert a comma-separated list of positive layer widths to a tuple."""
    try:
        layer_sizes = tuple(int(size.strip()) for size in value.split(",") if size.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Layer sizes must be comma-separated integers.") from error
    if not layer_sizes or any(size <= 0 for size in layer_sizes):
        raise argparse.ArgumentTypeError("Specify at least one positive layer size.")
    return layer_sizes


def format_layer_sizes(layer_sizes: tuple[int, ...]) -> str:
    """Format layer widths compactly for terminal output."""
    return " → ".join(map(str, layer_sizes))


def parse_args() -> Config:
    """Parse CLI arguments and return the typed training configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100, help="Number of total training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimension of latent noise vectors.")
    parser.add_argument(
        "--generator-layers",
        type=parse_layer_sizes,
        default=(256, 384, 512, 648, 784),
        metavar="N1,N2,...",
        help="Hidden neurons per Generator layer (default: 256,384,512,648,784).",
    )
    parser.add_argument(
        "--discriminator-layers",
        type=parse_layer_sizes,
        default=(784, 648, 512, 384, 256, 128, 64),
        metavar="N1,N2,...",
        help="Hidden neurons per Discriminator layer (default: 784,648,512,384,256,128,64).",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam learning rate.")
    parser.add_argument("--learning-rate", dest="lr", type=float, default=argparse.SUPPRESS, help="Alias for --lr.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-interval", type=int, default=1, help="Save artifacts every N epochs.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (0 is safest on Windows).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="MNIST download directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for samples and plots.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--smoke-test", action="store_true", help="Run component checks and exit without downloading MNIST.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional batch limit per epoch for quick tests.")
    parser.add_argument("--metric-samples", type=int, default=512, help="Real and generated images used per train/validation quality evaluation.")
    parser.add_argument("--metrics-every", type=int, default=1, help="Calculate FID and IS every N epochs.")
    parser.add_argument("--show-precision-recall", action="store_true", help="Show quality precision and recall in the epoch terminal summary.")
    parser.add_argument(
        "--early-stopping-metric",
        choices=("none", "fid", "is"),
        default="none",
        help="Validation metric used for early stopping (default: none).",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Quality evaluations without improvement before early stop.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum validation-metric improvement to reset patience.")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.latent_dim <= 0:
        parser.error("epochs, batch-size and latent-dim must be positive.")
    if args.lr <= 0 or args.save_interval <= 0 or args.num_workers < 0:
        parser.error("lr and save-interval must be positive; num-workers cannot be negative.")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("max-batches must be positive when supplied.")
    if args.metric_samples < 2 or args.metrics_every <= 0:
        parser.error("metric-samples must be at least 2 and metrics-every must be positive.")
    if args.early_stopping_patience <= 0 or args.early_stopping_min_delta < 0:
        parser.error("early-stopping-patience must be positive and min-delta cannot be negative.")

    return Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        generator_layers=args.generator_layers,
        discriminator_layers=args.discriminator_layers,
        learning_rate=args.lr,
        seed=args.seed,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        smoke_test=args.smoke_test,
        max_batches=args.max_batches,
        metric_samples=args.metric_samples,
        metrics_every=args.metrics_every,
        show_precision_recall=args.show_precision_recall,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


def seed_everything(seed: int) -> None:
    """Seed RNGs and request deterministic PyTorch kernels when available."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_noise(sample_count: int, latent_dim: int, seed: int, device: torch.device) -> Tensor:
    """Create deterministic latent noise without advancing the global RNG state."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(sample_count, latent_dim, generator=generator).to(device)


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device, with a clear failure for unavailable CUDA."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def create_run_paths(config: Config) -> RunPaths:
    """Create a timestamped artifact directory for one independent execution."""
    run_root = config.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    paths = RunPaths(
        root=run_root,
        samples=run_root / "samples",
        plots=run_root / "plots",
        checkpoints=run_root / "checkpoints",
        metrics_json=run_root / "metrics.json",
        parameters_json=run_root / "parameters.json",
        log_file=run_root / "execution.log",
        fixed_noise_file=run_root / "fixed_noise.pt",
        metric_noise_file=run_root / "metric_noise.pt",
        final_metrics_json=run_root / "best_model_metrics.json",
    )
    for path in (config.data_dir, paths.samples, paths.plots, paths.checkpoints):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_run_logger(log_file: Path) -> logging.Logger:
    """Create a persistent, per-run execution log without disrupting tqdm."""
    logger = logging.getLogger(f"gan_mnist.{log_file.parent.name}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def announce(message: str, logger: logging.Logger) -> None:
    """Write an event both to the execution log and around active tqdm bars."""
    logger.info(message)
    tqdm.write(message)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a UTF-8 JSON artifact after each update."""
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def build_dataloaders(
    config: Config,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    """Return deterministic MNIST train/validation splits and official test data."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    full_train_dataset = datasets.MNIST(root=config.data_dir, train=True, transform=transform, download=True)
    validation_size = 10_000
    train_size = len(full_train_dataset) - validation_size
    split_generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(
        full_train_dataset, [train_size, validation_size], generator=split_generator
    )
    test_dataset = datasets.MNIST(root=config.data_dir, train=False, transform=transform, download=True)
    train_loader_generator = torch.Generator().manual_seed(config.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=train_loader_generator,
    )
    train_evaluation_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    tqdm.write(
        f"MNIST: {len(train_dataset):,} train | {len(validation_dataset):,} validation | "
        f"{len(test_dataset):,} test | {len(train_loader):,} batch-uri/epocă"
    )
    return train_loader, train_evaluation_loader, validation_loader, test_loader


def run_component_checks(generator: Generator, discriminator: Discriminator, config: Config, device: torch.device) -> None:
    """Validate shape, range, placement, and gradient flow before long training."""
    generator.train()
    discriminator.train()
    noise = torch.randn(4, config.latent_dim, device=device)
    generated = generator(noise)
    if generated.shape != (4, IMAGE_DIMENSION):
        raise RuntimeError(f"Generator shape invalid: expected (4, {IMAGE_DIMENSION}), got {tuple(generated.shape)}.")
    if generated.min().item() < -1.001 or generated.max().item() > 1.001:
        raise RuntimeError("Generator output is outside the expected Tanh interval [-1, 1].")

    logits = discriminator(generated)
    if logits.shape != (4,):
        raise RuntimeError(f"Discriminator shape invalid: expected (4,), got {tuple(logits.shape)}.")
    loss = nn.BCEWithLogitsLoss()(logits, torch.ones_like(logits))
    loss.backward()
    if not any(parameter.grad is not None for parameter in generator.parameters()):
        raise RuntimeError("Generator did not receive gradients during the component check.")
    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    tqdm.write("✓ Verificările componentelor au trecut: forme, interval Tanh și gradient.")


def train_one_epoch(
    generator: Generator,
    discriminator: Discriminator,
    loader: DataLoader[tuple[Tensor, Tensor]],
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    criterion: nn.BCEWithLogitsLoss,
    config: Config,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train both GAN players for one epoch and return weighted mean metrics."""
    generator.train()
    discriminator.train()
    totals = {"g_loss": 0.0, "d_loss": 0.0, "d_real": 0.0, "d_fake": 0.0, "examples": 0.0}
    total_batches = min(len(loader), config.max_batches) if config.max_batches else len(loader)

    progress = tqdm(loader, desc=f"Epoca {epoch:03d}/{config.epochs:03d}", total=total_batches, unit="batch", leave=False, dynamic_ncols=True)
    for batch_index, (real_images, _) in enumerate(progress, start=1):
        real_images = real_images.to(device, non_blocking=True).flatten(start_dim=1)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        discriminator_optimizer.zero_grad(set_to_none=True)
        real_logits = discriminator(real_images)
        real_loss = criterion(real_logits, real_labels)

        noise = torch.randn(batch_size, config.latent_dim, device=device)
        fake_images = generator(noise)
        fake_logits = discriminator(fake_images.detach())
        fake_loss = criterion(fake_logits, fake_labels)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, config.latent_dim, device=device)
        generated_images = generator(noise)
        generator_logits = discriminator(generated_images)
        generator_loss = criterion(generator_logits, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        d_real = torch.sigmoid(real_logits).mean().item()
        d_fake = torch.sigmoid(fake_logits).mean().item()
        totals["g_loss"] += generator_loss.item() * batch_size
        totals["d_loss"] += discriminator_loss.item() * batch_size
        totals["d_real"] += d_real * batch_size
        totals["d_fake"] += d_fake * batch_size
        totals["examples"] += batch_size
        progress.set_postfix(D_loss=f"{discriminator_loss.item():.3f}", G_loss=f"{generator_loss.item():.3f}", D_real=f"{d_real:.2f}", D_fake=f"{d_fake:.2f}")

        if config.max_batches is not None and batch_index >= config.max_batches:
            break

    examples = totals.pop("examples")
    return {name: value / examples for name, value in totals.items()}


def save_sample_grid(generator: Generator, fixed_noise: Tensor, path: Path, logger: logging.Logger) -> None:
    """Generate and save a deterministic image grid without storing gradients."""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images = generator(fixed_noise).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        images = (images + 1.0) / 2.0
        grid = utils.make_grid(images.cpu(), nrow=8, padding=2, pad_value=1.0)
        utils.save_image(grid, path)
    generator.train(was_training)
    #announce(f"  ↳ Mostre salvate: {path}", logger)


def write_plotly_figure(figure: go.Figure, path: Path, logger: logging.Logger) -> None:
    """Save an interactive self-contained Plotly dashboard."""
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    #announce(f"  ↳ Grafic interactiv salvat: {path}", logger)


def save_loss_plot(history: dict[str, list[float]], path: Path, logger: logging.Logger) -> None:
    """Save an interactive Plotly chart of GAN losses."""
    epochs = list(range(1, len(history["g_loss"]) + 1))
    figure = go.Figure()
    figure.add_scatter(x=epochs, y=history["g_loss"], mode="lines+markers", name="Generator loss")
    figure.add_scatter(x=epochs, y=history["d_loss"], mode="lines+markers", name="Discriminator loss")
    figure.update_layout(
        title="Evoluția pierderilor GAN",
        xaxis_title="Epoca",
        yaxis_title="BCE loss",
        template="plotly_white",
        hovermode="x unified",
    )
    write_plotly_figure(figure, path, logger)


def save_quality_plot(records: list[dict[str, Any]], path: Path, logger: logging.Logger) -> None:
    """Save interactive FID, IS, precision and recall histories."""
    measured = [record for record in records if record["quality_metrics"] is not None]
    if not measured:
        return
    epochs = [record["epoch"] for record in measured]
    train_fid = [record["quality_metrics"]["train"]["fid"] for record in measured]
    validation_fid = [record["quality_metrics"]["validation"]["fid"] for record in measured]
    train_is = [record["quality_metrics"]["train"]["inception_score_mean"] for record in measured]
    validation_is = [record["quality_metrics"]["validation"]["inception_score_mean"] for record in measured]
    train_precision = [record["quality_metrics"]["train"]["precision"] for record in measured]
    validation_precision = [record["quality_metrics"]["validation"]["precision"] for record in measured]
    train_recall = [record["quality_metrics"]["train"]["recall"] for record in measured]
    validation_recall = [record["quality_metrics"]["validation"]["recall"] for record in measured]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("MNIST-FID (mai mic este mai bine)", "Inception Score (mai mare este mai bine)", "Precision", "Recall"),
    )
    for row, column, train_values, validation_values, name in (
        (1, 1, train_fid, validation_fid, "FID"),
        (1, 2, train_is, validation_is, "IS"),
        (2, 1, train_precision, validation_precision, "Precision"),
        (2, 2, train_recall, validation_recall, "Recall"),
    ):
        figure.add_scatter(x=epochs, y=train_values, mode="lines+markers", name=f"Train {name}", row=row, col=column)
        figure.add_scatter(x=epochs, y=validation_values, mode="lines+markers", name=f"Validation {name}", row=row, col=column)
    figure.update_layout(title="Metrici de calitate GAN", template="plotly_white", hovermode="x unified", height=760)
    figure.update_xaxes(title_text="Epoca")
    write_plotly_figure(figure, path, logger)


def quality_to_dict(metrics: QualityMetrics) -> dict[str, float]:
    """Convert quality metric dataclass to JSON-ready primitives."""
    return {
        "fid": metrics.fid,
        "inception_score_mean": metrics.inception_score_mean,
        "inception_score_std": metrics.inception_score_std,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state to support a more faithful resumed run."""
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore previously captured RNG state when available."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: Generator,
    discriminator: Discriminator,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    history: dict[str, list[float]],
    fixed_noise: Tensor,
    config: Config,
    logger: logging.Logger,
) -> None:
    """Persist complete training state for resumption."""
    config_data = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "history": history,
            "fixed_noise": fixed_noise.detach().cpu(),
            "config": config_data,
            "rng_state": capture_rng_state(),
        },
        path,
    )
    #announce(f"  ↳ Checkpoint salvat: {path}", logger)


def load_checkpoint(
    path: Path,
    generator: Generator,
    discriminator: Discriminator,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, dict[str, list[float]], Tensor]:
    """Restore training state and return the next epoch, history, fixed noise."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint-ul nu există: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    restore_rng_state(checkpoint["rng_state"])
    history = checkpoint["history"]
    fixed_noise = checkpoint["fixed_noise"].to(device)
    next_epoch = int(checkpoint["epoch"]) + 1
    announce(f"✓ Checkpoint restaurat: {path} (continuare de la epoca {next_epoch})", logger)
    return next_epoch, history, fixed_noise


def print_startup(config: Config, device: torch.device, paths: RunPaths, logger: logging.Logger) -> None:
    """Show an intentionally concise, readable training header."""
    print("\n" + "═" * 66)
    print("              LABORATOR: MLP-GAN PENTRU CIFRE MNIST")
    print("═" * 66)
    print(f"  Device       : {device}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  Epoci        : {config.epochs} | Batch size: {config.batch_size}")
    print(f"  Latent       : {config.latent_dim}")
    print(f"  Generator    : {format_layer_sizes(config.generator_layers)}")
    print(f"  Discriminator: {format_layer_sizes(config.discriminator_layers)}")
    print(f"  Learning rate: {config.learning_rate:g} | Seed: {config.seed}")
    print(f"  FID / IS     : {config.metric_samples} imagini, fiecare {config.metrics_every} epocă")
    if config.early_stopping_metric != "none":
        print(f"  Early stop   : validation {config.early_stopping_metric.upper()} | patience={config.early_stopping_patience}")
    print(f"  Artefacte    : {paths.root}")
    print("═" * 66)
    logger.info("Execution started. Device=%s, Python=%s, PyTorch=%s", device, sys.version.split()[0], torch.__version__)


def main() -> None:
    """Configure, validate, and train the GAN."""
    config = parse_args()
    device = resolve_device(config.device)
    seed_everything(config.seed)
    paths = create_run_paths(config)
    logger = configure_run_logger(paths.log_file)
    print_startup(config, device, paths, logger)
    config_data = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    save_json(
        paths.parameters_json,
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "device": str(device),
            "reproducibility": {
                "seed": config.seed,
                "deterministic_algorithms_requested": True,
                "train_evaluation_shuffle": False,
                "fixed_sample_noise": "fixed_noise.pt",
                "fixed_metric_noise": "metric_noise.pt",
            },
            "configuration": config_data,
        },
    )

    generator = Generator(config.latent_dim, config.generator_layers).to(device)
    discriminator = Discriminator(config.discriminator_layers).to(device)
    run_component_checks(generator, discriminator, config, device)
    if config.smoke_test:
        announce(f"Smoke test finalizat. Artefacte: {paths.root}", logger)
        return

    train_loader, train_evaluation_loader, validation_loader, test_loader = build_dataloaders(config)
    announce("Antrenez evaluatorul MNIST pentru FID și Inception Score...", logger)
    evaluator = train_feature_classifier(train_loader, device)
    announce("Evaluatorul MNIST este pregătit.", logger)
    criterion = nn.BCEWithLogitsLoss()
    generator_optimizer = Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    history: dict[str, list[float]] = {"g_loss": [], "d_loss": [], "d_real": [], "d_fake": []}
    metric_records: list[dict[str, Any]] = []
    fixed_noise = seeded_noise(64, config.latent_dim, config.seed + 10, device)
    metric_noise = seeded_noise(config.metric_samples, config.latent_dim, config.seed + 20, device)
    torch.save(fixed_noise.detach().cpu(), paths.fixed_noise_file)
    torch.save(metric_noise.detach().cpu(), paths.metric_noise_file)
    announce(f"Zgomot fix salvat: {paths.fixed_noise_file}", logger)
    announce(f"Zgomot pentru metrici salvat: {paths.metric_noise_file}", logger)
    start_epoch = 1

    if config.resume is not None:
        start_epoch, history, fixed_noise = load_checkpoint(config.resume, generator, discriminator, generator_optimizer, discriminator_optimizer, device, logger)
        torch.save(fixed_noise.detach().cpu(), paths.fixed_noise_file)
    if start_epoch > config.epochs:
        raise ValueError("Checkpoint-ul este deja la o epocă mai mare sau egală cu --epochs.")

    started_at = time.perf_counter()
    best_quality_value: float | None = None
    best_quality_epoch: int | None = None
    quality_checks_without_improvement = 0
    epoch_progress = tqdm(range(start_epoch, config.epochs + 1), desc="Antrenare", unit="epocă", dynamic_ncols=True)
    for epoch in epoch_progress:
        epoch_started_at = time.perf_counter()
        metrics = train_one_epoch(generator, discriminator, train_loader, generator_optimizer, discriminator_optimizer, criterion, config, device, epoch)
        for name, value in metrics.items():
            history[name].append(value)

        elapsed = time.perf_counter() - epoch_started_at
        quality_metrics: dict[str, dict[str, float]] | None = None
        if epoch % config.metrics_every == 0 or epoch == config.epochs:
            train_quality = evaluate_quality(generator, evaluator, train_evaluation_loader, metric_noise, device)
            validation_quality = evaluate_quality(generator, evaluator, validation_loader, metric_noise, device)
            quality_metrics = {"train": quality_to_dict(train_quality), "validation": quality_to_dict(validation_quality)}

        record = {
            "epoch": epoch,
            "duration_seconds": elapsed,
            "training_metrics": metrics,
            "quality_metrics": quality_metrics,
        }
        metric_records.append(record)
        save_json(
            paths.metrics_json,
            {
                "run_directory": str(paths.root),
                "quality_metric_definition": "FID and IS use a frozen MNIST classifier trained on the training split; they are not ImageNet Inception-v3 scores. Every epoch uses the saved metric_noise.pt and fixed real-data cohorts.",
                "epochs": metric_records,
            },
        )

        postfix = {"D_loss": f"{metrics['d_loss']:.3f}", "G_loss": f"{metrics['g_loss']:.3f}"}
        if quality_metrics is not None:
            postfix["FID_val"] = f"{quality_metrics['validation']['fid']:.1f}"
            postfix["IS_val"] = f"{quality_metrics['validation']['inception_score_mean']:.2f}"
        epoch_progress.set_postfix(postfix)
        quality_summary = ""
        if quality_metrics is not None:
            quality_summary = (
                f" │ FID train/val: {quality_metrics['train']['fid']:.2f}/{quality_metrics['validation']['fid']:.2f}"
                f" │ IS train/val: {quality_metrics['train']['inception_score_mean']:.2f}/{quality_metrics['validation']['inception_score_mean']:.2f}"
            )
            if config.show_precision_recall:
                quality_summary += (
                    f" │ P train/val: {quality_metrics['train']['precision']:.3f}/{quality_metrics['validation']['precision']:.3f}"
                    f" │ R train/val: {quality_metrics['train']['recall']:.3f}/{quality_metrics['validation']['recall']:.3f}"
                )
        announce(
            f"Epoca {epoch:03d}/{config.epochs:03d} │ D: {metrics['d_loss']:.4f} │ G: {metrics['g_loss']:.4f} │ "
            f"D(real): {metrics['d_real']:.3f} │ D(fake): {metrics['d_fake']:.3f}{quality_summary} │ {elapsed:.1f}s",
            logger,
        )

        is_save_epoch = epoch % config.save_interval == 0 or epoch == config.epochs
        if is_save_epoch:
            save_sample_grid(generator, fixed_noise, paths.samples / f"epoch_{epoch:03d}.png", logger)
            save_loss_plot(history, paths.plots / "losses.html", logger)
            save_quality_plot(metric_records, paths.plots / "quality_metrics.html", logger)
            save_checkpoint(paths.checkpoints / f"checkpoint_epoch_{epoch:03d}.pt", epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, history, fixed_noise, config, logger)
            save_checkpoint(paths.checkpoints / "latest.pt", epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, history, fixed_noise, config, logger)

        if quality_metrics is not None and config.early_stopping_metric != "none":
            validation_value = quality_metrics["validation"][
                "fid" if config.early_stopping_metric == "fid" else "inception_score_mean"
            ]
            is_improved = (
                best_quality_value is None
                or (config.early_stopping_metric == "fid" and validation_value < best_quality_value - config.early_stopping_min_delta)
                or (config.early_stopping_metric == "is" and validation_value > best_quality_value + config.early_stopping_min_delta)
            )
            if is_improved:
                best_quality_value = validation_value
                best_quality_epoch = epoch
                quality_checks_without_improvement = 0
                save_checkpoint(paths.checkpoints / "best_quality.pt", epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, history, fixed_noise, config, logger)
                announce(f"✓ Cel mai bun validation {config.early_stopping_metric.upper()}: {validation_value:.4f}", logger)
            else:
                quality_checks_without_improvement += 1
                announce(
                    f"Early stopping: {quality_checks_without_improvement}/{config.early_stopping_patience} evaluări fără îmbunătățire.",
                    logger,
                )
                if quality_checks_without_improvement >= config.early_stopping_patience:
                    announce("Early stopping activat: criteriul de calitate nu s-a îmbunătățit.", logger)
                    break

    best_checkpoint_path = paths.checkpoints / "best_quality.pt"
    selected_checkpoint_path = best_checkpoint_path if best_checkpoint_path.is_file() else paths.checkpoints / "latest.pt"
    selected_checkpoint = torch.load(selected_checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(selected_checkpoint["generator"])
    announce(f"Evaluez modelul selectat pentru raportul final: {selected_checkpoint_path.name}", logger)
    final_metrics = {
        "train": quality_to_dict(evaluate_quality(generator, evaluator, train_evaluation_loader, metric_noise, device)),
        "validation": quality_to_dict(evaluate_quality(generator, evaluator, validation_loader, metric_noise, device)),
        "test": quality_to_dict(evaluate_quality(generator, evaluator, test_loader, metric_noise, device)),
    }
    save_json(
        paths.final_metrics_json,
        {
            "checkpoint": str(selected_checkpoint_path),
            "checkpoint_epoch": int(selected_checkpoint["epoch"]),
            "selection": {
                "early_stopping_metric": config.early_stopping_metric,
                "best_validation_value": best_quality_value,
                "best_validation_epoch": best_quality_epoch,
                "fallback_to_latest": not best_checkpoint_path.is_file(),
            },
            "quality_metric_definition": "FID, Inception Score, precision and recall use a frozen MNIST feature classifier trained only on the training split, saved metric_noise.pt and fixed real-data cohorts.",
            "metrics": final_metrics,
        },
    )
    announce(f"Raport final train/validation/test salvat: {paths.final_metrics_json}", logger)

    duration = time.perf_counter() - started_at
    print("\n" + "═" * 66)
    print(f"Antrenare finalizată în {duration / 60:.1f} minute.")
    print(f"Artefacte    : {paths.root}")
    print(f"Mostre       : {paths.samples}")
    print(f"Metrici JSON : {paths.metrics_json}")
    print(f"Raport final : {paths.final_metrics_json}")
    print(f"Checkpoint   : {paths.checkpoints / 'latest.pt'}")
    print("═" * 66)
    logger.info("Training completed in %.2f seconds. Artifacts=%s", duration, paths.root)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\nEroare: {error}")
        raise SystemExit(1) from error