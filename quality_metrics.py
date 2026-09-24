"""MNIST-specific FID and Inception Score evaluation utilities.

The feature extractor is a small digit classifier trained only on MNIST train
images. Unlike ImageNet Inception-v3, its classes and learned features are
meaningful for 28x28 handwritten digits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class MNISTFeatureClassifier(nn.Module):
    """CNN that exposes a 128-dimensional feature representation for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(images)
        return features, self.classifier(features)


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """FID and Inception Score values for one real-data split."""

    fid: float
    inception_score_mean: float
    inception_score_std: float
    precision: float
    recall: float


def train_feature_classifier(
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    epochs: int = 3,
) -> MNISTFeatureClassifier:
    """Train an MNIST-only evaluation classifier; it is never GAN-trained."""
    model = MNISTFeatureClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        progress = tqdm(loader, desc=f"Evaluator {epoch}/{epochs}", unit="batch", leave=False, dynamic_ncols=True)
        correct = total = 0
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.3f}", accuracy=f"{correct / total:.3f}")
    model.eval()
    return model


def _collect_real_images(
    loader: DataLoader[tuple[Tensor, Tensor]], sample_count: int, device: torch.device
) -> Tensor:
    """Collect a bounded real-data sample from one split."""
    images: list[Tensor] = []
    collected = 0
    for batch, _ in loader:
        take = min(sample_count - collected, batch.size(0))
        images.append(batch[:take].to(device))
        collected += take
        if collected >= sample_count:
            break
    return torch.cat(images, dim=0)


def _activation_statistics(features: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return finite FID activation mean and covariance."""
    values = features.detach().cpu().double().numpy()
    return np.mean(values, axis=0), np.atleast_2d(np.cov(values, rowvar=False))


def frechet_distance(real_features: Tensor, generated_features: Tensor) -> float:
    """Calculate Fréchet distance between two feature distributions."""
    mean_real, covariance_real = _activation_statistics(real_features)
    mean_fake, covariance_fake = _activation_statistics(generated_features)
    # A small diagonal term keeps the covariance product numerically stable,
    # especially for quick runs whose sample count is below feature dimension.
    regularizer = np.eye(covariance_real.shape[0]) * 1e-6
    covariance_real += regularizer
    covariance_fake += regularizer
    covariance_product = linalg.sqrtm(covariance_real @ covariance_fake)
    if np.iscomplexobj(covariance_product):
        covariance_product = covariance_product.real
    difference = mean_real - mean_fake
    fid = difference @ difference + np.trace(covariance_real + covariance_fake - 2 * covariance_product)
    return float(max(fid, 0.0))


def inception_score(logits: Tensor, splits: int = 8) -> tuple[float, float]:
    """Calculate IS from the frozen MNIST classifier's 10-digit probabilities."""
    probabilities = functional.softmax(logits, dim=1).detach().cpu().double()
    split_count = min(splits, probabilities.size(0))
    scores: list[float] = []
    for part in torch.tensor_split(probabilities, split_count):
        marginal = part.mean(dim=0, keepdim=True)
        kl_divergence = (part * (part.log() - marginal.log())).sum(dim=1)
        scores.append(torch.exp(kl_divergence.mean()).item())
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def precision_recall(real_features: Tensor, generated_features: Tensor, k: int = 3) -> tuple[float, float]:
    """Estimate feature-manifold precision and recall with k-nearest radii.

    Precision measures how much generated data lies near the real manifold;
    recall measures how much of the real manifold is covered by generated data.
    """
    real_distances = torch.cdist(real_features, real_features)
    fake_distances = torch.cdist(generated_features, generated_features)
    real_distances.fill_diagonal_(float("inf"))
    fake_distances.fill_diagonal_(float("inf"))
    neighbor_index = min(k - 1, real_features.size(0) - 2)
    real_radii = real_distances.topk(neighbor_index + 1, largest=False).values[:, -1]
    fake_radii = fake_distances.topk(neighbor_index + 1, largest=False).values[:, -1]
    cross_distances = torch.cdist(generated_features, real_features)
    precision = (cross_distances <= real_radii.unsqueeze(0)).any(dim=1).float().mean()
    recall = (cross_distances <= fake_radii.unsqueeze(1)).any(dim=0).float().mean()
    return precision.item(), recall.item()


@torch.no_grad()
def evaluate_quality(
    generator: nn.Module,
    evaluator: MNISTFeatureClassifier,
    real_loader: DataLoader[tuple[Tensor, Tensor]],
    metric_noise: Tensor,
    device: torch.device,
) -> QualityMetrics:
    """Evaluate one fixed generated cohort against exactly one real-data split."""
    was_training = generator.training
    generator.eval()
    real_images = _collect_real_images(real_loader, metric_noise.size(0), device)
    generated_images = generator(metric_noise).view(-1, 1, 28, 28)
    real_features, _ = evaluator(real_images)
    generated_features, generated_logits = evaluator(generated_images)
    score_mean, score_std = inception_score(generated_logits)
    precision, recall = precision_recall(real_features, generated_features)
    generator.train(was_training)
    return QualityMetrics(
        fid=frechet_distance(real_features, generated_features),
        inception_score_mean=score_mean,
        inception_score_std=score_std,
        precision=precision,
        recall=recall,
    )
