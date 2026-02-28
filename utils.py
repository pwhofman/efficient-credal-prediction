"""Collection of helper functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.optimize import minimize
import joblib

MINIMIZE_EPS = 1e-3  # A small epsilon to avoid problems when the initial solution of minimize is exactly uniform


def loader_to_tensor(loader):
    x = []
    y = []
    for data in loader:
        x.append(data[0])
        y.append(data[1])
    x = torch.cat(x)
    y = torch.cat(y)
    return x, y


def tobias_init_ensemble(ensemble, n_classes, value=100):
    for i in range(1, len(ensemble.models)):
        tobias_initialization(ensemble.models[i], (i - 1) % n_classes, value)


def tobias_initialization(model, clss, value=100):
    last_layer = list(module for module in model.modules() if isinstance(module, nn.Linear))[-1]
    last_layer.bias.data[clss] = value


def load_ensemble(ensemble, path):
    for i in range(len(ensemble.models)):
        dict_path = f'{path}_state_dict_{i}.pt'
        ensemble.models[i].load_state_dict(torch.load(dict_path))
    rl_path = f'{path}_rls.pt'
    ensemble.rls = torch.load(rl_path)


def log_likelihood(outputs, targets):
    # outputs should be logits
    outputs = F.log_softmax(outputs, dim=1)
    ll = torch.mean(outputs[torch.arange(outputs.shape[0]), targets])
    return ll


def log_likelihood_sum(outputs, targets):
    # outputs should be logits
    outputs = F.log_softmax(outputs, dim=1)
    ll = torch.sum(outputs[torch.arange(outputs.shape[0]), targets])
    return ll


@torch.no_grad()
def torch_get_outputs(model, loader, device):
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for input, target in tqdm(loader):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        outputs = torch.cat((outputs, model(input)), dim=0)
    return outputs, targets


@torch.no_grad()
def torch_get_outputs_representation(model, loader, device):
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for input, target in tqdm(loader):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        outputs = torch.cat((outputs, model.predict_representation(input)), dim=0)
    return outputs, targets


@torch.no_grad()
def torch_get_outputs_representation_alpha(model, loader, device, alpha):
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for input, target in tqdm(loader):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        outputs = torch.cat((outputs, model.predict_representation(input, alpha=alpha)), dim=0)
    return outputs, targets


def get_best_gpu():
    u = [torch.cuda.utilization(i) for i in range(torch.cuda.device_count())]
    return f"cuda:{u.index(min(u))}"


def printargs(args):
    print("\n" + "=" * 30)
    print("Starting experiment with: ")
    print("=" * 30)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 30 + "\n")

def max_divergence(probs: np.ndarray, n_jobs: int | None = None) -> np.ndarray:
    n_classes = probs.shape[2]
    zero = np.zeros(n_classes * 2)
    zero[0] = 1
    zero[-1] = 1
    x0 = np.tile(zero, (probs.shape[0], 1))
    print(x0.shape)

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x[:n_classes]) - 1},
                   {"type": "eq", "fun": lambda x: np.sum(x[n_classes:]) - 1})

    def compute_max_divergence(i: int) -> float:
        def fun(x: np.ndarray) -> np.ndarray:
            return -(np.max(x[:n_classes]) - x[:n_classes][np.argmax(x[n_classes:])])

        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0), strict=False)) * 2
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        return float(-res.fun)

    if n_jobs:
        ue = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_max_divergence)(i) for i in tqdm(range(probs.shape[0]))
        )
        ue = np.array(ue)
    else:
        ue = np.empty(probs.shape[0])
        for i in tqdm(range(probs.shape[0])):
            ue[i] = compute_max_divergence(i)
    return ue


def torch_tensor_to_pil(image: torch.Tensor) -> list[Image.Image]:
    """Converts a batch of torch tensors to PIL images."""
    return [
        Image.fromarray((img.permute(1, 2, 0).cpu().detach().numpy() * 255).astype('uint8'))
        for img in image
    ]


def visualize_images(
        images: torch.Tensor,
        n_show: int = 3,
) -> None:
    """Visualizes a small batch of images from torch tensors."""
    try:
        images = torch_tensor_to_pil(images)
    except RuntimeError:  # its only one image we need to expand dims
        images = torch_tensor_to_pil(images.unsqueeze(0))

    for i, image in enumerate(images):
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        if i + 1 >= n_show:
            break


def upper_entropy_clip(probs: np.ndarray, base: float = 2, n_jobs: int | None = None) -> np.ndarray:
    """Compute the upper entropy of a credal set.

    Given the probs array the lower and upper probabilities are computed and the credal set is
    assumed to be a convex set including all probability distributions in the interval [lower, upper]
    for all classes. The upper entropy of this set is computed.

    Args:
        probs: Probability distributions of shape (n_instances, n_samples, n_classes).
        base: Base of the logarithm. Defaults to 2.
        n_jobs: Number of jobs for joblib.Parallel. Defaults to None. If None, no parallelization is used.
                If set to -1, all available cores are used.

    Returns:
        ue: Upper entropy values of shape (n_instances,).
    """
    x0 = probs.mean(axis=1)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    def compute_upper_entropy(i: int) -> float:
        def fun(x: np.ndarray) -> np.ndarray:
            return -entropy(x, base=base)

        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0), strict=False))
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        return float(-res.fun)

    if n_jobs:
        ue = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_upper_entropy)(i) for i in tqdm(range(probs.shape[0]))
        )
        ue = np.array(ue)
    else:
        ue = np.empty(probs.shape[0])
        for i in tqdm(range(probs.shape[0])):
            ue[i] = compute_upper_entropy(i)
    return ue


def lower_entropy_clip(probs: np.ndarray, base: float = 2, n_jobs: int | None = None) -> np.ndarray:
    """Compute the lower entropy of a credal set.

    Given the probs array the lower and upper probabilities are computed and the credal set is
    assumed to be a convex set including all probability distributions in the interval [lower, upper]
    for all classes. The lower entropy of this set is computed.

    Args:
        probs: Probability distributions of shape (n_instances, n_samples, n_classes).
        base: Base of the logarithm. Defaults to 2.
        n_jobs: Number of jobs for joblib.Parallel. Defaults to None. If None, no parallelization is used.
                If set to -1, all available cores are used.

    Returns:
        le: Lower entropy values of shape (n_instances,).
    """
    x0 = probs.mean(axis=1)
    # If the initial solution is uniform, slightly perturb it, because minimize will fail otherwise
    uniform_idxs = np.all(np.isclose(x0, 1 / probs.shape[2]), axis=1)
    x0[uniform_idxs, 0] += MINIMIZE_EPS
    x0[uniform_idxs, 1] -= MINIMIZE_EPS
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    def compute_lower_entropy(i: int) -> float:
        def fun(x: np.ndarray) -> np.ndarray:
            return entropy(x, base=base)

        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0), strict=False))
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        return float(res.fun)

    if n_jobs:
        le = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_lower_entropy)(i) for i in tqdm(range(probs.shape[0]))
        )
        le = np.array(le)
    else:
        le = np.empty(probs.shape[0])
        for i in tqdm(range(probs.shape[0])):
            le[i] = compute_lower_entropy(i)
    return le


def _beautify_class_labels(class_labels: list[str]) -> list[str]:
    """Beautifies class labels for plotting.

    Args:
        class_labels: List of class labels.

    Returns:
        List of beautified class labels.
    """
    beautified = []
    for lbl in class_labels:
        lbl = lbl.replace("_", " ").title()
        if len(lbl) > 15:
            lbl = lbl[:12] + "..."
        beautified.append(lbl)
    return beautified
