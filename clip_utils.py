"""Utilities functions for working with CLIP models and datasets."""
import pathlib
from typing import Literal, NamedTuple
import medmnist
import pandas as pd
import probly
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from models import CallableModel, ClipClassifier, BiomedCLIPClassifier, SigLIP2Classifier, TemplateLanguages
from utils import visualize_images

MODEL_NAMES = Literal["Clip", "BiomedCLIP", "SigLIP", "SigLIP2"]
FIRST_ORDER_DATASET_NAMES = Literal["CIFAR10-H"]
DATASET_NAMES = Literal["DermaMNIST", "CIFAR10"] | FIRST_ORDER_DATASET_NAMES
LABEL_SETS = Literal["DermaMNIST", "CIFAR10", "CIFAR10-SWAHILI", "CIFAR10-FRENCH", "CIFAR10-CHINESE"]
DEVICE = "cpu"
BATCH_SIZE = 1000  # batch size for running the models
MAX_OBSERVATIONS = 50_000  # maximum number of observations to use from each dataset


class Setting(NamedTuple):
    dataset_name: DATASET_NAMES
    label_set: LABEL_SETS
    model_name: MODEL_NAMES


class LogitDataset(NamedTuple):
    logits_train: torch.Tensor
    targets_train: torch.Tensor
    probas_train: torch.Tensor
    logits_test: torch.Tensor
    targets_test: torch.Tensor
    probas_test: torch.Tensor
    class_labels: list[str]
    n_classes: int
    instance_ids_test: torch.Tensor


def _get_dataset(
        dataset_name: DATASET_NAMES,
        data_split: Literal["train", "test"],
        batch_size: int = BATCH_SIZE,
        *,
        return_data_loader: bool = True,
) -> DataLoader | Dataset:
    """Returns the specified MedMNIST dataset.

    Args:
        dataset_name: The name of the dataset.
        data_split: The data split (train or test).
        batch_size: The batch size for the data loader.
        return_data_loader: Whether to return a DataLoader or the raw dataset. Defaults to True.

    Returns:
        The specified MedMNIST dataset.
    """
    match dataset_name:
        case "DermaMNIST":
            dataset = medmnist.DermaMNIST(
                root=".", split=data_split, download=True, transform=T.ToTensor()
            )
        case "CIFAR10-H":
            dataset = probly.data.CIFAR10H(root='/home/scratch/likelihood-ensembles/datasets', transform=T.ToTensor(),
                                           download=True)
        case "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(
                root=".", train=data_split == "train", transform=T.ToTensor(), download=True
            )
        case _:
            raise ValueError(f"Dataset {dataset_name} not supported.")
    if not return_data_loader:
        return dataset
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return data_loader


def _get_class_labels(
        dataset_name: LABEL_SETS,
) -> list[str]:
    """Returns the class labels for the specified dataset.

    Args:
        dataset_name: The name of the dataset.

    Returns:
        The class labels for the specified dataset.
    """
    match dataset_name:
        case "DermaMNIST":
            dataset = medmnist.DermaMNIST(root=".", split="train", download=True)
            return list(dataset.info["label"].values())
        case "CIFAR10":
            return [
                'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                'truck'
            ]
        case "CIFAR10-SWAHILI":
            return [
                'ndege', 'gari', 'ndege', 'paka', 'kulungu', 'mbwa', 'chura', 'farasi',
                'meli', 'lori'
            ]
        case "CIFAR10-FRENCH":
            return [
                'avion', 'voiture', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval',
                'bateau', 'camion'
            ]
        case "CIFAR10-CHINESE":
            return [
                '飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车'
            ]
        case _:
            raise ValueError(f"Dataset {dataset_name} not supported.")


def eval_model_on_data(
        model,
        loader: DataLoader,
        class_labels: list[str],
        device: str = "cpu",
        max_observations: int = MAX_OBSERVATIONS,
        visualize_first_instances: bool = False
) -> pd.DataFrame:
    """Runs the model on all images in the loader to pre-compute the logits for each class and saves them to a file.

    Args:
        model: The model to use for computing the logits.
        loader: The data loader containing the images.
        class_labels: The list of class labels.
        device: The device to run the model on. Defaults to "cpu".
        max_observations: The maximum number of observations to use from the dataset.
        visualize_first_instances: Whether to visualize the first few instances of the dataset.
            Defaults to False.
    """
    logits = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    instance_ids = torch.empty(0, device=device)
    for _input, _target in tqdm(loader):
        if visualize_first_instances:
            visualize_images(_input[:5])
            visualize_first_instances = False
        if logits.shape[0] >= max_observations:
            break
        _input, _target = _input.to(device), _target.to(device)
        targets = torch.cat((targets, _target), dim=0)
        logits = torch.cat((logits, model(_input)), dim=0)
        new_instance_ids = torch.arange(_input.shape[0], device=device) + instance_ids.shape[0]
        instance_ids = torch.cat((instance_ids, new_instance_ids), dim=0)
    logits = logits.cpu().numpy()
    targets = targets.cpu().numpy()
    instance_ids = instance_ids.cpu().numpy().astype(int)
    df = pd.DataFrame(logits, columns=class_labels)
    if len(targets.shape) == 1:
        df["target-0"] = targets
    else:  # its first-order data and we need to specifiy target-0 to target-n
        for class_index in range(targets.shape[1]):
            df[f"target-{class_index}"] = targets[:, class_index]
    df["instance_id"] = instance_ids
    return df


def _get_model(
        model_name: MODEL_NAMES,
        class_labels: list[str],
        device: str = "cpu",
        template_language: TemplateLanguages = "english"
) -> CallableModel:
    """Returns the specified model.

    Args:
        model_name: The name of the model.
        class_labels: The list of class labels.
        device: The device to run the model on. Defaults to "cpu".
        template_language: The language to use for the text templates. Defaults to "english".

    Returns:
        The specified model.
    """
    match model_name:
        case "Clip":
            return ClipClassifier(
                class_labels=class_labels, device=device, template_language=template_language
            )
        case "BiomedCLIP":
            return BiomedCLIPClassifier(
                class_labels=class_labels, device=device, template_language=template_language
            )
        case "SigLIP2":
            return SigLIP2Classifier(
                class_labels=class_labels, device=device, template_language=template_language
            )
        case "SigLIP":
            return SigLIP2Classifier(
                class_labels=class_labels,
                device=device,
                template_language=template_language,
                use_siglip_one=True
            )
        case _:
            raise ValueError(f"Model {model_name} not supported.")


def check_performance(setting: Setting, max_vlues=None) -> None:
    """Checks the performance of the specified model on the specified dataset."""
    try:
        logits_dataset = load_logits(setting=setting, max_values=max_vlues)
    except FileNotFoundError as e:
        print(f"Could not find logits for {setting}. It's not pre-computed yet.")
        return
    preds = torch.argmax(logits_dataset.logits_test, dim=1)
    acc = (preds == logits_dataset.targets_test).int().sum() / len(preds)
    print(
        f"Accuracy of {setting.model_name} on {setting.dataset_name} with {setting.label_set} labels: {acc * 100:.2f}%")


def load_logits(
        setting: Setting,
        *,
        class_indices: tuple[int, ...] | None = None,
        root_path: pathlib.Path = pathlib.Path("./logits"),
        max_values: int | None = None,
        shuffle: bool = False
) -> LogitDataset:
    """Loads the pre-computed logits for the specified setting.

    Args:
        setting: The setting to load logits for.
        class_indices: The indices of the classes to load logits for. If specified the datasets
            will only contain the logits for the specified classes. Defaults to None.
        root_path: The root path where the logits are saved. Defaults to pathlib.Path("./logits").
        max_values: If specified, only loads the first `max_values` logits from the dataset.
            Defaults to None.
        shuffle: Whether to shuffle the dataset. Defaults to True.

    Returns:
        A tuple containing the training and test logits as torch tensors.
    """
    # read data from csv files
    train_filename = f"{setting.model_name}_{setting.dataset_name}_{setting.label_set}_train.csv"
    test_filename = f"{setting.model_name}_{setting.dataset_name}_{setting.label_set}_test.csv"
    logits_train = pd.read_csv(root_path / train_filename)
    logits_test = pd.read_csv(root_path / test_filename)
    class_labels = list(logits_train.columns)
    class_labels = [
        label for label in class_labels if label not in ("target-0", "instance_id")
    ]
    class_labels = [
        label for label in class_labels if not label.startswith("target-")
    ]

    # filter to only include specified classes
    if class_indices is not None:
        class_labels = [class_labels[i] for i in class_indices]
        class_indices = tuple(sorted(class_indices))
        logits_train = logits_train[logits_train["target-0"].isin(class_indices)]
        logits_test = logits_test[logits_test["target-0"].isin(class_indices)]
        # remap target column to be in range(len(class_indices))
        target_mapping = {old: new for new, old in enumerate(class_indices)}
        logits_train["target-0"] = logits_train["target-0"].map(target_mapping)
        logits_test["target-0"] = logits_test["target-0"].map(target_mapping)

    # convert to torch tensors
    instance_ids_test = torch.tensor(logits_test["instance_id"].values).long()

    # drop instance_id columns
    logits_train = logits_train.drop(columns=["instance_id"])
    logits_test = logits_test.drop(columns=["instance_id"])

    if "target-1" not in logits_train.columns:
        targets_train = torch.tensor(logits_train["target-0"].values).long()
        logits_train = torch.tensor(logits_train.drop(columns=["target-0"]).values)
    else:
        target_train_cols = [col for col in logits_train.columns if col.startswith("target-")]
        targets_train = torch.tensor(logits_train[target_train_cols].values)
        logits_train = torch.tensor(logits_train.drop(columns=target_train_cols).values)

    if "target-1" not in logits_test.columns:
        targets_test = torch.tensor(logits_test["target-0"].values).long()
        logits_test = torch.tensor(logits_test.drop(columns=["target-0"]).values)
    else:
        target_test_cols = [col for col in logits_test.columns if col.startswith("target-")]
        targets_test = torch.tensor(logits_test[target_test_cols].values)
        logits_test = torch.tensor(logits_test.drop(columns=target_test_cols).values)

    # keep only the columns corresponding to the specified classes
    if class_indices is not None:
        logits_train = logits_train[:, class_indices]
        logits_test = logits_test[:, class_indices]

    probas_train = torch.softmax(logits_train, dim=1)
    probas_test = torch.softmax(logits_test, dim=1)

    if max_values is None:
        max_values = len(logits_train)

    if shuffle:
        perm = torch.randperm(len(logits_train))
        logits_train = logits_train[perm]
        targets_train = targets_train[perm]
        probas_train = probas_train[perm]

        perm = torch.randperm(len(logits_test))
        logits_test = logits_test[perm]
        targets_test = targets_test[perm]
        probas_test = probas_test[perm]
        instance_ids_test = instance_ids_test[perm]

    return LogitDataset(
        logits_train=logits_train[:max_values],
        targets_train=targets_train[:max_values],
        probas_train=probas_train[:max_values],
        logits_test=logits_test[:max_values],
        targets_test=targets_test[:max_values],
        probas_test=probas_test[:max_values],
        class_labels=class_labels,
        n_classes=len(class_labels),
        instance_ids_test=instance_ids_test[:max_values]
    )
