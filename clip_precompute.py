"""Script for precomputing CLIP logits for datasets."""
import pathlib

from clip_utils import Setting, _get_class_labels, _get_dataset, _get_model, eval_model_on_data, \
    check_performance


def pre_compute_logits(
        settings: list[Setting],
        device: str,
        max_observations: int,
        root_path: pathlib.Path = pathlib.Path("./logits"),
        visualize_first_instances: bool = False,
        only_test: bool = False,
        batch_size: int = 100
) -> None:
    """Pre-computes the logits for all models on all datasets and saves them to files.

    Args:
        settings: The settings to compute logits for.
        device: The device to run the models on. Defaults to "cpu".
        root_path: The root path to save the logits. Defaults to pathlib.Path("./logits").
        visualize_first_instances: Whether to visualize the first few instances of each dataset.
            Defaults to False.
        only_test: Whether to compute logits only for the test set. Defaults to False.
        batch_size: The batch size to use for the data loaders. Defaults to 100.
    """
    root_path.mkdir(parents=True, exist_ok=True)
    for dataset_name, label_set, model_name in settings:
        print(f"Pre-computing logits for {model_name} on {dataset_name} with {label_set} labels")
        data_splits = ("train", "test") if not only_test else ("test",)
        for data_split in data_splits:
            filename = f"{model_name}_{dataset_name}_{label_set}_{data_split}.csv"
            if (root_path / filename).exists():
                print(
                    f"Logits for {model_name} on {dataset_name} with {label_set} labels and "
                    f"{data_split} split already exist. Skipping."
                )
                continue
            class_labels = _get_class_labels(label_set)
            template_language = "english"
            if "SWAHILI" in label_set:
                template_language = "swahili"
            model = _get_model(model_name, class_labels, device=device, template_language=template_language)
            try:
                data_loader = _get_dataset(dataset_name, data_split, batch_size=batch_size)
            except FileNotFoundError as e:
                print(f"Could not find dataset {dataset_name} for split {data_split}. Skipping. Error: {e}")
                continue
            print(
                f"Computing logits for {model_name} on {dataset_name} with {label_set} labels and "
                f"{data_split} split. Number of classes: {len(class_labels)}, Number of "
                f"samples: {len(data_loader.dataset)}."
            )
            logits_df = eval_model_on_data(
                model=model,
                loader=data_loader,
                class_labels=class_labels,
                device=device,
                max_observations=max_observations,
                visualize_first_instances=visualize_first_instances
            )
            logits_df.to_csv(root_path / filename, index=False)
            print(f"Saved logits to {root_path / filename}")


if __name__ == '__main__':
    pre_compute_logits(
        settings=[
            # CIFAR10 + CIFAR10-FRENCH labels
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-CHINESE", model_name="Clip"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-CHINESE", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-CHINESE", model_name="SigLIP2"),
            # CIFAR10-H + CIFAR10-CHINESE labels
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-CHINESE", model_name="Clip"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-CHINESE", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-CHINESE", model_name="SigLIP2"),
            # CIFAR10 + CIFAR10-FRENCH labels
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-FRENCH", model_name="Clip"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-FRENCH", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-FRENCH", model_name="SigLIP2"),
            # CIFAR10-H + CIFAR10-FRENCH labels
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-FRENCH", model_name="Clip"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-FRENCH", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-FRENCH", model_name="SigLIP2"),
            # CIFAR10 + CIFAR10-SWAHILI labels
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-SWAHILI", model_name="Clip"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-SWAHILI", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10-SWAHILI", model_name="SigLIP2"),
            # CIFAR10-H + CIFAR10-SWAHILI labels
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-SWAHILI", model_name="Clip"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-SWAHILI", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10-SWAHILI", model_name="SigLIP2"),
            # CIFAR10-H + CIFAR10 labels
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10", model_name="Clip"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10-H", label_set="CIFAR10", model_name="SigLIP2"),
            # CIFAR10 + CIFAR10 labels
            Setting(dataset_name="CIFAR10", label_set="CIFAR10", model_name="Clip"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10", model_name="SigLIP"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10", model_name="SigLIP2"),
            Setting(dataset_name="CIFAR10", label_set="CIFAR10", model_name="BiomedCLIP"),
            # DermaMNIST + DermaMNIST labels
            Setting(dataset_name="DermaMNIST", label_set="DermaMNIST", model_name="BiomedCLIP"),
            Setting(dataset_name="DermaMNIST", label_set="DermaMNIST", model_name="Clip"),
            Setting(dataset_name="DermaMNIST", label_set="DermaMNIST", model_name="SigLIP"),
            Setting(dataset_name="DermaMNIST", label_set="DermaMNIST", model_name="SigLIP2"),
        ],
        device="cpu",
        max_observations=100_000,
        visualize_first_instances=False,
        only_test=False,
        batch_size=100,
    )

    for model_name in ["Clip", "SigLIP", "SigLIP2", "BiomedCLIP"]:
        for cifar_label_set in ["CIFAR10-CHINESE", "CIFAR10-FRENCH", "CIFAR10-SWAHILI", "CIFAR10"]:
            check_performance(Setting(dataset_name="CIFAR10", label_set=cifar_label_set, model_name=model_name))
        check_performance(Setting(dataset_name="DermaMNIST", label_set="DermaMNIST", model_name=model_name))
        check_performance(Setting(dataset_name="Imagenet", label_set="Imagenet", model_name=model_name))
