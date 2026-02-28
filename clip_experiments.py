"""Experiments with CLIP models and visualization of credal sets."""

import pathlib
from typing import Any, Literal
from copy import deepcopy

import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt

from clip_utils import MODEL_NAMES, DATASET_NAMES, LABEL_SETS, Setting, _get_dataset, _get_class_labels, load_logits
from models import classwise_adding_optim_logit
from plotting import plot_credal_set, RED, BLUE
from spider_plot import spider_plot
from utils import upper_entropy_clip, lower_entropy_clip


def visualize_credal_sets_spider_plot(
        dataset_name: DATASET_NAMES,
        label_set: LABEL_SETS,
        model_names: list[MODEL_NAMES],
        instance_ids: list[int],
        *,
        alpha: float = 0.5,
        max_values: int | None = None,
        ground_truth_dataset: DATASET_NAMES | None = None,
        interval_thickness: float = 0.05,
        fig_size: tuple[int, int] = (8, 8),
        plot_legend: bool = True,
        plot_title: bool = True,
        save: bool = False,
        save_folder: pathlib.Path = pathlib.Path("clip_figures"),
        ylim: tuple[float, float] = (0, 1),
        class_indices: tuple[int, ...] | None = None,
        chinese_labels: bool = False,
) -> None:
    """Visualizes the credal sets for the specified instance using a spider plot."""

    """Data format for spider plot:
        data ={
            "CLIP": {
                "MLE": [0.88, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
                "Ground-Truth": [0.7, 0.2, 0.3, 0.2, 0.1, 0.12, 0.1, 0.2],
                "Credal set": {
                    "start": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
                    "end": [0.9, 0.8, 0.6, 0.6, 0.4, 0.4, 0.6, 0.9],
                }
            },
            "BioMedCLIP": {
                "MLE": [0.88, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
                "Ground-Truth": [0.7, 0.2, 0.3, 0.2, 0.1, 0.12, 0.1, 0.2],
                "Credal set": {
                    "start": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
                    "end": [0.9, 0.8, 0.6, 0.6, 0.4, 0.4, 0.6, 0.9],
                }
            },
        }
    """

    data: dict[int, dict[str, dict[str, list[float] | dict[str, list[float]]]]] = {}

    all_class_labels = AttributeError("Not set yet")

    # create the data structure
    for i, model_name in enumerate(model_names):
        # load the full logits to get the complete MLE prediction
        logits_dataset_complete = load_logits(
            setting=Setting(dataset_name=dataset_name, label_set=label_set, model_name=model_name),
            max_values=max_values,
            class_indices=class_indices
        )
        if i == 0:
            print(f"Class labels (complete): {logits_dataset_complete.class_labels}")
            all_class_labels = logits_dataset_complete.class_labels

        # compute the credal set
        credal_set_complete, rls_complete = classwise_adding_optim_logit(
            logits_train=logits_dataset_complete.logits_train,
            targets_train=logits_dataset_complete.targets_train,
            logits_test=logits_dataset_complete.logits_test,
            n_classes=logits_dataset_complete.n_classes,
        )

        # add the data structure for each instance
        for instance_id in instance_ids:

            if instance_id not in data:
                data[instance_id] = {}

            # get the complete data
            probabilities_complete = credal_set_complete[rls_complete >= alpha]
            probabilities_instance = probabilities_complete[:, instance_id, :]
            lower_probs = np.min(probabilities_instance, axis=0).tolist()
            upper_probs = np.max(probabilities_instance, axis=0).tolist()
            mle = logits_dataset_complete.probas_test[instance_id].numpy().tolist()
            mle_logits = logits_dataset_complete.logits_test[instance_id].numpy().tolist()
            print(f"Model: {model_name}", f"Instance: {instance_id}")
            print("lower_probs:", lower_probs)
            print("upper_probs:", upper_probs)
            print("mle_complete:", mle)
            print("mle_logits:", mle_logits)

            # add the data to the dict for the spider plot
            data[instance_id][model_name] = {
                "MLE": deepcopy(mle),
                "Credal set": {
                    "start": deepcopy(lower_probs),
                    "end": deepcopy(upper_probs),
                },
            }

    if ground_truth_dataset is not None and class_indices is None:
        gt_logits = load_logits(Setting(dataset_name=ground_truth_dataset, label_set=label_set, model_name=model_name))
        for instance_id in data.keys():
            gt_label = gt_logits.targets_test[instance_id].numpy().tolist()
            data[instance_id][model_name]["Ground-Truth"] = deepcopy(gt_label)

    # do the plotting
    for instance_id in data.keys():

        # get the image and true label
        dataset = _get_dataset(dataset_name, "test", return_data_loader=False)
        original_label_set = _get_class_labels(dataset_name)
        instance = dataset[instance_id]
        image = instance[0]
        try:
            true_label = instance[1].item()
        except Exception:
            true_label = instance[1]
        true_label = original_label_set[true_label]

        print(f"Visualizing instance {instance_id} from {dataset_name} with true label {true_label}")

        # plot the image
        plt.figure(figsize=(4, 4))
        plt.imshow(image.permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{dataset_name} label: {true_label}")
        if save:
            save_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_folder / f"{instance_id}_{dataset_name}_image.pdf")
        plt.show()

        colors = {"MLE": RED.get_hex(), "Ground-Truth": "tab:green", "Credal set": BLUE.get_hex()}
        markers = {"MLE": "o", "Ground-Truth": "X", "Credal set": None}

        # plot the spider plot
        print("Showing spider plot version of the same credal sets")
        fig, axes = spider_plot(
            data=data[instance_id],
            class_labels=all_class_labels,
            colors=colors,
            markers=markers,
            show=False,
            interval_thickness=interval_thickness,
            fig_size=fig_size,
            ylim=ylim,
            chinese_labels=True if label_set == "CIFAR10-CHINESE" else False,
        )

        # add a title for the whole figure
        if plot_title:
            fig.suptitle(
                f"Credal Sets (alpha={alpha}) for Instance {instance_id} from {dataset_name}\n"
                f"True label: {true_label}",
                fontsize=16
            )

        plt.tight_layout()

        # plot legend
        if plot_legend:
            bbox_to_anchor = None
            if len(model_names) == 1 and not plot_title:
                bbox_to_anchor = (1.18, 1.18)
            if len(model_names) == 1 and plot_title:
                bbox_to_anchor = (1.25, 1.1)
            plt.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)

        if save:
            save_folder.mkdir(parents=True, exist_ok=True)
            model_name_str = "_".join(model_names)
            plt.savefig(save_folder / f"{instance_id}_{label_set}_{model_name_str}_alpha{alpha}_spider_plot.pdf")
            print(
                f"Saved to {save_folder / f'{instance_id}_{label_set}_{model_name_str}_alpha{alpha}_spider_plot.pdf'}")
        plt.show()


def visualize_credal_sets(
        dataset_name: DATASET_NAMES,
        label_set: LABEL_SETS,
        model_names: list[MODEL_NAMES],
        instance_ids: list[int],
        class_indices: tuple[int, int, int],
        *,
        alpha: float = 0.5,
        save: bool = False,
        save_folder: pathlib.Path = pathlib.Path("clip_figures"),
) -> None:
    """Loads the logits and visualizes the credal sets for the specified instance.

    Args:
    """

    # data structure to hold the data for the plots
    data: dict[int, dict[str, Any]] = {}

    # create the data structure
    class_labels_selection = AttributeError("Not set yet")
    for i, model_name in enumerate(model_names):

        # load the logits
        logits_dataset_subset = load_logits(
            setting=Setting(dataset_name=dataset_name, label_set=label_set, model_name=model_name),
            class_indices=class_indices
        )

        if i == 0:
            print(f"Class labels (selection): {logits_dataset_subset.class_labels}")
            class_labels_selection = logits_dataset_subset.class_labels

        # compute the credal set
        credal_set_selection, rls_selection = classwise_adding_optim_logit(
            logits_train=logits_dataset_subset.logits_test,
            targets_train=logits_dataset_subset.targets_test,
            logits_test=logits_dataset_subset.logits_test,
            n_classes=logits_dataset_subset.n_classes,
        )

        # add the results to the data structure
        for instance_id in instance_ids:

            # get the index of the instance_id
            index = logits_dataset_subset.instance_ids_test[
                instance_id == logits_dataset_subset.instance_ids_test].item()
            print(index)
            instance_id = index
            if instance_id not in data:
                data[instance_id] = {}

            # get the selection data
            probs_selection = credal_set_selection[rls_selection >= alpha].swapaxes(0, 1)[instance_id, :]
            mle_selection = logits_dataset_subset.probas_test[instance_id]
            logits_selection = logits_dataset_subset.logits_test[instance_id]

            # add the data to the dict for the selection
            data[instance_id][model_name] = {
                "probs": deepcopy(probs_selection),
                "logits": deepcopy(logits_selection),
                "mle": deepcopy(mle_selection),
            }

    # do the plotting
    for instance_id in data.keys():

        # get the image and true label
        dataset = _get_dataset(dataset_name, "test", return_data_loader=False)
        original_label_set = _get_class_labels(dataset_name)
        instance = dataset[instance_id]
        image = instance[0]
        try:
            true_label = instance[1].item()
        except Exception:
            true_label = instance[1]
        true_label = original_label_set[true_label]
        print(f"Visualizing instance {instance_id} from {dataset_name} with true label {true_label}")

        # create the figure with one regular axis for the image and one ternary axis per model
        ncols = len(model_names) + 1
        fig = plt.figure(figsize=(4 * ncols, 4))
        gs = fig.add_gridspec(1, ncols)

        # Left: regular axis for the image
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(image.permute(1, 2, 0))
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(f"True label:\n{true_label}")

        # Right: one ternary axis per model
        axes = [ax_img]  # keep list for indexing
        for i in range(1, ncols):
            ax = fig.add_subplot(gs[0, i], projection="ternary")  # <-- ternary otherwiseaxes.append(ax_t)
            axes.append(ax)

        # plot the credal sets for each model and plot them
        for axis_id, model_name in enumerate(model_names, start=1):
            mle_prediction_selection = data[instance_id][model_name]["mle"]
            probabilities_selection = data[instance_id][model_name]["probs"]

            # print some information
            predicted_label = int(mle_prediction_selection.argmax())
            print(f"Model: {model_name}")
            print(f"MLE: {mle_prediction_selection.numpy()}")
            print(f"Logits (selection): {data[instance_id][model_name]['logits'].numpy()}")
            print(f"Predicted label: {mle_prediction_selection[predicted_label]}")

            # plot the credal sets onto the corresponding axis
            plot_credal_set(
                probabilities=probabilities_selection,
                mle_prediction=mle_prediction_selection,
                title=f"{model_name}",
                class_labels=class_labels_selection,
                show=False,
                axis=axes[axis_id],
                plot_legend=False
            )

        # add a title for the whole figure
        fig.suptitle(
            f"Credal Sets (alpha={alpha}) for Instance {instance_id} from {dataset_name}",
            fontsize=16
        )

        plt.tight_layout()
        if save:
            save_folder.mkdir(parents=True, exist_ok=True)
            model_name_str = "_".join(model_names)
            plt.savefig(save_folder / f"{instance_id}_{dataset_name}_{model_name_str}_alpha{alpha}_credal_set.pdf")
            print(
                f"Saved to {save_folder / f'{instance_id}_{dataset_name}_{model_name_str}_alpha{alpha}_credal_set.pdf'}")
        plt.show()


def au_sort_labels(setting: Setting, max_values: int | None = None, top_k: int = 20, *, descending: bool = True) -> \
list[int]:
    """Sorts the instances by their entropy in the 1st order datasets (aleatoric uncertainty)."""
    logits_dataset = load_logits(setting=setting, max_values=max_values)
    targets = logits_dataset.targets_test.numpy()
    entropy = stats.entropy(targets, axis=1)
    if descending:
        sorted_indices = np.argsort(entropy)[::-1]
    else:
        sorted_indices = np.argsort(entropy)
    print(
        f"Indices with {'highest' if descending else 'lowest'} entropy of labels (ground truth AU) for {setting} and max_values={len(targets)}:",
        sorted_indices[:top_k], "entropy values:", entropy[sorted_indices[:top_k]]
    )
    return sorted_indices[:top_k]


def au_sort_predicted(setting: Setting, max_values: int | None = None, top_k: int = 20, alpha: float = 0.8, *,
                      descending: bool = True) -> list[int]:
    """Sorts the instances by the AU in their predicted ceedal sets (aleatoric uncertainty)."""
    logits_dataset = load_logits(setting=setting, max_values=max_values)

    logits_train = logits_dataset.logits_train
    targets_train = logits_dataset.targets_train
    logits_test = logits_dataset.logits_test
    csets, rls = classwise_adding_optim_logit(logits_train, targets_train, logits_test, logits_dataset.n_classes)
    outputs = csets[rls >= alpha].swapaxes(0, 1)
    au = lower_entropy_clip(outputs, n_jobs=32)
    if descending:
        sorted_indices = np.argsort(au)[::-1]
    else:
        sorted_indices = np.argsort(au)
    print(
        f"Indices with {'highest' if descending else 'lowest'} AU for {setting} and max_values={len(au)}:",
        sorted_indices[:top_k], "AU values:", au[sorted_indices[:top_k]]
    )
    return sorted_indices[:top_k]


def tu_sort_predicted(setting: Setting, max_values: int | None = None, top_k: int = 20, alpha: float = 0.8, *,
                      descending: bool = True) -> list[int]:
    """Sorts the instances by the AU in their predicted ceedal sets (aleatoric uncertainty)."""
    logits_dataset = load_logits(setting=setting, max_values=max_values)

    logits_train = logits_dataset.logits_train
    targets_train = logits_dataset.targets_train
    logits_test = logits_dataset.logits_test
    csets, rls = classwise_adding_optim_logit(logits_train, targets_train, logits_test, logits_dataset.n_classes)
    outputs = csets[rls >= alpha].swapaxes(0, 1)
    au = upper_entropy_clip(outputs, n_jobs=32)
    if descending:
        sorted_indices = np.argsort(au)[::-1]
    else:
        sorted_indices = np.argsort(au)
    print(
        f"Indices with {'highest' if descending else 'lowest'} AU for {setting} and max_values={len(au)}:",
        sorted_indices[:top_k], "AU values:", au[sorted_indices[:top_k]]
    )
    return sorted_indices[:top_k]


def eu_sort_predicted(setting: Setting, max_values: int | None = None, top_k: int = 20, alpha: float = 0.8, *,
                      descending: bool = True) -> list[int]:
    """Sorts the instances by the EU in their predicted ceedal sets (epistemic uncertainty)."""
    logits_dataset = load_logits(setting=setting, max_values=max_values)

    logits_train = logits_dataset.logits_train
    targets_train = logits_dataset.targets_train
    logits_test = logits_dataset.logits_test
    csets, rls = classwise_adding_optim_logit(logits_train, targets_train, logits_test, logits_dataset.n_classes)
    outputs = csets[rls >= alpha].swapaxes(0, 1)
    eu = upper_entropy_clip(outputs, n_jobs=32) - lower_entropy_clip(outputs, n_jobs=32)
    if descending:
        sorted_indices = np.argsort(eu)[::-1]
    else:
        sorted_indices = np.argsort(eu)
    print(
        f"Indices with {'highest' if descending else 'lowest'} EU for {setting} and max_values={len(eu)}:",
        sorted_indices[:top_k], "EU values:", eu[sorted_indices[:top_k]]
    )
    return sorted_indices[:top_k]


def _instance_db_clip(
        kind: Literal["au_predicted"] | Literal["eu_predicted"],
        dataset_name: DATASET_NAMES,
        label_set: LABEL_SETS,
        alpha: float,
) -> list[int]:
    if kind == "au_predicted" and alpha == 0.8 and dataset_name == "CIFAR10" and label_set == "CIFAR10":
        return [3070, 5626, 9936, 9346, 2520, 3412, 5250, 8322, 8376, 9621, 5424, 9613, 6475, 5759, 943, 1406, 5806,
                9688, 4977, 8526]
    if kind == "eu_predicted" and alpha == 0.8 and dataset_name == "CIFAR10" and label_set == "CIFAR10":
        return [2777, 5829, 5881, 1082, 1600, 2658, 655, 9544, 3596, 753, 6518, 4295, 4800, 6006, 384, 3483, 9391, 3056,
                7518, 7423]
    raise ValueError("Setting not covered.")


def _instance_db_siglip(
        kind: Literal["au_predicted"] | Literal["eu_predicted"],
        dataset_name: DATASET_NAMES,
        label_set: LABEL_SETS,
        alpha: float,
) -> list[int]:
    if kind == "au_predicted" and alpha == 0.8 and dataset_name == "CIFAR10" and label_set == "CIFAR10":
        return [6345, 2931, 587, 1095, 5224, 632, 3870, 8538, 8218, 3357, 5361, 9350, 7061, 7176, 8323, 2232, 9216,
                8220, 9255, 3970]
    if kind == "eu_predicted" and alpha == 0.8 and dataset_name == "CIFAR10" and label_set == "CIFAR10":
        return [3871, 689, 949, 9015, 8501, 8159, 8666, 7803, 4258, 6162, 4586, 3289, 428, 2788, 8679, 9054, 8431, 7453,
                1613, 2130]
    raise ValueError("Setting not covered.")


def _instance_db_siglip2(
        kind: Literal["au_predicted"] | Literal["eu_predicted"],
        dataset_name: DATASET_NAMES,
        label_set: LABEL_SETS,
        alpha: float,
) -> list[int]:
    if kind == "au_predicted" and alpha == 0.8 and dataset_name == "CIFAR10" and label_set == "CIFAR10":
        return [3494, 4652, 232, 5213, 7784, 4814, 7238, 2270, 2635, 1594, 4012, 6127, 3864, 8902, 4210, 8971, 8318,
                4730, 5733, 8840]
    if kind == "eu_predicted" and alpha == 0.8 and dataset_name == "CIFAR10" and label_set == "CIFAR10":
        return [2153, 7765, 3871, 5034, 9000, 8843, 4294, 2118, 2374, 8732, 6794, 242, 6432, 9015, 3698, 7193, 722, 496,
                8247, 6433]
    raise ValueError("Setting not covered.")


def _instance_db_biomedclip(
        kind: Literal["au_predicted"] | Literal["eu_predicted"],
        dataset_name: DATASET_NAMES,
        label_set: LABEL_SETS,
        alpha: float,
) -> list[int]:
    raise ValueError("Setting not covered.")


def instance_db(
        kind: Literal["au_labels"] | Literal["au_predicted"] | Literal["eu_predicted"],
        dataset_name: DATASET_NAMES,
        label_set: LABEL_SETS,
        model_name: MODEL_NAMES,
) -> list[int]:
    """Returns a DataFrame with the logits, probabilities, and target for the specified instance."""
    if kind == "au_target":
        if dataset_name == "CIFAR10" or dataset_name == "CIFAR10-H":
            return [6750, 8153, 6792, 86, 2232, 5840, 3463, 5369, 6197, 5227, 3391, 4821, 8855, 5734, 3357, 7238, 2855,
                    5837, 6024, 3113, 86, 357, 46, 313, 441, 68, 356, 982, 314, 59, 637, 237, 224, 127, 250, 910, 223,
                    426, 418, 966]
        raise ValueError("Setting not covered.")
    if model_name == "CLIP":
        return _instance_db_clip(kind=kind, dataset_name=dataset_name, label_set=label_set, alpha=alpha)
    if model_name == "SigLIP":
        return _instance_db_siglip(kind=kind, dataset_name=dataset_name, label_set=label_set, alpha=alpha)
    if model_name == "SigLIP2":
        return _instance_db_siglip2(kind=kind, dataset_name=dataset_name, label_set=label_set, alpha=alpha)
    if model_name == "BiomedCLIP":
        return _instance_db_biomedclip(kind=kind, dataset_name=dataset_name, label_set=label_set, alpha=alpha)
    raise ValueError("Setting not covered.")


def get_indices(kind: Literal["au_labels"] | Literal["au_predicted"] | Literal["eu_predicted"] | Literal["random"],
                setting: Setting, alpha: float, *, descending: bool = True, max_plots: int = 20) -> list[int]:
    if kind == "tu_predicted":
        return tu_sort_predicted(setting=setting, alpha=alpha, top_k=max_plots, descending=descending)
    if kind == "au_labels":
        return au_sort_labels(setting=setting, top_k=max_plots, descending=descending)
    if kind == "au_predicted":
        return au_sort_predicted(setting=setting, alpha=alpha, top_k=max_plots, descending=descending)
    if kind == "eu_predicted":
        return eu_sort_predicted(setting=setting, alpha=alpha, top_k=max_plots, descending=descending)
    if kind == "random":
        np.random.seed(42)
        n_rows = len(load_logits(setting=setting).logits_test)
        all_indices = np.arange(0, n_rows)
        np.random.shuffle(all_indices)
        return all_indices[:max_plots].tolist()
    raise ValueError("Invalid kind.")


if __name__ == '__main__':
    # settings for the experiments
    do_visualization_spider_plots = True
    do_sort_uncertainty = True
    do_visualization_credal_sets = False  # requires different settings

    # credal set settings:
    alpha = 0.9

    # model settings:
    model_names = ["CLIP", "SigLIP",
                   "SigLIP2"]  # ["CLIP"], ["SigLIP"], ["SigLIP2"], ["BiomedCLIP"], ["CLIP", "SigLIP", "SigLIP2"]

    # data settings:
    dataset_name = "CIFAR10"  # "CIFAR10", "DermaMNIST"
    label_set = "CIFAR10"  # "CIFAR10", "CIFAR10-CHINESE", "CIFAR10-FRENCH", "CIFAR10-SWAHILI"
    ground_truth_set = "CIFAR10-H"  # None or "CIFAR10-H"

    # sort / selection settings:
    kind = "tu_predicted"  # "tu_predicted", "au_labels", "au_predicted", "eu_predicted", "random"
    sort_descending = True  # whether to sort in descending (high uncertainty first) or ascending order (low uncertainty first)
    max_plots = 100
    max_idx = None  # only for visualize_credal_sets
    model_to_index = "CLIP"  # only for sorting: "CLIP", "SigLIP", "SigLIP2", "BiomedCLIP"

    # credal set settings:
    class_indices_spider = None
    class_indices_credal = (3, 5, 6)

    # instance ids
    instance_ids = get_indices(kind=kind, setting=Setting(dataset_name=dataset_name, label_set=label_set,
                                                          model_name=model_to_index), alpha=alpha,
                               descending=sort_descending)

    # plotting settings
    save = False
    plot_title = not save
    plot_legend = save

    if do_visualization_spider_plots:
        visualize_credal_sets_spider_plot(
            dataset_name=dataset_name,
            label_set=label_set,
            model_names=model_names,
            instance_ids=instance_ids,
            alpha=alpha,
            ground_truth_dataset=ground_truth_set,
            interval_thickness=0.2,
            fig_size=(4.9 * len(model_names), 4.75),
            plot_legend=plot_legend,
            plot_title=plot_title,
            save=save,
            save_folder=pathlib.Path("clip_figures") / "spider_plots",
            class_indices=class_indices_spider
        )

        if do_visualization_credal_sets:
            visualize_credal_sets(
                dataset_name=dataset_name,
                label_set=label_set,
                model_names=model_names,
                instance_ids=instance_ids,
                class_indices=class_indices_credal,
                alpha=alpha,
                save=save,
                save_folder=pathlib.Path("clip_figures") / "credal_sets"
            )

    if do_sort_uncertainty:
        model_name = model_to_index
        au_sort_labels(Setting(dataset_name="CIFAR10-H", label_set="CIFAR10", model_name=model_name))
        au_sort_predicted(Setting(dataset_name="CIFAR10", label_set="CIFAR10-CHINESE", model_name=model_name),
                          alpha=alpha)
        eu_sort_predicted(Setting(dataset_name="CIFAR10", label_set="CIFAR10-CHINESE", model_name=model_name),
                          alpha=alpha)

    if do_sort_uncertainty:
        model_name = "BiomedCLIP"
        au_sort_predicted(Setting(dataset_name="DermaMNIST", label_set="DermaMNIST", model_name=model_name),
                          alpha=alpha)
        eu_sort_predicted(Setting(dataset_name="DermaMNIST", label_set="DermaMNIST", model_name=model_name),
                          alpha=alpha)
