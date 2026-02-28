"""Script containing plotting functions."""
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # <-- use alias, avoid shadowing
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import pandas as pd

from colour import Color

from utils import _beautify_class_labels

SAVE_FOLDER = pathlib.Path("all_plots")
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER = "./results/"

# colors
NEUTRAL = Color("#ffffff").hex
COLOR_LINES = "lightgrey"
RED = Color("#ff0d57")
BLUE = Color("#1e88e5")
PURPLE = Color("#895cb5")
LINES = Color("#cccccc")

# colors for methods
MLE_COLOR = Color("#ff0d57").hex
MAIN_METHOD_COLOR = Color("#1e88e5").hex
MAIN_METHOD_COLOR_LIGHT = Color("#8ec3f2").hex
RL_BASELINE_COLOR = 'tab:green'
BASELINES_COLOR = 'tab:orange'

BASELINES_EDGECOLOR = 'white'
BASELINES_MARKERSIZE = 80
ALPHA_MARKER_SIZE = 90
MAIN_METHOD_LINE_WIDTH = 3.5

MARKER_OURS = 'o'
MARKER_RL_BASELINE = 'o'
MARKER_CREDALWRAPPER = '^'
MARKER_CREDALENSEMBLING = 'o'
MARKER_CREDALBNN = 's'
MARKER_CREDALNET = 'D'

OOD_MARKERS = {
    "SVHN": "o",
    "Places365": "s",
    "CIFAR-100": "D",
    "FMNIST": "v",
    "ImageNet": "P",
}

FONT_SIZE = 12


DATA_DICT = {'chaosnli': 'ChaosNLI',
             'cifar10': 'CIFAR-10H',
             'qualitymri': 'QualityMRI',
             "svhn": "SVHN",
             "places365": "Places365",
             "cifar100": "CIFAR-100",
             "fmnist": "Fashion-MNIST",
             "imagenet": "ImageNet", }
COLOR_DICT = {'ours': '#2ca02c', 'credalwrapper': '#2ca02c', 'credalensembling': '#2ca02c', 'credalbnn': '#2ca02c'}
RESULTS_DICT = {'ours': {}, 'credalwrapper': {}, 'credalensembling': {}, 'credalbnn': {}, 'credalnet': {}}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DATASETS = ['chaosnli', 'cifar10', 'qualitymri']
OOD_DATASETS = ['svhn', 'places365', 'cifar100', 'fmnist', 'imagenet']
SEED_DICT = {'chaosnli': [4, 5, 6], 'cifar10': [1, 2, 3], 'qualitymri': [1, 2, 3]}
CLASS_DICT = {'chaosnli': 3, 'cifar10': 10, 'qualitymri': 2}
NETWORK_DICT = {'chaosnli': 'fcnet', 'cifar10': 'resnet', 'qualitymri': 'torchresnet'}

tobias = 100
n_members = 20
METHODS = ['credalrl', 'credalwrapper', 'credalensembling', 'credalbnn', 'credalnet']
SINGLE_METHODS = ["evidential", "deterministic"]
cmap = plt.get_cmap('tab10')
edgecolor = 'tab:blue'
colors = [cmap(i / (n_members - 1)) for i in range(n_members)]
line_width = 2

s = 90
n = 10
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
marker_size = 80


def print_ood_table_results_baselines():
    members = 5
    seeds = SEED_DICT['cifar10']
    aurocs = np.empty((len(OOD_DATASETS), len(METHODS), len(seeds)))
    for i, dataset in enumerate(OOD_DATASETS):
        for j, method in enumerate(METHODS):
            for k, seed in enumerate(seeds):
                if method == 'credalensembling':
                    alpha = 0.0
                    auroc = np.load(
                        f'{RESULTS_FOLDER}ood_cifar10_{dataset}_{method}_{members}_{alpha}_{seed}.npy')
                elif method == 'credalrl':
                    alpha = 0.95
                    auroc = np.load(
                        f'{RESULTS_FOLDER}ood_cifar10_{dataset}_{method}_{members}_{alpha}_{seed}.npy')
                else:
                    auroc = np.load(f'{RESULTS_FOLDER}ood_cifar10_{dataset}_{method}_{members}_{seed}.npy')
                aurocs[i, j, k] = auroc
    aurocs_mu = np.mean(aurocs, axis=2)
    aurocs_std = np.std(aurocs, axis=2)
    output = ''
    for j, method in enumerate(METHODS):
        for i, dataset in enumerate(OOD_DATASETS):
            output += rf'& ${aurocs_mu[i, j]:.3f} \scriptstyle{{\pm {aurocs_std[i, j]:.3f}}}$ '
        output += r'\\ '
        print(output)
        output = ''
    print('\n')

def print_ood_table_results_single_baselines():
    seeds = SEED_DICT['cifar10']
    aurocs = np.empty((len(OOD_DATASETS), len(SINGLE_METHODS), len(seeds)))
    for i, dataset in enumerate(OOD_DATASETS):
        for j, method in enumerate(SINGLE_METHODS):
            for k, seed in enumerate(seeds):
                auroc = np.load(f'{RESULTS_FOLDER}ood_cifar10_{dataset}_{method}_{seed}.npy')
                aurocs[i, j, k] = auroc
    aurocs_mu = np.mean(aurocs, axis=2)
    aurocs_std = np.std(aurocs, axis=2)
    output = ''
    for j, method in enumerate(SINGLE_METHODS):
        for i, dataset in enumerate(OOD_DATASETS):
            output += rf'& ${aurocs_mu[i, j]:.3f} \scriptstyle{{\pm {aurocs_std[i, j]:.3f}}}$ '
        output += r'\\ '
        print(output)
        output = ''
    print('\n')


def print_ood_table_results():
    seeds_ood = SEED_DICT['cifar10']
    res = np.empty((len(seeds_ood), len(OOD_DATASETS), len(ALPHAS)))
    for i, dataset_ood in enumerate(OOD_DATASETS):
        for j, seed in enumerate(seeds_ood):
            res_ = np.load(
                f'{RESULTS_FOLDER}classwise_optim_logit_ood_cifar10_{dataset_ood}_resnet_{seed}.npy')
            res[j, i, :] = res_[1, :]
    for i, alpha in enumerate(ALPHAS):
        print(f'\nAlpha: {alpha} \n{OOD_DATASETS}')
        for j, dataset_ood in enumerate(OOD_DATASETS):
            print(rf'${res[:, j, i].mean(0):.3f} \scriptstyle{{\pm {res[:, j, i].std(0):.3f}}}$ & ', end="")
    print('\n')


def _apply_cov_eff_styling() -> None:
    plt.xlabel("Coverage", fontsize=FONT_SIZE)
    plt.ylabel("Efficiency", fontsize=FONT_SIZE)

    # add grid manually at all 0.2 intervals on x and y axis
    for i in np.arange(0, 1.1, 0.2):
        plt.axhline(y=i, color=COLOR_LINES, linestyle='dotted', linewidth=0.5, zorder=0)
        plt.axvline(x=i, color=COLOR_LINES, linestyle='dotted', linewidth=0.5, zorder=0)

    # add xticks and yticks at every 0.1 interval but labels only at 0.2 interval
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.gca().set_xticklabels(
        [f'{x:.1f}' if i % 2 == 0 else '' for i, x in enumerate(np.arange(0, 1.1, 0.1))])
    plt.gca().set_yticklabels(
        [f'{y:.1f}' if i % 2 == 0 else '' for i, y in enumerate(np.arange(0, 1.1, 0.1))])
    plt.tight_layout()


def plot_cov_efficiency(dataset, figsize, save: bool = True, plot_legend: bool = True) -> None:
    _ = plt.figure(figsize=figsize)

    plt.plot([], [], label='Ours', color='#ffffff')

    # #CLASS ADDING
    res = []
    base = NETWORK_DICT[dataset]
    for seed in SEED_DICT[dataset]:
        res.append(np.load(f'{RESULTS_FOLDER}classwise_optim_logit_cov_eff_{dataset}_{base}_{seed}.npy'))
    res = np.array(res)

    plt.plot(res[:, 1, :].mean(0), res[:, 2, :].mean(0), label='EffCre', color=MAIN_METHOD_COLOR, linewidth=3,
             zorder=80, marker=MARKER_OURS)
    plt.scatter(res[:, 1, :].mean(0), res[:, 2, :].mean(0), cmap='Greys', c=ALPHAS, marker='o', s=ALPHA_MARKER_SIZE,
                zorder=81, edgecolors=MAIN_METHOD_COLOR, linewidths=0.5)
    for x, y, xe, ye in zip(res[:, 1, :].mean(0), res[:, 2, :].mean(0), res[:, 1, :].std(0), res[:, 2, :].std(0)):
        plt.errorbar(x, y, xerr=xe, yerr=ye, fmt='none', ecolor='gray', elinewidth=1, capsize=3, zorder=15)

    # CRERL
    with open(f'{RESULTS_FOLDER}/{dataset}_result.pkl', 'rb') as f:
        results_dict = pickle.load(f)
    plt.plot([], [], label='Baselines', color='#ffffff')
    plt.plot([0], [1], color=colors[0])
    plt.plot(results_dict['ours']['cov'], results_dict['ours']['eff'], label='CreRL', color=RL_BASELINE_COLOR,
             linewidth=2, marker='o', zorder=78)
    plt.scatter(results_dict['ours']['cov'], results_dict['ours']['eff'], cmap='Greys', c=ALPHAS, marker='o',
                s=ALPHA_MARKER_SIZE, zorder=79, vmin=0, vmax=1, edgecolors=RL_BASELINE_COLOR, linewidths=0.5)
    for x, y, xe, ye in zip(results_dict['ours']['cov'], results_dict['ours']['eff'], results_dict['ours']['cov_std'],
                            results_dict['ours']['eff_std']):
        plt.errorbar(x, y, xerr=xe, yerr=ye, fmt='none', ecolor='gray', elinewidth=1, capsize=3, zorder=15)

    # CREDALENSEMBLING
    plt.plot(results_dict['credalensembling']['cov'], results_dict['credalensembling']['eff'], label='CreEns',
             color=BASELINES_COLOR, linewidth=2, marker=MARKER_CREDALENSEMBLING, zorder=4)
    plt.scatter(results_dict['credalensembling']['cov'], results_dict['credalensembling']['eff'], cmap='Greys',
                c=ALPHAS[:-1], marker='o', s=ALPHA_MARKER_SIZE, zorder=5, edgecolors=BASELINES_COLOR, linewidths=0.5)
    for x, y, xe, ye in zip(results_dict['credalensembling']['cov'], results_dict['credalensembling']['eff'],
                            results_dict['credalensembling']['cov_std'], results_dict['credalensembling']['eff_std']):
        plt.errorbar(x, y, xerr=xe, yerr=ye, fmt='none', ecolor='gray', elinewidth=1, capsize=3, zorder=1)

    # CREDALWRAPPER
    plt.scatter(results_dict['credalwrapper']['cov'], results_dict['credalwrapper']['eff'], label='CreWra',
                color=BASELINES_COLOR, edgecolors=BASELINES_EDGECOLOR, linewidths=0.5, s=BASELINES_MARKERSIZE,
                marker=MARKER_CREDALWRAPPER, zorder=99)
    plt.errorbar(results_dict['credalwrapper']['cov'], results_dict['credalwrapper']['eff'],
                 xerr=results_dict['credalwrapper']['cov_std'], yerr=results_dict['credalwrapper']['eff_std'],
                 fmt='none', ecolor='gray', elinewidth=1, capsize=3, zorder=1)

    # CREDALBNN
    plt.scatter(results_dict['credalbnn']['cov'], results_dict['credalbnn']['eff'], label='CreBNN',
                color=BASELINES_COLOR, edgecolors=BASELINES_EDGECOLOR, linewidths=0.5, s=BASELINES_MARKERSIZE,
                marker=MARKER_CREDALBNN, zorder=99)
    plt.errorbar(results_dict['credalbnn']['cov'], results_dict['credalbnn']['eff'],
                 xerr=results_dict['credalbnn']['cov_std'], yerr=results_dict['credalbnn']['eff_std'], fmt='none',
                 ecolor='gray', elinewidth=1, capsize=3, zorder=1)

    # CREDALNET
    plt.scatter(results_dict['credalnet']['cov'], results_dict['credalnet']['eff'], label='CreNet',
                color=BASELINES_COLOR, edgecolors=BASELINES_EDGECOLOR, linewidths=0.5, s=BASELINES_MARKERSIZE,
                marker=MARKER_CREDALNET, zorder=99)
    plt.errorbar(results_dict['credalnet']['cov'], results_dict['credalnet']['eff'],
                 xerr=results_dict['credalnet']['cov_std'], yerr=results_dict['credalnet']['eff_std'], fmt='none',
                 ecolor='gray', elinewidth=1, capsize=3, zorder=1)

    if plot_legend:
        legend = plt.legend(loc="lower left")
        legend.get_texts()[0].set_fontweight('bold')
        legend.get_texts()[2].set_fontweight('bold')
    plt.title(f'{DATA_DICT[dataset]}', fontsize=FONT_SIZE, weight='bold')

    _apply_cov_eff_styling()

    if save:
        SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
        plt.savefig(SAVE_FOLDER / f'cov_eff_{dataset}.pdf', bbox_inches='tight')
    plt.show()


def plot_cov_efficiency_no_baseline(
        *,
        figsize=None,  # e.g., COV_EFF_FIGSIZE
        series=None,  # list of dicts (see below)
        seeds=(1, 2, 3),
        base_color=None,  # e.g., MAIN_METHOD_COLOR ("#rrggbb" or (r,g,b))
        use_shades=True,  # generate shades per ID from base_color
        shade_min=0.3,  # [0..1], lower = more toward white/black
        shade_max=1.0,  # [0..1], 1 keeps original
        lighten=True,  # True=blend toward white, False=toward black
        title=None,  # fallback: from series if length==1
        legend_title="Datasets",
        save=True,
):
    """
    General coverage–efficiency plotter. All curves are computed as mean±std over seeds.

    Each item in `series` is a dict with:
      - 'pattern' (str): file path pattern with {seed} and optionally {id}, e.g.
            f"{RESULT_FOLDER}/classwise_optim_logit_cov_eff_{id}_{base}_{seed}.npy"
        or without {id} for single-curve series:
            f"{RESULT_FOLDER}/classwise_optim_logit_cov_eff_cifar10_siglip2_{seed}.npy"
      - 'ids' (iterable|None): if provided, plots one curve per ID (substitute {id}); labels default to str(id)
      - 'label' (str|callable|None): string label for single-curve series, or callable(id)->str for multi-curve
      - 'color' (str|tuple|None): explicit color for this series (overrides shades); for multi-curve, you can pass
                                  a list of colors matching ids.

    Global variables used if present in your module:
      RESULT_FOLDER, MARKER_OURS, alphas, edgecolor, _apply_cov_eff_styling
    """
    # --- safe fallbacks for globals used in your environment ---
    marker = globals().get("MARKER_OURS", "o")
    alphas = globals().get("alphas", None)
    edgecolor = globals().get("edgecolor", "black")

    if figsize is None:
        figsize = (6, 4)

    if series is None:
        series = []

    def _to_rgb(color_in):
        if isinstance(color_in, str):
            return mcolors.to_rgb(color_in)
        if isinstance(color_in, (list, tuple)) and len(color_in) == 3:
            c = tuple(float(x) for x in color_in)
            if max(c) > 1.0:
                c = tuple(x / 255.0 for x in c)
            return c
        raise ValueError("base_color must be hex string or RGB tuple/list of length 3")

    def _generate_shades(color_in, n, *, lighten, t_min, t_max):
        base_rgb = _to_rgb(color_in)
        if n <= 1:
            return [mcolors.to_hex(base_rgb)]
        shades = []
        for i in range(n):
            t = t_min + (t_max - t_min) * (i / (n - 1))
            if lighten:
                new_rgb = tuple(1.0 - (1.0 - c) * t for c in base_rgb)  # toward white
            else:
                new_rgb = tuple(c * t for c in base_rgb)  # toward black
            shades.append(mcolors.to_hex(new_rgb))
        return shades

    def _load_stack(pattern, ids, seeds):
        """
        Returns: list of np.ndarray with shape (n_seeds, 3, n_points),
        one entry per id (or a singleton list for id=None).
        """
        id_list = [None] if ids is None else list(ids)
        stacks = []
        for _id in id_list:
            arrs = []
            for s in seeds:
                path = pattern.format(seed=s, id=_id)
                arrs.append(np.load(path))
            stacks.append(np.asarray(arrs))
        return id_list, stacks

    def _plot_one(res, label, color):
        x_mean = res[:, 1, :].mean(axis=0)
        y_mean = res[:, 2, :].mean(axis=0)
        x_std = res[:, 1, :].std(axis=0)
        y_std = res[:, 2, :].std(axis=0)

        plt.plot(x_mean, y_mean, label=label, color=color, linewidth=MAIN_METHOD_LINE_WIDTH, marker=marker)
        plt.scatter(x_mean, y_mean, c=ALPHAS, cmap="Greys", marker="o",
                    s=ALPHA_MARKER_SIZE, zorder=99, edgecolors=edgecolor, linewidths=0.5)
        for x, y, xe, ye in zip(x_mean, y_mean, x_std, y_std):
            plt.errorbar(x, y, xerr=xe, yerr=ye, fmt='none',
                         ecolor='gray', elinewidth=1, capsize=3, zorder=1)

    # --- figure ---
    plt.figure(figsize=figsize)

    # guard + normalize shade params
    shade_min = float(shade_min)
    shade_max = float(shade_max)
    assert 0.0 <= shade_min <= 1.0 and 0.0 <= shade_max <= 1.0 and shade_min <= shade_max

    # If a single base_color is given and use_shades=True, we assign shades per *ID* within each series.
    # If series[i]['color'] is provided, it overrides the shade for that series (or per id if list).
    for ser in series:
        pattern = ser["pattern"]
        ids = ser.get("ids", None)
        label_spec = ser.get("label", None)
        explicit_color = ser.get("color", None)

        id_list, stacks = _load_stack(pattern, ids, seeds)

        # build labels
        if callable(label_spec):
            labels = [label_spec(_id) for _id in id_list]
        elif isinstance(label_spec, str) and ids is None:
            labels = [label_spec]
        else:
            labels = [str(_id) if _id is not None else (label_spec or "Series") for _id in id_list]

        # build colors
        if explicit_color is not None:
            if isinstance(explicit_color, (list, tuple)) and ids is not None and len(explicit_color) == len(id_list):
                colors_for_ids = list(explicit_color)
            else:
                colors_for_ids = [explicit_color] * len(id_list)
        elif base_color is not None and use_shades:
            colors_for_ids = _generate_shades(base_color, len(id_list),
                                              lighten=lighten, t_min=shade_min, t_max=shade_max)
        else:
            # fall back to Matplotlib cycle
            colors_for_ids = [None] * len(id_list)

        # plot
        for label, color, res in zip(labels, colors_for_ids, stacks):
            _plot_one(res, label, color)

    # colorbar for alpha scale (uses global `alphas`)
    # labels / title
    plt.xlabel('Coverage')
    plt.ylabel('Efficiency')
    if title is None and len(series) == 1:
        # try to extract something nice from single-series patterns
        title = ser.get("title", None)
    plt.title(title or "Coverage–Efficiency", fontsize=FONT_SIZE, weight='bold')

    # legend + styling
    ncol = 2
    if len(series[0].get("ids")) <= 3:
        ncol = 1
    plt.legend(title=legend_title, loc="lower left", ncol=ncol)
    _apply_cov_eff_styling()  # optional, if you have it

    if save:
        SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
        plt.savefig(SAVE_FOLDER / f'cov_eff_{title.replace(" ", "_")}.pdf', bbox_inches='tight')
    plt.show()


def plot_active_learning(dataset_id, figsize, save: bool = True, ylim=(0.5, 1), main_paper: bool = False) -> None:
    import scipy.stats as ss

    fig = plt.figure(figsize=figsize)

    dataset = dataset_id  # 46941 46941 46963 46937 46930
    alpha = 0.8
    # seeds = [1,2,3,4,5,6,7,8,9,10]
    seeds = [1, 2, 3]
    pred = "mle"

    ress_entropy = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_entropy_{alpha}_{pred}_{seed}.npy')
        ress_entropy.append(res)
    ress_entropy = np.array(ress_entropy)
    plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), color="white",
             linewidth=MAIN_METHOD_LINE_WIDTH, zorder=19)
    plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), label='EffCre (entropy)',
             color=MAIN_METHOD_COLOR, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25, zorder=20)
    plt.fill_between(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0) - ss.sem(ress_entropy[:, :, 0], 0),
                     ress_entropy[:, :, 0].mean(0) + ss.sem(ress_entropy[:, :, 0], 0), alpha=0.2,
                     color=MAIN_METHOD_COLOR, zorder=1)

    ress_zero_one = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_zero_one_{alpha}_{pred}_{seed}.npy')
        ress_zero_one.append(res)
    ress_zero_one = np.array(ress_zero_one)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), color="white", zorder=9,
             linewidth=MAIN_METHOD_LINE_WIDTH)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), label='EffCre (zero-one)',
             color=MAIN_METHOD_COLOR_LIGHT, zorder=10, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25)
    plt.fill_between(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0) - ss.sem(ress_zero_one[:, :, 0], 0),
                     ress_zero_one[:, :, 0].mean(0) + ss.sem(ress_zero_one[:, :, 0], 0), alpha=0.2,
                     color=MAIN_METHOD_COLOR_LIGHT, zorder=1)

    ress_random = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_random_{seed}.npy')
        ress_random.append(res)
    ress_random = np.array(ress_random)
    plt.plot(ress_random[:, :, 1].mean(0), ress_random[:, :, 0].mean(0), color="white",
             linewidth=MAIN_METHOD_LINE_WIDTH - 1, zorder=1)
    plt.plot(ress_random[:, :, 1].mean(0), ress_random[:, :, 0].mean(0), label='MLE (random)', color=BASELINES_COLOR,
             linewidth=MAIN_METHOD_LINE_WIDTH - 2.25, zorder=2)
    plt.fill_between(ress_random[:, :, 1].mean(0), ress_random[:, :, 0].mean(0) - ss.sem(ress_random[:, :, 0], 0),
                     ress_random[:, :, 0].mean(0) + ss.sem(ress_random[:, :, 0], 0), alpha=0.2, color=BASELINES_COLOR,
                     zorder=0)

    # add grid normally
    plt.grid(color=COLOR_LINES, linestyle='dotted', linewidth=0.5, zorder=0)

    plt.ylim(ylim)
    plt.xlabel("Number of Instances", fontsize=FONT_SIZE)
    plt.ylabel("Accuracy", fontsize=FONT_SIZE)
    title = f"Active In-Context Learning (ID = {dataset})"
    weight = 'bold' if main_paper else 'normal'
    plt.title(title, fontsize=FONT_SIZE, weight=weight)
    plt.legend()
    plt.tight_layout()
    if save:
        SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
        paper_flag = ""
        if main_paper:
            paper_flag = "_mainpaper"
        plt.savefig(SAVE_FOLDER / f'cov_eff_{title.replace(" ", "_")}{paper_flag}.pdf', bbox_inches='tight')
    plt.show()


def plot_active_learning_ablation(dataset_id, figsize, save: bool = True, ylim=(0.5, 1),
                                  main_paper: bool = False) -> None:
    import scipy.stats as ss

    def generate_color_shades(base_color, num_colors=4, lighten=True, shade_min=0.3, shade_max=1.0):
        """
        Generate _num_colors_ colors starting from a base color and creating a gradient
        toward white (if lighten=True) or black (if lighten=False).

        Parameters
        ----------
        base_color : str | tuple
            Hex string "#rrggbb" or RGB tuple/list in [0,1] or [0,255].
        num_colors : int, optional
            Number of colors to generate. Default is 4.
        lighten : bool
            True -> blend toward white, False -> toward black.
        shade_min : float
            Minimum blend factor (0..1). Lower = closer to white/black.
        shade_max : float
            Maximum blend factor (0..1). 1 keeps the original color.

        Returns
        -------
        list of str
            4 hex colors (e.g. ["#aabbcc", ...])
        """

        # helper to convert to RGB in [0,1]
        def _to_rgb(color_in):
            if isinstance(color_in, str):
                return mcolors.to_rgb(color_in)
            if isinstance(color_in, (list, tuple)) and len(color_in) == 3:
                c = tuple(float(x) for x in color_in)
                if max(c) > 1.0:  # probably 0–255
                    c = tuple(x / 255.0 for x in c)
                return c
            raise ValueError("base_color must be hex string or RGB tuple/list of length 3")

        base_rgb = _to_rgb(base_color)

        # compute 4 evenly spaced factors between shade_min and shade_max
        colors = []
        for i in range(num_colors):
            t = shade_min + (shade_max - shade_min) * (i / (num_colors - 1))
            if lighten:
                new_rgb = tuple(1.0 - (1.0 - c) * t for c in base_rgb)  # toward white
            else:
                new_rgb = tuple(c * t for c in base_rgb)  # toward black
            colors.append(mcolors.to_hex(new_rgb))

        return colors

    colors = generate_color_shades(MAIN_METHOD_COLOR, num_colors=6, lighten=True, shade_min=0.3, shade_max=1.0)
    # colors = colors[::-1]  # dark to light

    fig = plt.figure(figsize=figsize)

    dataset = dataset_id  # 46941 46963 46930
    # alphas = [0.6, 0.8, 0.9, 0.95]
    seeds = [1, 2, 3]
    pred = "mle"

    # ress_entropy = []
    # for seed in seeds:
    #     res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_entropy_0.6_{pred}_{seed}.npy')
    #     ress_entropy.append(res)
    # ress_entropy = np.array(ress_entropy)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), color="white", linewidth=MAIN_METHOD_LINE_WIDTH, zorder=19)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), label=r'EffCre ($\alpha=0.6$)', color=MAIN_METHOD_COLOR,linewidth=MAIN_METHOD_LINE_WIDTH-1.25, zorder=20)
    # plt.fill_between(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0) - ss.sem(ress_entropy[:, :, 0], 0), ress_entropy[:, :, 0].mean(0) + ss.sem(ress_entropy[:, :, 0], 0), alpha=0.2, color=MAIN_METHOD_COLOR, zorder=1)
    #
    # ress_entropy = []
    # for seed in seeds:
    #     res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_entropy_0.8_{pred}_{seed}.npy')
    #     ress_entropy.append(res)
    # ress_entropy = np.array(ress_entropy)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), color="white", linewidth=MAIN_METHOD_LINE_WIDTH, zorder=19)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), label='EffCre (entropy)', color=MAIN_METHOD_COLOR,linewidth=MAIN_METHOD_LINE_WIDTH-1.25, zorder=20)
    # plt.fill_between(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0) - ss.sem(ress_entropy[:, :, 0], 0), ress_entropy[:, :, 0].mean(0) + ss.sem(ress_entropy[:, :, 0], 0), alpha=0.2, color=MAIN_METHOD_COLOR, zorder=1)
    #
    # ress_entropy = []
    # for seed in seeds:
    #     res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_entropy_0.9_{pred}_{seed}.npy')
    #     ress_entropy.append(res)
    # ress_entropy = np.array(ress_entropy)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), color="white", linewidth=MAIN_METHOD_LINE_WIDTH, zorder=19)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), label='EffCre (entropy)', color=MAIN_METHOD_COLOR,linewidth=MAIN_METHOD_LINE_WIDTH-1.25, zorder=20)
    # plt.fill_between(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0) - ss.sem(ress_entropy[:, :, 0], 0), ress_entropy[:, :, 0].mean(0) + ss.sem(ress_entropy[:, :, 0], 0), alpha=0.2, color=MAIN_METHOD_COLOR, zorder=1)
    #
    # ress_entropy = []
    # for seed in seeds:
    #     res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_entropy_0.95_{pred}_{seed}.npy')
    #     ress_entropy.append(res)
    # ress_entropy = np.array(ress_entropy)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), color="white", linewidth=MAIN_METHOD_LINE_WIDTH, zorder=19)
    # plt.plot(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0), label='EffCre (entropy)', color=MAIN_METHOD_COLOR,linewidth=MAIN_METHOD_LINE_WIDTH-1.25, zorder=20)
    # plt.fill_between(ress_entropy[:, :, 1].mean(0), ress_entropy[:, :, 0].mean(0) - ss.sem(ress_entropy[:, :, 0], 0), ress_entropy[:, :, 0].mean(0) + ss.sem(ress_entropy[:, :, 0], 0), alpha=0.2, color=MAIN_METHOD_COLOR, zorder=1)

    ress_zero_one = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}al_{dataset}_zero_one_0.2_{pred}_{seed}.npy')
        ress_zero_one.append(res)
    ress_zero_one = np.array(ress_zero_one)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), color="white", zorder=9,
             linewidth=MAIN_METHOD_LINE_WIDTH)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), label=r'EffCre ($\alpha=0.2$)',
             color=colors[0], zorder=10, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25)
    plt.fill_between(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0) - ss.sem(ress_zero_one[:, :, 0], 0),
                     ress_zero_one[:, :, 0].mean(0) + ss.sem(ress_zero_one[:, :, 0], 0), alpha=0.2,
                     color=colors[0], zorder=1)

    ress_zero_one = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_zero_one_0.4_{pred}_{seed}.npy')
        ress_zero_one.append(res)
    ress_zero_one = np.array(ress_zero_one)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), color="white", zorder=9,
             linewidth=MAIN_METHOD_LINE_WIDTH)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), label=r'EffCre ($\alpha=0.4$)',
             color=colors[1], zorder=10, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25)
    plt.fill_between(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0) - ss.sem(ress_zero_one[:, :, 0], 0),
                     ress_zero_one[:, :, 0].mean(0) + ss.sem(ress_zero_one[:, :, 0], 0), alpha=0.2,
                     color=colors[1], zorder=1)

    ress_zero_one = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_zero_one_0.6_{pred}_{seed}.npy')
        ress_zero_one.append(res)
    ress_zero_one = np.array(ress_zero_one)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), color="white", zorder=9,
             linewidth=MAIN_METHOD_LINE_WIDTH)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), label=r'EffCre ($\alpha=0.6$)',
             color=colors[2], zorder=10, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25)
    plt.fill_between(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0) - ss.sem(ress_zero_one[:, :, 0], 0),
                     ress_zero_one[:, :, 0].mean(0) + ss.sem(ress_zero_one[:, :, 0], 0), alpha=0.2,
                     color=colors[1], zorder=1)

    ress_zero_one = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_zero_one_0.8_{pred}_{seed}.npy')
        ress_zero_one.append(res)
    ress_zero_one = np.array(ress_zero_one)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), color="white", zorder=9,
             linewidth=MAIN_METHOD_LINE_WIDTH)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), label=r'EffCre ($\alpha=0.8$)',
             color=colors[2], zorder=10, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25)
    plt.fill_between(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0) - ss.sem(ress_zero_one[:, :, 0], 0),
                     ress_zero_one[:, :, 0].mean(0) + ss.sem(ress_zero_one[:, :, 0], 0), alpha=0.2,
                     color=colors[2], zorder=1)

    ress_zero_one = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_zero_one_0.9_{pred}_{seed}.npy')
        ress_zero_one.append(res)
    ress_zero_one = np.array(ress_zero_one)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), color="white", zorder=9,
             linewidth=MAIN_METHOD_LINE_WIDTH)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), label=r'EffCre ($\alpha=0.9$)',
             color=colors[3], zorder=10, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25)
    plt.fill_between(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0) - ss.sem(ress_zero_one[:, :, 0], 0),
                     ress_zero_one[:, :, 0].mean(0) + ss.sem(ress_zero_one[:, :, 0], 0), alpha=0.2,
                     color=colors[3], zorder=1)

    ress_zero_one = []
    for seed in seeds:
        res = np.load(f'{RESULTS_FOLDER}/al_{dataset}_zero_one_0.95_{pred}_{seed}.npy')
        ress_zero_one.append(res)
    ress_zero_one = np.array(ress_zero_one)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), color="white", zorder=9,
             linewidth=MAIN_METHOD_LINE_WIDTH)
    plt.plot(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0), label=r'EffCre ($\alpha=0.95$)',
             color=colors[4], zorder=10, linewidth=MAIN_METHOD_LINE_WIDTH - 1.25)
    plt.fill_between(ress_zero_one[:, :, 1].mean(0), ress_zero_one[:, :, 0].mean(0) - ss.sem(ress_zero_one[:, :, 0], 0),
                     ress_zero_one[:, :, 0].mean(0) + ss.sem(ress_zero_one[:, :, 0], 0), alpha=0.2,
                     color=colors[4], zorder=1)

    # add grid normally
    plt.grid(color=COLOR_LINES, linestyle='dotted', linewidth=0.5, zorder=0)

    plt.ylim(ylim)
    plt.xlabel("Number of Instances", fontsize=FONT_SIZE)
    plt.ylabel("Accuracy", fontsize=FONT_SIZE)
    title = f"Active In-Context Learning (ID = {dataset})"
    weight = 'bold' if main_paper else 'normal'
    plt.title(title, fontsize=FONT_SIZE, weight=weight)
    plt.legend()
    plt.tight_layout()
    if save:
        SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
        paper_flag = ""
        if main_paper:
            paper_flag = "_mainpaper"
        plt.savefig(SAVE_FOLDER / f'cov_eff_ablation_{title.replace(" ", "_")}{paper_flag}.pdf', bbox_inches='tight')
    plt.show()


def plot_alpha_colorbar(figsize) -> None:
    """Plots a standalone polt as a colorbar for the alpha values."""
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_visible(False)
    axins = inset_axes(ax, width="5%", height="100%", loc='center')
    cmap = plt.get_cmap('Greys')
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=axins, orientation='vertical')
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_ticklabels([f'{x:.1f}' for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(0.5)
    cb.set_label(r'$\alpha$', rotation=0, labelpad=15, weight='bold')
    plt.savefig(SAVE_FOLDER / 'alpha_colorbar.pdf')
    plt.show()


def plot_ood_runtime(figsize) -> None:
    def _apply_ood_runtime_styling() -> None:
        plt.xlabel("Runtime in hours (log) and models trained", fontsize=FONT_SIZE)
        plt.ylabel("AUROC", fontsize=FONT_SIZE)

        # add grid manually
        plt.axhline(0.8, alpha=1, linestyle="--", linewidth=1, color=COLOR_LINES, zorder=0)
        plt.axvline(0.5, alpha=1, linestyle="--", linewidth=3, zorder=0, color=COLOR_LINES)
        plt.axvline(5.93, alpha=1, linestyle="--", linewidth=3, zorder=0, color=COLOR_LINES)

        plt.ylim([0.5, 1])
        # plt.xlim([0, 9])
        plt.xscale('log')
        # add xticks and yticks
        x_ticks = [0.5, 1.1, 2, 3, 4, 5, 6, 7, 8]
        plt.xticks(x_ticks, [f"{val}h" for val in range(0, 9)], rotation=0)
        plt.gca().xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
        plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
        plt.xlabel("Runtime in hours (log) and models trained", fontsize=FONT_SIZE)
        plt.ylabel("AUROC", fontsize=FONT_SIZE)
        # add grid only to x-axis
        plt.grid(axis='x', color=COLOR_LINES, linestyle='dotted', linewidth=0.5, zorder=0)

        plt.title("Out-of-Distribution Detection", fontsize=FONT_SIZE, weight="bold")
        plt.tight_layout()

    _ = plt.figure(figsize=figsize)
    ax = plt.gca()

    method_colors = [Color("#1e88e5").hex, 'tab:green', 'tab:orange', 'tab:orange',
                     'tab:orange', 'tab:orange', 'tab:orange']
    datasets = ["SVHN", "Places365", "CIFAR-100", "FMNIST", "ImageNet"]

    # load data
    df = pd.read_csv(f"{RESULTS_FOLDER}/ood_runtime.csv")
    # transform to hours
    df["runtime_mean"] = df["runtime_mean"] / 3600
    df["runtime_std"] = df["runtime_std"] / 3600

    df.loc[0, 'runtime_mean'] = 0.5
    df.loc[0, 'runtime_std'] = 0

    # --- Scatter plots ---
    for dataset in datasets:
        plt.scatter(
            df["runtime_mean"],
            df[f"{dataset}_mean"],
            color="darkgrey",
            marker=OOD_MARKERS[dataset],
            s=80,
            alpha=1,
            zorder=10,
            edgecolors="black"
        )

    # --- Bounding boxes ---
    x_pads = [0.044, 0.3, 0.45, 0.55, 0.66, 0.55]
    y_pads = [0.02] * len(df)

    for i, method in enumerate(df["Method"]):
        if i == 2:  # skip method 2
            continue

        xs = [df.loc[i, "runtime_mean"]] * len(datasets)
        ys = [df.loc[i, f"{ds}_mean"] for ds in datasets if not pd.isna(df.loc[i, f"{ds}_mean"])]

        if ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            rect = patches.Rectangle(
                (x_min - x_pads[i] + 1e-5, y_min - y_pads[i]),
                (x_max - x_min) + 2 * x_pads[i],
                (y_max - y_min) + 2 * y_pads[i],
                linewidth=1,
                edgecolor="gray",
                facecolor=method_colors[i % len(method_colors)],
                alpha=0.8,
                linestyle="--",
                zorder=2
            )
            ax.add_patch(rect)

    # --- Annotations ---
    offsets = {
        0: (-0.01, 0.10),  # EffCre
        1: (-1.5, 0.09),  # CreRL
        2: (-2.14, -0.06),  # CreWra
        3: (-2.14, -0.08),  # CreEns
        4: (-2.1, -0.057),  # CreBNN
        5: (-0.7, 0.087),  # CreNet
    }

    for i, method in enumerate(df["Method"]):
        if i not in offsets:
            continue
        dx, dy = offsets[i]
        x_val = df.loc[i, "runtime_mean"] + dx
        y_val = df.loc[i, f"{datasets[-1]}_mean"] + dy  # last dataset only
        plt.annotate(
            f"${method}$",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=FONT_SIZE
        )

    # annotate model number
    plt.annotate(f"1 model\ntrained", (df.loc[0, "runtime_mean"] + 0.15, 0.7), textcoords="offset points",
                 xytext=(5, 5), fontsize=FONT_SIZE,
                 color='black', alpha=0.5, fontweight='bold', horizontalalignment='center')
    plt.annotate(f"10 models\ntrained", (df.loc[3, "runtime_mean"] - 2.2, 0.7), textcoords="offset points",
                 xytext=(5, 5), fontsize=FONT_SIZE,
                 color='black', alpha=0.5, fontweight='bold', horizontalalignment='center')

    # Legends: one for datasets (colors), one for methods (markers)
    dataset_legend = [
        Line2D(
            [0], [0],
            marker=OOD_MARKERS[ds],
            color='w',
            label=ds,
            markerfacecolor='darkgrey',
            markersize=9,
            markeredgecolor='k'
        )
        for ds in datasets
    ]

    leg = plt.legend(
        handles=dataset_legend,
        loc='lower left',
        fontsize=FONT_SIZE,
        title="Datasets",
        title_fontsize=FONT_SIZE,
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.1  # 👈 two-column layout
    )
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"

    # plot grid normally
    _apply_ood_runtime_styling()
    plt.savefig(SAVE_FOLDER / 'ood_runtime.pdf', bbox_inches='tight')
    plt.show()


def plot_ood_vs_ensemble_size(dataset="svhn"):
    # Hardcoded results from LaTeX table: AUROC vs ensemble size
    results = {
        "EffCre_0.95": {
            "svhn": {1: (0.885, 0.003)},
            "places365": {1: (0.862, 0.005)},
            "cifar100": {1: (0.854, 0.003)},
            "fmnist": {1: (0.907, 0.002)},
            "imagenet": {1: (0.826, 0.004)},
        },
        "CreRL_0.95": {
            "svhn": {2: (0.892, 0.023), 5: (0.917, 0.012), 10: (0.921, 0.010), 20: (0.917, 0.013)},
            "places365": {2: (0.860, 0.006), 5: (0.894, 0.002), 10: (0.905, 0.002), 20: (0.910, 0.001)},
            "cifar100": {2: (0.855, 0.002), 5: (0.885, 0.002), 10: (0.896, 0.001), 20: (0.901, 0.000)},
            "fmnist": {2: (0.900, 0.004), 5: (0.928, 0.004), 10: (0.940, 0.002), 20: (0.945, 0.004)},
            "imagenet": {2: (0.832, 0.001), 5: (0.863, 0.002), 10: (0.872, 0.002), 20: (0.878, 0.002)},
        },
        "CreWra": {
            "svhn": {2: (0.922, 0.008), 5: (0.943, 0.006), 10: (0.953, 0.004), 20: (0.957, 0.003)},
            "places365": {2: (0.879, 0.001), 5: (0.904, 0.001), 10: (0.911, 0.001), 20: (0.916, 0.001)},
            "cifar100": {2: (0.880, 0.001), 5: (0.905, 0.001), 10: (0.912, 0.000), 20: (0.916, 0.000)},
            "fmnist": {2: (0.912, 0.001), 5: (0.939, 0.001), 10: (0.948, 0.001), 20: (0.952, 0.000)},
            "imagenet": {2: (0.855, 0.001), 5: (0.879, 0.001), 10: (0.886, 0.001), 20: (0.890, 0.001)},
        },
        "CreEns_0.0": {
            "svhn": {2: (0.901, 0.007), 5: (0.938, 0.007), 10: (0.949, 0.001), 20: (0.955, 0.001)},
            "places365": {2: (0.872, 0.002), 5: (0.898, 0.001), 10: (0.907, 0.001), 20: (0.913, 0.000)},
            "cifar100": {2: (0.872, 0.001), 5: (0.900, 0.001), 10: (0.909, 0.002), 20: (0.914, 0.001)},
            "fmnist": {2: (0.904, 0.004), 5: (0.929, 0.001), 10: (0.941, 0.002), 20: (0.949, 0.001)},
            "imagenet": {2: (0.848, 0.003), 5: (0.874, 0.001), 10: (0.883, 0.001), 20: (0.888, 0.000)},
        },
        "CreBNN": {
            "svhn": {2: (0.795, 0.008), 5: (0.843, 0.006), 10: (0.880, 0.009), 20: (0.907, 0.006)},
            "places365": {2: (0.764, 0.005), 5: (0.829, 0.006), 10: (0.856, 0.002), 20: (0.885, 0.002)},
            "cifar100": {2: (0.763, 0.004), 5: (0.831, 0.007), 10: (0.859, 0.002), 20: (0.880, 0.002)},
            "fmnist": {2: (0.812, 0.008), 5: (0.851, 0.007), 10: (0.886, 0.001), 20: (0.935, 0.002)},
            "imagenet": {2: (0.747, 0.004), 5: (0.809, 0.007), 10: (0.838, 0.001), 20: (0.859, 0.002)},
        },
        "CreNet": {
            "svhn": {2: (0.929, 0.007), 5: (0.938, 0.003), 10: (0.944, 0.001), 20: (0.943, 0.003)},
            "places365": {2: (0.888, 0.003), 5: (0.908, 0.001), 10: (0.915, 0.001), 20: (0.918, 0.000)},
            "cifar100": {2: (0.876, 0.002), 5: (0.900, 0.001), 10: (0.908, 0.001), 20: (0.912, 0.000)},
            "fmnist": {2: (0.925, 0.007), 5: (0.941, 0.003), 10: (0.949, 0.001), 20: (0.951, 0.002)},
            "imagenet": {2: (0.849, 0.002), 5: (0.871, 0.002), 10: (0.881, 0.001), 20: (0.884, 0.001)},
        }
    }
    plt.figure(figsize=(8, 4))
    for method, ds_vals in results.items():
        if dataset not in ds_vals:
            continue
        members = sorted(ds_vals[dataset].keys())
        means = [ds_vals[dataset][m][0] for m in members]
        stds = [ds_vals[dataset][m][1] for m in members]
        size = 10
        if method == "EffCre_0.95":
            plt.errorbar(members, means, yerr=stds, marker=MARKER_OURS, capsize=4, label=r'EffCre$_{{0.95}}$',
                         color=MAIN_METHOD_COLOR, markeredgecolor='k', markersize=size)
        elif method == "CreRL_0.95":
            plt.errorbar(members[1:], means[1:], yerr=stds[1:], marker=MARKER_RL_BASELINE, capsize=4,
                         label=r'CreRL$_{{0.95}}$', color=RL_BASELINE_COLOR, markeredgecolor='k', markersize=size)
        elif method == "CreEns_0.0":
            plt.errorbar(members[1:], means[1:], yerr=stds[1:], marker=MARKER_CREDALENSEMBLING, capsize=4,
                         label=r'CreEns$_{{0}}$', color=BASELINES_COLOR, markeredgecolor='k', markersize=size)
        elif method == "CreBNN":
            plt.errorbar(members[1:], means[1:], yerr=stds[1:], marker=MARKER_CREDALBNN, capsize=4, label=method,
                         color=BASELINES_COLOR, markeredgecolor='k', markersize=size)
        elif method == "CreNet":
            plt.errorbar(members[1:], means[1:], yerr=stds[1:], marker=MARKER_CREDALNET, capsize=4, label=method,
                         color=BASELINES_COLOR, markeredgecolor='k', markersize=size)
        elif method == "CreWra":
            plt.errorbar(members[1:], means[1:], yerr=stds[1:], marker=MARKER_CREDALWRAPPER, capsize=4, label=method,
                         color=BASELINES_COLOR, markeredgecolor='k', markersize=size)
        else:
            raise ValueError(f"Unknown method: {method}")

    plt.xticks(range(21), fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([0.75, 1])
    # plt.xscale('log')
    plt.xlabel("Number of Ensemble Members", fontsize=20)
    plt.ylabel("AUROC", fontsize=20)
    # plt.title(f"OOD Detection on CIFAR-10 (ID) and {dataset_label[dataset]} (OOD)", fontsize=20)
    plt.grid(False)  # , linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_FOLDER / f'ood_cifar10_{dataset}_members.pdf', bbox_inches='tight')
    plt.show()


def plot_credal_set(
        probabilities: torch.Tensor,
        mle_prediction: torch.Tensor,
        *,
        title: str | None = None,
        class_labels: list[str] | None = None,
        show: bool = True,
        axis: plt.Axes | None = None,
        plot_legend: bool = True,
        pad_factor: float = 1.65
) -> tuple[plt.Figure, plt.Axes] | None:
    """Plots a credal set in a ternary plot.

    Args:
        probabilities: Tensor of shape (n_models, 3) containing the class probabilities.
        mle_prediction: Tensor of shape (3,) containing the MLE prediction.
        title: An optional title for the plot.
        class_labels: Optional list of class labels for the three classes.
        show: Whether to show the plot immediately or return the figure and axis.
        axis: Optional matplotlib axis to plot on. If None, a new figure and axis are created.
        plot_legend: Whether to plot the legend onto the figure. Default is True.

    Returns:
        If `show` is False, returns a tuple of (figure, axis). Otherwise, returns None.
    """
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(projection="ternary")
    else:
        fig = axis.figure

    probs = probabilities
    lower_probs = np.min(probs, axis=0)
    upper_probs = np.max(probs, axis=0)
    lower_idxs = np.argmin(probs, axis=0)
    upper_idxs = np.argmax(probs, axis=0)
    vertices_ = []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
        for x in [lower_probs[i], upper_probs[i]]:
            for y in [lower_probs[j], upper_probs[j]]:
                z = 1 - x - y
                if lower_probs[k] <= z <= upper_probs[k]:
                    prob = [0, 0, 0]
                    prob[i] = x
                    prob[j] = y
                    prob[k] = z
                    vertices_.append(prob)
    vertices = np.array(vertices_)

    if len(vertices) > 0:
        center = np.mean(vertices, axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        vertices = vertices[np.argsort(angles)]
        axis.scatter(
            mle_prediction[0], mle_prediction[1], mle_prediction[2],
            zorder=99,
            label='MLE (TS)',
            c=RED.get_hex(),
            marker='o',
            s=100,
            clip_on=False  # for visibility even if outside the triangle
        )
        vertices_closed = np.vstack([vertices, vertices[0]])
        axis.fill(
            vertices_closed[:, 0], vertices_closed[:, 1], vertices_closed[:, 2],
            label=r'Temperature Sc.',
            alpha=0.5,
            color=BLUE.get_hex(),
            zorder=2,
            linewidth=1,
        )
    else:
        msg = "The set of vertices is empty. Please check the probabilities in the credal set."
        raise ValueError(msg)

    if plot_legend:
        fig.legend()
    if title is not None:
        axis.set_title(title)

    if class_labels is not None:
        class_labels = _beautify_class_labels(class_labels)
        if len(class_labels) != 3:
            raise ValueError("class_labels must have length 3.")

        edge_midpoints = [
            ((0.5, 0.5, 0.0), class_labels[1]),
            ((0.0, 0.5, 0.5), class_labels[2]),
            ((0.5, 0.0, 0.5), class_labels[0]),
        ]

        # rotation angles for the labels
        rotations = [60, 0, -60]  # approximate intended angles

        # Padding factor: >1 pushes labels outward from the center of the triangle
        center = np.array([1 / 3, 1 / 3, 1 / 3])

        for (coords, lbl), rot in zip(edge_midpoints, rotations):
            coords = np.array(coords)
            # move a bit away from the triangle center
            coords_padded = center + pad_factor * (coords - center)

            axis.text(
                *coords_padded,
                lbl,
                ha="center",
                va="center",
                fontsize=12,
                weight="bold",
                rotation=rot,  # not perfect, but provides orientation
                zorder=100,
                clip_on=False,
            )

    # finalize plot
    axis.grid(visible=True, color=LINES.get_hex(), linestyle='dotted', linewidth=0.5)

    # visualize and show
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axis


if __name__ == '__main__':
    # print ood
    # print_ood_table_results()
    # print_ood_table_results_baselines()
    print_ood_table_results_single_baselines()

    exit(0)
    # ood runtime
    plot_ood_runtime(figsize=(5.5, 4.5))

    # active learning
    plot_active_learning(dataset_id=46941, figsize=(6, 3.2), ylim=(0.57, 0.87), main_paper=True)
    # active learning appendix
    plot_active_learning(dataset_id=46941, figsize=(4.5, 4.5), ylim=(0.57, 0.87))
    plot_active_learning(dataset_id=46963, figsize=(4.5, 4.5), ylim=(0.76, 0.925))
    plot_active_learning(dataset_id=46930, figsize=(4.5, 4.5), ylim=(0.86, 0.97))

    # active learning ablation
    plot_active_learning_ablation(dataset_id=46941, figsize=(4.5, 4.5), ylim=(0.57, 0.87))
    plot_active_learning_ablation(dataset_id=46963, figsize=(4.5, 4.5), ylim=(0.57, 0.925))
    plot_active_learning_ablation(dataset_id=46930, figsize=(4.5, 4.5), ylim=(0.65, 0.97))

    # alpha colorbar
    plot_alpha_colorbar(figsize=(3, 4.5))

    # ood ensemble size
    plot_ood_vs_ensemble_size(dataset="svhn")

    # cov-eff
    plot_cov_efficiency(dataset='cifar10', figsize=(4.5, 4.5), plot_legend=False)
    plot_cov_efficiency(dataset='qualitymri', figsize=(4.5, 4.5))
    plot_cov_efficiency(dataset='chaosnli', figsize=(4.5, 4.5), plot_legend=False)

    # clip cov-eff
    models = ("clip", "siglip", "siglip2")
    labels = {"clip": "CLIP + EffCre", "siglip": "SigLIP + EffCre", "siglip2": "SigLIP-2 + EffCre"}
    plot_cov_efficiency_no_baseline(
        figsize=(6, 3.5),
        seeds=(1,),
        # auto-generate shades across models if you like:
        base_color=MAIN_METHOD_COLOR, use_shades=True, lighten=True, shade_min=0.6, shade_max=1.0,
        legend_title="Models",
        title="CIFAR-10",
        series=[
            dict(
                pattern=f"{RESULTS_FOLDER}/classwise_optim_logit_cov_eff_cifar10_{{id}}_{{seed}}.npy",
                ids=models,
                label=lambda m: labels[m],
                # or fix exact colors instead of shades:
                # color=["tab:blue", "tab:orange", "tab:green"],
            )
        ],
    )

    # tabpfn cov-eff
    plot_cov_efficiency_no_baseline(
        figsize=(6, 3.2),
        seeds=(1, 2, 3),
        base_color=MAIN_METHOD_COLOR,
        use_shades=True,
        lighten=True,
        shade_min=0.3,
        shade_max=1.0,
        legend_title="TabArena Datasets",
        title="TabPFN + EffCre",
        series=[
            dict(
                pattern=f"{RESULTS_FOLDER}/classwise_optim_logit_cov_eff" + "_{id}_tabpfn_{seed}.npy",
                ids=(46941, 46958, 46960, 46963, 46906, 46980,),
                # label per id (optional): label=lambda i: f"{i}"
            )
        ],
    )
