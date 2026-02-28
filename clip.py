"""Script for creating credal sets based on CLIP for coverage-efficiency experiments."""
from clip_experiments import load_logits
from clip_utils import Setting
import numpy as np
import torch
from tqdm import tqdm
import models
from probly.metrics import coverage, efficiency

# set seeds
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

MODEL_FOLDER = './checkpoints/'
RESULTS_FOLDER = "./results/"

method = "classwise_optim_logit"
model = "Clip"  # Clip, SigLIP, or SigLIP2
dataset = 'imagenet'  # cifar10 or imagenet
dataset_dict = {'cifar10': 'CIFAR10', 'imagenet': 'Imagenet'}
dataset_test = {'cifar10': 'CIFAR10-H', 'imagenet': 'Imagenet-ReaL'}
generate = False
bs = 1000

# load precomputed logits
if dataset == 'cifar10':
    data_train = load_logits(
        Setting(dataset_name=dataset_dict[dataset], label_set=dataset_dict[dataset], model_name=model))
elif dataset == 'imagenet':
    data_train = load_logits(
        Setting(dataset_name=dataset_dict[dataset], label_set=dataset_dict[dataset], model_name=model),
        max_values=10_000, shuffle=True)
data_test = load_logits(Setting(dataset_name=dataset_test[dataset], label_set=dataset_dict[dataset], model_name=model))

logits_train = data_train.logits_train
print(f"Logits train shape: {logits_train.shape}, max value: {logits_train.max().abs()}")
targets_train = data_train.targets_train
logits_test = data_test.logits_test
print("Logits test shape:", logits_test.shape)
targets_test = data_test.targets_test
targets_test = targets_test.cpu().detach().numpy()

print("Creating credal sets...")
if dataset == 'cifar10':
    csets, rls = models.classwise_adding_optim_logit(logits_train, targets_train, logits_test, data_train.n_classes)
elif dataset == 'imagenet':
    if generate:
        bounds, rls = models.classwise_adding_optim_logit_clip(logits_train, targets_train, data_train.n_classes)
        np.save(f'{MODEL_FOLDER}{method}_bounds_{dataset}_{model.lower()}_{seed}.npy', bounds)
    else:
        bounds = np.load(f'{MODEL_FOLDER}{method}_bounds_{dataset}_{model.lower()}_{seed}.npy')
else:
    raise ValueError("Unknown dataset")

print("Done with creating credal sets.")

alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]

covs = np.empty(len(alphas))
effs = np.empty(len(alphas))

print("Going into alphas...")

if dataset == 'cifar10':
    for i, alpha in tqdm(enumerate(alphas)):
        outputs = csets[rls >= alpha].swapaxes(0, 1)
        cov = coverage(outputs, targets_test)
        eff = efficiency(outputs)
        covs[i] = cov
        effs[i] = eff
        print(f"Alpha {alpha}: cov {cov}, eff {eff}, num models {outputs.shape}")
elif dataset == 'imagenet':
    for i, alpha in tqdm(enumerate(alphas)):
        # do batching
        cov = 0
        eff = 0
        for start in tqdm(range(0, logits_test.shape[0], bs)):
            batch_logits_test = logits_test[start:start + bs]
            batch_targets_test = targets_test[start:start + bs]
            outputs = []
            for k in range(data_train.n_classes):
                # both ``directions''
                for d in range(2):
                    logits_test_T = batch_logits_test
                    logits_test_T[:, k] += bounds[i, k, d]
                    outputs.append(torch.softmax(logits_test_T, dim=1).cpu().detach().numpy())
            outputs = np.stack(outputs, axis=1)
            cov += coverage(outputs, batch_targets_test) * batch_logits_test.shape[0]
            eff += efficiency(outputs) * batch_logits_test.shape[0]
        cov /= logits_test.shape[0]
        eff /= logits_test.shape[0]
        covs[i] = cov
        effs[i] = eff
        print(f"Alpha {alpha}: cov {cov}, eff {eff}, num models {outputs.shape}")
else:
    raise ValueError("Unknown dataset")

res = np.vstack((alphas, covs, effs))
np.save(f'{RESULTS_FOLDER}{method}_cov_eff_{dataset}_{model.lower()}_{seed}.npy', res)
