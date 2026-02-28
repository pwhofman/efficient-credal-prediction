"""Script for running coverage-efficiency and OOD experiments."""
import numpy as np
import torch
import data
import utils
from tqdm import tqdm
import models
from probly.metrics import coverage, efficiency, zero_one_loss
import argparse
from probly.quantification.classification import upper_entropy, lower_entropy
from probly.tasks import out_of_distribution_detection

MODEL_FOLDER = './checkpoints/'
MODEL_FOLDER = '/home/scratch/efficient-credal-sets/checkpoints/'
RESULTS_FOLDER = "./results/"


def batched_entropy_diff(probs, batch_size=128, n_jobs=-1):
    n_instances = probs.shape[0]
    results = []
    for start in range(0, n_instances, batch_size):
        end = min(start + batch_size, n_instances)
        batch = probs[start:end]

        ue = upper_entropy(batch, n_jobs=n_jobs)
        le = lower_entropy(batch, n_jobs=n_jobs)
        results.append(ue - le)

    return np.concatenate(results, axis=0)


def main(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set device
    if torch.cuda.is_available():
        device = utils.get_best_gpu()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = 'cpu'

    print(f'Using device: {device} and the following args:')
    utils.printargs(args)

    train_loader, _, _ = data.get_data_train(args.dataset, args.seed, validation=False, batch_size=512)

    model = models.get_model(args.model, args.classes)
    model.load_state_dict(torch.load(f'{MODEL_FOLDER}{args.dataset}_{args.model}_{args.seed}.pt'))
    model.eval()
    model = model.to(device)

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]

    if args.experiment == "cov_eff":
        test_loader = data.get_data_task(args.dataset, seed=args.seed, first_order=True, batch_size=512)
        logits_train, targets_train = utils.torch_get_outputs(model, train_loader, device)
        logits_test, targets_test = utils.torch_get_outputs(model, test_loader, device)
        targets_train = targets_train.long()
        targets_test = targets_test.cpu().detach().numpy()

        csets, rls = models.classwise_adding_optim_logit(logits_train, targets_train, logits_test, args.classes)

        covs = np.empty(len(alphas))
        effs = np.empty(len(alphas))
        for i, alpha in tqdm(enumerate(alphas)):
            outputs = csets[rls >= alpha].swapaxes(0, 1)
            cov = coverage(outputs, targets_test)
            eff = efficiency(outputs)
            covs[i] = cov
            effs[i] = eff
            print(f"Alpha {alpha}: cov {cov}, eff {eff}, num models {outputs.shape}")
        res = np.vstack((alphas, covs, effs))
        np.save(f'{RESULTS_FOLDER}classwise_optim_logit_{args.experiment}_{args.dataset}_{args.model}_{args.seed}.npy', res)
    elif args.experiment == "ood":
        id_loader, ood_loader = data.get_data_ood(args.dataset, args.dataset_ood, seed=args.seed, batch_size=512)
        logits_train, targets_train = utils.torch_get_outputs(model, train_loader, device)
        logits_id, _ = utils.torch_get_outputs(model, id_loader, device)
        logits_ood, _ = utils.torch_get_outputs(model, ood_loader, device)
        targets_train = targets_train.long()
        csets_id, csets_ood, rls = models.classwise_adding_optim_logit_ood(logits_train, targets_train, logits_id,
                                                                           logits_ood, args.classes)

        aurocs = np.empty(len(alphas))
        for i, alpha in tqdm(enumerate(alphas)):
            print(f"Calculating AUROC for alpha {alpha}")
            eu = []
            for cset in [csets_id, csets_ood]:
                outputs = cset[rls >= alpha].swapaxes(0, 1)
                eu.append(batched_entropy_diff(outputs, batch_size=2000, n_jobs=32))
            print(f'Alpha {alpha}: Mean EU in ID: {np.mean(eu[0])}, Mean EU in OOD: {np.mean(eu[1])}')
            aurocs[i] = out_of_distribution_detection(eu[0], eu[1])
        res = np.vstack((alphas, aurocs))
        np.save(
            f'{RESULTS_FOLDER}classwise_optim_logit_{args.experiment}_{args.dataset}_{args.dataset_ood}_{args.model}_{args.seed}.npy',
            res)
    else:
        raise ValueError("Unknown experiment type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, help='Experiment')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--dataset_ood', type=str, help='Dataset OoD')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--seed', type=int, help='Seed')
    args = parser.parse_args()
    main(args)

# python eval_cov_eff_ood.py --experiment ood --dataset cifar10 --classes 10 --model resnet --seed 1 --dataset_ood svhn
