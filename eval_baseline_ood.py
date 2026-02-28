"""Script to run baselines for the OOD experiments."""
import argparse
import json
import pickle
import time

import torch
import numpy as np
from probly.tasks import out_of_distribution_detection

from models import get_model, LikelihoodEnsemble, DesterckeEnsemble, CaprioEnsemble, WangEnsemble, Ensemble
import utils
from data import get_data_ood
from probly.quantification.classification import upper_entropy, lower_entropy, upper_entropy_convex_hull, \
    lower_entropy_convex_hull

MODEL_FOLDER = './checkpoints/'
MODEL_FOLDER = '/home/scratch/likelihood-ensembles/checkpoints/'
RESULTS_FOLDER = "./results/"


# select members
def sample_evenly(lst, x):
    n = len(lst)
    if x <= 1:
        return [0]
    indices = np.linspace(0, n - 1, x, dtype=int)  # evenly spaced indices
    return indices


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        device = utils.get_best_gpu()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = 'cpu'

    print(f'Using device: {device} and the following args:')
    utils.printargs(args)

    id_loader, ood_loader = get_data_ood(name_id=args.dataset, name_ood=args.ooddata, seed=args.seed, batch_size=500)

    if args.method == 'credalrl':
        ensemble = LikelihoodEnsemble(get_model(args.model, args.classes), args.classes, n_members=20,
                                      tobias_value=args.tobias)
        utils.load_ensemble(ensemble,
                            f'{MODEL_FOLDER}{args.dataset}_{args.model}_20_{args.tobias}_{args.alpha}_{args.seed}')
    else:
        if args.method == "credalwrapper":
            ensemble = Ensemble(get_model(args.model, args.classes), 20)
        elif args.method == "credalensembling":
            ensemble = DesterckeEnsemble(get_model(args.model, args.classes), 20)
        elif args.method == "credalbnn":
            ensemble = CaprioEnsemble(get_model(args.model, args.classes), 20, args.prior_mu, args.prior_std)
        elif args.method == "credalnet":
            ensemble = WangEnsemble(get_model(args.model, args.classes), 20, args.delta, args.classes)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        ensemble.load_state_dict(
            torch.load(f'{MODEL_FOLDER}baseline_{args.dataset}_{args.method}_20_{args.seed}'))

    # sample from ensemble if smaller size is desired
    if args.n_members < 20:
        idx = sample_evenly(ensemble.models, args.n_members)
        for i in range(args.n_members):
            ensemble.models[i] = ensemble.models[idx[i]]
        ensemble.models = ensemble.models[:args.n_members]

    print(len(ensemble.models))

    ensemble = ensemble.to(device)
    ensemble.eval()

    times = []
    for run in range(5):
        start = time.time()
        if args.method == 'credalensembling':
            id_outputs, _ = utils.torch_get_outputs_representation_alpha(ensemble, id_loader, device, alpha=args.alpha)
            ood_outputs, _ = utils.torch_get_outputs_representation_alpha(ensemble, ood_loader, device, alpha=args.alpha)
        else:
            id_outputs, _ = utils.torch_get_outputs_representation(ensemble, id_loader, device)
            ood_outputs, _ = utils.torch_get_outputs_representation(ensemble, ood_loader, device)

        id_outputs = id_outputs.detach().cpu().numpy()
        ood_outputs = ood_outputs.detach().cpu().numpy()
        times.append(time.time() - start)
    times = np.array(times)
    print(np.mean(times), np.std(times))


    if args.method in ['credalrl', 'credalwrapper', 'credalnet']:
        id_upper_entropy = upper_entropy(id_outputs)
        ood_upper_entropy = upper_entropy(ood_outputs)
        id_lower_entropy = lower_entropy(id_outputs)
        ood_lower_entropy = lower_entropy(ood_outputs)
    elif args.method in ['credalensembling', 'credalbnn']:
        id_upper_entropy = upper_entropy_convex_hull(id_outputs)
        ood_upper_entropy = upper_entropy_convex_hull(ood_outputs)
        id_lower_entropy = lower_entropy_convex_hull(id_outputs)
        ood_lower_entropy = lower_entropy_convex_hull(ood_outputs)

    ood_dict = {
        'id_upper_entropy': id_upper_entropy,
        'ood_upper_entropy': ood_upper_entropy,
        'id_lower_entropy': id_lower_entropy,
        'ood_lower_entropy': ood_lower_entropy,
    }

    if args.method in ['credalrl', 'credalensembling']:
        with open(
                f'{RESULTS_FOLDER}ood_{args.dataset}_{args.ooddata}_{args.method}_{args.n_members}_{args.alpha}_{args.seed}.pkl',
                'wb') as f:
            pickle.dump(ood_dict, f)
    else:
        with open(f'{RESULTS_FOLDER}ood_{args.dataset}_{args.ooddata}_{args.method}_{args.n_members}_{args.seed}.pkl',
                  'wb') as f:
            pickle.dump(ood_dict, f)

    id_unc = ood_dict['id_upper_entropy'] - ood_dict['id_lower_entropy']
    ood_unc = ood_dict['ood_upper_entropy'] - ood_dict['ood_lower_entropy']
    auroc = out_of_distribution_detection(id_unc, ood_unc)

    if args.method in ['credalrl', 'credalensembling']:
        np.save(
            f'{RESULTS_FOLDER}ood_cifar10_{args.dataset}_{args.method}_{args.n_members}_{args.alpha}_{args.seed}.npy',
            auroc)
    else:
        np.save(
            f'{RESULTS_FOLDER}ood_cifar10_{args.dataset}_{args.method}_{args.n_members}_{args.seed}.npy',
            auroc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='iD Dataset')
    parser.add_argument('--ooddata', type=str, help='OoD Dataset')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--method', type=str, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--n_members', type=int, help='Number of members to use')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--tobias', type=int, help='Tobias <3 value')
    parser.add_argument('--alpha', type=float, help='Alpha value to use')
    parser.add_argument('--delta', type=float, help='Delta value to use')
    parser.add_argument('--prior_mu', type=json.loads, help='Prior mean')
    parser.add_argument('--prior_std', type=json.loads, help='Prior std')
    args = parser.parse_args()
    main(args)

# python eval_baseline_ood.py --dataset cifar10 --ooddata svhn --classes 10 --method credalrl --model resnet --n_members 2 --seed 1 --alpha 0.8 --tobias 100
