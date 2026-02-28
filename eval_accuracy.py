import data
import utils
import models
import argparse
import numpy as np
import torch
from probly import metrics
import torch.nn.functional as F
from tqdm import tqdm
from models import DesterckeEnsemble, CaprioEnsemble, WangEnsemble, LikelihoodEnsemble, Ensemble, get_model

MODEL_FOLDER = './checkpoints/'

# select members
def sample_evenly(lst, x):
    n = len(lst)
    if x <= 1:
        return [0]
    indices = np.linspace(0, n - 1, x, dtype=int)  # evenly spaced indices
    return indices

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

    test_loader = data.get_data_task(args.dataset, seed=args.seed, first_order=False, batch_size=512)

    seeds = [1,2,3] if args.seed == 1 else [4,5,6]
    accs = []
    eces = []
    for s in seeds:
        if args.method == 'credalrl':
            ensemble = LikelihoodEnsemble(get_model(args.model, args.classes), args.classes, n_members=20,
                                          tobias_value=100)
            utils.load_ensemble(ensemble,
                                f'{MODEL_FOLDER}{args.dataset}_{args.model}_20_100_0.8_{s}')
        elif args.method == 'effcre':
            ensemble ={'models':[]}
            model = models.get_model(args.model, args.classes)
            model.load_state_dict(torch.load(f'{MODEL_FOLDER}{args.dataset}_{args.model}_{s}.pt'))
            ensemble['models'].append(model)
        else:
            if args.method == "credalwrapper":
                ensemble = Ensemble(get_model(args.model, args.classes), 20)
            elif args.method == "credalensembling":
                ensemble = DesterckeEnsemble(get_model(args.model, args.classes), 20)
            elif args.method == "credalbnn":
                ensemble = CaprioEnsemble(get_model(args.model, args.classes), 20, [0,1],[0,1])
            elif args.method == "credalnet":
                ensemble = WangEnsemble(get_model(args.model, args.classes), 20, 0.5, args.classes)
            else:
                raise ValueError(f"Unknown method: {args.method}")
            ensemble.load_state_dict(
                torch.load(f'{MODEL_FOLDER}baseline_{args.dataset}_{args.method}_20_{s}'))

        # sample from ensemble if smaller size is desired
        if args.n_members < 20 and args.method != "effcre":
            idx = sample_evenly(ensemble.models, args.n_members)
            for i in range(args.n_members):
                ensemble.models[i] = ensemble.models[idx[i]]
            ensemble.models = ensemble.models[:args.n_members]

        print(len(ensemble.models))

        outputs = torch.empty(0, device=device)
        targets = torch.empty(0, device=device)
        for input, target in tqdm(test_loader):
            input, target = input.to(device), target.to(device)
            targets = torch.cat((targets, target), dim=0)
            outputs = torch.cat((outputs, ensemble.predict_pointwise(input)), dim=0)

        print(outputs.shape)
        print(targets.shape)

        ens_acc, ens_ece = [], []
        for model in ensemble.models:
            model.eval()
            model = model.to(device)
            logits_test, targets_test = utils.torch_get_outputs(model, test_loader, device)
            probs_test = torch.softmax(logits_test, dim=1).cpu().detach().numpy()
            targets_test = targets_test.cpu().detach().numpy()

            # compute ECE
            ens_ece.append(metrics.expected_calibration_error(probs_test, targets_test))

            # compute accuracy
            ens_acc.append(1 - metrics.zero_one_loss(probs_test, targets_test))
        ens_acc = np.array(ens_acc)
        ens_ece = np.array(ens_ece)
        print(ens_acc.shape)
        accs.append(ens_acc)
        eces.append(ens_ece)
    accs = np.array(accs)
    eces = np.array(eces)
    print(accs.shape)
    print(np.mean(accs, axis=0), np.std(accs, axis=0))
    print(np.mean(eces, axis=0), np.std(eces, axis=0))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='Method')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--n_members', type=int, help='Number of members to use')
    args = parser.parse_args()
    main(args)

# python eval_accuracy.py --dataset cifar10 --classes 10 --model resnet --seed 1 --method credalrl --n_members 10