
import time
import numpy as np
import torch
import utils
import data
import models
import argparse
from scipy.optimize import minimize
from torch.nn import functional as F

MODEL_FOLDER = './checkpoints/'

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

    alpha = 0.8

    train_loader, _, _ = data.get_data_train(args.dataset, args.seed, validation=False, batch_size=512)

    model = models.get_model(args.model, args.classes)
    model.load_state_dict(torch.load(f'{MODEL_FOLDER}{args.dataset}_{args.model}_{args.seed}.pt'))
    model.eval()
    model = model.to(device)

    id_loader, ood_loader = data.get_data_ood(args.dataset, args.dataset_ood, seed=args.seed, batch_size=512)

    logits_train, targets_train = utils.torch_get_outputs(model, train_loader, device)
    targets_train = targets_train.long()


    mll = utils.log_likelihood(logits_train, targets_train).cpu().detach().item()
    bounds = []
    for k in range(args.classes):
        # 1 is finding minimum, -1 is finding maximum
        bound = []
        for direction in [1, -1]:
            def fun(x):
                return direction * x[k]

            def const(x) -> float:
                c = torch.tensor(x, device=logits_train.device)
                logits_train_T = logits_train + c
                lik = utils.log_likelihood(logits_train_T, targets_train).cpu().detach().item()
                rel_lik = np.exp(lik - mll)
                return rel_lik

            x0 = np.zeros(args.classes)

            optim_bounds = [(0.0, 0.0)] * args.classes
            optim_bounds[k] = (None, None)

            constraints = {'type': 'ineq', 'fun': lambda x: const(x) - alpha}
            res = minimize(fun, x0, constraints=constraints, bounds=optim_bounds)
            bound.append(res.x)
        bounds.append(bound)


    times = []
    for run in range(5):
        csets_id = []
        csets_ood = []
        rls = []
        start = time.time()
        logits_id, _ = utils.torch_get_outputs(model, id_loader, device)
        logits_ood, _ = utils.torch_get_outputs(model, ood_loader, device)
        # add the bounds to the logits_test to make predictions
        for k in range(args.classes):
            # both ``directions''
            for d in range(2):
                logits_id_T = logits_id + torch.tensor(bounds[k][d], device=logits_id.device)
                csets_id.append(F.softmax(logits_id_T, dim=1).cpu().detach().numpy())
                logits_ood_T = logits_ood + torch.tensor(bounds[k][d], device=logits_ood.device)
                csets_ood.append(F.softmax(logits_ood_T, dim=1).cpu().detach().numpy())
        rls.append([alpha] * (2 * args.classes))
        csets_id = np.array(csets_id)
        csets_ood = np.array(csets_ood)
        rls = np.array(rls).flatten()
        times.append(time.time() - start)

    times = np.array(times)
    print(np.mean(times), np.std(times))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--dataset_ood', type=str, help='Dataset OoD')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--seed', type=int, help='Seed')
    args = parser.parse_args()
    main(args)

# python timer.py --dataset cifar10 --classes 10 --model resnet --seed 1 --dataset_ood svhn
