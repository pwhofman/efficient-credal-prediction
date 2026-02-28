"""Script to run active learning experiments."""
import torch
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from tabpfn import TabPFNClassifier
import models
from tqdm import tqdm

from utils import max_divergence
from probly.quantification.classification import upper_entropy, lower_entropy
from probly.utils import intersection_probability
import argparse
import utils

RESULTS_FOLDER = "./results/"


def main(args):
    # Set seeds
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

    X, y = fetch_openml(data_id=args.data, return_X_y=True)
    X = X.values
    y = y.values
    X = OrdinalEncoder().fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y))
    class_nums = np.bincount(y)
    print("DATASET SUMMARY")
    print(f"Data ID: {args.data}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {class_nums}")

    budget = 2 * n_classes

    # If there is a class with less than args.class_init instances, we cannot use the dataset.
    if np.min(class_nums) < args.class_init:
        print("Dataset has a class with less than 2 instances. Exiting.")
        return

    init_idx = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        chosen = np.random.choice(idx, args.class_init, replace=False)
        init_idx.extend(chosen)

    # Create initial training set based on init_idx and remove them from the remaining / pool data
    X_train = X[init_idx]
    y_train = y[init_idx]
    X_pool = np.delete(X, init_idx, axis=0)
    y_pool = np.delete(y, init_idx, axis=0)
    # Create test set
    X_pool, X_test, y_pool, y_test = train_test_split(
        X_pool, y_pool, test_size=int(len(X) * args.split), random_state=args.seed, shuffle=True, stratify=y_pool
    )

    # If the number of iterations is -1, we run until the pool is empty
    if args.iterations == -1:
        args.iterations = len(X_pool) // budget
        print(f"Setting number of iterations to {args.iterations}")

    # If the pool is smaller than the budget * iterations, we cannot use the dataset.
    if len(X_pool) < budget * args.iterations:
        print("Pool is smaller than budget * iterations. Exiting.")
        return

    res = np.empty((args.iterations, 2))
    for it in tqdm(range(args.iterations)):
        # Train TabPFN and predict
        pfn = TabPFNClassifier(random_state=args.seed, device=device)
        pfn.fit(X_train, y_train)

        logits_train = pfn.predict_logits(X_train)
        logits_test = pfn.predict_logits(X_test)
        if args.measure in ['random']:
            acc = accuracy_score(y_test, logits_test.argmax(1))
        else:
            logits_train = torch.tensor(logits_train)
            if args.credal_pred == "mle":
                acc = accuracy_score(y_test, logits_test.argmax(1))
            elif args.credal_pred == "intersection":
                logits_test = torch.tensor(logits_test)
                csets, rls = models.classwise_adding_optim_logit(logits_train, y_train, logits_test, n_classes)
                outputs = csets[rls >= args.alpha].swapaxes(0, 1)
                preds_test = intersection_probability(outputs)
                acc = accuracy_score(y_test, preds_test.argmax(1))

        # Log accuracy and number of instances the model was trained on
        res[it, 0] = acc
        res[it, 1] = X_train.shape[0]

        # Sample new instances by considering the most uncertain instances
        if it < args.iterations - 1:
            if args.measure == 'random':
                us = np.random.rand(X_pool.shape[0])
            else:
                logits_pool = pfn.predict_logits(X_pool)
                logits_pool = torch.tensor(logits_pool)
                csets, rls = models.classwise_adding_optim_logit(logits_train, y_train, logits_pool, n_classes)
                outputs = csets[rls >= args.alpha].swapaxes(0, 1)
                if args.measure == 'entropy':
                    us = upper_entropy(outputs, n_jobs=32) - lower_entropy(outputs, n_jobs=32)
                elif args.measure == 'zero_one':
                    us = max_divergence(outputs, n_jobs=32)
                elif args.measure == 'total_entropy':
                    us = upper_entropy(outputs, n_jobs=32)

            indices = np.argsort(us)
            print("======STATS======")
            print(f"uncertainty of most uncertain sample: {us[indices[-1]]}")
            print(f"average uncertainty of pool: {np.mean(us)}")
            print("=================")

            X_pool = X_pool[indices]
            y_pool = y_pool[indices]

            X_train = np.vstack((X_train, X_pool[-budget:]))
            y_train = np.hstack((y_train, y_pool[-budget:]))

            X_pool = X_pool[:-budget]
            y_pool = y_pool[:-budget]

    if args.measure in ['random']:
        np.save(f'{RESULTS_FOLDER}al_{args.data}_{args.measure}_{args.seed}.npy', res)
    else:
        np.save(f'{RESULTS_FOLDER}al_{args.data}_{args.measure}_{args.alpha}_{args.credal_pred}_{args.seed}.npy', res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed for reproducibility')
    parser.add_argument('--data', type=int, help='Dataset to use')
    parser.add_argument('--measure', type=str, help='Uncertainty measure to use for sampling')
    parser.add_argument('--alpha', type=float, help='Alpha values for credal set')
    parser.add_argument('--split', type=float, help='Test split size', default=0.3)
    parser.add_argument('--iterations', type=int, help='Number of iterations', default=50)
    parser.add_argument('--class_init', type=int, help='Number of initial instances per class', default=2)
    parser.add_argument('--credal_pred', type=str, help='How to predict based on a credal set? mle or intersection',
                        default="mle")
    args = parser.parse_args()
    utils.printargs(args)
    main(args)

# python eval_active_learning.py --data 46941 --measure entropy --alpha 0.8 --seed 1 --iterations -1
