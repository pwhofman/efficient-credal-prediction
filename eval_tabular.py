"""Script for experiments on tabular datasets with TabPFN."""
import torch
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from probly.metrics import coverage, efficiency
import models
from tqdm import tqdm
import argparse

RESULTS_FOLDER = "./results/"
split = 0.3


def main(args):
    print("Using seed", args.seed)
    X, y = fetch_openml(data_id=args.data_id, return_X_y=True)
    X = X.values
    y = y.values
    X = OrdinalEncoder().fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    n_classes = len(np.unique(y))
    X_train0, X_test0, y_train0, y_test0 = train_test_split(
        X, y, test_size=split, random_state=args.seed
    )
    print("DATASET SUMMARY")
    print(f"Data ID: {args.data_id}")
    print(f"Number of samples: {X_train0.shape[0] + X_test0.shape[0]}")
    print(f"Number of features: {X_train0.shape[1]}")
    print(f"Number of classes: {n_classes}")

    print("Fit Random Forest")
    rf = RandomForestClassifier(max_depth=5, random_state=args.seed)
    rf.fit(X_train0, y_train0)
    X_train1 = X_train0
    X_test1 = X_test0
    y_train1 = rf.predict_proba(X_train1)
    y_train10 = np.array([np.random.choice(n_classes, p=probs) for probs in y_train1])
    y_test1 = rf.predict_proba(X_test1)
    print("Obtained first-order data")

    print("Fit TabPFN")
    pfn = TabPFNClassifier(random_state=args.seed)
    pfn.fit(X_train0, y_train10)

    print("Predict with TabPFN")
    y_pred1 = pfn.predict_proba(X_test1)
    acc = accuracy_score(y_test0, y_pred1.argmax(1))
    print("Accuracy", acc)

    logits_train = pfn.predict_logits(X_train1)
    logits_test = pfn.predict_logits(X_test1)
    logits_train = torch.tensor(logits_train, dtype=torch.float32)
    logits_test = torch.tensor(logits_test, dtype=torch.float32)
    y_train10 = torch.tensor(y_train10, dtype=torch.long)

    print(logits_train.shape, logits_test.shape)

    csets, rls = models.classwise_adding_optim_logit(logits_train, y_train10, logits_test, n_classes)
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    covs = np.empty(len(alphas))
    effs = np.empty(len(alphas))
    for i, alpha in tqdm(enumerate(alphas)):
        outputs = csets[rls >= alpha].swapaxes(0, 1)
        cov = coverage(outputs, y_test1)
        eff = efficiency(outputs)
        covs[i] = cov
        effs[i] = eff
    res = np.vstack((alphas, covs, effs))
    np.save(f'{RESULTS_FOLDER}classwise_optim_logit_cov_eff_{args.data_id}_tabpfn_{args.seed}.npy', res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, help='Data ID')
    parser.add_argument('--seed', type=int, help='Seed')
    args = parser.parse_args()
    main(args)

# python eval_tabular.py --data_id 46906 --seed 1
