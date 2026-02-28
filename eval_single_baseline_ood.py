"""Script to run baselines for the OOD experiments."""
import argparse
import json
import pickle

import torch
import numpy as np
from probly.tasks import out_of_distribution_detection

from models import get_model
from probly.representation.classification import Evidential
import utils
from data import get_data_ood
from probly.quantification.classification import evidential_uncertainty
from ddu.utils.gmm_utils import gmm_evaluate, gmm_fit, get_embeddings
from ddu.metrics.uncertainty_confidence import logsumexp, entropy
from ddu.metrics.ood_metrics import get_roc_auc_logits
from data import get_data_train

MODEL_FOLDER = '/home/scratch/likelihood-ensembles/checkpoints/'
RESULTS_FOLDER = "./results/"


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

    if args.method == 'evidential':
        model = Evidential(get_model(args.model, args.classes))
    elif args.method == "deterministic":
        model = get_model(args.model, args.classes)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    if args.method == 'deterministic' and args.model == 'resnet':
        model.load_state_dict(torch.load(f'/home/scratch/efficient-credal-sets/checkpoints/{args.dataset}_{args.model}_{args.seed}.pt'))
        print(f"Loaded model: /home/scratch/efficient-credal-sets/checkpoints/{args.dataset}_{args.model}_{args.seed}.pt")
    else:
        model.load_state_dict(
                torch.load(f'{MODEL_FOLDER}baseline_{args.dataset}_{args.model}_{args.method}_{args.seed}'))
        print(f"Loaded model: {MODEL_FOLDER}baseline_{args.dataset}_{args.model}_{args.method}_{args.seed}")
    model = model.to(device)
    model.eval()

    if args.method == "evidential":
        id_outputs, _ = utils.torch_get_outputs(model, id_loader, device)
        ood_outputs, _ = utils.torch_get_outputs(model, ood_loader, device)

        id_outputs = id_outputs.detach().cpu().numpy()
        ood_outputs = ood_outputs.detach().cpu().numpy()

        id_eu = evidential_uncertainty(id_outputs)
        ood_eu = evidential_uncertainty(ood_outputs)

    elif args.method == "deterministic":
        train_loader, _, _ = get_data_train(args.dataset, args.seed, validation=False)
        print("in deterministic")
        # Evaluate a GMM model
        print("GMM Model")
        embeddings, labels = get_embeddings(
            model,
            train_loader,
            num_dim=512, # number of features
            dtype=torch.double,
            device=device,
            storage_device=device,
        )

        gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=args.classes)

        logits, labels = gmm_evaluate(
            model, gaussians_model, id_loader, device=device, num_classes=args.classes, storage_device=device,
        )

        ood_logits, ood_labels = gmm_evaluate(
            model, gaussians_model, ood_loader, device=device, num_classes=args.classes, storage_device=device,
        )

        ood_eu, id_eu = get_roc_auc_logits(
            logits, ood_logits, logsumexp, device, confidence=True
        ) # order of ood_eu, id_eu is correct. this is done to align the setup with theirs using confidence=True

        id_eu = id_eu.cpu().detach().numpy()
        ood_eu = ood_eu.cpu().detach().numpy()

        # (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(
        #     logits, ood_logits, logsumexp, device, confidence=True
        # )
        # (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(logits, ood_logits, entropy, device)


    ood_dict = {
        'id_eu': id_eu,
        'ood_eu': ood_eu,
    }

    with open(f'{RESULTS_FOLDER}ood_{args.dataset}_{args.ooddata}_{args.model}_{args.method}_{args.seed}.pkl',
              'wb') as f:
        pickle.dump(ood_dict, f)

    auroc = out_of_distribution_detection(id_eu, ood_eu)
    print(f"The AUROC is {auroc}")

    np.save(
        f'{RESULTS_FOLDER}ood_{args.dataset}_{args.ooddata}_{args.model}_{args.method}_{args.seed}.npy',
        auroc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='iD Dataset')
    parser.add_argument('--ooddata', type=str, help='OoD Dataset')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--method', type=str, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--seed', type=int, help='Seed')

    args = parser.parse_args()
    main(args)