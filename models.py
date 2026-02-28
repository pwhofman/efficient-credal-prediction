"""Collection of all models used in experiments."""
from typing import Literal
from utils import log_likelihood
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from probly.representation import Ensemble, Bayesian
from utils import tobias_init_ensemble, torch_tensor_to_pil
import numpy as np
from scipy.optimize import minimize
from ddu.net.resnet import resnet18


def get_model(base, n_classes):
    if base == 'resnet':
        model = ResNet18()
        model.linear = nn.Linear(512, n_classes)
    elif base == 'fcnet':
        model = FCNet(768, n_classes)
    elif base == 'torchresnet':
        model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif base == "creresnet":
        model = ResNet18()
        model.linear = nn.Sequential(
            nn.Linear(model.linear.in_features, 2 * n_classes),
            nn.BatchNorm1d(2 * n_classes),
            IntSoftmax()
        )
    elif base == "crefcnet":
        model = CreFCNet(768, n_classes)
    elif base == "cretorchresnet":
        model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2 * n_classes),
            nn.BatchNorm1d(2 * n_classes),
            IntSoftmax()
        )
    elif base == 'resnetddu':
        model = resnet18(num_classes=n_classes)
    else:
        raise ValueError(f"Unknown base model: {base}")
    return model


class LikelihoodEnsemble(Ensemble):
    def __init__(self, base, n_classes, n_members, tobias_value=100):
        super().__init__(base, n_members)
        self.n_members = n_members
        self.rls = [1.0]
        self.tobias_value = tobias_value
        if self.tobias_value:
            tobias_init_ensemble(self, n_classes, tobias_value)


class CaprioEnsemble(Ensemble):
    def __init__(self, base, n_members, prior_mu, prior_sigma):
        super().__init__(base, n_members)
        self.n_members = n_members
        mus = np.linspace(prior_mu[0], prior_mu[1], endpoint=True, num=n_members)
        sigmas = np.random.uniform(prior_sigma[0], prior_sigma[1], size=n_members)
        for i in range(n_members):
            self.models[i] = Bayesian(base, prior_mean=mus[i], prior_std=sigmas[i])


class DesterckeEnsemble(Ensemble):
    def __init__(self, base, n_members):
        super().__init__(base, n_members)
        self.n_members = n_members

    @torch.no_grad()
    def predict_representation(self, x: torch.Tensor, alpha: float, distance: str = 'euclidean',
                               logits: bool = False) -> torch.Tensor:
        x = super().predict_representation(x, logits=logits)
        if distance == 'euclidean':
            # when the distance is euclidean the mean is the representative probability distribution
            representative = torch.mean(x, dim=1)
            # compute distances to the representative distribution
            dists = torch.cdist(x, torch.unsqueeze(representative, 1), p=2)
            # discard alpha percent of the predictions with the largest distances
            # sort the distances
            sorted_indices = torch.argsort(dists.squeeze(), dim=1)
            # get the indices of the predictions to keep
            keep_indices = sorted_indices[:, :int(round((1 - alpha) * self.n_members))]
            # get the predictions to keep
            keep_predictions = torch.gather(x, 1, keep_indices.unsqueeze(2).expand(-1, -1, x.shape[2]))
        else:
            raise ValueError(f"Unknown distance metric: {distance}")
        return keep_predictions


class WangEnsemble(Ensemble):
    def __init__(self, base, n_members, delta, n_classes):
        super().__init__(base, n_members)
        self.n_members = n_members
        self.delta = delta
        self.n_classes = n_classes

    def predict_pointwise(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        if logits:
            raise ValueError('Logits not possible for credal nets')
        outputs = torch.stack([model(x) for model in self.models], dim=1).mean(dim=1)
        return outputs.reshape(outputs.shape[0], 2, -1).mean(dim=1)

    def predict_representation(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        if logits:
            raise ValueError('Logits not possible for credal nets')
        outputs = torch.stack([model(x) for model in self.models], dim=1).mean(dim=1)
        return outputs.reshape(outputs.shape[0], 2, -1)


class FCNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x


class CreFCNet(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2 * n_classes)
        self.bn = nn.BatchNorm1d(2 * n_classes)
        self.act = nn.ReLU()
        self.int_softmax = IntSoftmax()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        x = self.bn(x)
        x = self.int_softmax(x)
        return x


class CreResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.base = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(self.base.children())[:-2])
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2 * n_classes)
        self.bn = nn.BatchNorm1d(2 * n_classes)
        self.act = nn.ReLU()
        self.int_softmax = IntSoftmax()

    def forward(self, x):
        x = self.upsample(x)
        x = self.base(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        x = self.int_softmax(x)
        return x


class IntSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Extract number of classes
        n_classes = int(x.shape[-1] / 2)

        # Extract center and the radius
        center = x[:, :n_classes]
        radius = x[:, n_classes:]

        # Ensure the nonnegativity of radius
        radius_nonneg = F.softplus(radius)

        # Compute upper and lower probabilities
        exp_center = torch.exp(center)
        exp_center_sum = torch.sum(exp_center, dim=-1, keepdim=True)

        lo = torch.exp(center - radius_nonneg) / (exp_center_sum - exp_center + torch.exp(center - radius_nonneg))
        hi = torch.exp(center + radius_nonneg) / (exp_center_sum - exp_center + torch.exp(center + radius_nonneg))

        # Generate output
        output = torch.cat([lo, hi], dim=-1)

        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# clip based models --------------------------------------------------------------------------------
class CallableModel:
    """ A callable model interface. """

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("The forward method must be implemented by the specific model.")

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)


ENGLISH_TEMPLATE = "This is a photo of a {}"
SWAHILI_TEMPLATE = "Hii ni picha ya {}"
CHINESE_TEMPLATE = "这是一张{}的照片"
FRENCH_TEMPLATE = "Ceci est une photo de {}"
TemplateLanguages = Literal["english", "swahili"]


class ClipClassifier(CallableModel):
    """ A CLIP-based zero-shot classifier. """

    def __init__(
            self,
            class_labels: list[str],
            device: str = "cpu",
            template_language: TemplateLanguages = "english",
    ) -> None:
        """ Initialize the CLIP-based classifier.

        Args:
            class_labels: List of class labels for classification.
        """
        from transformers import CLIPProcessor, CLIPModel
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._model.to(device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        template = ENGLISH_TEMPLATE
        if template_language == "swahili":
            template = SWAHILI_TEMPLATE
        if template_language == "chinese":
            template = CHINESE_TEMPLATE
        if template_language == "french":
            template = FRENCH_TEMPLATE
        self._template = [template.format(c) for c in class_labels]

    def _preprocess(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = torch_tensor_to_pil(images)
        inputs = self._processor(
            text=self._template, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        return inputs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = self._preprocess(images)
        with torch.no_grad():
            output = self._model(**inputs)
        return output.logits_per_image

    def get_embeddings(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._preprocess(images)
        with torch.no_grad():
            output = self._model(**inputs)
        image_embeddings = output.image_embeds
        text_embeddings = output.text_embeds
        return image_embeddings, text_embeddings


class BiomedCLIPClassifier(CallableModel):
    """ A BioMedCLIP-based zero-shot classifier. """

    def __init__(
            self,
            class_labels: list[str],
            device: str = "cpu",
            template_language: TemplateLanguages = "english",
    ) -> None:
        """ Initialize the BioMedCLIP-based classifier.

        Args:
            class_labels: List of class labels for classification.
        """
        from open_clip import create_model_from_pretrained, get_tokenizer

        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self._model = model
        self._model.to(device)
        self._model.eval()
        self._processor = preprocess
        template = ENGLISH_TEMPLATE
        if template_language == "swahili":
            template = SWAHILI_TEMPLATE
        if template_language == "chinese":
            template = CHINESE_TEMPLATE
        if template_language == "french":
            template = FRENCH_TEMPLATE
        self._template = [template.format(c) for c in class_labels]
        self._template = tokenizer(self._template)
        self._template.to(device)
        self.device = device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = torch_tensor_to_pil(images)
        images = torch.stack([self._processor(image) for image in images]).to(self.device)
        with torch.no_grad():
            image_features, text_features, logit_scale = self._model(images, self._template)
            logits = (logit_scale * image_features @ text_features.t())
        return logits


class SigLIP2Classifier(CallableModel):
    """ A SigLIP2-based zero-shot classifier. """

    def __init__(
            self,
            class_labels: list[str],
            device: str = "cpu",
            template_language: TemplateLanguages = "english",
            use_siglip_one: bool = False,
    ) -> None:
        """ Initialize the SigLIP-based classifier.

        Args:
            class_labels: List of class labels for classification.
            device: Device to use.
            template_language: Language for the text templates.
        """
        from transformers import AutoModel, AutoProcessor

        model_name = "google/siglip2-base-patch16-224"
        if use_siglip_one:
            model_name = "google/siglip-base-patch16-224"

        self.device = device
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model.to(device)

        template = ENGLISH_TEMPLATE
        if template_language == "swahili":
            template = SWAHILI_TEMPLATE
        if template_language == "chinese":
            template = CHINESE_TEMPLATE
        if template_language == "french":
            template = FRENCH_TEMPLATE
        self._template = [template.format(c) for c in class_labels]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = torch_tensor_to_pil(images)
        inputs = self._processor(
            images=images,
            text=self._template,
            return_tensors="pt",
            padding="max_length",
            max_length=64
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.logits_per_image


alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]


def classwise_adding_optim_logit(logits_train, targets_train, logits_test, n_classes):
    csets = []
    rls = []
    mll = log_likelihood(logits_train, targets_train).cpu().detach().item()
    for alpha in tqdm(alphas, desc='Alphas'):
        bounds = []
        for k in range(n_classes):
            # 1 is finding minimum, -1 is finding maximum
            bound = []
            for direction in [1, -1]:
                def fun(x):
                    return direction * x[k]
                    # c = torch.tensor(x, device=logits_train.device)
                    # logits_train_T = logits_train + c
                    # probs = F.softmax(logits_train_T, dim=1).cpu().detach().numpy()
                    # return direction * np.mean(probs[:, k], axis=0)

                def const(x) -> float:
                    c = torch.tensor(x, device=logits_train.device)
                    logits_train_T = logits_train + c
                    lik = log_likelihood(logits_train_T, targets_train).cpu().detach().item()
                    rel_lik = np.exp(lik - mll)
                    return rel_lik

                x0 = np.zeros(n_classes)
                optim_bounds = [(0.0, 0.0)] * n_classes
                # T_abs = int(np.max((abs(torch.max(logits_train).cpu().numpy()), abs(torch.min(logits_train).cpu().numpy()))))
                # optim_bounds[k] = (-2 * T_abs, 2 * T_abs)
                optim_bounds[k] = (None, None)
                constraints = {'type': 'ineq', 'fun': lambda x: const(x) - alpha}
                res = minimize(fun, x0, constraints=constraints, bounds=optim_bounds)
                bound.append(res.x)
            bounds.append(bound)

        # add the bounds to the logits_test to make predictions
        for k in range(n_classes):
            # both ``directions''
            for d in range(2):
                logits_test_T = logits_test + torch.tensor(bounds[k][d], device=logits_test.device)
                csets.append(F.softmax(logits_test_T, dim=1).cpu().detach().numpy())
        rls.append([alpha] * (2 * n_classes))
    csets = np.array(csets)
    rls = np.array(rls).flatten()
    return csets, rls


def classwise_adding_optim_logit_clip(logits_train, targets_train, n_classes):
    """This is a special function for clip-based models. It will return the bounds, hence not the csets, because
    this is too expensive.
    """
    rls = []
    bounds = []
    logits_train = logits_train.cpu().detach()
    targets_train = targets_train.cpu().detach()
    mll = log_likelihood(logits_train, targets_train).cpu().detach().numpy()

    def optimize_for_class(k, alpha):
        # optimize the bounds for class k with alpha value alpha
        bound = []
        for direction in [1, -1]:
            def fun(x):
                # return direction * x
                return direction * x

            def const(x) -> float:
                # c = torch.tensor(x, device=logits_train.device)
                # logits_train_T = logits_train + c
                # logits_train_T = logits_train.copy()
                logits_train[:, k] += x
                # logits_train[:, k] += x
                lik = log_likelihood(logits_train, targets_train).cpu().detach().numpy()
                rel_lik = np.exp(lik - mll)
                logits_train[:, k] -= x
                # del logits_train_T
                # gc.collect()
                return rel_lik

            # x0 = np.zeros(n_classes)
            x0 = np.array([0])
            # optim_bounds = [(0.0, 0.0)] * n_classes
            optim_bounds = ((-10000.0, 10000.0),)
            constraints = {'type': 'ineq', 'fun': lambda x: const(x) - alpha}
            res = minimize(fun, x0, constraints=constraints, bounds=optim_bounds)
            # res = minimize(fun, x0, constraints=constraints)
            bound.append(res.x)
        return bound

    for alpha in alphas:
        print(f"Running with alpha {alpha}")
        # bounds = Parallel(n_jobs=32, backend='loky')(delayed(optimize_for_class)(k, alpha) for k in tqdm(range(n_classes)))
        alpha_bounds = [optimize_for_class(k, alpha) for k in tqdm(range(n_classes))]
        bounds.append(alpha_bounds)
        rls.append([alpha] * (2 * n_classes))
    bounds = np.squeeze(np.array(bounds))
    rls = np.array(rls).flatten()
    return bounds, rls


def classwise_adding_optim_logit_ood(logits_train, targets_train, logits_id, logits_ood, n_classes):
    csets_id = []
    csets_ood = []
    rls = []
    mll = log_likelihood(logits_train, targets_train).cpu().detach().item()
    for alpha in tqdm(alphas, desc='Alphas'):
        bounds = []
        for k in range(n_classes):
            # 1 is finding minimum, -1 is finding maximum
            bound = []
            for direction in [1, -1]:
                def fun(x):
                    return direction * x[k]

                def const(x) -> float:
                    c = torch.tensor(x, device=logits_train.device)
                    logits_train_T = logits_train + c
                    lik = log_likelihood(logits_train_T, targets_train).cpu().detach().item()
                    rel_lik = np.exp(lik - mll)
                    return rel_lik

                x0 = np.zeros(n_classes)

                optim_bounds = [(0.0, 0.0)] * n_classes
                optim_bounds[k] = (None, None)

                constraints = {'type': 'ineq', 'fun': lambda x: const(x) - alpha}
                res = minimize(fun, x0, constraints=constraints, bounds=optim_bounds)
                bound.append(res.x)
            bounds.append(bound)

        # add the bounds to the logits_test to make predictions
        for k in range(n_classes):
            # both ``directions''
            for d in range(2):
                logits_id_T = logits_id + torch.tensor(bounds[k][d], device=logits_id.device)
                csets_id.append(F.softmax(logits_id_T, dim=1).cpu().detach().numpy())
                logits_ood_T = logits_ood + torch.tensor(bounds[k][d], device=logits_ood.device)
                csets_ood.append(F.softmax(logits_ood_T, dim=1).cpu().detach().numpy())
        rls.append([alpha] * (2 * n_classes))
    csets_id = np.array(csets_id)
    csets_ood = np.array(csets_ood)
    rls = np.array(rls).flatten()
    return csets_id, csets_ood, rls
