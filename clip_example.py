"""Example script showing how to run the clip-based classifiers on CIFAR-10 images."""
import torchvision
import torchvision.transforms as T
import torch

from models import SigLIP2Classifier, BiomedCLIPClassifier, ClipClassifier
from utils import visualize_images

if __name__ == '__main__':
    # load CIFAR-10 without transforms (keeps PIL images)
    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=T.ToTensor()  # note: no normalization here this is left to the processing inside the models
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=3)
    print(dataset.classes)

    # a check that shows the images can be loaded in the same order as the data loader from dataset
    for i in range(3):
        instance = dataset[i]
        print(instance[0].shape, instance[1], dataset.classes[instance[1]])
        visualize_images(instance[0])

    # initialize the clip classifier
    siglip = SigLIP2Classifier(class_labels=dataset.classes, device="cpu", use_siglip_one=True)
    siglip2 = SigLIP2Classifier(class_labels=dataset.classes, device="cpu")
    biomed = BiomedCLIPClassifier(class_labels=dataset.classes, device="cpu")
    clip = ClipClassifier(class_labels=dataset.classes, device="cpu")

    # example batch
    for input, target in data_loader:
        visualize_images(input)

        # run siglip
        outputs = siglip(input)
        print(outputs)
        print(target.cpu().numpy())

        # run siglip2
        outputs = siglip2(input)
        print(outputs)
        print(target.cpu().numpy())

        # run biomed
        outputs = biomed(input).cpu().numpy()
        print(outputs)
        print(target.cpu().numpy())

        # run og. clip
        outputs = clip(input).cpu().numpy()
        print(outputs)
        print(target.cpu().numpy())
        break
