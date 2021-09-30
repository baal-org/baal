import numpy as np
import torch.backends
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.transforms import transforms

try:
    import flash
except ImportError:
    raise ImportError(
        "`lightning-flash` library is required to run this example."
        " pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git#egg=lightning-flash[image]'"
    )
from flash import DataModule
from flash.image import ImageClassifier
from flash.core.classification import Probabilities
from flash.image.classification.integrations.baal import ActiveLearningDataModule, ActiveLearningLoop

from baal.active import get_heuristic

IMG_SIZE = 128

train_transforms = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
test_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010))])


class DataModule_(DataModule):

    @property
    def num_classes(self):
        return 10


train_set = datasets.CIFAR10(".", train=True, transform=train_transforms, download=True)
test_set = datasets.CIFAR10(".", train=False, transform=test_transforms, download=True)
dm = DataModule_(train_dataset=train_set, test_dataset=test_set)
active_dm = ActiveLearningDataModule(dm,
                                     heuristic=get_heuristic('bald'),
                                     num_label_randomly=1024,
                                     val_split=0.1)


loss_fn = CrossEntropyLoss()
head = torch.nn.Sequential(
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(512, active_dm.num_classes),
)
model = ImageClassifier(num_classes=10,
                        head=head,
                        backbone="vgg16",
                        pretrained=True,
                        loss_fn=loss_fn,
                        optimizer=torch.optim.SGD,
                        optimizer_kwargs={"lr": 0.001,
                                          "momentum": 0.9,
                                          "weight_decay": 0},
                        learning_rate=1,  # we don't use learning rate here since it is initialized in the optimizer.
                        serializer=Probabilities(),)

if __name__ == "__main__":
    trainer = flash.Trainer(max_epochs=2500,
                            default_root_dir='./ckpt')
    active_learning_loop = ActiveLearningLoop(label_epoch_frequency=40,
                                              inference_iteration=20)
    active_learning_loop.connect(trainer.fit_loop)
    trainer.fit_loop = active_learning_loop
    trainer.finetune(model, datamodule=active_dm, strategy="freeze")