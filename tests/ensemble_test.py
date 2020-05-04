import pytest
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset

from baal.ensemble import EnsembleModelWrapper, ensemble_prediction

N_CLASS = 3


class DummyDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, item):
        return torch.randn(3, 32, 32), item % N_CLASS


class AModel(nn.Module):
    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 5, 5),
            nn.AdaptiveAvgPool2d(5),
            self.Flatten(),
            nn.Linear(125, N_CLASS)
        )

    def forward(self, x):
        return self.seq(x)

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)

@pytest.mark.parametrize(
    ("use_cuda", "n_ensemble"),
    [
        (False, 4),
        pytest.param(
            True, 4, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='No GPU')
        )
    ],
)
def test_prediction(use_cuda, n_ensemble):
    model = AModel()
    ensemble = EnsembleModelWrapper(model, nn.CrossEntropyLoss())
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    dataset = DummyDataset()
    if use_cuda:
        model.cuda()

    for i in range(n_ensemble):
        model.apply(weight_init)
        ensemble.train_on_dataset(dataset, optimizer, 1, 2, use_cuda)
        ensemble.add_checkpoint()
    assert len(ensemble._weights) == n_ensemble

    out = ensemble.predict_on_batch(dataset[0][0].unsqueeze(0), cuda=use_cuda)
    assert not all(torch.eq(out[..., 0], out[..., i]).all() for i in range(1, n_ensemble))

    out = ensemble_prediction(model=model, data=dataset[1][0].unsqueeze(0),
                              weights=ensemble._weights,
                              cuda=use_cuda)
    assert not all(torch.eq(out[..., 0], out[..., i]).all() for i in range(1, n_ensemble))


if __name__ == '__main__':
    pytest.main()
