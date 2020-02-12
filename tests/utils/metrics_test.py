import os
from pathlib import Path
import numpy as np
import pytest
import shutil
import torch
from hypothesis import given
from torch_hypothesis import classification_logits_and_labels

from baal.utils.metrics import Loss, Accuracy, Precision, ECE, ClassificationReport, PRAuC, ECE_PerCLs


def test_loss():
    loss_calculator = Loss(average=True)
    loss = list()
    loss_calculator.reset()
    for _ in range(10):
        epoch_loss = np.random.uniform(0, 1)
        loss.append(epoch_loss)
        loss_calculator.update(epoch_loss)

    assert np.around(np.array(loss).mean(), decimals=4) == np.around(loss_calculator.value,
                                                                     decimals=4)

    loss_calculator = Loss(average=False)
    loss = list()
    loss_calculator.reset()
    for _ in range(10):
        epoch_loss = np.random.uniform(0, 1)
        loss.append(epoch_loss)
        loss_calculator.update(epoch_loss)

    assert np.allclose(np.array(loss), loss_calculator.value)


@given(y_ypred=classification_logits_and_labels(batch_size=(1, 32), n_classes=(2, 50)))
def test_that_accuracy_metric_calculates_multiclass_correctly(y_ypred):
    predicted, true = y_ypred
    predicted = predicted.softmax(-1) + torch.randn_like(predicted) / 100
    # test adding a batch:
    accuracy_calculator = Accuracy(average=True, topk=(1,))
    accuracy_calculator.update(predicted, true)
    assert np.allclose(
        accuracy_calculator.value, (true == predicted.argmax(-1)).float().mean().item()
    )
    # test adding all elements in a batch separately
    new_accuracy_calculator = Accuracy(average=True, topk=(1,))
    for x, y in zip(predicted, true):
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        new_accuracy_calculator.update(x, y)
    assert np.allclose(new_accuracy_calculator.value, accuracy_calculator.value)


@given(y_ypred=classification_logits_and_labels(batch_size=(1, 32), n_classes=(2, 50)))
def test_that_accuracy_metric_calculates_top_k_correctly(y_ypred):
    predicted, true = y_ypred
    predicted = predicted.softmax(-1) + torch.randn_like(predicted) / 100
    last_accuracy = 0
    for k in range(1, predicted.size(1) + 1):
        accuracy_calculator = Accuracy(average=True, topk=(k,))
        accuracy_calculator.update(predicted, true)
        assert 1 >= accuracy_calculator.value >= last_accuracy
        last_accuracy = accuracy_calculator.value

    accuracy_calculator = Accuracy(average=True, topk=tuple(range(1, predicted.size(1) + 1)))
    accuracy_calculator.update(predicted, true)
    assert all(
        val_1 >= val_0
        for val_0, val_1 in zip(accuracy_calculator.value[::2], accuracy_calculator.value[1::2])
    )


def test_that_accuracy_raises_errors_when_shapes_dont_match():
    predicted = torch.randn(5, 3)
    true = torch.tensor([0, 0, 0])
    accuracy_calculator = Accuracy(average=True, topk=(1,))
    with pytest.raises(ValueError):
        accuracy_calculator.update(predicted, true)


@given(y_ypred=classification_logits_and_labels(batch_size=(1, 32), n_classes=(2, 50)))
def test_that_precision_metric_calculates_multiclass_correctly(y_ypred):
    predicted, true = y_ypred
    predicted = predicted.softmax(-1) + torch.randn_like(predicted) / 100
    # test adding a batch:
    precision = Precision(num_classes=predicted.size(1), average=True)
    precision.update(predicted, true)
    true_positive = (
        (torch.zeros_like(predicted).scatter(1, predicted.argmax(-1).unsqueeze(-1), 1) > 0)
        & (torch.zeros_like(predicted).scatter(1, true.unsqueeze(-1), 1) > 0)
    ).sum(dim=0)
    all_positive = (torch.zeros_like(predicted).scatter(1, true.unsqueeze(-1), 1) > 0).sum(dim=0)
    manual_precision = true_positive.float() / all_positive.float()
    manual_precision[torch.isnan(manual_precision)] = 0
    # breakpoint()
    assert np.allclose(precision.value, manual_precision.mean())
    # test adding all elements in a batch separately
    new_precision = Precision(num_classes=predicted.size(1), average=True)
    for x, y in zip(predicted, true):
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        new_precision.update(x, y)
    assert np.allclose(new_precision.value, precision.value)


@given(y_ypred=classification_logits_and_labels(batch_size=(1, 32), n_classes=(2, 50)))
def test_that_precision_string_repr_doesnt_throw_errors(y_ypred):
    predicted, true = y_ypred
    predicted = predicted.softmax(-1) + torch.randn_like(predicted) / 100
    precision = Precision(num_classes=predicted.size(1), average=True)
    precision.update(predicted, true)
    assert "±" in str(precision)


@given(y_ypred=classification_logits_and_labels(batch_size=(1, 32), n_classes=(2, 50)))
def test_that_accuracy_string_repr_doesnt_throw_errors(y_ypred):
    predicted, true = y_ypred
    predicted = predicted.softmax(-1) + torch.randn_like(predicted) / 100
    accuracy = Accuracy(average=True, topk=(1, 2))
    accuracy.update(predicted, true)
    assert "±" in str(accuracy)


@given(y_ypred=classification_logits_and_labels(batch_size=(1, 32), n_classes=(2, 50)))
def test_auc(y_ypred):
    predicted, true = y_ypred
    num_classes = predicted.shape[1]
    met = PRAuC(num_classes=num_classes, n_bins=10, average=False)
    met.update(predicted, true)
    met.update(predicted, true)
    assert len(met.value) == num_classes

    met = PRAuC(num_classes=num_classes, n_bins=10, average=True)
    met.update(predicted, true)
    met.update(predicted, true)
    assert isinstance(met.value, float)


def test_accuracy():
    acc_calculator = Accuracy(average=False, topk=(2,))

    # start with multiclass classification
    pred = torch.FloatTensor([[0.4, 0.5, 0.1], [0.1, 0.8, 0.1]])
    target = torch.LongTensor([[2], [1]])

    assert pred.size() == torch.Size([2, 3])

    # check for length of the data
    for i in range(2):
        acc_calculator.update(output=pred[i, :].unsqueeze(0), target=target[i, :])

    assert acc_calculator.value.shape == (2, 1)
    assert np.allclose(acc_calculator.value, np.array([[0], [1]]))

    acc_calculator = Accuracy(average=True, topk=(2,))

    # check for length of the data
    for i in range(2):
        acc_calculator.update(output=pred[i, :].unsqueeze(0), target=target[i, :])

    assert acc_calculator.value == 0.5


def test_precision():
    prec_calculator = Precision(num_classes=3, average=True)

    # start with multiclass classification
    pred = torch.FloatTensor([[0.4, 0.5, 0.1], [0.1, 0.8, 0.1]])
    target = torch.LongTensor([[2], [1]])

    for i in range(2):
        prec_calculator.update(output=pred[i, :].unsqueeze(0), target=target[i, :].unsqueeze(0))

    assert round(prec_calculator.value, 2) == 0.33

    prec_calculator = Precision(num_classes=3, average=False)

    # start with multiclass classification
    pred = torch.FloatTensor([[0.4, 0.5, 0.1], [0.1, 0.8, 0.1]])
    target = torch.LongTensor([[2], [1]])

    for i in range(2):
        prec_calculator.update(output=pred[i, :].unsqueeze(0), target=target[i, :].unsqueeze(0))

    assert np.allclose(np.array(prec_calculator.value), np.array([0, 1, 0]))


def test_ece():
    ece_calculator = ECE(n_bins=3)

    # start with multiclass classification
    pred = torch.FloatTensor([[-40, 50, 10], [10, 80, 10]])
    target = torch.LongTensor([[2], [1]])

    for i in range(2):
        ece_calculator.update(output=pred[i, :].unsqueeze(0), target=target[i, :].unsqueeze(0))

    assert np.allclose(ece_calculator.samples, [0, 0, 2])
    assert np.allclose(ece_calculator.tp, [0, 0, 1])
    assert round(ece_calculator.value, 2) == 0.5

    pth = 'tmp'
    Path(pth).mkdir(exist_ok=True)
    ece_calculator.plot(pth=os.path.join(pth, 'figure.png'))
    assert os.path.exists(os.path.join(pth, 'figure.png'))
    shutil.rmtree(pth)

    ece_calculator.reset()

    # start with multiclass classification
    pred = torch.FloatTensor([[0.4, 0.5, 0.1], [0.1, 0.8, 0.1]])
    target = torch.LongTensor([[2], [1]])

    for i in range(2):
        ece_calculator.update(output=pred[i, :].unsqueeze(0), target=target[i, :].unsqueeze(0))

    assert np.allclose(ece_calculator.samples, [0, 1, 1])
    assert np.allclose(ece_calculator.tp, [0, 0, 1])
    assert round(ece_calculator.value, 2) == 0.25

def test_ece_percls():
    ece_calculator = ECE_PerCLs(n_cls=3, n_bins=3)

    # start with multiclass classification
    pred = torch.FloatTensor([[-40, 50, 10], [10, 80, 10]])
    target = torch.LongTensor([[2], [1]])

    for i in range(2):
        ece_calculator.update(output=pred[i, :].unsqueeze(0), target=target[i, :].unsqueeze(0))

    assert np.allclose(ece_calculator.samples, np.array([[0, 0, 0], [0, 0, 2], [0, 0, 0]]))
    assert np.allclose(ece_calculator.tp, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
    assert np.allclose(ece_calculator.value, np.array([0, 0.5, 0]))

    pth = 'tmp'
    Path(pth).mkdir(exist_ok=True)
    print(pth)
    ece_calculator.plot(pth=os.path.join(pth, 'figure.png'))
    assert os.path.exists(os.path.join(pth, 'figure.png'))
    shutil.rmtree(pth)

def test_classification_report():
    met = ClassificationReport(num_classes=3)
    pred = torch.FloatTensor([[0.4, 0.5, 0.1], [0.1, 0.8, 0.1]])
    target = torch.LongTensor([2, 1])
    met.update(pred, target)
    pred = torch.FloatTensor([[0.1, 0.5, 0.4], [0.8, 0.1, 0.1]])
    target = torch.LongTensor([1, 3])
    met.update(pred, target)
    assert np.allclose(met.value['accuracy'], [1.0, 0.666666666, 0.666666666])


if __name__ == '__main__':
    pytest.main()
