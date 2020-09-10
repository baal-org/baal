# API Reference

## ModelWrapper

```eval_rst
.. autoclass:: baal.ModelWrapper
    :members:
```

## Active learning functionality

```eval_rst
.. autoclass:: baal.active.ActiveLearningDataset
    :members:

.. autoclass:: baal.active.ActiveLearningLoop
    :members:

.. autoclass:: baal.active.FileDataset
    :members:
```

## Calibration Wrapper

```eval_rst
.. autoclass:: baal.calibration.DirichletCalibrator
    :members:
```

## Heuristics

```eval_rst
.. autoclass:: baal.active.heuristics.AbstractHeuristic
    :members:

.. autoclass:: baal.active.heuristics.BALD

.. autoclass:: baal.active.heuristics.Random

.. autoclass:: baal.active.heuristics.Entropy
```
    
## Pytorch Lightning Compatibility

 ```eval_rst
.. autoclass:: baal.utils.pytorch_lightning.ActiveLearningMixin
    :members: predict_step, pool_loader

.. autoclass:: baal.utils.pytorch_lightning.ResetCallback
    :members: on_train_start

.. autoclass:: baal.utils.pytorch_lightning.BaalTrainer
    :members: predict_on_dataset, predict_on_dataset_generator
```