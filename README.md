# Bayesian Active Learning (Baal)
[![CircleCI](https://circleci.com/gh/ElementAI/baal.svg?style=svg&circle-token=aa12d3134798ff2bf8a49cebe3c855b96a776df1)](https://circleci.com/gh/ElementAI/baal)  [![Documentation Status](https://readthedocs.org/projects/baal/badge/?version=latest)](https://baal.readthedocs.io/en/latest/?badge=latest) [![Gitter](https://badges.gitter.im/eai-baal/community.svg)](https://gitter.im/eai-baal/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
---

<p align="left">
  <img height=15% width=25% src="https://github.com/ElementAI/baal/blob/master/docs/literature/images/repo_logo_25_no_corner.svg">
</p>

BaaL is an active learning library developed at
[ElementAI](https://www.elementai.com/). This repository contains techniques
and reusable components to make active learning accessible for all.

Read the documentation at https://baal.readthedocs.io.

Our paper can be read on [arXiv](https://arxiv.org/abs/2006.09916). It includes tips and tricks to make active learning usable in production.

In this [blog post](https://www.elementai.com/news/2019/element-ai-makes-its-bayesian-active-learning-library-open-source), we present our library.

For a quick introduction to BaaL and Bayesian active learning, please see this [presentation](https://drive.google.com/file/d/1icbTSbhl-Cs1X4k5XKYOEWfhkx9wBPdw/view?usp=sharing).




## Installation and requirements

BaaL requires `Python>=3.6`.

To install baal using pip: `pip install baal`

To install baal from source: `pip install -e .`

For requirements please see: _[requirements.txt](requirements.txt)_.

## What is Active Learning?

Active learning is a special case of machine learning in which a learning
algorithm is able to interactively query the user (or some other information
source) to obtain the desired outputs at new data points
(to understand the concept in more depth, refer to our [tutorial](https://baal.readthedocs.io/en/latest/)).


## BaaL Framework

At the moment BaaL supports the following methods to perform active learning.

- Monte-Carlo Dropout (Gal et al. 2015)
- MCDropConnect (Mobiny et al. 2019)
- Deep ensembles
- Semi-supervised learning

If you want to propose new methods, please submit an issue.


The **Monte-Carlo Dropout** method is a known approximation for Bayesian neural
networks. In this method, the dropout layer is used both in training and test
time. By running the model multiple times whilst randomly dropping weights, we calculate the uncertainty of the prediction using one of the uncertainty measurements in [heuristics.py](src/baal/active/heuristics/heuristics.py).

The framework consists of four main parts, as demonstrated in the flowchart below:

- ActiveLearningDataset
- Heuristics
- ModelWrapper
- ActiveLearningLoop

<p align="center">
  <img src="./docs/literature/images/Baalscheme.svg">
</p>

To get started, wrap your dataset in our _[**ActiveLearningDataset**](src/baal/active/dataset.py)_ class. This will ensure that the dataset is split into
`training` and `pool` sets. The `pool` set represents the portion of the training set which is yet
to be labelled.


We provide a lightweight object _[**ModelWrapper**](src/baal/modelwrapper.py)_ similar to `keras.Model` to make it easier to train and test the model. If your model is not ready for active learning, we provide Modules to prepare them. 

For example, the _[**MCDropoutModule**](src/baal/bayesian/dropout.py)_ wrapper changes the existing dropout layer
to be used in both training and inference time and the `ModelWrapper` makes
the specifies the number of iterations to run at training and inference.

In conclusion, your script should be similar to this:
```python
dataset = ActiveLearningDataset(your_dataset)
dataset.label_randomly(INITIAL_POOL)  # label some data
model = MCDropoutModule(your_model)
model = ModelWrapper(model, your_criterion)
active_loop = ActiveLearningLoop(dataset,
                                 get_probabilities=model.predict_on_dataset,
                                 heuristic=heuristics.BALD(shuffle_prop=0.1),
                                 ndata_to_label=NDATA_TO_LABEL)
for al_step in range(N_ALSTEP):
    model.train_on_dataset(dataset, optimizer, BATCH_SIZE, use_cuda=use_cuda)
    if not active_loop.step():
        # We're done!
        break
```


For a complete experiment, we provide _[experiments/](experiments/)_ to understand how to
write an active training process. Generally, we use the **ActiveLearningLoop**
provided at _[src/baal/active/active_loop.py](src/baal/active/active_loop.py)_.
This class provides functionality to get the predictions on the unlabeled pool
after each (few) epoch(s) and sort the next set of data items to be labeled
based on the calculated uncertainty of the pool.


### Re-run our Experiments

```bash
nvidia-docker build [--target base_baal] -t baal .
nvidia-docker run --rm baal python3 experiments/vgg_mcdropout_cifar10.py 
```

### Use BaaL for YOUR Experiments

Simply clone the repo, and create your own experiment script similar to the
example at [experiments/vgg_experiment.py](experiments/vgg_experiment.py). Make sure to use the four main parts
of BaaL framework. _Happy running experiments_

### Dev install

Simply build the Dockerfile as below:

```bash
git clone git@github.com:ElementAI/baal.git
nvidia-docker build [--target base_baal] -t baal-dev .
```

Now you have all the requirements to start contributing to BaaL. _**YEAH!**_

### Contributing!

To contribute, see [CONTRIBUTING.md](./CONTRIBUTING.md).


### Who We Are!

"There is passion, yet peace; serenity, yet emotion; chaos, yet order."

At ElementAI, the BaaL team tests and implements the most recent papers on uncertainty estimation and active learning.
The BaaL team is here to serve you!

- [Parmida Atighehchian](mailto:parmida@elementai.com)
- [Frédéric Branchaud-Charron](mailto:frederic.branchaud-charron@elementai.com)
- [Jan Freyberg](mailto:jan.freyberg@gmail.com)
- [Rafael Pardinas](mailto:rafael.pardinas@elementai.com)
- [Lorne Schell](mailto:lorne.schell@elementai.com)

### How to cite

If you used BaaL in one of your project, we would greatly appreciate if you cite this library using this Bibtex:

```
@misc{atighehchian2019baal,
  title={BaaL, a bayesian active learning library},
  author={Atighehchian, Parmida and Branchaud-Charron, Frederic and Freyberg, Jan and Pardinas, Rafael and Schell, Lorne},
  year={2019},
  howpublished={\url{https://github.com/ElementAI/baal/}},
}
```

### Licence
To get information on licence of this API please read [LICENCE](./LICENSE)
