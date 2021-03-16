# User guide

Welcome to BaaL's user guide where we will learn on bayesian deep learning and bayesian active learning.

In addition, we propose a [cheat sheet](./baal_cheatsheet.md) that will help users translate an equation to the equivalent BaaL representation.

### Notations and glossary

* Training dataset ``$`D_L`$``
* Pool, the unlabelled portion of the dataset ``$`D_U`$``
* Heuristic, the function that computes the uncertainty (ex. BALD) ``$`U `$``
* Active learning step, the sequence of training, selecting and labelling one or many examples.
* BALD, an heuristic that works well with deep learning models that are overconfident.
* Query size, the number of items to label between retraining.
* Iterations, number of Monte Carlo sampling to do.

## Active learning

Active learning is a field of machine learning where we reduce the labelling cost by only labelling the most informative examples. Datasets, especially in industry, contain many similar examples. Labelling these brings no information to the model and therefore are not useful.

To select the next example to label, we first train a machine learning model on the trained dataset. Then we compute the model's uncertainty on all unlabelled examples. The most uncertain is selected to be labelled.


## Bayesian active learning

Bayesian active learning builds upon active learning by framing the problem from a Bayesian point-of-view. In this case, we want to maximize the mutual information between a model's predictions and its parameters.

In addition, we will do this by sampling from the posterior distribution allowing us to better estimate the uncertainty. As an example, it is common to use MC-Dropout (Gal and Ghahramani, 2016) and BALD (Houlsby et al. 2013) to do this. The former allows us to draw from the posterior distribution and the latter estimates the mutual information. In recent years, new approaches were suggested to improve BALD such as BatchBALD (Kirsch et al, 2019) or ICAL (Jain et al. 2020), but they work on similar principles. 

## Open challenges

Active learning is a challenging field, many techniques work only on classification or are sensitive the data distribution. Simply doing random selection is strong in some cases such as semantic segmentation. 

In this context, it is vital to have some domain knowledge on the problem.

### Consequences of using AL

The effect of using active learning is an understudied problem.

While we know that AL creates more balanced datasets, better calibrated models and such. We do not know what is the effect of sampling bias in all settings. 

At ICLR 2020, Farquhar et al. showed that sampling bias produces biased risk estimators and they propose a new unbiased estimator that gets good results on simple models. We hope that work in this area continues so that we can better understand the impact of active learning.


```eval_rst
.. toctree::
    :caption: Resources
    :maxdepth: 1

    Literature review <../literature/index>
    Active learning dataset and training loop classes <../notebooks/active-learning.ipynb>
    Methods for approximating bayesian posteriors <../notebooks/posteriors.ipynb>
    Full active learning example <../notebooks/active_learning_process>
```

**References**
Kirsch, Andreas, Joost Van Amersfoort, and Yarin Gal. "Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning." NeurIPS (2019).
Jain, Siddhartha, Ge Liu, and David Gifford. "Information Condensing Active Learning." arXiv preprint arXiv:2002.07916 (2020).
Houlsby, Neil, et al. "Bayesian active learning for classification and preference learning." arXiv preprint arXiv:1112.5745 (2011).
Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.

---