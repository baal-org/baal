# Welcome to Baal (**ba**yesian **a**ctive **l**earning)

Baal is a Bayesian active learning library.
We provide methods to estimate sampling from the posterior distribution
in order to maximize the efficiency of labelling during active learning. Our library is suitable for research and industrial applications.

To know more on what is Bayesian active learning, see our [User guide](user_guide/index.md).

We are a member of Pytorch's ecosystem, and we welcome contributions from the community.
If you have any question, we are reachable on [Slack](https://join.slack.com/t/baal-world/shared_invite/zt-z0izhn4y-Jt6Zu5dZaV2rsAS9sdISfg).

## Support

For support, we have several ways to help you:

* Our [FAQ](support/faq.md)
* Submit an issue on Github [here](https://github.com/ElementAI/baal/issues/new/choose)
* Join our [Slack](https://join.slack.com/t/baal-world/shared_invite/zt-z0izhn4y-Jt6Zu5dZaV2rsAS9sdISfg)!

## :material-file-tree: Learn more about Baal

* [:material-link: User Guide](user_guide)
* [:material-book-education: Active learning dataset and training loop classes](notebooks/fundamentals/active-learning.ipynb)
* [:material-book-education: Methods for approximating bayesian posteriors](notebooks/fundamentals/posteriors.ipynb)
* [:material-link: API Index](api)
* [:material-help: FAQ](support/faq.md)

## :material-file-tree: Industry
* [:material-book-education: Active learning dataset and training loop classes](notebooks/fundamentals/active-learning.ipynb)

.. toctree ::
    :caption: Tutorials
    :maxdepth: 1
    
    How to use BaaL with Label Studio <tutorials/label-studio.md>
    How to do research and plot progress <notebooks/active_learning_process.ipynb>
    How to use in production <notebooks/baal_prod_cls.ipynb>
    How to use deep ensembles <notebooks/deep_ensemble.ipynb>

.. toctree ::
    :caption: Compatibility with other libraries
    :maxdepth: 1
    
    How to use with Pytorch Lightning <https://devblog.pytorchlightning.ai/active-learning-made-simple-using-flash-and-baal-2216df6f872c>
    How to use with HuggingFace <notebooks/compatibility/nlp_classification.ipynb>
    How to use with Scikit-Learn <notebooks/compatibility/sklearn_tutorial.ipynb>
    
.. toctree ::
    :caption: Technical Reports
    :maxdepth: 1
    
    Combining calibration and variational inference for active learning <reports/dirichlet_calibration>
    Double descend in active learning <reports/double_descend.md>
    Can active learning mitigate bias in datasets <notebooks/fairness/ActiveFairness.ipynb>

.. toctree::
    :caption: Literature and support
    :maxdepth: 2

    Background literature <literature/index>
    Cheat Sheet <user_guide/baal_cheatsheet>
```
