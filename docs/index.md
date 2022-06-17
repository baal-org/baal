```eval_rst
.. baal documentation master file, created by
   sphinx-quickstart on Thu Apr  4 14:15:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
```

# Welcome to the documentation for baal (**ba**yesian **a**ctive **l**earning)

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/ElementAI/baal" data-size="large" data-show-count="true" aria-label="Star ElementAI/baal on GitHub">Star</a>

Baal is a Bayesian active learning library.
We provide methods to estimate sampling from the posterior distribution
in order to maximize the efficiency of labelling during active learning. Our library is suitable for research and industrial applications.

To know more on what is Bayesian active learning, see our [User guide](user_guide/index.md).

We are a member of Pytorch's ecosystem, and we welcome contributions from the community.
If you have any question, we are reachable on [Slack](https://join.slack.com/t/baal-world/shared_invite/zt-z0izhn4y-Jt6Zu5dZaV2rsAS9sdISfg).

## Support

For support, we have several ways to help you:

* Our [FAQ](faq.md)
* Submit an issue on Github [here](https://github.com/ElementAI/baal/issues/new/choose)
* Join our [Slack](https://join.slack.com/t/baal-world/shared_invite/zt-z0izhn4y-Jt6Zu5dZaV2rsAS9sdISfg)!

```eval_rst
.. toctree::
    :caption: Learn more about Baal
    :maxdepth: 1

    User guide <user_guide/index>
    Active learning dataset and training loop classes <notebooks/fundamentals/active-learning.ipynb>
    Methods for approximating bayesian posteriors <notebooks/fundamentals/posteriors.ipynb>
    API Index <api/index>
    FAQ <faq>

.. toctree ::
    :caption: Tutorials
    :maxdepth: 1
    
    How to use Baal with Label Studio <tutorials/label-studio.md>
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
   
## Indices and tables

```eval_rst
* :ref:`genindex`
* :ref:`search`
```
