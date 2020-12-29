```eval_rst
.. baal documentation master file, created by
   sphinx-quickstart on Thu Apr  4 14:15:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
```

# Welcome to the documentation for baal (**ba**yesian **a**ctive **l**earning)

<a href="https://github.com/ElementAI/baal">
<img src="_static/images/GitHub-Mark-64px.png" style="width:30px;height:30px;" /></a>

BaaL is a Bayesian active learning library. We provide methods to estimate sampling from the posterior distribution
in order to maximize the effiency of labelling during active learning.

To know more on what is Bayesian active learning, see our [User guide](user_guide/index.md).

We are member of Pytorch's ecosystem and we welcome contributions from the community.
If you have any question, we are reachable on [Gitter](https://gitter.im/eai-baal/community#).

# Support

For support, we have several ways to help you:

* Our [FAQ](faq.md)
* Submit an issue on Github [here](https://github.com/ElementAI/baal/issues/new/choose)
* Join our [Gitter](https://gitter.im/eai-baal/community#)!

Baal has several components:

```eval_rst
.. toctree::
    :caption: Components of baal
    :maxdepth: 3

    Active learning dataset and training loop classes <notebooks/active-learning.ipynb>
    Methods for approximating bayesian posteriors <notebooks/posteriors.ipynb>
    API Index <api/index>
    FAQ <faq>

.. toctree ::
    :caption: Tutorials
    :maxdepth: 1
    
    Use BaaL in production (Classification) <notebooks/baal_prod_cls.ipynb>
    Use BaaL with Pytorch Lightning <notebooks/pytorch_lightning.ipynb>
    Use BaaL with SkLearn (Classification) <notebooks/sklearn_tutorial.ipynb>
    
.. toctree ::
    :caption: Reports
    :maxdepth: 1
    
    Combining calibration and variational inference for active learning <reports/dirichlet_calibration>

.. toctree::
    :caption: Literature and support
    :maxdepth: 1

    Background literature <literature/index>
```
   
## Indices and tables

```eval_rst
* :ref:`genindex`
* :ref:`search`
```
