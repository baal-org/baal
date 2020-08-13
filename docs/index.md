```eval_rst
.. baal documentation master file, created by
   sphinx-quickstart on Thu Apr  4 14:15:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
```

# Welcome to the documentation for baal (**ba**yesian **a**ctive **l**earning)

<a href="https://github.com/ElementAI/baal">
<img src="http://www.pngall.com/wp-content/uploads/2016/04/Github-Free-PNG-Image.png" width=64 /></a>

baal aims to implement active learning using metrics of uncertainty derived
from approximations of bayesian posteriors in neural networks.

Therefore, baal has several components:

```eval_rst
.. toctree::
    :caption: Components of baal
    :maxdepth: 2

    Active learning dataset and training loop classes <notebooks/active-learning.ipynb>
    Methods for approximating bayesian posteriors <notebooks/posteriors.ipynb>
    API Index <api/index>

.. toctree ::
    :caption: Tutorials
    :maxdepth: 1
    
    Use BaaL in production (Classification) <notebooks/baal_prod_cls.ipynb>
    Use BaaL with SkLearn (Classification) <notebooks/sklearn_tutorial.ipynb>
    
.. toctree ::
    :caption: Reports
    :maxdepth: 1
    
    Combining calibration and variational inference for active learning <reports/dirichlet_calibration>

.. toctree::
    :caption: Literature and support
    :maxdepth: 2

    Background literature <literature/index>
```
   
## Indices and tables

```eval_rst
* :ref:`genindex`
* :ref:`search`
```