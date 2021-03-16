# The theory behind Bayesian active learning

In this document, we keep a list of the papers to get you started in Bayesian deep learning and Bayesian active learning.

We hope to include a summary for each of then in the future, but for now we have this list with some notes.


### How to estimate uncertainty in Deep Learning networks

* [Excellent tutorial from AGW on Bayesian Deep Learning](https://icml.cc/virtual/2020/tutorial/5750)
  * This is inspired by his publication [Bayesian Deep Learning and a Probabilistic Perspective of Generalization](https://arxiv.org/abs/2002.08791)
* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf) (Gal and Ghahramani, 2016)
  * This describes Monte-Carlo Dropout, a way to estimate uncertainty through stochastic dropout at test time
* [Bayesian Uncertainty Estimation for Batch Normalized Deep Networks](https://arxiv.org/abs/1802.06455) (Teye et al. 2018)
  * This describes Monte-Carlo BatchNorm, a way to estimate uncertainty through random batch norm parameters at test time
* [Bayesian Deep Learning and a Probabilistic Perspective of Generalization](https://arxiv.org/abs/2002.08791) (Gordon Wilson and Izmailov, 2020)
  * Presentation of multi-SWAG a mix between VI and Ensembles.
* [Advances in Variational inference](https://arxiv.org/pdf/1711.05597.pdf) (Zhang et al, 2018)
  * Gives a quick introduction to VI and the most recent advances.
* [A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/1902.02476) (Maddox et al. 2019)
  * Presents SWAG, an easy way to create ensembles.

    
    

### Bayesian active learning
* [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf) (Gal and Islam and Ghahramani, 2017)
  * Fundamental paper on how to do Bayesian active learning. A must read.
* [Sampling bias in active learning](http://cseweb.ucsd.edu/~dasgupta/papers/twoface.pdf) (Dasgupta 2009)
  * Presents sampling bias and how to solve it by combining heuristics and random selection.

* [Bayesian Active Learning for Classification and  Preference Learning](https://arxiv.org/pdf/1112.5745.pdf) (Houlsby et al. 2011)
  * Fundamental paper on one of the main heuristic BALD.


### Bayesian active learning on NLP

* [Deep Bayesian Active Learning for Natural Language Processing: Results of a Large-Scale Empirical Study](https://arxiv.org/abs/1808.05697) (Siddhant and Lipton, 2018)
  * Experimental paper on how to use Bayesian active learning on NLP tasks.
