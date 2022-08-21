# Active learning heuristics

**Heuristics** take a set of predictions and outputs the order in which they should be labelled.

A simple heuristic would be to prioritize items where the model had low confidence.
We will cover the two main heuristics: **Entropy** and **BALD**.


### Entropy

The goal of this heuristic is to maximize information. To do so, we will compute the entropy of each prediction before ordering them.

Let $p_{c}(x)$ be the probability of input $x$ to be from class $c$. The entropy can be computed as:

$$
H(x) = \sum_c^C p_c(x)
$$

This score reflects the informativeness of knowing the true label of $x$.
Naturally the next item to label would be $argmax_{x \in {\cal D}} H(x)$, where ${\cal D} is our dataset$ 

A drawback of this method is that it doesn't differentiate between *aleatoric* uncertainty and *epistemic* uncertainty.
To do so, we will use BALD

### BALD

Bayesian active learning by disagreement or BALD (Houslby et al. 2013) is the basis of most modern active learning heuristics.

From a Bayesian model $f$, we draw $I$ predictions per sample $x$.

Then, we want to maximize the mutual information between a prediction and the model's parameters. This is done by looking at how the predictions are disagreeing with each others.
If the prediction "flips" often, it means that the item is close to a decision boundary and thus hard to fit.

 ${\cal I}[y, \theta \mid x, {\cal D}] = {\cal H}[y \mid x, {\cal D}] - {\cal E}_{p(\theta \mid {\cal D})}[{\cal H}[y \mid x, \theta]]$ 
