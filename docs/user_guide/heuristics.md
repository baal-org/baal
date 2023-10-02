# Active learning heuristics

**Heuristics** take a set of predictions and outputs the order in which they should be labelled.

A simple heuristic would be to prioritize items where the model had low confidence.
We will cover the **PowerBALD**, **BALD** and **Entropy**.

### PowerBALD :material-check-decagram:

PowerBALD and PowerEntropy (Kirsch et al. 2022) are the latest state-of-the-art in Bayesian active learning.
The idea is simple; BALD suffers from a lack of diversity in its recommendation. If your dataset has many near-duplicates, they will be recommended at the same time.
This is a side-effect of Batch active learning, where we label multiple items between training.

To fix this, PowerBALD adds noise to the uncertainty:

$$
PowerBALD(x) = BALD(x) + \epsilon, \epsilon \in {\cal Gumbel(0,T)}
$$

$T$ is an hyperparameter for the temperature of the distribution and is set to 1 by default.

Another way of looking at it, we sample from uncertainty distribution.

$$
acquisitions = Categorical(\frac{BALD(x_i)^T}{\sum_j BALD(x_j)}), i \in D_{pool}.
$$

By doing this, we can add a little bit of diversity to our selection and greatly improve our performance on most datasets.
This approach is on-par with methods like BADGE or BatchBALD, while being hundreds of time faster.

??? Code

    To use PowerBALD in your code, it is as simple as:

    ```
    from baal.active.heuristics.aliases import PowerBALD

    heuristic = PowerBALD(query_size=QUERY_SIZE, temperature=1.0)
    ```

### BALD

Bayesian active learning by disagreement or BALD (Houslby et al. 2013) is the basis of most modern active learning heuristics.

From a Bayesian model $f$, we draw $I$ predictions per sample $x$.

Then, we want to maximize the mutual information between a prediction and the model's parameters.
This is done by looking at how the predictions are disagreeing with each others.
If the prediction "flips" often, it means that the item is close to a decision boundary and thus hard to fit.

### Entropy

The goal of this heuristic is to maximize information. To do so, we will compute the entropy of each prediction before ordering them.

Let $p_{c}(x)$ be the probability of input $x$ to be from class $c$. The entropy can be computed as:

$$
H(x) = \sum_c^C p_c(x)
$$

This score reflects the informativeness of knowing the true label of $x$.
Naturally the next item to label would be $argmax_{x \in {\cal D}} H(x)$, where ${\cal D} is our dataset$

A drawback of this method is that it doesn't differentiate between _aleatoric_ uncertainty and _epistemic_ uncertainty.
To do so, we will use BALD
