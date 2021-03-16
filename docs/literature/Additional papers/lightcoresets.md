# Scalable k-Means Clustering via Lightweight Coresets

**Note** This review will focus on the Coreset approch and not so much on the k-means.

[Paper pdf](https://ai.google/research/pubs/pub46906.pdf)

This paper presents a novel Coreset algorithm called *Light Coreset*. 

Let ``$`X`$`` be the dataset, ``$`d`$`` a distance function and ``$`\mu(X)`$`` the mean of the dataset per feature.

We compute the distribution ``$`q`$``with:

``$`q(x) = 0.5 * \frac{1}{\vert X  \vert} + 0.5 * \frac{d(x, \mu(X))^2}{\sum_{x' \in X} d(x', \mu(X))^2}`$``,
where ``$`x \in X`$``. 

We can then select ``$`m`$`` samples by sampling from this distribution. For their experiments, they used the L2 distance for *d*.

Let A be the first part of the equation ``$`q`$`` and B the second. The authors offers the following explanation :

>The first component (A) is the uniform distribution and ensures
that all points are sampled with nonzero probability. The second
component (B) samples points proportionally to their squared
distance to the mean of the data. The intuition is that the points
that are far from the mean of the data have a potentially large
impact on the quantization error of a clustering. The component
(B) ensures that these potentially important points are sampled
frequently enough.

#### Advantages
* Their method is by order of magnitude faster than standard coresets and when tested on some UCI datasets, they perform better as well.
