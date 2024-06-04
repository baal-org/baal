# Uncertainty Estimation Using a Single Deep Deterministic Neural Network

Authors: Joost van Amersfoort,  Lewis Smith, Yee Whye Teh and Yarin Gal

Paper: https://arxiv.org/abs/2003.02037
Code: https://github.com/y0ast/deterministic-uncertainty-quantification

## Problematic

Using MC-Dropout or ensembles are expensive to run. In this paper, the authors proposes to add a gradient penalty to reliably detect out-of-distribution data. In addition, the inference phase is fast because we can perform the uncertainty estimation in a single pass.

## Deterministic Uncertainty Quantification (DUQ)

DUQ uses a RBF Network to compute centroids for each class. The model is trained by minimizing the number of false assignation between a class centroid and its members.

### Loss funtion
 
For a model f, a centroid matrix W and a centroid e, we compute the similarity using a RBF kernel. Theta is a hyper parameter.

$K_c(f_\theta, e_c) = exp(-\frac{\frac{1}{n}\mid \mid W_cf_\theta(x) - e_c\mid\mid^2_2}{2\sigma^2})$

with this similarity we can make a prediction by selecting the centroid with the highest similarity.

The loss function is now simply

$L(x,y) = - \sum_c y_clog(K_c) + (1 - y_c)log(1-K_c)$,

where $K_c(f_\theta, e_c)=K_c$

After each batch, we update the centroid matrix using an exponential moving average.

### Regularization

To avoid feature collapse, the authors introduce a gradient penalty directly applied to $K_c$: 
$\lambda* (\mid\mid \nabla_x \sum_c K_c\mid\mid^2_2 - 1)^2$
where 1 is the Lipschitz constant. In their experiments, they use $\lambda=0.05$.

In summary, this simple technique is faster and better than ensembles. It also shows that RBF networks work on large datasets.
