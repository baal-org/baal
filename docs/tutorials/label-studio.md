# Using Baal with Label Studio

*By: Frédéric Branchaud-Charron (@Dref360)*

In this tutorial, we will see how to use Baal inside of Label Studio, a widely known labelling tool.

By using Bayesian active learning in your labelling setup, you will be able to label only the most informative examples.
This will avoid labelling duplicates and easy examples.

This is also a good way to start the conversation between your labelling team and your machine learning team as they
need to communicate early in the process!

We will built upon Label
Studio's [Pytorch transfer learning](https://github.com/heartexlabs/label-studio-ml-backend/blob/master/label_studio_ml/examples/pytorch_transfer_learning.py)
example, so be sure to download it and try to run it before adding Baal to it. The full example can be
found [here](https://gist.github.com/Dref360/288845b2fbb0504e4cfc216a76b547e7).

More info:

* [Baal documentation](https://baal.readthedocs.io/en/latest/)
* [Bayesian Deep Learning cheatsheet](https://baal.readthedocs.io/en/latest/user_guide/baal_cheatsheet.html)

Support:

* [Github](https://github.com/ElementAI/baal)
* [Gitter](https://gitter.im/eai-baal/community)

## Installing Baal

To install Baal, you will need to add `baal` in
the [generated `Dockerfile`](https://github.com/heartexlabs/label-studio-ml-backend/blob/master/label_studio_ml/default_configs/Dockerfile)
.

```dockerfile
# Dockerfile
RUN pip install --no-cache \
                -r requirements.txt \
                uwsgi==2.0.19.1 \
                supervisor==4.2.2 \
                label-studio==1.0.2 \
                baal \
                click==7.1.2 \
                git+https://github.com/heartexlabs/label-studio-ml-backend
```

and when developing, you should install Baal in your local environment.

`pip install baal==1.3.0`

## Modifying `pytorch_transfer_learning.py`

The overall changes are pretty minor, so we will go step by step, specifying the class and method we are modifying.
Again, the full script is available [here](https://gist.github.com/Dref360/288845b2fbb0504e4cfc216a76b547e7).

### Model

The simplest way of doing Bayesian uncertainty estimation in active learning is MC-Dropout (Gal and Ghahramani, 2015)
which requires Dropout layers. To use this, we use VGG-16 instead of the default ResNet-18.

```python
from baal.bayesian.dropout import patch_module

# ImageClassifier.__init__
self.model = models.vgg16(pretrained=True)
last_layer_idx = 6
num_ftrs = self.model.classifier[last_layer_idx].in_features
self.model.classifier[last_layer_idx] = nn.Linear(num_ftrs, num_classes)
# Set Dropout layers for MC-Dropout
self.model = patch_module(self.model)
```

Next, we will wrap our model using `baal.modelwrapper.ModelWrapper` from Baal which will simplify the different loops.
If you use another framework, feel free to checkout
our [Pytorch Lightning integration](https://baal.readthedocs.io/en/latest/notebooks/compatibility/pytorch_lightning.html)
and our [HuggingFace integration](https://baal.readthedocs.io/en/latest/notebooks/compatibility/nlp_classification.html)
.

```python
# ImageClassifier.__init__
self.wrapper = ModelWrapper(self.model, self.criterion)
```

### Training loop

We can simplify the training loop by using `ModelWrapper`.

**NOTE:** `train` now receives a `torch.utils.data.Dataset` instead of a `Dataloader`.

```python
# ImageClassifier
def train(self, dataset, num_epochs=5):
    since = time.time()
    self.wrapper.train_on_dataset(dataset, self.optimizer, batch_size=32,
                                  epoch=num_epochs,
                                  use_cuda=use_cuda)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return self.model
```

### Prediction

We can draw multiple predictions from the model's parameter distribution using MC-Dropout. In this script we will make
20 predictions per example:

```python 
# ImageClassifier
def predict(self, image_urls):
    images = torch.stack([get_transformed_image(url) for url in image_urls])
    with torch.no_grad():
        return self.wrapper.predict_on_batch(images, iterations=20, cuda=use_cuda)

```

In `ImageClassifierAPI` we will leverage this set of predictions and BALD (Houlsby et al, 2013) to estimate the model's
uncertainty and to get the "average prediction" which would be more trustworthy:

```python
# ImageClassifierAPI.predict

logits = self.model.predict(image_urls)
average_prediction = logits.mean(-1)
predicted_label_indices = np.argmax(average_prediction, axis=1)
# Get the uncertainty from the predictions.
predicted_scores = BALD().get_uncertainties(logits)
```

## Launching LabelStudio

Following Label Studio tutorial, you can start your ML Backend as usual.
In the Settings, do not forget to checkbox all boxes:

![](https://i.imgur.com/4vcj2u8.png)

and to use active learning, order by Predictions score:

![](https://i.imgur.com/cGVngqw.png)

## Labeling in action!

To test this setup, we imported in Label Studio a subset of [MIO-TCD](http://podoce.dinf.usherbrooke.ca/), a dataset
that is similar to real production data. This dataset suffers from heavy class imbalance, the class *car* represents 90%
of all images in the dataset.

After labelling randomly 100 images, I start training my model. On a subset of 10k unlabelled images, we get the
following most uncertain predictions:

| ![](https://i.imgur.com/7LuI4qf.jpg) | ![](https://i.imgur.com/YjViSz6.jpg) | ![]( https://i.imgur.com/9SyYMfR.jpg) |
|--------------------------------------|--------------------------------------|---------------------------------------|
| Articulated Truck                    | Bicycle                              | Background                            |

The model has seen enough cars, and wants to label new classes as they would be the most informatives. If we continue
labelling, we will see a similar behavior, where the class *car* is undersampled and the others are oversampled.

In [Atighehchian et al. 2019](https://arxiv.org/abs/2006.09916), we compare BALD to Uniform sampling on this dataset and
we get better performance on underrepresented classes.
In the image below, we have the F1 for two underrepresented classes:

![](https://i.imgur.com/dWP7QIJ.png)

**In conlusion**, we can now use Bayesian active learning in Label Studio which would help your labelling process be
more efficient. Please do not hesitate to reach out on our Gitter or on Label
Studio's [Slack](http://slack.labelstud.io.s3-website-us-east-1.amazonaws.com/?source=site-header) if you have feedback
or questions.
