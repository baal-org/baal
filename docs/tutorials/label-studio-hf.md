# Text Classification

*By: Frédéric Branchaud-Charron (@Dref360)*

In this tutorial, we will see how to use Baal inside of Label Studio, a widely known labelling tool.

By using Bayesian active learning in your labelling setup, you will be able to label only the most informative examples
and avoid duplicates and easy examples.

This is also a good way to start the conversation between your labelling team and your machine learning team as they
need to communicate early in the process!

We will built upon Label
Studio's [Text classification](https://github.com/heartexlabs/label-studio-ml-backend/tree/master/label_studio_ml/examples/simple_text_classifier)
example, so be sure to download it and try to run it before adding Baal to it. The full example can be
found [here](https://gist.github.com/Dref360/448d1d72e0f6f050b154cdb5a1ad909e).

More info:

* [Baal documentation](https://baal.readthedocs.io/en/latest/)
* [Bayesian Deep Learning cheatsheet](https://baal.readthedocs.io/en/latest/user_guide/baal_cheatsheet/)

Support:

* [Github](https://github.com/baal-org/baal)
* [Slack](https://join.slack.com/t/baal-world/shared_invite/zt-z0izhn4y-Jt6Zu5dZaV2rsAS9sdISfg)

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

`pip install baal[nlp]`

## Modifying `simple_text_classifier.py`

The overall changes are pretty minor, so we will go step by step, specifying the class and method we are modifying.
Again, the full script is available [here](https://gist.github.com/Dref360/448d1d72e0f6f050b154cdb5a1ad909e).

### Model

The simplest way of doing Bayesian uncertainty estimation in active learning is MC-Dropout (Gal and Ghahramani, 2015)
which requires Dropout layers. Fortunately, HuggingFace models come with one Dropout layer, but feel free to add more!

```python
from baal.bayesian.dropout import patch_module

# SimpleTextClassifier
def reset_model(self):
    BASE_MODEL = 'distilbert-base-uncased'
    use_cuda = torch.cuda.is_available()
    
    # Load model using distilbert as base.
    self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=BASE_MODEL,
                                                                    num_labels=self.num_classes)
    self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=BASE_MODEL)
    self.model = patch_module(self.model)
    if use_cuda:
        self.model.cuda()
    
    # Use BaalTransformerTrainer to replace HF Trainer.
    self.trainer = BaalTransformersTrainer(model=self.model)
```

### Create dataset

We modify the function `make_dataset` to be fed to HuggingFace:

```python
from baal.active.dataset.nlp_datasets import HuggingFaceDatasets
from datasets import Dataset

def make_dataset(self, texts, labels):
    dataset = Dataset.from_dict({
        'text': texts,
        'label': labels
    })
    return HuggingFaceDatasets(dataset, self.tokenizer, target_key='label', input_key='text', max_seq_len=128, )
```

### Training loop

We can simplify the training loop by using `HuggingFace`.
There is a lot of data manipulation that needs to be done, but the actual training can be done as such:

```python
# SimpleTextClassifier
def train(self, annotations, num_epochs=5):
    ...
    # train the model
    print(f'Start training on {len(input_texts)} samples')
    self.reset_model()
    self.trainer.train_dataset = self.make_dataset(input_texts, output_labels_idx)
    self.trainer.train()
    ...
```

### Prediction

We draw multiple predictions from the model's parameter distribution using MC-Dropout. In this script we will make
20 predictions per example. Next, we use BALD (Houlsby et al, 2013) to estimate the epistemic uncertainty of each item. 

Finally, we notify Label Studio to prioritise uncertain items by adding a `score` field to the response.

```python 
# SimpleTextClassifier
NUM_DRAWS = 20
def predict(self, tasks, **kwargs):
    # collect input texts
    input_texts = []
    for task in tasks:
        input_text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        input_texts.append(input_text)
    dataset = self.make_dataset(input_texts, [0] * len(input_texts))
    
    # get model predictions
    probabilities = self.trainer.predict_on_dataset(dataset, NUM_DRAWS)
    uncertainties = BALD().get_uncertainties(probabilities).tolist()
    predictions = probabilities.mean(-1)
    predicted_label_indices = np.argmax(predictions, axis=1).tolist()
    predictions = []
    for idx, score in zip(predicted_label_indices, uncertainties):
        predicted_label = self.labels[idx]
        # prediction result for the single task
        result = [{
            'from_name': self.from_name,
            'to_name': self.to_name,
            'type': 'choices',
            'value': {'choices': [predicted_label]}
        }]

        # expand predictions with their scores for all tasks
        predictions.append({'result': result, 'score': score})

    return predictions
```

## Launching LabelStudio

Following Label Studio tutorial, you can start your ML Backend as usual:

Environment:

* `export LABEL_STUDIO_HOSTNAME=http://localhost:8080`
* `export LABEL_STUDIO_ML_BACKEND_V2=True`
* `export API_KEY=${YOUR_API_KEY}` 

Dependencies:

* `pip install baal[nlp]`

How to:

* Run `label-studio-ml init my_ml_backend --script label_studio_baal_hf.py --force`
* Run `label-studio-ml start my_ml_backend`
* Run `label-studio start my-annotation-project --init --ml-backend http://localhost:9090`

In the Settings, do not forget to checkbox all boxes:

![](https://i.imgur.com/4vcj2u8.png)

and to use active learning, order by Predictions score:

![](https://i.imgur.com/cGVngqw.png)

## Results!

To test our methodology, we used the same parameters to perform active learning on [CLINC-OOS](https://huggingface.co/datasets/clinc_oos).

In [Kirsch et al. 2022](https://arxiv.org/abs/2106.12059), we compare Entropy to Uniform sampling on this dataset:

<p align="center">
    <img src="https://i.imgur.com/t1nNyZL.png" width="50%">
</p>

Note that for this particular dataset, BALD is not recommended.

**In conlusion**, we can now use Bayesian active learning in Label Studio which would help your labelling process be
more efficient. Please do not hesitate to reach out on our Slack or on Label
Studio's [Slack](http://slack.labelstud.io.s3-website-us-east-1.amazonaws.com/?source=site-header) if you have feedback
or questions.
