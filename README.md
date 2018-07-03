# NIPS Adversarial Vision Challenge

### Installation

To install the package simply run:

`pip install adversarial-vision-challenge`

We test using Python 2.7, 3.4, 3.5 and 3.6. 
Other Python versions might work as well. 
**We recommend using Python 3**!

The test-scripts you can run locally before submission, will work with Python `> 3.4` though due to the `crowdai-repo2docker` dependency. See `Running Tests Scripts`-section below for more detailed information.


### Implementing a model server

To run a model server, load your model and wrap it as a [foolbox model](https://foolbox.readthedocs.io/en/latest/modules/models.html).
Then pass the foolbox model to the `model_server` function.

```python
from adversarial_vision_challenge import model_server

foolbox_model = load_your_foolbox_model()
model_server(foolbox_model)
```

### Implementing an attack

To run an attack, use the `load_model` method to get a model instance that is callable to get the predicted labels.

```python
from adversarial_vision_challenge.utils import read_images, store_adversarial
from adversarial_vision_challenge.utils import load_model

model = load_model()

for (file_name, image, label) in read_images():
    # model is callable and returns the predicted class,
    # i.e. 0 <= model(image) < 200

    # run your adversarial attack
    adversarial = your_attack(model, image, label)

    # store the adversarial
    store_adversarial(file_name, adversarial)
    
 ### Running the tests scripts
```

### Running Tests Scripts

Although you can use this package with python 2 if you develop in python 2, 
you need to install and run the the **`avc-*` binaries** within a **python >3.4 environment**, otherwise they'll immediately fail.  

- To test a model, run the following: `avc-test-model .`
- To test an untargeted attack, run the following: `avc-test-untargeted-attack .`
- To test an targeted attack, run the following: `avc-test-targeted-attack .`

within the folders you want to test.

In order for the attacks to work, your models / attack folders need to have the following structure:
- for models: https://gitlab.crowdai.org/adversarial-vision-challenge/nips18-avc-model-template
- for attacks: https://gitlab.crowdai.org/adversarial-vision-challenge/nips18-avc-attack-template

