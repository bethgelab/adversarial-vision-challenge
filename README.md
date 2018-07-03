# NIPS Adversarial Vision Challenge

### Installation

To install the package simply run:

`pip install adversarial-vision-challenge`

This package contains helper functions to implement models and attacks that can be used with Python 2.7, 3.4, 3.5 and 3.6. Other Python versions might work as well. **We recommend using Python 3**!

Furthermore, this package also contains test scripts that should be used before submission to perform local tests of your model or attack. These test scripts are Python 3 only, because they depend on `crowdai-repo2docker`. See `Running Tests Scripts` section below for more detailed information.


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

The test scripts (running on your host machine) will need Python 3. Your model or attack running inside a docker container and using this package can use Python 2 or 3.

- To test a model, run the following: `avc-test-model .`
- To test an untargeted attack, run the following: `avc-test-untargeted-attack .`
- To test an targeted attack, run the following: `avc-test-targeted-attack .`

within the folders you want to test.

In order for the attacks to work, your models / attack folders need to have the following structure:
- for models: https://gitlab.crowdai.org/adversarial-vision-challenge/nips18-avc-model-template
- for attacks: https://gitlab.crowdai.org/adversarial-vision-challenge/nips18-avc-attack-template

