# NIPS Adversarial Vision Challenge

### Serving a model

To run a model server, load your model and wrap it as a [foolbox model](https://foolbox.readthedocs.io/en/latest/modules/models.html).
Then pass the foolbox model to the `model_server` function.

```python
from adversarial_vision_challenge import model_server

foolbox_model = load_your_foolbox_model()
model_server(foolbox_model)
```

### Running an attack

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
```
