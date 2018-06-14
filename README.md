# adversarial-vision-challenge
NIPS Adversarial Vision Challenge



### UTILS

To iterate over the given images and store your adversarials while running an attack,
use the following utility methods:

```
from adversarial_vision_challenge.utils import read_images, store_adversarial


for (file_name, img, label) in read_images():

    # run your adversarial attack
    adversarial = your_attack(img, label)

    store_adversarial(file_name, adversarial)

```


### RUN A MODEL SERVER

To run a model server, load your model and wrap it into a foolbox model.
Than pass the fool model to the `model_server` method.

```
from adversarial_vision_challenge import model_server


foolbox_model = load_your_foolbox_model()
model_server(foolbox_model)
```


### RUN AN ATTACK

To run an attack, use the `load_model` method, which returnsa properly pre-configured model instance.

```
from adversarial_vision_challenge.utils import read_images, store_adversarial
from adversarial_vision_challenge.utils import load_model

model = load_model()

for (file_name, image, label) in read_images():
    attack = YourAttack()
    criterion = YourCriteria()
    adversarial = foolbox.Adversarial(model, criterion, image, label)
    attack(adversarial)
    print(adversarial.distance.value)
```