import foolbox
from adversarial_vision_challenge import load_model, read_images, store_adversarial


def run_attack(model, image, label):
    attack = foolbox.attacks.AdditiveGaussianNoiseAttack()
    criterion = foolbox.criteria.Misclassification()
    adversarial = foolbox.Adversarial(model, criterion, image, label)
    attack(adversarial)
    print(adversarial.distance.value)
    return adversarial.image


def main():
    model = load_model()
    for (file_name, image, label) in read_images():
        print('predicted model for: ', file_name, model(image))
        adversarial = run_attack(model, image, label)
        store_adversarial(file_name, adversarial)


if __name__ == '__main__':
    main()
