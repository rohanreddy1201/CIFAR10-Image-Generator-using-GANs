# CIFAR-10 Image Generator using Generative Adversarial Networks (GANs)

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate realistic images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The GAN framework consists of a generator and a discriminator network trained adversarially to produce high-quality images resembling specific classes from the dataset.

## Project Structure

The project structure is as follows:

- `data/`: Directory containing the CIFAR-10 dataset.
- `gan_model.py`: Python script defining the Generator and Discriminator models.
- `training_script.py`: Python script for training the GAN model.
- `evaluate.py`: Python script for generating images using the trained GAN model.

## How to Use

1. **Setup Environment**: Make sure you have Python installed along with the necessary dependencies.

2. **Dataset**: Download the CIFAR-10 dataset and place it in the `data/` directory.

3. **Training**: Run the `training_script.py` script to train the GAN model. Adjust hyperparameters as needed in the script.

4. **Evaluation**: After training, run the `evaluate.py` script to generate new images using the trained model.

## Disclaimer

This project is for educational purposes only. The code provided here should not be used for any commercial or production purposes. We do not guarantee the accuracy, reliability, or suitability of this code for any purpose.
