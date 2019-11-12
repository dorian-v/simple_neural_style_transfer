import imageio
import numpy as np
from config import CONFIG
from PIL import Image


def generate_noise_image(content_image, noise_ratio=CONFIG.NOISE_RATIO):
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20,
                                    (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype(
        'float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image


def reshape_and_normalize_image(image):
    # Reshape image to mach expected input of VGG19
    image = np.array(Image.fromarray(image).resize((CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT)))
    image = np.reshape(image, ((1,) + image.shape))

    # Subtract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS

    return image


def save_image(path, image):
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS

    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    imageio.imwrite(path, image)
