import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [512, 512])
    img = img[tf.newaxis, :]
    return img

def style_transfer(content_image, style_image):
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

def display_image(image):
    image = image[0]  # Remove batch dimension
    image = np.array(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_image(image, output_path):
    image = image[0]  # Remove batch dimension
    image = np.array(image)
    plt.imsave(output_path, image)

# Load your images
content_image = load_image('bird.jpeg')
style_image = load_image('painting.jpeg')

# Apply style transfer
host = style_transfer(content_image, style_image)

# Display the stylized image
display_image(host)
save_image(host, 'host.png')
