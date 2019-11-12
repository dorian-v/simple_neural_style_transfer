import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from vgg_model import load_vgg_model
from image_tools import *
from get_directory import get_directory

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

CONTENT_LAYER = 'conv4_2'

alpha = 1
beta = 10
gamma = 10 ** -3


def get_content_image_activations(sess, model, content_image):
    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model[CONTENT_LAYER]

    # Set a_C to be the hidden layer activation from the layer we have selected with content image as input
    a_C = sess.run(out)
    return a_C, out


def compute_content_cost(a_C, a_G):
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

    return J_content


def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def get_style_image_activations(sess, model, STYLE_LAYERS, style_image):

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))

    a_S = {}
    for layer_name, _ in STYLE_LAYERS:
        out = model[layer_name]
        a_S[layer_name] = sess.run(out)

    return a_S

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * n_C **2 * (n_W * n_H) ** 2)

    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS, a_S):
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Will be evaluated later with session run
        a_G = out

        # Add weighted style cost of layer to overall style cost
        J_style += coeff * compute_layer_style_cost(a_S[layer_name], a_G)

    return J_style


def compute_total_variation_regularization(image):
    return tf.image.total_variation(image)[0]


def total_cost(J_content, J_style, J_TV, alpha=10, beta=40, gamma=10**-3):
    J = alpha * J_content + beta * J_style + gamma * J_TV
    return J


def define_total_cost_function(model, STYLE_LAYERS, a_C, a_S, a_G):
    J_content = compute_content_cost(a_C, a_G)
    J_style = compute_style_cost(model, STYLE_LAYERS, a_S)
    J_TV = compute_total_variation_regularization(model['input'])
    J = total_cost(J_content, J_style, J_TV, alpha, beta, gamma)
    return J_content, J_style, J_TV, J


def model_nn(sess, model, train_step, input_image, num_iterations=200):

    sess.run(tf.compat.v1.global_variables_initializer())
    # Run the noisy input image (initial generated image) through the model
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        _ = sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Save generated image every 20 iteration.
        if i % 100 == 0:
            #Jt, Jc, Js, Jtv = sess.run([J, J_content, J_style, J_TV])
            #print("Iteration " + str(i) + " :")
            #print("total cost = " + str(Jt))
            #print("content cost = " + str(Jc))
            #print("style cost = " + str(Js))
            #print('total variation cost = ' + str(Jtv))
            # save current generated image in the "/output" directory
            save_image(get_directory() + "/output/" + str(i) + ".png", generated_image)


    # save last generated image
    save_image(get_directory() + '/output/generated_image.jpg', generated_image)

    return generated_image


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('Usage: python ' + sys.argv[0] + ' content_image_path style_image_path')
        exit(1)

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()

    # Load and prepare content and style images
    content_image = plt.imread(sys.argv[1])
    content_image = reshape_and_normalize_image(content_image)
    style_image = plt.imread(sys.argv[2])
    style_image = reshape_and_normalize_image(style_image)

    # Initialize output image
    generated_image = generate_noise_image(content_image)

    # Load pre-trained VGG19 model
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    # Set a_C to be the hidden layer activation from the layer we have selected with content image as input
    # Set a_G to be the activations from the same layer, but not yet evaluated
    a_C, a_G = get_content_image_activations(sess, model, content_image)

    # Set a_S to be the hidden layers activations from the layers we have selected with style image as input
    a_S = get_style_image_activations(sess, model, STYLE_LAYERS, style_image)

    # define cost function, optimizer and training steps
    J_content, J_style, J_TV, J = define_total_cost_function(model, STYLE_LAYERS, a_C, a_S, a_G)
    optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    model_nn(sess, model, train_step, generated_image, 1000)