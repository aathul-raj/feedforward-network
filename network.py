#### Libraries
# Standard library
import random, mnist_loader
import time

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes, lambda_reg=0.01):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
        # For l2 reg
        self.lambda_reg = lambda_reg

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = leaky_relu(np.dot(w, a)+b)

        # Apply softmax to the last layer
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        return softmax(z)


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            current_eta = eta * (0.95 ** (j // 5))
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, current_eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds, {4:.2f}% accuracy".format(
                    j, self.evaluate(test_data), n_test, time2-time1, (self.evaluate(test_data) / n_test) * 100))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (self.lambda_reg / len(mini_batch))) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = leaky_relu(z)
            activations.append(activation)
        # Output layer
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = leaky_relu_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z, axis=0)

def relu(z):
    """The ReLU function."""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of ReLU"""
    return (z > 0).astype(float) + 1e-7

def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)

def leaky_relu_prime(z, alpha=0.01):
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = np.zeros_like(image)
    
    # Ensure dx and dy are within bounds
    dx = max(-28, min(28, dx))
    dy = max(-28, min(28, dy))
    
    if dx < 0:
        shifted_image[:, :dx] = image[:, -dx:]
    elif dx > 0:
        shifted_image[:, dx:] = image[:, :-dx]
    else:
        shifted_image = image.copy()
    
    if dy < 0:
        shifted_image[:dy, :] = shifted_image[-dy:, :]
    elif dy > 0:
        shifted_image[dy:, :] = shifted_image[:-dy, :]
    
    return shifted_image.reshape([-1, 1])

def augment_data(training_data):
    augmented_data = []
    for x, y in training_data:
        augmented_data.append((x, y))
        augmented_data.append((shift_image(x, 1, 0), y))
        augmented_data.append((shift_image(x, -1, 0), y))
        augmented_data.append((shift_image(x, 0, 1), y))
        augmented_data.append((shift_image(x, 0, -1), y))
    return augmented_data