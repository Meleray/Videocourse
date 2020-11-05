from interface import *
import math
import numpy as np
import sys

# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "ReLU"

    def num_of_parameters(self):
        return 0

    def forward(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values,
                    n - batch size, ... - arbitrary input shape
            :return: np.array((n, ...)), output values,
                    n - batch size, ... - arbitrary output shape (same as input)
        """
        return np.maximum(inputs, np.zeros(np.shape(inputs)))

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs,
                    n - batch size, ... - arbitrary output shape
            :return: np.array((n, ...)), dLoss/dInputs,
                    n - batch size, ... - arbitrary input shape (same as output)
        """
        dydx = self.forward_inputs > 0
        return np.multiply(grad_outputs, dydx)
        


# ============================== 2.1.2 Softmax ===============================
class Softmax(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Softmax"

    def num_of_parameters(self):
        return 0

    def forward(self, inputs):
        """
            :param inputs: np.array((n, d)), input values,
                    n - batch size, d - number of units
            :return: np.array((n, d)), output values,
                    n - batch size, d - number of units
        """
        inputs = inputs - np.max(inputs) + 1e-10
        return np.exp(inputs) /  np.sum(np.exp(inputs), axis=1)[:, None]

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                    n - batch size, d - number of units
            :return: np.array((n, d)), dLoss/dInputs,
                    n - batch size, d - number of units
        """
        return np.multiply(grad_outputs, self.forward_outputs) - self.forward_outputs * np.sum(np.multiply(grad_outputs, self.forward_outputs), axis=1)[:, None]


# =============================== 2.1.3 Dense ================================
class Dense(Layer):

    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_shape = (units,)
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None
        self.name = "Dense"

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units, = self.output_shape

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            #initializer=normal_initializer()
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

    def num_of_parameters(self):
        return self.input_shape[0] * self.output_shape[0] + self.input_shape[0]

    def forward(self, inputs):
        """
            :param inputs: np.array((n, d)), input values,
                    n - batch size, d - number of input units
            :return: np.array((n, c)), output values,
                    n - batch size, c - number of output units
        """
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs,
                    n - batch size, c - number of output units
            :return: np.array((n, d)), dLoss/dInputs,
                    n - batch size, d - number of input units
        """
        # Don't forget to update current gradients:
        # dLoss/dWeights
        self.weights_grad = np.dot(np.matrix.transpose(self.forward_inputs), grad_outputs) / len(grad_outputs)
        # dLoss/dBiases
        self.biases_grad = np.mean(grad_outputs, axis=0)    

        return np.dot(grad_outputs, np.matrix.transpose(self.weights))
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def __call__(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values
            :return: np.array((n,)), loss scalars for batch
        """
        # your code here \/
        return -np.sum(y_gt * np.log(y_pred + 1e-10), axis=1)
        # your code here /\

    def gradient(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values
            :return: np.array((n, d)), gradient loss to y_pred
        """
        # your code here \/
        return -y_gt / (y_pred + 1e-10)
        # your code here /\


# ================================ 2.3.1 SGD =================================
class SGD(Optimizer):
    def __init__(self, lr):
        self._lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter
            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam
                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - parameter_grad * self._lr
            # your code here /\

        return updater


# ============================ 2.3.2 SGDMomentum =============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self._lr = lr
        self._momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter
            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam
                :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self._momentum * updater.inertia + self._lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ======================= 2.4 Train and test on MNIST ========================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.05, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    x_train = np.clip(x_train, 0.0, 1.0)
    x_valid = np.clip(x_valid, 0.0, 1.0)
    model.add(Dense(100, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    model.print_parameters()

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid, batch_size=128, epochs=20)

    return model

# ------------------------ Conv2D 3.1 ----------------------------------

"""
    Method which calculates the padding based on the specified output shape and the
    shape of the filters
"""
def determine_padding(filter_shape, output_shape):
    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1) / 2))
        pad_h2 = int(math.ceil((filter_height - 1) / 2))
        pad_w1 = int(math.floor((filter_width - 1) / 2))
        pad_w2 = int(math.ceil((filter_width - 1) / 2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)

def get_im2col_indices(images_shape, filter_shape, padding, stride):
    """
        :param images_shape: (n, c, h, w), images shape,
                n - batch size, c - number of input channels
                (h, w) - input image shape
        :param filter_shape: (fh, fw),
        :param padding: (int, int),
                Specifies how much to pad edges
        :param stride: int
                Specifies how far the convolution window moves for each step.
        :return: np.array((n, oc, oh, ow)), output values,
                n - batch size, oc - number of output channels
                (oh, ow) - output image shape
    """
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

    # your code here \/
    ir = np.tile(np.repeat(np.arange(filter_height), filter_width), channels)
    ic = np.repeat(np.arange(out_height), out_width) * stride
    i = np.reshape(ir, (-1, 1)) + np.reshape(ic, (1, -1))
    jr = np.tile(np.arange(filter_width), filter_height * channels)
    jc = np.tile(np.arange(out_width), out_height) * stride
    j = np.reshape(jr, (-1, 1)) + np.reshape(jc, (1, -1))
    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    return (k, i, j)
    # your code here /\

"""
    Method which turns the image shaped input to column shape.
    Used during the forward pass.
"""
def im2col(images, filter_shape, stride, output_shape):
    """
        :param images: np.ndarray((n, c, h, w)), images ,
                n - batch size, c - number of input channels
                (h, w) - input image shape
        :param filter_shape: (fh, fw)
        :param stride: int
                Specifies how far the convolution window moves for each step.
        :param output_shape: "same" or "valid"
                "same" - add padding so that the output height and width matches the
                input height and width. "valid" - no padding is added
        :return: np.array((oc, os)), image in column shape
                oc - fh * fw * c, os - output size
    """
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode = 'constant')

    # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)
    channels = images.shape[1]

    # your code here \/
    
    cols = images_padded[:, k, i, j]
    return cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    # your code here /\
"""
    Method which turns the column shaped input to image shape.
    Used during the backward pass.
"""
def col2im(cols, images_shape, filter_shape, stride, output_shape):
    """
        :param cols: np.ndarray((ic, is)), images in column shape
        :param filter_shape: (fh, fw)
        :param stride: int
                Specifies how far the convolution window moves for each step.
        :param output_shape: "same" or "valid"
                "same" - add padding so that the output height and width matches the
                input height and width. "valid" - no padding is added
        :return: np.ndarray((n, c, h, w)), images ,
                n - batch size, c - number of input channels
                (h, w) - input image shape
    """
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.zeros((batch_size, channels, height_padded, width_padded), dtype = cols.dtype)

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    # your code here \/
    cols = cols.reshape(channels * filter_shape[0] * filter_shape[1], -1, batch_size).transpose(2, 0, 1)
    np.add.at(images_padded, (slice(None), k, i, j), cols)
    # Return image without padding
    return images_padded[:, :, pad_h[0]:height_padded - pad_h[1], pad_w[0]:width_padded - pad_w[1]]
    # your code here /\


class Conv2D(Layer):
    def __init__(self, filters, kernel_size = 3, padding = "same", strides = 1, *args, **kwargs):
        """
            filters: int
                Number of filters, number of channels of the output shape
            kernel_size: int
                Kernel size
            padding: "valid" or "same". 
                "same" - add padding so that the output height and width matches the
                input height and width. "valid" - no padding is added
            strides: int
                Specifies how far the convolution window moves for each step.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = "Conv2D"

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter
        self.filter_shape = (self.kernel_size, self.kernel_size)

        self.output_shape = self.get_output_shape()

        self.weights, self.weights_grad = self.add_parameter(
            name = "weights",
            shape = (self.filters, self.input_shape[0], *self.filter_shape),
            initializer = normal_initializer()
        )

        self.biases, self.biases_grad = self.add_parameter(
            name = 'biases',
            shape = (self.filters, 1),
            initializer = np.zeros
        )

    # returns the number of parameters
    def num_of_parameters(self):
        return self.kernel_size * self.kernel_size * self.filters + self.filters

    # returns shape of the output tensor
    def get_output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.strides + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.strides + 1
        return self.filters, int(output_height), int(output_width)

    def forward(self, inputs):
        """
            :param inputs: np.array((n, ic, ih, iw)), input values,
                    n - batch size, ic - number of input channels
                    (ih, iw) - input image shape
            :return: np.array((n, oc, oh, ow)), output values,
                    n - batch size, oc - number of output channels
                    (oh, ow) - output image shape
        """
        batch_size = inputs.shape[0]
        # your code here \/
        cols = im2col(inputs, self.filter_shape, self.strides, self.padding)
        filter_matrix = np.array([filter.flatten() for filter in self.weights], dtype='float')
        outputs = np.dot(filter_matrix, cols).reshape(self.output_shape[0], self.output_shape[1], self.output_shape[2], batch_size).transpose(3, 0, 1, 2)
        for i in range(batch_size):
            outputs[i] = np.array([outputs[i, c] + self.biases[c] for c in range(self.output_shape[0])])
        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, oc, oh, ow)), dLoss/dOutputs,
                    n - batch size, oc - number of output filters
                    (oh, ow) - output image shape
            :return: np.array((n, ic, ih, iw)), dLoss/dInputs,
                    n - batch size, ic - number of input filters
                    (ih, iw) - input image shape
        """
        # Reshape accumulated gradient into column shape
        batch_size = grad_outputs.shape[0]
        # your code here \/
        
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3)).reshape(-1, 1) 
        grad_outputs = grad_outputs.transpose(1, 2, 3, 0).reshape(self.filters, -1)
        cols = im2col(self.forward_inputs, self.filter_shape, self.strides, self.padding)
        self.weights_grad = np.dot(grad_outputs, np.matrix.transpose(cols)).reshape(np.shape(self.weights))
        grad_inputs = np.dot(np.matrix.transpose(self.weights.reshape(self.filters, -1)), grad_outputs)
        return col2im(grad_inputs, tuple([batch_size]) + self.input_shape, self.filter_shape, self.strides, self.padding)
        # your code here /\

# ======================= 3.2 MaxPooling Layer ==========================

class MaxPooling2D(Layer):
    def __init__(self, pool_size = 2, strides = 1, *args, **kwargs):
        """
            pool_size: int
                Size of the pooling window
            strides: int
                Specifies how far the pooling window moves for each step.
        """
        super().__init__(*args, **kwargs)
        self.pool_size = (pool_size, pool_size)
        self.strides = strides
        self.name = "MaxPool2D"
        self.saved_indexes = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.output_shape = self.get_output_shape()

    def num_of_parameters(self):
        return 0

    # returns shape of the output tensor
    def get_output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_size[0]) / self.strides + 1
        out_width = (width - self.pool_size[1]) / self.strides + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, ih, iw)), input values,
                    n - batch size, c - number of input channels
                    (ih, iw) - input image shape
            :return: np.array((n, c, oh, ow)), output values,
                    n - batch size, c - number of input channels
                    (oh, ow) - output image shape
        """
        batch_size, channels, height, width = inputs.shape
        _, out_height, out_width = self.output_shape
        outputs = np.zeros((batch_size, channels, out_height, out_width))
        self.saved_indexes = np.zeros((batch_size, channels, out_height, out_width, 2))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, height, self.strides):
                    for j in range(0, width, self.strides):
                        window = inputs[b, c, i:min(height, i + self.pool_size[0]):, j:min(width, j+self.pool_size[1]):]
                        outputs[b, c, i // self.strides, j // self.strides] = np.max(window)
                        self.saved_indexes[b, c, i // self.strides, j // self.strides] = np.argwhere(window==window.max())
        self.saved_indexes = self.saved_indexes.astype('int')
        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, oh, ow)), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (oh, ow) - output image shape
            :return: np.array((n, c, ih, iw)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (ih, iw) - input image shape
        """
        batch_size, _, out_height, out_width = grad_outputs.shape
        channels, height, width = self.input_shape
        # your code here \/
        
        grad_inputs = np.zeros((batch_size, channels, height, width))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        grad_inputs[b, c, i * self.strides + self.saved_indexes[b, c, i, j, 0], j * self.strides + self.saved_indexes[b, c, i, j, 1]] = grad_outputs[b, c, i, j]
        return grad_inputs
        # your code here /\

# ============================ 3.3 Global Average Pooling ========================

class GlobalAveragePooling(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        c, h, w = self.input_shape
        self.output_shape = (c,)

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, ih, iw)), input values,
                    n - batch size, c - number of channels
                    (ih, iw) - input image shape
            :return: np.array((n, c)), output values,
                    n - batch size, c - number of channels
        """
        # your code here \/

        outputs = np.array([[np.average(chn) for chn in img] for img in inputs])
        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                    n - batch size, d - number of channels
            :return: np.array((n, ih, iw, d)), dLoss/dInputs,
                    n - batch size, d - number of channels
                    (ih, iw) - input image shape
        """
        batch_size = grad_outputs.shape[0]
        # your code here \/
        c, h, w = self.input_shape
        grad_inputs = np.array([[np.full((h, w), chn / (h * w)) for chn in img] for img in grad_outputs])
        return grad_inputs
        # your code here /\

# ============================ 3.4 Batch Normalization ========================

class BatchNormalization(Layer):
    def __init__(self, momentum = 0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BatchNorm"
        self.momentum = momentum
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.gamma, self.gamma_grad = self.add_parameter(
            name = "gamma",
            shape = (self.input_shape),
            initializer = np.ones
        )

        self.beta, self.beta_grad = self.add_parameter(
            name = 'beta',
            shape = (self.input_shape),
            initializer = np.zeros
        )

    def num_of_parameters(self):
        return self.gamma.size + self.beta.size

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, h, w)), input values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), output values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        batch_size = inputs.shape[0]
        # your code here \/
        mean = np.mean(inputs, axis = 0)
        var = np.var(inputs, axis = 0)
        inputs = (inputs - mean) / np.sqrt(var + self.eps)
        self.running_mean = mean
        self.running_var = var
        return self.gamma * inputs + self.beta
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        batch_size = grad_outputs.shape[0]
        x = self.forward_inputs - self.running_mean
        # your code here \/
        dldx = self.gamma * grad_outputs
        dvar = np.sum(dldx * x * -0.5 * np.power(self.running_var + self.eps, -1.5), axis = 0)
        dmean = dvar * np.sum(-2 * x, axis = 0) / batch_size + np.sum(dldx / -np.sqrt(self.running_var + self.eps), axis = 0)
        grad_inputs = dldx / np.sqrt(self.running_var + self.eps) + dvar * 2 * x / batch_size + dmean / batch_size
        self.gamma_grad = np.sum(grad_outputs * x / np.sqrt(self.running_var + self.eps), axis = 0)
        self.beta_grad = np.sum(grad_outputs, axis = 0)
        return grad_inputs
        # your code here /\


# ============================= 3.5 Dropout ===============================

class Dropout(Layer):
    def __init__(self, p = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Dropout"
        self.p = p
        self.mask = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        # When testing on unittests, please uncomment next two lines
        np.random.seed(1)
        self.training = True

    def num_of_parameters(self):
        return 0

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, h, w)), input values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), output values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            for conv layers
            or
            :param inputs: np.array((n, c)), input values,
                    n - batch size, c - number of units
            :return: np.array((n, c)), output values,
                    n - batch size, c - number of units
            for dense layers
        """
        batch_size = inputs.shape[0]
        self.mask = np.random.sample(np.shape(inputs)) > self.p
        # your code here \/
        if not self.training:
            return inputs * (1 - self.p)
        else:
            return np.multiply(inputs, self.mask)
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        batch_size = grad_outputs.shape[0]
        return np.multiply(grad_outputs, self.mask)
        # your code here /\

# ============================ 3.6 Flatten Layer ============================

class Flatten(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Flatten"

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.output_shape = (np.array(self.input_shape[:]).prod(), )

    def num_of_parameters(self):
        return 0

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, h, w)), input values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, (c * h * w))), output values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        self.input_shape = np.shape(inputs)
        batch_size = inputs.shape[0]

        # your code here \/
        return np.array([ch.flatten() for ch in inputs])
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (c * h * w))), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        batch_size = grad_outputs.shape[0]

        # your code here \/
        return grad_outputs.reshape(self.input_shape)
        # your code here /\

# ======================= 3.7 Train Cifar10 Conv Model ========================

def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(
        ...
    )
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(...)

    model.print_parameters()

    # 3) Train and validate the model using the provided data
    model.fit(...)

    # your code here /\
    return model