import numpy as np


# From PDF

class Initializer:
    def __call__(self, shape):
        raise NotImplementedError()


# Based on Initializer

class RandomUniform(Initializer):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, shape):
        return np.random.uniform(self.min_val, self.max_val, shape)


# From PDF

class RunningStatistic:
    def __init__(self):
        self.total = 0
        self.count = 0

    def __call__(self, y, t):
        return self.call(y, t)

    def call(self, y, t):
        self.reset_state()
        self.update_state(y, t)
        return self.result()

    def reset_state(self):
        self.total = 0
        self.count = 0
        return self

    def update_state(self, y, t):
        self.count += len(t)
        self.update(y, t)
        return self

    def result(self):
        return self.total / self.count

    def update(self, y, t):
        raise NotImplementedError()


# Based on RunningStatistic

class MeanSquaredError(RunningStatistic):
    def update(self, y, t):
        error = np.sum(np.subtract(y, t) ** 2)
        self.total += error


# Based on RunningStatistic

class RootMeanSquaredError(MeanSquaredError):
    def result(self):
        # Pull result from RunningStatistic

        return np.sqrt(super().result())

# Based on RunningStatistic
# DONE Categorical Accuracy Statistic
class CategoricalAccuracy(RunningStatistic):
    def update(self, y, t):
        correct = np.sum(np.argmax(y, axis=-1) == np.argmax(t, axis=-1))
        self.total += correct

# From PDF

class Layer:
    def __init__(self):
        self.d_output = None
        self.input = None
        self.output = None

    def __call__(self, input_, training=False):
        self.input = input_
        self.output = self.call(input_)
        self.training = training
        return self.output

    def backprop(self, d_output):
        self.d_output = d_output
        return self.d_input(d_output)

    def call(self, input_, training=False):
        raise NotImplementedError()

    def d_input(self, d_output):
        raise NotImplementedError()

    def update(self, alpha):
        pass


# PDF function inheriting from Layer

class Activation(Layer):
    pass


# Based on Layer (Is an activation function)

class ReLU(Activation):
    def call(self, input_):
        return np.maximum(0, input_)
    def d_input(self, d_output):
        return d_output * (self.output > 0)


# Based on Layer

# DONE add the 'weight regularization' functionality
# DONE Add store and restore params for weights
class Dense(Layer):
    def __init__(self, num_inputs, num_outputs, weights_initializer, weights_regularizer=None):
        super().__init__()
        self.weights = weights_initializer((num_inputs, num_outputs))
        self.dw = None
        self.weights_regularizer = weights_regularizer
        self.stored_weights = None

    def call(self, input_):
        return np.dot(input_, self.weights)

    def d_input(self, d_output):
        self.dw = np.dot(self.input.T, d_output)
        if self.weights_regularizer != None:
            self.dw += self.weights_regularizer(self.weights)
        return np.dot(d_output, self.weights.T)

    def update(self, alpha):
        self.weights -= alpha * self.dw
        
    def store_params(self):
        self.stored_weights = np.copy(self.weights)
        
    def restore_params(self):
        self.weights = self.stored_weights

#TODO New layer
class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def call(self, input, training=False):
        if training:
            self.mask = np.random.binomial(1, 1-self.rate, input.shape) / (1 - self.rate)
            return input * self.mask
        else:
            return input

    def d_input(self, d_output):
        if self.mask is not None:
            return d_output * self.mask
        else:
            return d_output
        

# Superclass for regularizers
# DONE Regularizer, L1, L2

class Regularizer:
    def __call__(self, weights):
        """
        Returns the derivative of the penalty term w.r.t. the weights.
        @param weights: the weights
        @ret
        """
        raise NotImplementedError('NIE')

class L1(Regularizer):
    def __init__(self, lambda_):
        self.lambda_ = lambda_
    
    def __call__(self, weights):
        return self.lambda_ * np.sign(weights)

class L2(Regularizer):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, weights):
        return self.lambda_ * weights


class Sequential:
    def __init__(self, layers=None):
        if (layers is not None):
            self.layers = layers
        else:
            self.layers = []
        self.loss = None
        self.metric = None

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    # DONE Add metric argument which if provided replaces the default choice for Metric
    def compile(self, metric=RootMeanSquaredError()):
        self.loss = MeanSquaredError()  # Our loss fn, from PDF
        self.metric = metric  # Our performance metric, from PDF

    def backprop(self, y, t, alpha):
        # Compute initial gradient
        d_output = (y - t)

        # Backprop the layers in reverse
        for layer in reversed(self.layers):
            d_output = layer.backprop(d_output)

        # Update weights
        for layer in self.layers:
            layer.update(alpha)

    def batch_generator(x, t, batch_size, idx):
        # DONE
        n = len(x)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:min(i+batch_size,n)]
            yield x[batch_idx], t[batch_idx]

    # DONE Modify to support batch_size
    def fit(self, x, target, alpha, max_iter, shuffle, batch_size=1, validation_data=None):
        n = len(x)
        idx = np.arange(n)  # From PDF specs
        history = []

        for iter in range(max_iter):
            # Reset states at the start of an epoch

            self.loss.reset_state()
            self.metric.reset_state()

            if shuffle:
                np.random.shuffle(idx)

            for i in range(0, n, batch_size):
                end = i + batch_size
                batch_i = idx[i:end]
                x_i = x[batch_i]
                t_i = target[batch_i]
                y_i = self.predict(x_i)  # Forward pass

                # Update statistics

                self.loss.update_state(y_i, t_i)
                self.metric.update_state(y_i, t_i)

                self.backprop(y_i, t_i, alpha)  # Backprop

            print(f"i=%d, " % (iter))
            return dict(loss = self.loss.result(), metric = self.metric.result())
            
            if validation_data:
                loss = self.loss.result()
                metric = self.metric.result()
            
                x_valid, t_valid = validation_data
                y_valid = self.predict(x_valid)
                
                valid_loss = self.loss(y_valid, t_valid)
                valid_metric = self.metric(y_valid, t_valid)
                
                print(f"loss: %.6g, metric: %.6g, valid_loss: %.6g, valid_metric: %.6g" % (loss, metric, valid_loss, valid_metric))
            else:
                print("")
                


n, m, k = 3, 2, 1
x = np.arange(n * m).reshape((n,m)).astype(float)
t = np.arange(n * k).reshape((n, k)).astype(float)
def weights_initializer(shape):
    return np.arange(np.prod(shape)).reshape(shape).astype(float)
model = Sequential([Dense(m, k, weights_initializer)])
model.compile()
history = model.fit(x, t, alpha=0, max_iter=1, shuffle=False)
print(history)