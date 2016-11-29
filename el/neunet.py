import numpy as np
import theano
import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, input_vecs, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
            ),
            name='weights',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                    n_out,
                    dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input_vecs, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input_vec = input_vecs

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, w_init=None, b_init=None,
                 activation=T.tanh):
        self.activation = activation

        if w_init is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            w_init = theano.shared(value=W_values, name='W', borrow=True)

        if b_init is None:
            b_values = np.zeros(n_out, dtype=theano.config.floatX)
            b_init = theano.shared(value=b_values, name='b', borrow=True)

        self.W = w_init
        self.b = b_init
        self.params = [self.W, self.b]

    def get_output(self, input_vecs):
        lin_output = T.dot(input_vecs, self.W) + self.b
        return lin_output if self.activation is None else self.activation(lin_output)


class MLP(object):
    def __init__(self, rng, input_vecs, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(
            rng=rng,
            # input_vecs=input_vecs,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.log_regression_layer = LogisticRegression(
            # input_vecs=self.hidden_layer.output,
            input_vecs=self.hidden_layer.get_output(input_vecs),
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.log_regression_layer.W).sum()
        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.log_regression_layer.W ** 2).sum()
        self.negative_log_likelihood = self.log_regression_layer.negative_log_likelihood
        self.errors = self.log_regression_layer.errors

        self.params = self.hidden_layer.params + self.log_regression_layer.params

        self.input_vecs = input_vecs
