import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from util import ZFilter
import matplotlib.pyplot as plt

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400


class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.sess = tf.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        K.set_session(self.sess)

        # Log variance bounds
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # Create and initialize your model
        self.models = []
        for i in range(num_nets):
            model = self.create_network()
            model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=self.nll_loss(), metrics=[self.rmse_error()])
            self.models.append(model)
        self.train_losses = []
        self.train_RMSEs = []

    def nll_loss(self):
        def loss(y_true, y_pred):
            mean, logvar = self.get_output(y_pred)
            nll = (y_true - mean) * (1/K.exp(logvar)) * (y_true - mean)
            return nll
        return loss

    def rmse_error(self):
        def rmse(y_true, y_pred):
            mean, logvar = self.get_output(y_pred)
            return K.sqrt(K.mean(K.square(mean - y_true)))
        return rmse

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(input=I, output=O)
        return model

    def train(self, inputs, targets, batch_size=128, epochs=5, plot=True):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        num_transitions = len(inputs)
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        sample_inputs = []
        sample_targets = []
        if len(self.models) > 1:
            for _ in self.models:
                sample_idxs = np.random.random_integers(0, num_transitions - 1, num_transitions)
                train_inputs = inputs[sample_idxs]
                train_targets = targets[sample_idxs]
                sample_inputs.append(train_inputs)
                sample_targets.append(train_targets)
        else:
            sample_inputs = [inputs]
            sample_targets = [targets]

        for e in range(epochs):
            for m, model in enumerate(self.models):
                train_inputs = sample_inputs[m]
                train_targets = sample_targets[m]
                history = model.fit(train_inputs, train_targets, batch_size=batch_size, epochs=1, shuffle=True)
        self.train_losses.append(history.history["loss"][0])
        self.train_RMSEs.append(history.history["rmse"][0])


        if plot:
            plt.figure()
            plt.title('train loss')
            plt.plot(self.train_losses)
            plt.savefig("train_loss.png")

            plt.figure()
            plt.title('train RMSE')
            plt.plot(self.train_RMSEs)
            plt.savefig("train_rmse.png")

    def predict(self, input):
        model_id = np.random.random_integers(0, len(self.models)-1)
        output = self.models[model_id].predict(input)
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = np.clip(raw_v, -7, -3)
        return mean, logvar