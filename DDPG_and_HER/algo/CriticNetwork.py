import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_critic_network(state_size, action_size, learning_rate):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """
    state_input=tf.keras.Input(shape=(state_size,))
    action_input=tf.keras.Input(shape=(action_size,))
    concate=tf.keras.layers.concatenate([state_input,action_input])
    hidden1 = Dense(HIDDEN1_UNITS, activation='relu')(concate)
    hidden2 = Dense(HIDDEN2_UNITS, activation='relu')(hidden1)
    value= Dense(1, activation='linear')(hidden2)

    model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model, state_input, action_input


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.sess=sess
        self.state_size=state_size
        self.action_size=action_size
        self.batch_size=batch_size
        self.tau=tau
        self.learning_rate=learning_rate
        
        self.model,self.state,self.action=create_critic_network(state_size, action_size, learning_rate)
        self.target_model,self.target_state,self.target_action=create_critic_network(state_size, action_size, learning_rate)
        self.grad=tf.gradients(self.model.output, self.action)
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        grads=self.sess.run(self.grad,feed_dict={self.state: states, self.action: actions})
        return grads[0]

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        target_weights=self.target_model.get_weights()
        weights=self.model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i]=self.tau*weights[i]+(1-self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)
