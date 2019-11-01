import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_actor_network(state_size, action_size):
	"""Creates an actor network.

	Args:
		state_size: (int) size of the input.
		action_size: (int) size of the action.
	Returns:
		model: an instance of tf.keras.Model.
		state_input: a tf.placeholder for the batched state.
	"""
	input_layer = Input(shape=(state_size,))
	layer = Dense(HIDDEN1_UNITS, activation='relu')(input_layer)
	layer = Dense(HIDDEN2_UNITS, activation='relu')(layer)
	output_layer = Dense(action_size, activation='tanh')(layer)
	model = Model(inputs=input_layer, outputs=output_layer)

	return model, input_layer


class ActorNetwork(object):
	def __init__(self, sess, state_size, action_size, batch_size,
				 tau, learning_rate):
		"""Initialize the ActorNetwork.
		This class internally stores both the actor and the target actor nets.
		It also handles training the actor and updating the target net.

		Args:
			sess: A Tensorflow session to use.
			state_size: (int) size of the input.
			action_size: (int) size of the action.
			batch_size: (int) the number of elements in each batch.
			tau: (float) the target net update rate.
			learning_rate: (float) learning rate for the critic.
		"""
		self.sess = sess
		self.batch_size = batch_size
		self.tau = tau
		self.learning_rate = learning_rate
		self.action_size = action_size

		self.model, self.state = create_actor_network(state_size, action_size)
		self.target_model, self.target_state  = create_actor_network(state_size, action_size)

		self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
		self.params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradient)
		grads = zip(self.params_grad, self.model.trainable_weights)

		self.optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
		self.sess.run(tf.initialize_all_variables())

	def train(self, states, action_grads):
		"""Updates the actor by applying dQ(s, a) / da.

		Args:
			states: a batched numpy array storing the state.
			actions: a batched numpy array storing the actions.
			action_grads: a batched numpy array storing the
				gradients dQ(s, a) / da.
		"""
		self.sess.run(self.optimizer, feed_dict={self.state: states,
												 self.action_gradient: action_grads})

	def update_target(self):
		"""Updates the target net using an update rate of tau."""
		actor_weights = self.model.get_weights()
		actor_target_weights = self.target_model.get_weights()
		for i in range(len(actor_weights)):
			actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
		self.target_model.set_weights(actor_target_weights)
