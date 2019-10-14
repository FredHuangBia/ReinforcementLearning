#!/usr/bin/env python
import numpy as np, gym, sys, copy, argparse, os, collections, random, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class QNetwork(nn.Module):

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.
		super(QNetwork, self).__init__()
		self.env_name = environment_name
		env = gym.make(environment_name)
		self.num_action = env.action_space.n
		self.num_observe = env.observation_space.shape[0]
		self.fc1 = nn.Linear(self.num_observe, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, self.num_action)

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		model_file = suffix + self.env_name + ".pth"
		torch.save(self.state_dict(), model_file)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		self.model = torch.load(model_file)

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		self.load_state_dict(torch.load(weight_file))

	def forward(self, x):
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		x = self.fc3(x)
		return x


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
		# Hint: you might find this useful:
		# 		collections.deque(maxlen=memory_size)
		self.memory_size = memory_size
		self.burn_in = burn_in
		self.memory = collections.deque(maxlen=memory_size)
		self.now_size = 0
		self.bad_size = 0

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		return random.sample(self.memory, batch_size)

	def append(self, transition):
		# Appends transition to the memory. 	
		if self.now_size == self.memory_size:
			poped = self.memory.popleft()
			self.now_size -= 1
			if (poped[3] is None):
				self.bad_size -= 1
				if (self.bad_size < self.memory_size * 0.03) and transition[3] is not None:
					transition = poped
		self.memory.append(transition)
		self.now_size += 1
		if (transition[3] is None):
			self.bad_size += 1


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.environment_name = environment_name
		if environment_name == "CartPole-v0":
			self.lr = 2e-4
			self.gamma = 0.99
			self.num_episodes = 3000
			self.memory = Replay_Memory(100000)
			self.eps_start = 0.5
			self.eps_end = 0.05
			self.eps_decay = 100000
		elif environment_name == "MountainCar-v0":
			self.lr = 1e-3
			self.gamma = 1.0
			self.num_episodes = 10000
			self.memory = Replay_Memory(10000)
			self.eps_start = 0.5
			self.eps_end = 0.05
			self.eps_decay = 100000
		else:
			print("Unknown environment")
			return
		self.policy_DQN = QNetwork(environment_name)
		############  Uncomment this block for Double-Q learning ############
		# self.target_DQN = QNetwork(environment_name)
		# self.target_DQN.load_state_dict(self.policy_DQN.state_dict())
		# self.target_DQN.eval()
		# self.C = 100
		#####################################################################
		self.optimizer = optim.Adam(self.policy_DQN.parameters(), lr=self.lr)
		self.env = gym.make(environment_name)
		self.num_action = self.env.action_space.n
		self.num_observe = self.env.observation_space.shape[0]
		self.env.reset()
		self.itr = 0
		self.episode = 0
		self.batch_size = 32
		self.eval_period = 100
		self.test_rewards = [[], []]
		self.TD_errors = [[], []]

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.             
		sample = random.random()
		eps_threshold = self.eps_start + (self.eps_end - self.eps_start) * (self.itr / self.eps_decay)
		if sample > eps_threshold:
			with torch.no_grad():
				return q_values.max(1)[1].view(1, 1)
		else:
			return torch.tensor([[random.randrange(self.num_action)]], dtype=torch.long)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		with torch.no_grad():
			return q_values.max(1)[1].view(1, 1)

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		saved0 = False
		saved1 = False
		saved2 = False
		saved3 = False
		self.burn_in_memory()
		while self.episode < self.num_episodes:
			# Initialize the environment and state
			observation = self.env.reset()
			done = False
			while not done:
				# Select and perform an action
				observation_tensor = torch.tensor([observation], dtype=torch.float32)
				self.policy_DQN.eval()
				q_values = self.policy_DQN(observation_tensor)
				action = self.epsilon_greedy_policy(q_values)
				new_observation, reward, done, _ = self.env.step(action.item())
				if (done):
					new_observation = None
					if self.environment_name == "CartPole-v0":
						reward = 0
				# Store the transition in memory
				self.memory.append([observation, action, reward, new_observation])
				observation = new_observation

				self.policy_DQN.train()
				transitions = self.memory.sample_batch(self.batch_size)
				self.optimize(transitions)
				self.itr += 1

			self.episode += 1
			############  Uncomment this block for Double-Q learning ############
			# if self.episode % self.C == 0:
			# 	print("update target")
			# 	self.target_DQN.load_state_dict(self.policy_DQN.state_dict())
			# 	self.target_DQN.eval()
			#####################################################################

			if (self.episode % self.eval_period) == 0:
				avg_reward = self.test()
				if (self.environment_name == "CartPole-v0"):
					if (avg_reward < 200 * 1 / 3 and not saved0):
						self.policy_DQN.save_model_weights("CartPole_0_")
						saved0 = True
					elif (200 * 1 / 3 < avg_reward < 200 * 2 / 3 and not saved1):
						self.policy_DQN.save_model_weights("CartPole_1_")
						saved1 = True
					elif (200 * 2 / 3 < avg_reward < 200 * 3 / 3 and not saved2):
						self.policy_DQN.save_model_weights("CartPole_2_")
						saved2 = True
					elif (avg_reward == 200 and not saved3):
						self.policy_DQN.save_model_weights("CartPole_3_")
						saved3 = True
				elif (self.environment_name == "MountainCar-v0"):
					if (avg_reward < -170 and not saved0):
						self.policy_DQN.save_model_weights("MountainCar_0_")
						saved0 = True
					elif (-170 < avg_reward < -140 and not saved1):
						self.policy_DQN.save_model_weights("MountainCar_1_")
						saved1 = True
					elif (-140 < avg_reward < -110 and not saved2):
						self.policy_DQN.save_model_weights("MountainCar_2_")
						saved2 = True
					elif (-110 <= avg_reward and not saved3):
						self.policy_DQN.save_model_weights("MountainCar_3_")
						saved3 = True

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.policy_DQN.eval()
		num_episodes = 100
		total_reward = 0
		for i_epidode in range(num_episodes):
			observation = self.env.reset()
			done = False
			while not done:
				# Select and perform an action
				observation_tensor = torch.tensor([observation], dtype=torch.float32)
				q_values = self.policy_DQN(observation_tensor)
				action = self.greedy_policy(q_values)
				new_observation, reward, done, _ = self.env.step(action.item())
				observation = new_observation
				total_reward += reward
		avg_reward = total_reward / num_episodes
		self.test_rewards[0].append(self.episode)
		self.test_rewards[1].append(avg_reward)
		print("Test result: ", avg_reward)
		return avg_reward

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		while self.memory.now_size < self.memory.burn_in:
			observation = self.env.reset()
			done = False
			while not done:
				# perform a random action
				action = self.env.action_space.sample()
				new_observation, reward, done, _ = self.env.step(action)
				# Store the transition in memory
				self.memory.append([observation, action, reward, new_observation])
				observation = new_observation

	def optimize(self, transitions):
		batch = Transition(*zip(*transitions))

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
												batch.next_state)), dtype=torch.bool)

		non_final_next_states = [torch.tensor([s], dtype=torch.float32)
								 for s in batch.next_state if s is not None]

		non_final_next_states = torch.cat(non_final_next_states)
		state_batch = [torch.tensor([s], dtype=torch.float32) for s in batch.state]
		state_batch = torch.cat(state_batch)
		action_batch = [torch.tensor([[1*(a==i) for i in range(self.env.action_space.n)]], dtype=torch.bool) for a in batch.action]
		action_batch = torch.cat(action_batch)
		reward_batch = [torch.tensor([r], dtype=torch.float32) for r in batch.reward]
		reward_batch = torch.cat(reward_batch)

		state_action_values = self.policy_DQN(state_batch)
		state_action_values = state_action_values.masked_select(action_batch).unsqueeze(1)

		next_state_values = torch.zeros(self.batch_size)
		next_state_values[non_final_mask] = self.policy_DQN(non_final_next_states).max(1)[0].detach()
		############  Uncomment this block for Double-Q learning ############
		# next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).max(1)[0].detach()
		#####################################################################

		expected_state_action_values = (next_state_values * self.gamma) + reward_batch
		expected_state_action_values = expected_state_action_values.unsqueeze(1)
		loss = F.mse_loss(state_action_values, expected_state_action_values)
		self.optimizer.zero_grad()
		loss.backward()
		# for param in self.policy_DQN.parameters():
		# 	param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

		if self.itr % 10000 == 0:
			TD_error = state_action_values.detach() - expected_state_action_values.detach()
			TD_error = np.mean(TD_error.numpy())
			self.TD_errors[0].append(self.itr)
			self.TD_errors[1].append(TD_error)

	def plot(self):
		f1 = plt.figure(1)
		plt.plot(self.test_rewards[0], self.test_rewards[1])
		plt.title("Avg Test Reward - Training Episodes")
		f1.savefig("reward.png")
		f2 = plt.figure(2)
		plt.plot(self.TD_errors[0], self.TD_errors[1])
		plt.title("Avg TD Error - Training Iterations")
		f2.savefig("err.png")
		# plt.show()

# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi):
	# Usage: 
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
	#       	test_video(self, self.environment_name, episode)
	save_path = "./videos-%s-%s" % (env, epi)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	# To create video
	env = gym.wrappers.Monitor(agent.env, save_path, force=True)
	reward_total = []
	state = env.reset()
	agent.policy_DQN.eval()
	done = False
	while not done:
		env.render()
		state_tensor = torch.tensor([state], dtype=torch.float32)
		q_values = agent.policy_DQN(state_tensor)
		action = agent.greedy_policy(q_values)
		next_state, reward, done, info = env.step(action.item())
		state = next_state
		reward_total.append(reward)
	print("reward_total: {}".format(np.sum(reward_total)))
	agent.env.close()


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str,default='DQN')
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	environment_name = args.env

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	agent = DQN_Agent(environment_name)
	agent.train()
	agent.plot()

	# agent.policy_DQN.load_model_weights("results1/MountainCar_3_MountainCar-v0.pth")
	# test_video(agent, agent.environment_name, "3")




if __name__ == '__main__':
	main(sys.argv)

