import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


# class ActorNet(nn.Module):
# 	def __init__(self):
# 		super(ActorNet, self).__init__()
# 		self.num_action = 4
# 		self.num_state = 8
# 		self.fc1 = nn.Linear(self.num_state, 16)
# 		self.bn1 = nn.BatchNorm1d(16)
# 		self.fc2 = nn.Linear(16, 16)
# 		self.bn2 = nn.BatchNorm1d(16)
# 		self.fc3 = nn.Linear(16, 16)
# 		self.bn3 = nn.BatchNorm1d(16)
# 		self.fc4 = nn.Linear(16, self.num_action)
# 		nn.init.zeros_(self.fc1.bias)
# 		nn.init.zeros_(self.fc2.bias)
# 		nn.init.zeros_(self.fc3.bias)
# 		nn.init.zeros_(self.fc4.bias)
# 		nn.init.uniform_(self.fc1.weight, -np.sqrt(3 / 12), np.sqrt(3 / 12))
# 		nn.init.uniform_(self.fc2.weight, -np.sqrt(3 / 16), np.sqrt(3 / 16))
# 		nn.init.uniform_(self.fc3.weight, -np.sqrt(3 / 16), np.sqrt(3 / 16))
# 		nn.init.uniform_(self.fc4.weight, -np.sqrt(3 / 10), np.sqrt(3 / 10))
#
# 	def save_model_weights(self, suffix):
# 		model_file = suffix + ".pth"
# 		torch.save(self.state_dict(), model_file)
#
# 	def load_model(self, model_file):
# 		self.model = torch.load(model_file)
#
# 	def load_model_weights(self,weight_file):
# 		self.load_state_dict(torch.load(weight_file))
#
# 	def forward(self, x):
# 		x = F.relu(self.fc1(x))
# 		x = self.bn1(x)
# 		x = F.relu(self.fc2(x))
# 		x = self.bn2(x)
# 		x = F.relu(self.fc3(x))
# 		x = self.bn3(x)
# 		x = F.softmax(self.fc4(x))
# 		return x
#
#
# class CriticNet(nn.Module):
# 	def __init__(self):
# 		super(CriticNet, self).__init__()
# 		self.num_state = 8
# 		self.fc1 = nn.Linear(self.num_state, 16)
# 		self.bn1 = nn.BatchNorm1d(16)
# 		self.fc2 = nn.Linear(16, 16)
# 		self.bn2 = nn.BatchNorm1d(16)
# 		self.fc3 = nn.Linear(16, 16)
# 		self.bn3 = nn.BatchNorm1d(16)
# 		self.fc4 = nn.Linear(16, 1)
# 		nn.init.zeros_(self.fc1.bias)
# 		nn.init.zeros_(self.fc2.bias)
# 		nn.init.zeros_(self.fc3.bias)
# 		nn.init.zeros_(self.fc4.bias)
# 		nn.init.uniform_(self.fc1.weight, -np.sqrt(3 / 12), np.sqrt(3 / 12))
# 		nn.init.uniform_(self.fc2.weight, -np.sqrt(3 / 16), np.sqrt(3 / 16))
# 		nn.init.uniform_(self.fc3.weight, -np.sqrt(3 / 16), np.sqrt(3 / 16))
# 		nn.init.uniform_(self.fc4.weight, -np.sqrt(6 / 17), np.sqrt(6 / 17))
#
# 	def save_model_weights(self, suffix):
# 		model_file = suffix + ".pth"
# 		torch.save(self.state_dict(), model_file)
#
# 	def load_model(self, model_file):
# 		self.model = torch.load(model_file)
#
# 	def load_model_weights(self,weight_file):
# 		self.load_state_dict(torch.load(weight_file))
#
# 	def forward(self, x):
# 		x = F.relu(self.fc1(x))
# 		x = self.bn1(x)
# 		x = F.relu(self.fc2(x))
# 		x = self.bn2(x)
# 		x = F.relu(self.fc3(x))
# 		x = self.bn3(x)
# 		x = self.fc4(x)
# 		return x


class ActorNet(nn.Module):
	def __init__(self):
		super(ActorNet, self).__init__()
		self.num_action = 4
		self.num_state = 8
		self.fc1 = nn.Linear(self.num_state, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, self.num_action)
		nn.init.zeros_(self.fc1.bias)
		nn.init.zeros_(self.fc2.bias)
		nn.init.zeros_(self.fc3.bias)
		nn.init.uniform_(self.fc1.weight, -np.sqrt(3 / 36), np.sqrt(3 / 36))
		nn.init.uniform_(self.fc2.weight, -np.sqrt(3 / 64), np.sqrt(3 / 64))
		nn.init.uniform_(self.fc3.weight, -np.sqrt(3 / 34), np.sqrt(3 / 34))

	def save_model_weights(self, suffix):
		model_file = suffix + ".pth"
		torch.save(self.state_dict(), model_file)

	def load_model(self, model_file):
		self.model = torch.load(model_file)

	def load_model_weights(self,weight_file):
		self.load_state_dict(torch.load(weight_file))

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = F.softmax(x)
		return x


class CriticNet(nn.Module):
	def __init__(self):
		super(CriticNet, self).__init__()
		self.num_state = 8
		self.fc1 = nn.Linear(self.num_state, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 1)
		nn.init.zeros_(self.fc1.bias)
		nn.init.zeros_(self.fc2.bias)
		nn.init.zeros_(self.fc3.bias)
		nn.init.uniform_(self.fc1.weight, -np.sqrt(3 / 36), np.sqrt(3 / 36))
		nn.init.uniform_(self.fc2.weight, -np.sqrt(3 / 64), np.sqrt(3 / 64))
		nn.init.uniform_(self.fc3.weight, -np.sqrt(6 / 65), np.sqrt(6 / 65))

	def save_model_weights(self, suffix):
		model_file = suffix + ".pth"
		torch.save(self.state_dict(), model_file)

	def load_model(self, model_file):
		self.model = torch.load(model_file)

	def load_model_weights(self,weight_file):
		self.load_state_dict(torch.load(weight_file))

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class A2C():
	# Implementation of N-step Advantage Actor Critic.
	# This class inherits the Reinforce class, so for example, you can reuse
	# generate_episode() here.

	def __init__(self, model, lr, critic_model, critic_lr, n=20):
		# Initializes A2C.
		# Args:
		# - model: The actor model.
		# - lr: Learning rate for the actor model.
		# - critic_model: The critic model.
		# - critic_lr: Learning rate for the critic model.
		# - n: The value of N in N-step A2C.
		self.model = model
		self.critic_model = critic_model
		self.n = n
		self.lr = lr
		self.critic_lr = critic_lr
		self.gamma = 0.99
		self.gamma_n = self.gamma ** self.n
		self.env = gym.make("LunarLander-v2")
		self.env.seed(1)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)

	def stochastic_policy(self, q_values):
		actions = [i for i in range(self.model.num_action)]
		probabilities = q_values.detach().numpy().squeeze()
		action = np.random.choice(actions, 1, p=probabilities)
		return action[0]

	def train(self):
		# Trains the model on a single episode using A2C.
		#       method generate_episode() to generate training data.
		states, actions, rewards = self.generate_episode()
		T = len(states)
		cumulative_r = 0
		returns = []
		for t in reversed(range(T)):
			if t + self.n >= T:
				V_end = 0
			else:
				state_tensor = torch.tensor([states[t+self.n]], dtype=torch.float32)
				self.critic_model.eval()
				with torch.no_grad():
					V_end = self.critic_model(state_tensor)
				cumulative_r -= (self.gamma_n / self.gamma ) * rewards[t+self.n]
			cumulative_r = rewards[t] + self.gamma * cumulative_r
			R_t = self.gamma_n * V_end + cumulative_r
			returns.append(R_t)
		returns.reverse()

		self.model.train()
		self.critic_model.train()
		states_tensor = torch.tensor(states, dtype=torch.float32)
		num_act = self.model.num_action
		action_mask = [torch.tensor([[1*(a==i) for i in range(num_act)]], dtype=torch.bool) for a in actions]
		action_mask = torch.cat(action_mask)
		# calculate model loss
		with torch.no_grad():
			advantage = torch.tensor(returns) - self.critic_model(states_tensor).squeeze()
		q_values = torch.clamp(self.model(states_tensor), 1e-7, 1)
		loss_model = advantage * torch.log(q_values.masked_select(action_mask))
		loss_model = -torch.mean(loss_model)
		# calculate critic loss
		advantage = torch.tensor(returns) - self.critic_model(states_tensor).squeeze()
		loss_critic = advantage.pow(2)
		loss_critic = torch.mean(loss_critic)

		loss = loss_model + loss_critic
		self.optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.critic_optimizer.step()

	def test(self, episodes=100, render=False):
		self.model.eval()
		self.critic_model.eval()
		sum_rewards = []
		for i in range(episodes):
			sum_reward = 0
			state = self.env.reset()
			done = False
			while not done:
				if render and i % 20 == 0:
					self.env.render()
				state_tensor = torch.tensor([state], dtype=torch.float32)
				self.model.eval()
				q_values = torch.clamp(self.model(state_tensor), 1e-7, 1)
				action = self.stochastic_policy(q_values)
				new_state, reward, done, _ = self.env.step(action)
				sum_reward += reward
				state = new_state
			sum_rewards.append(sum_reward)
		avg_reward = np.mean(sum_rewards)
		std_reward = np.std(sum_rewards)
		return avg_reward, std_reward

	def generate_episode(self, render=False):
		# Generates an episode by executing the current policy in the given env.
		# Returns:
		# - a list of states, indexed by time step
		# - a list of actions, indexed by time step
		# - a list of rewards, indexed by time step
		states = []
		actions = []
		rewards = []
		state = self.env.reset()
		done = False
		while not done:
			if render:
				self.env.render()
			state_tensor = torch.tensor([state], dtype=torch.float32)
			self.model.eval()
			q_values = self.model(state_tensor)
			action = self.stochastic_policy(q_values)
			new_state, reward, done, _ = self.env.step(action)
			states.append(state)
			actions.append(action)
			rewards.append(reward * 1e-2)
			state = new_state
		return states, actions, rewards


def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=50000, help="Number of episodes to train on.")
	parser.add_argument('--lr', dest='lr', type=float,
						default=5e-4, help="The actor's learning rate.")
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--n', dest='n', type=int,
						default=20, help="The value of N in N-step A2C.")

	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
							  action='store_true',
							  help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
							  action='store_false',
							  help="Whether to render the environment.")
	parser.set_defaults(render=False)

	return parser.parse_args()


def main(args):
	# Parse command-line arguments.
	args = parse_arguments()
	num_episodes = args.num_episodes
	lr = args.lr
	critic_lr = args.critic_lr
	n = args.n
	render = args.render

	# Create the environment.
	env = gym.make('LunarLander-v2')

	model = ActorNet()
	critic_model = CriticNet()
	a2c = A2C(model, 1e-3, critic_model, 1e-3, n=20)
	avg_rewards = []
	std_rewards = []
	trained_epi = []
	for i in range(50000):
		a2c.train()
		if i % 500 == 0:
			avg_reward, std_reward = a2c.test(render=False)
			trained_epi.append(i)
			avg_rewards.append(avg_reward)
			std_rewards.append(std_reward)
			print("Average reward and std after %d episodes: %f  %f"%(i, avg_reward, std_reward))
	f1 = plt.figure(1)
	plt.errorbar(trained_epi, avg_rewards, std_rewards)
	plt.title("Avg Test Reward - Training Episodes")
	f1.savefig("reward_%d.png"%a2c.n)


if __name__ == '__main__':
	main(sys.argv)
