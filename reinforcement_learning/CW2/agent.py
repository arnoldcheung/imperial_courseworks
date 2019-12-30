############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch

import collections

# Defining the Q-Network
class Network(torch.nn.Module):

    def __init__(self, dimensions):
        super(Network, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dimensions[i], dimensions[i+1]) for i in range(len(dimensions) - 1)])

    def forward(self, input):
        out = torch.nn.functional.relu(self.layers[0](input))
        for layer in self.layers[1:-1]:
            out = torch.nn.functional.relu(layer(out))
        return self.layers[-1](out)


class DQN:

    def __init__(self, dimensions, lr=0.001, gamma=1.0):
        self.dimensions = dimensions
        self.lr = lr
        self.gamma = gamma

        # Main Q-Network
        self.q_network = Network(self.dimensions)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Target network
        self.target_network = Network(self.dimensions)
        self.update_network(self.target_network, self.q_network)

        # Best network (saves the most recent network that arrives at the goal)
        self.best_network = Network(self.dimensions)
        self.update_network(self.best_network, self.q_network)

    # Reset the network (Not used)
    def reset(self):
        self.q_network = Network(self.dimensions)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.target_network = Network(self.dimensions)
        self.update_network(self.target_network, self.q_network)

    # Main training function of the Q-Network
    def train_q_network(self, transition, use_target_network=True):
        self.optimiser.zero_grad()
        loss, TD_error = self._calculate_loss(transition, use_target_network)

        loss.backward()
        self.optimiser.step()

        # Returns the loss and the TD error (for prioritised experience replay)
        return loss.item(), TD_error

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, use_target_network):

        # Input of the sampled transitions from experience replay buffer

        # States
        batch_input = [transition[0] for transition in transitions]
        batch_input_tensor = torch.tensor(batch_input)

        # Actions
        batch_action = [[transition[1]] for transition in transitions]
        batch_action_tensor = torch.tensor(batch_action)

        # Instant rewards
        batch_R = [[transition[2]] for transition in transitions]
        batch_R_tensor = torch.tensor(batch_R)

        # Next states
        batch_next_states = [transition[3] for transition in transitions]
        batch_next_states_tensor = torch.tensor(batch_next_states)

        # Q-Network output Q-value reward predictions for input states and actions
        network_prediction = self.q_network.forward(batch_input_tensor)
        network_prediction_for_action = torch.gather(network_prediction, 1, batch_action_tensor)

        # Target network
        if use_target_network:
            next_states_prediction = self.target_network.forward(batch_next_states_tensor)
        else:
            next_states_prediction = self.q_network.forward(batch_next_states_tensor)

        # Chooses optimal action / action value for Q-network or target network
        target_optimal_Q_tensor, optimal_a_tensor = next_states_prediction.max(1)


        # Double Q-learning
        if use_target_network:
            next_states_prediction = self.q_network.forward(batch_next_states_tensor)
            target_optimal_Q_tensor = torch.gather(next_states_prediction, 1, optimal_a_tensor.unsqueeze(1))


        TD_target = batch_R_tensor + self.gamma * target_optimal_Q_tensor

        TD_error = TD_target - network_prediction_for_action
        # print(TD_error)

        return torch.nn.MSELoss()(network_prediction_for_action, TD_target), TD_error

    def update_network(self, to_network, from_network):
        # print('----- Updating target network -----')
        weights = from_network.state_dict()
        to_network.load_state_dict(weights)


class ReplayBuffer:
    def __init__(self, capacity, mini_batch_size, alpha):
        self.capacity = capacity
        self.container = collections.deque(maxlen=self.capacity)

        self.mini_batch_size = mini_batch_size

        self.alpha = alpha
        self.weights = collections.deque(maxlen=self.capacity)
        self.wdenom = None  # denominator
        self.p = []

    def reset(self):
        self.container = collections.deque(maxlen=self.capacity)

    def add_transitions(self, transition):
        self.container.append(transition)

        if self.weights:
            self.weights.append(max(self.weights))
        else:
            self.weights.append(1)

        self.wdenom = sum([w**self.alpha for w in self.weights])

    def sample_mini_batch(self):
        batch_ind = np.random.choice(range(len(self.container)), self.mini_batch_size, p=self.p, replace=False)
        # print(batch_ind)
        return batch_ind, [self.container[ind] for ind in batch_ind]

    def update_weights(self, ind, TD_error, epsilon=0.01):
        for i, e in zip(ind, TD_error):
            weight = abs(e) + epsilon
            self.weights[i] = weight.item()

    def update_p(self):
        self.p = [w ** self.alpha / self.wdenom for w in self.weights]

    def get_size(self):
        return len(self.container)


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 100000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Previous data (Not used)
        self.previous_distance_to_goal = 1000
        self.previous_advanced_distance = 0
        self.previous_state = None
        self.previous_reward = 0

        # Hyper parameters
        # Actions
        self.step_size = 0.02

        # Q-network
        dqn_sizes = [2, 50, 50, 8]
        lr = 0.01
        gamma = 0.9
        self.dqn = DQN(dqn_sizes, lr, gamma)
        self.training_interval = 1

        # Experience replay buffer
        buffer_capacity = 1000
        self.mini_batch_size = 300
        alpha = 1  # Affects weighting / probability
        self.num_exploration_steps = 5000  # Number of steps before delta starts decreasing
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.mini_batch_size, alpha)
        self.stop_training = False

        # Target network
        self.use_target_network = True
        self.target_network_update_interval = 100

        # Greedy epsilon parameters
        self.delta = 0.0001
        self.epsilon = 0.875
        self.minimum_epsilon = 0.7

        # Best_network
        self.has_best_network = False

        # Counters (resets agent when stuck counter == 20)
        self.current_episode_step_counter = 0
        self.stuck_counter = 0
        self.goal_counter = 0
        self.consecutive_goals = 0
        self.seen_goal = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):

        if self.has_best_network:
            print('Replaced q-network with best network')
            self.dqn.update_network(self.dqn.q_network, self.dqn.best_network)

        if self.num_steps_taken % self.episode_length == 0:
            self.consecutive_goals = 0
            self.previous_distance_to_goal = None

            self.current_episode_step_counter = 0
            return True

        # Check if the agent is stuck, reset if stuck
        if self.stuck_counter > 20:
            self.consecutive_goals = 0
            self.stuck_counter = 0

            if self.epsilon < 0.15:
                self.epsilon = 0.15

            # Replace network with last best network if stuck
            if self.has_best_network:
                print('Replaced q-network with best network')
                self.dqn.update_network(self.dqn.q_network, self.dqn.best_network)
            print('Agent is stuck - terminating episode')

            self.current_episode_step_counter = 0
            return True

        if self.goal_counter > 20 and self.num_steps_taken > self.num_exploration_steps:
            print('Reached goal - terminating episode')

            self.goal_counter = 0
            if self.epsilon == 0:
                self.consecutive_goals += 1
                print('Consecutive goals: {}'.format(self.consecutive_goals))

            if self.consecutive_goals == 2:
                self.dqn.update_network(self.dqn.best_network, self.dqn.q_network)
                print('Updated best network')
                self.has_best_network = True
                self.stop_training = True

            self.current_episode_step_counter = 0
            return True

        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state, kind='greedy_epsilon'):

        # Epsilon greedy action
        if kind == 'greedy_epsilon':
            # Choose an action.

            # 8 discrete actions, including diagonals
            discrete_actions = [0, 1, 2, 3, 4, 5, 6, 7]

            # Get Q-value predictions from Q-Network
            Q_values = self.dqn.q_network.forward(torch.tensor(state))

            # Get action with highest value
            max_Q_value, argmax = Q_values.max(0)

            # Calculate probability of each action base on epsilon
            p = [self.epsilon / (len(discrete_actions) - 1)] * len(discrete_actions)
            p[argmax.item()] = 1 - self.epsilon

            # Choose epsilon greedy action based on p
            discrete_action = np.random.choice(discrete_actions, p=p)
            discrete_action = np.random.choice(discrete_actions, p=p)

            # print('Current Epsilon = {}'.format(self.epsilon))
            # if self.num_steps_taken >= self.num_exploration_steps and self.seen_goal:
            #     print('End of random exploration: epsilon can now decrease below minimum epsilon')

            # Start reducing delta
            if self.num_steps_taken > self.num_exploration_steps:
                self.epsilon -= self.delta
                if self.epsilon < self.minimum_epsilon:
                    self.epsilon = self.minimum_epsilon

            # Check if the agent is stuck bouncing between two states
            if self.epsilon < 0.2 and self.current_episode_step_counter > 500:
                print('Agent stuck detected: return epsilon to 0.5')
                self.stuck_counter = 21
                self.epsilon = 0.2

            # Convert the discrete action into a continuous action.
            action = self._discrete_action_to_continuous(discrete_action)

            # Store the action; this will be used later, when storing the transition
            self.action = discrete_action

        # Random action (not used)
        elif kind == 'random':
            # Random action
            discrete_actions = [0, 1, 2, 3, 4, 5, 6, 7]
            discrete_action = np.random.choice(discrete_actions)

            # Convert the discrete action into a continuous action.
            action = self._discrete_action_to_continuous(discrete_action)

            # Store the action; this will be used later, when storing the transition
            self.action = discrete_action

        # Continuous action (not used)
        elif kind == 'continuous':
            action = np.random.uniform(low=-self.step_size, high=self.step_size, size=2).astype(np.float32)
            pass

        # Previous states (not used)
        if not self.num_steps_taken % self.episode_length == 0:
            self.previous_state = self.state

        # Update target network
        if self.num_steps_taken % self.target_network_update_interval == 0:
            print('Step: {} - Updating target network'.format(self.num_steps_taken))
            print('Current epsilon = {}'.format(self.epsilon))
            self.dqn.update_network(self.dqn.target_network, self.dqn.q_network)

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        self.current_episode_step_counter += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # return action
        return action

    def _discrete_action_to_continuous(self, discrete_action):

        # Calculate max diagonal travel distance using radius = 0.02
        diag_step = self.step_size / np.sqrt(2)

        if discrete_action == 0:  # Move right
            continuous_action = np.array([self.step_size, 0], dtype=np.float32)
        elif discrete_action == 1:  # Move down
            continuous_action = np.array([0, -self.step_size], dtype=np.float32)
        elif discrete_action == 2:  # Move left
            continuous_action = np.array([-self.step_size, 0], dtype=np.float32)
        elif discrete_action == 3:  # Move up
            continuous_action = np.array([0, self.step_size], dtype=np.float32)

        elif discrete_action == 4:  # Move down right
            continuous_action = np.array([diag_step, -diag_step], dtype=np.float32)
        elif discrete_action == 5:  # Move down left
            continuous_action = np.array([-diag_step, -diag_step], dtype=np.float32)
        elif discrete_action == 6:  # Move up left
            continuous_action = np.array([-diag_step, diag_step], dtype=np.float32)
        elif discrete_action == 7:  # Move up right
            continuous_action = np.array([diag_step, diag_step], dtype=np.float32)

        return continuous_action

    # Testing different reward functions
    def get_reward_1(self, distance_to_goal, next_state):
        if not self.previous_distance_to_goal:
            advanced_distance = 0.02
        else:
            advanced_distance = self.previous_distance_to_goal - distance_to_goal
        self.previous_advanced_distance = advanced_distance
        self.previous_distance_to_goal = distance_to_goal

        critereon = 0.01

        reward = 0

        if advanced_distance > - critereon and not (self.state == next_state).all():

            if 0.8 < distance_to_goal:
                reward += 200 * advanced_distance

            elif 0.6 < distance_to_goal <= 0.8:
                reward += 400 * advanced_distance

            elif 0.4 < distance_to_goal <= 0.6:
                reward += 600 * advanced_distance

            elif 0.2 < distance_to_goal <= 0.4:
                reward += 1200 * advanced_distance

            elif 0.05 < distance_to_goal <= 0.2:
                reward += 2000 * advanced_distance

            elif distance_to_goal <= 0.05:
                reward += 6000 * advanced_distance

            reward = reward * (3 - distance_to_goal)

        else:
            reward = - 100

        return reward

    # def get_reward_2(self, distance_to_goal, next_state):
    #     # print('Previous distace to goal = {}'.format(self.previous_distance_to_goal))
    #     # print('Distance to goal = {}'.format(distance_to_goal))
    #
    #     advanced_distance = self.previous_distance_to_goal - distance_to_goal
    #
    #     if advanced_distance > 0:
    #         reward = 3 - distance_to_goal
    #
    #     elif abs(advanced_distance) < self.step_size / 1.5:
    #         if advanced_distance == 0:
    #             reward = -10 * (1 + self.stuck_counter)
    #         else:
    #             reward = 0
    #
    #     else:
    #         reward = - 10
    #
    #     if distance_to_goal < 0.03:
    #         reward = 500
    #
    #     print('Reward = {}'.format(reward))
    #
    #     return reward
    #
    # def get_reward_3(self, distance_to_goal, next_state):
    #
    #     advanced_distance = self.previous_distance_to_goal - distance_to_goal
    #
    #     print(advanced_distance)
    #
    #     if advanced_distance == 0 or (advanced_distance < - self.step_size / 1.5):
    #         reward = -200
    #
    #     elif abs(advanced_distance) < self.step_size / 1.5:
    #         reward = 20
    #
    #     else:
    #         reward = 1000 * advanced_distance * (5 - distance_to_goal)
    #
    #     if distance_to_goal < 0.05:
    #         reward = 500
    #
    #     print('Reward = {}'.format(reward))
    #
    #     return reward
    #
    # def get_reward_4(self, distance_to_goal, next_state):
    #     reward = 10 * (3 - distance_to_goal) ** 2
    #
    #     # multiplier = np.random.uniform(low=0.9, high=1.1)
    #
    #     if (self.state == next_state).all():
    #         reward += -10 * (1 + self.stuck_counter)
    #
    #     # reward = reward * multiplier
    #
    #     if distance_to_goal < 0.05:
    #         reward = 100
    #
    #     print('Reward = {}'.format(reward))
    #
    #     return reward

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # check stuck
        if (self.state == next_state).all():
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if distance_to_goal < 0.05:
            self.goal_counter += 1
            self.seen_goal = True
            self.minimum_epsilon = 0
            print('The agent has seen the goal')
        else:
            self.goal_counter = 0

        # Convert the distance to a reward
        reward = self.get_reward_1(distance_to_goal, next_state)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Now you can do something with this transition ...

        self.replay_buffer.add_transitions(transition)

        if self.replay_buffer.get_size() >= self.mini_batch_size and not self.stop_training:
            if self.num_steps_taken % self.training_interval == 0:

                # print('Training ... ')

                self.replay_buffer.update_p()

                ind, mini_batch = self.replay_buffer.sample_mini_batch()
                loss, TD_error = self.dqn.train_q_network(mini_batch, use_target_network=self.use_target_network)
                # print('Loss = {}'.format(loss))

                self.replay_buffer.update_weights(ind, TD_error)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        Q_values = self.dqn.q_network.forward(torch.tensor(state))
        max_Q_value, argmax = Q_values.max(0)
        # Choose action with max value
        discrete_action = argmax.item()
        # Convert the discrete action into a continuous action.
        action = self._discrete_action_to_continuous(discrete_action)

        return action

