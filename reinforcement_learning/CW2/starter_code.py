# Import some modules from other libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import collections
import time

# Import the environment module
from environment import Environment

# policy_states = []


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment, epsilon=1.0):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Policy
        self.policy = []
        # Epsilon greedy
        self.epsilon = epsilon
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0
        # Policy
        self.policy = []

    # Function to make the agent take one step in the environment.
    def step(self, reward_f):
        # Choose an action.
        discrete_actions = [0, 1, 2, 3]
        discrete_action = np.random.choice(discrete_actions)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this action.

        reward = reward_f(self.state, next_state, distance_to_goal)

        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    def greedy_step(self, network, reward_f):
        Q_values = network.q_network.forward(torch.tensor(self.state))
        max_Q_value, argmax = Q_values.max(0)
        # Choose action with max value
        discrete_action = argmax.item()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this action.

        reward = reward_f(self.state, next_state, distance_to_goal)

        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    def epsilon_greedy_step(self, network, delta, reward_f):
        discrete_actions = [0, 1, 2, 3]
        Q_values = network.q_network.forward(torch.tensor(self.state))
        max_Q_value, argmax = Q_values.max(0)
        # Initialise p for epsilon greedy
        p = [self.epsilon/3] * 4
        p[argmax.item()] = 1 - self.epsilon
        # Choose epsilon greedy action based on p
        discrete_action = np.random.choice(discrete_actions, p=p)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this action.

        reward = reward_f(self. state, next_state, distance_to_goal)

        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Update epsilon value by delta
        # print('Current epsilon = {:.2f}    Greedy action probability = {:.2f}%'.format(self.epsilon, (1 - self.epsilon) * 100))
        self.epsilon -= delta
        if self.epsilon < 0:
            self.epsilon = 0
        # Return the transition
        return transition

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def original_reward(self, state, next_state, distance_to_goal):
        reward = 1 - distance_to_goal
        return reward

    def custom_reward(self, state, next_state, distance_to_goal):
        current_L1_distance = np.linalg.norm(state - self.environment.goal_state)
        next_L1_distance = np.linalg.norm(next_state - self.environment.goal_state)

        if next_L1_distance >= current_L1_distance:
            reward = -10

        else:
            reward = 15

        if next_L1_distance < 0.05:
            reward = 100

        return reward

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:  # Move down
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        elif discrete_action == 2:  # Move left
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 3:  # Move up
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        return continuous_action


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Create a target network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        # Initialise target network
        self.update_target_network()

    def reset(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Create a target network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        # Initialise target network
        self.update_target_network()

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition, use_target_network=True):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition, use_target_network)
        # print(loss)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, use_target_network, gamma=0.9):
        # s, a, r, ns = transition

        batch_input = [transition[0] for transition in transitions]
        batch_input_tensor = torch.tensor(batch_input)

        batch_action = [[transition[1]] for transition in transitions]
        batch_action_tensor = torch.tensor(batch_action)

        batch_R = [[transition[2]] for transition in transitions]
        batch_R_tensor = torch.tensor(batch_R)

        batch_next_states = [transition[3] for transition in transitions]
        batch_next_states_tensor = torch.tensor(batch_next_states)

        network_prediction = self.q_network.forward(batch_input_tensor)
        network_prediction_for_action = torch.gather(network_prediction, 1, batch_action_tensor)

        # target network
        if use_target_network:
            next_states_prediction = self.target_network.forward(batch_next_states_tensor)
        else:
            next_states_prediction = self.q_network.forward(batch_next_states_tensor)

        target_optimal_Q_tensor, optimal_a_tensor = next_states_prediction.max(1)

        TD_target = batch_R_tensor + gamma * target_optimal_Q_tensor.unsqueeze(1)

        return torch.nn.MSELoss()(network_prediction_for_action, TD_target)

    def update_target_network(self):
        # print('----- Updating target network -----')
        weights = self.q_network.state_dict()
        self.target_network.load_state_dict(weights)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = collections.deque(maxlen=self.capacity)

    def reset(self):
        self.container = collections.deque(maxlen=self.capacity)

    def add_transitions(self, transition):
        self.container.append(transition)

    def sample_mini_batch(self, mini_batch_size):
        batch_ind = np.random.choice(range(len(self.container)), mini_batch_size, replace=False)
        # print(batch_ind)
        return [self.container[ind] for ind in batch_ind]

    def get_size(self):
        return len(self.container)


class Visualiser:
    def __init__(self, magnification, network):
        self.network = network
        self.height = 1.0
        self.width = 1.0
        self.magnification = magnification
        self.image = np.zeros([int(self.magnification * self.height), int(self.magnification * self.width), 3], dtype=np.uint8)

    def draw_Q(self, save=False):
        # Create a black image
        self.image.fill(0)
        # Triangles
        for x in range(10):
            for y in range(10):

                state_x = x/10 + 0.05
                state_y = (10 - y)/10 - 0.05

                centre_x = state_x
                centre_y = y/10 + 0.05

                colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

                Q_values = self.network.q_network.forward(torch.tensor([state_x, state_y]))

                max_Q_value, argmax = Q_values.max(0)
                max_Q_value = max_Q_value.item()

                # print(state_x, state_y, argmax)

                min_Q_value, argmin = Q_values.min(0)
                min_Q_value = min_Q_value.item()

                yellow = (0, 255, 255)
                blue = (255, 0, 0)

                for ind in range(len(Q_values)):
                    Q_value = Q_values[ind].item()
                    # print(Q_value)
                    frac = (Q_value - min_Q_value) / (max_Q_value - min_Q_value)
                    colours[ind] = tuple((y - b) * frac + b for y, b in zip(yellow, blue))

                centre = (int(centre_x * self.magnification), int(centre_y * self.magnification))
                left = int((x/10) * self.magnification)
                right = int((x/10 + 0.1) * self.magnification)
                top = int((y/10) * self.magnification)
                bottom = int((y/10 + 0.1) * self.magnification)

                top_left = (left, top)
                top_right = (right, top)
                bottom_left = (left, bottom)
                bottom_right = (right, bottom)

                cv2.fillConvexPoly(self.image, np.asarray([centre, top_left, top_right]), color=colours[3])  # Up (3)
                cv2.fillConvexPoly(self.image, np.asarray([centre, top_left, bottom_left]), color=colours[2])  # Left (2)
                cv2.fillConvexPoly(self.image, np.asarray([centre, bottom_left, bottom_right]), color=colours[1])  # Down (1)
                cv2.fillConvexPoly(self.image, np.asarray([centre, bottom_right, top_right]), color=colours[0])  # Right (0)

                cv2.polylines(self.image, [np.asarray([centre, top_left, top_right], dtype='int32')], color=(0, 0, 0), isClosed=True, thickness=1)  # Up (3)
                cv2.polylines(self.image, [np.asarray([centre, top_left, bottom_left], dtype='int32')], color=(0, 0, 0), isClosed=True, thickness=1)  # Left (2)
                cv2.polylines(self.image, [np.asarray([centre, bottom_left, bottom_right], dtype='int32')], color=(0, 0, 0), isClosed=True, thickness=1)  # Down (1)
                cv2.polylines(self.image, [np.asarray([centre, bottom_right, top_right], dtype='int32')], color=(0, 0, 0), isClosed=True, thickness=1)  # Right (0)


        # White gridlines
        line_colour = (255, 255, 255)
        for i in range(11):
            vertical_line_start = (int(self.width * i * self.magnification/10), 0)
            vertical_line_end = (int(self.width * i * self.magnification/10), int(self.height * self.magnification))
            cv2.line(self.image, vertical_line_start, vertical_line_end, color=line_colour, thickness=2)

            horizontal_line_start = (0, int(self.height * i * self.magnification / 10))
            horizontal_line_end = (int(self.width * self.magnification), int(self.height * i * self.magnification / 10))
            cv2.line(self.image, horizontal_line_start, horizontal_line_end, color=line_colour, thickness=2)

        cv2.imshow('Q-Value Visualisation', self.image)

        if save:
            cv2.imwrite('Figure_3.png', self.image)

        cv2.waitKey(0)

    def draw_policy(self, agent, save=False):
        # Create a black image
        self.image.fill(0)
        # White gridlines
        line_colour = (255, 255, 255)
        for i in range(11):
            vertical_line_start = (int(self.width * i * self.magnification / 10), 0)
            vertical_line_end = (int(self.width * i * self.magnification / 10), int(self.height * self.magnification))
            cv2.line(self.image, vertical_line_start, vertical_line_end, color=line_colour, thickness=2)

            horizontal_line_start = (0, int(self.height * i * self.magnification / 10))
            horizontal_line_end = (int(self.width * self.magnification), int(self.height * i * self.magnification / 10))
            cv2.line(self.image, horizontal_line_start, horizontal_line_end, color=line_colour, thickness=2)

        start_x_pos = int(agent.policy[0][0][0] * self.magnification)
        start_y_pos = int((1- agent.policy[0][0][1]) * self.magnification)

        end_x_pos = int(agent.policy[-1][0][0] * self.magnification)
        end_y_pos = int((1 - agent.policy[-1][0][1]) * self.magnification)

        cv2.circle(self.image, (start_x_pos, start_y_pos), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(self.image, (end_x_pos, end_y_pos), 10, (0, 255, 0), cv2.FILLED)

        step = 0
        for transition in agent.policy:

            step += 1
            red = (0, 0, 255)
            green = (0, 255, 0)

            frac = step / len(agent.policy)

            colour = tuple((g - r) * frac + r for g, r in zip(green, red))

            from_state = transition[0]
            to_state = transition[3]

            from_x_pos = int(from_state[0] * self.magnification)
            from_y_pos = int((1 - from_state[1]) * self.magnification)

            to_x_pos = int(to_state[0] * self.magnification)
            to_y_pos = int((1 - to_state[1]) * self.magnification)

            # centre = (x_pos, y_pos)

            cv2.line(self.image, (from_x_pos, from_y_pos), (to_x_pos, to_y_pos), color=colour, thickness=5)

        cv2.imshow('Policy Visualisation', self.image)
        if save:
            cv2.imwrite('Figure_4.png', self.image)
        cv2.waitKey(0)


def plot_delta(delta):
    # Set the random seed for both NumPy and Torch
    # You should leave this as 0, for consistency across different runs (Deep Reinforcement Learning is highly sensitive to different random seeds, so keeping this the same throughout will help you debug your code).
    CID = 184493
    np.random.seed(CID)
    torch.manual_seed(CID)

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop PCs, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=750)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    # Create a replay buffer
    replay_buffer = ReplayBuffer(1000000)
    # Hyper-parameters
    mini_batch_size = 50
    num_episode = 25
    episode_steps = 20

    steps = []
    losses = []
    episode = 0
    step_count = 0

    while episode < num_episode:
        episode += 1
        # print('---------------------------Episode {} ---------------------------'.format(episode))
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(episode_steps):
            step_count += 1
            # Step the agent once, and get the transition tuple for this step
            # transition = agent.step()
            transition = agent.epsilon_greedy_step(dqn, delta, agent.custom_reward)

            replay_buffer.add_transitions(transition)

            # loss = dqn._calculate_loss(transition)

            if replay_buffer.get_size() >= mini_batch_size:
                mini_batch = replay_buffer.sample_mini_batch(mini_batch_size)
                loss = dqn.train_q_network(mini_batch, use_target_network=True)

                steps.append(step_count)
                losses.append(loss)

        # Update target network
        dqn.update_target_network()

    # Compute optimal (greedy) policy after training
    agent.reset()

    for greedy_step_num in range(episode_steps):
        transition = agent.greedy_step(dqn, agent.custom_reward)
        agent.policy.append(transition)
    # print total reward for current episode
    print("Delta = {}".format(delta))
    print("Total greedy reward = {}".format(agent.total_reward))

    return agent.total_reward


def plot_final_distance(custom=False):
    # Set the random seed for both NumPy and Torch
    # You should leave this as 0, for consistency across different runs (Deep Reinforcement Learning is highly sensitive to different random seeds, so keeping this the same throughout will help you debug your code).
    CID = 184493
    np.random.seed(CID)
    torch.manual_seed(CID)

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop PCs, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=750)
    greedy_environment = Environment(display=False, magnification=750)
    # Create an agent
    agent = Agent(environment)
    greedy_agent = Agent(greedy_environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    # Create a replay buffer
    replay_buffer = ReplayBuffer(1000000)
    # Hyper-parameters
    mini_batch_size = 50
    num_episode = 25
    episode_steps = 20
    delta = 0.05

    steps = []
    losses = []
    episode = 0
    step_count = 0

    final_distances = []

    while episode < num_episode:
        episode += 1
        # print('---------------------------Episode {} ---------------------------'.format(episode))
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(episode_steps):
            step_count += 1
            steps.append(step_count)

            # Step the agent once, and get the transition tuple for this step
            # transition = agent.step()
            if custom:
                transition = agent.epsilon_greedy_step(dqn, delta, agent.custom_reward)
            else:
                transition = agent.epsilon_greedy_step(dqn, delta, agent.original_reward)

            replay_buffer.add_transitions(transition)

            if replay_buffer.get_size() >= mini_batch_size:
                mini_batch = replay_buffer.sample_mini_batch(mini_batch_size)
                loss = dqn.train_q_network(mini_batch, use_target_network=True)

                losses.append(loss)

            # Compute optimal (greedy) policy after training
            greedy_agent.reset()

            for greedy_step_num in range(episode_steps):
                if custom:
                    greedy_transition = greedy_agent.greedy_step(dqn, greedy_agent.custom_reward)
                else:
                    greedy_transition = greedy_agent.greedy_step(dqn, greedy_agent.original_reward)
                agent.policy.append(greedy_transition)

            final_distance = np.linalg.norm(greedy_transition[3] - greedy_environment.goal_state)
            final_distances.append(final_distance)

        # Update target network
        dqn.update_target_network()

    if custom:
        plt.plot(steps, final_distances, label='Custom Reward')
    else:
        plt.plot(steps, final_distances, label='Original Reward')


# Main entry point
if __name__ == "__main__":

    # Set the random seed for both NumPy and Torch
    # You should leave this as 0, for consistency across different runs (Deep Reinforcement Learning is highly sensitive to different random seeds, so keeping this the same throughout will help you debug your code).
    CID = 184493
    np.random.seed(CID)
    torch.manual_seed(CID)

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop PCs, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=750)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    # Create a replay buffer
    replay_buffer = ReplayBuffer(1000000)
    # Hyper-parameters
    mini_batch_size = 50
    num_episode = 25
    episode_steps = 20
    # delta value for epsilon greedy
    delta = 0.05
    # Loop over episodes
    deltas = []
    total_rewards = []

    steps = []
    losses = []
    episode = 0
    step_count = 0

    while episode < num_episode:
        episode += 1
        print('---------------------------Episode {} ---------------------------'.format(episode))
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(episode_steps):
            step_count += 1
            # Step the agent once, and get the transition tuple for this step
            # transition = agent.step()
            transition = agent.epsilon_greedy_step(dqn, delta, agent.custom_reward)

            replay_buffer.add_transitions(transition)

            if replay_buffer.get_size() >= mini_batch_size:
                mini_batch = replay_buffer.sample_mini_batch(mini_batch_size)
                loss = dqn.train_q_network(mini_batch, use_target_network=True)

                steps.append(step_count)
                losses.append(loss)

        # Update target network
        dqn.update_target_network()

    # Compute optimal (greedy) policy after training
    agent.reset()

    for greedy_step_num in range(episode_steps):
        transition = agent.greedy_step(dqn, agent.custom_reward)
        agent.policy.append(transition)
        time.sleep(0.5)

    # print total reward for current episode
    print("Delta = {}".format(delta))
    print("Total greedy reward = {}".format(agent.total_reward))

    deltas.append(delta)
    total_rewards.append(agent.total_reward)

    visualiser = Visualiser(500, dqn)
    # visualiser.draw_Q()
    visualiser.draw_policy(agent)

    # plt.plot(steps, losses)
    # plt.title('Log Loss Against Steps Taken With Target Network')
    # plt.xlabel('Steps taken')
    # plt.ylabel('Loss (Log)')
    # plt.xlim(0, num_episode * episode_steps)
    # plt.yscale('log')
    # plt.show()

