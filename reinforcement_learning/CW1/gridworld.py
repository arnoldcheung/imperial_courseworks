import numpy as np

num_rows = 4
num_cols = 4
reward_states = [(0, 1)]  # define reward state position
penalty_states = [(3, 2)]  # define penalty state poistion

walls = [(1, 2), (2, 0), (3, 0), (3, 1), (3, 3)]  # define walls position

# The environment
class Game:
    def __init__(self, rows, cols, walls, reward_states, penalty_states, start_pos, action_success_prob):
        self.rows = rows
        self.cols = cols

        self.board = np.ndarray((rows, cols), dtype=object)
        for wall in walls:
            self.board[wall] = -1  # mark walls (-1)

        for reward_state in reward_states:
            self.board[reward_state] = 1  # mark reward state (1)

        for penalty_state in penalty_states:
            self.board[penalty_state] = 2  # mark penalty state (2)

        state_num = 1

        # replace markings with state state objects
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.board[i, j]:  # normal states are unmarked
                    self.board[i, j] = Position(state_num, -1, is_state=True)
                    state_num += 1
                elif self.board[i, j] == -1:
                    self.board[i, j] = Position(None, None, is_wall=True)
                elif self.board[i, j] == 1:
                    self.board[i, j] = Position(state_num, 10, is_state=True, is_terminal=True)
                    state_num += 1
                elif self.board[i, j] == 2:
                    self.board[i, j] = Position(state_num, -100, is_state=True, is_terminal=True)
                    state_num += 1

        self.actions = ["N", "S", "W", "E"]  # define possible actions
        self.action_success_prob = action_success_prob  # define p (personalised p from CID)

        #  give a starting position if needed (not used in value iteration)
        if start_pos:
            self.current_pos = start_pos

        #  score of the current episode (not used)
        self.score = 0

    # returns the next position for a given current position and successfully executed action)
    def get_next_position(self, from_pos, action):

        if action == "N":
            next_pos = (from_pos[0] - 1, from_pos[1])
        elif action == "S":
            next_pos = (from_pos[0] + 1, from_pos[1])
        elif action == "W":
            next_pos = (from_pos[0], from_pos[1] - 1)
        else:
            next_pos = (from_pos[0], from_pos[1] + 1)

        # check if next position is legal
        if (next_pos[0] >= 0) and (next_pos[0] <= 3):
            if (next_pos[1] >= 0) and (next_pos[1] <= 3):
                if self.board[next_pos].is_state:
                    return next_pos

        return from_pos  # returns current position if move is not legal (i.e move into a wall)

    # returns the actual executed action (non-deterministic policy execution)
    def get_next_action(self, intended_action):
        success = np.random.choice([1, 0], p=[self.action_success_prob, 1 - self.action_success_prob])
        if success:
            return intended_action
        else:
            remaining_actions = self.actions.remove(intended_action)
            return np.random.choice(remaining_actions)

    # returns the transition reward for moving into a certain next state
    def get_transition_reward(self, next_pos):
        return self.board[next_pos].reward

    # returns a transition probability for value iteration
    def get_transition_prob(self, intended_action, actual_action):
        if intended_action == actual_action:
            return self.action_success_prob
        else:
            return (1 - self.action_success_prob) / (len(self.actions) - 1)

    # show the grid world
    def show_board(self):
        for i in range(self.rows):
            print('-------------------------')
            out = '| '
            for j in range(self.cols):
                box = str(self.board[i, j])
                box = box.ljust(3, ' ')
                out += box + ' | '
            print(out)
        print('-------------------------')


# State, Wall or Terminal states
class Position:
    def __init__(self, state_num, reward, is_state=False, is_terminal=False, is_wall=False):

        self.is_state = is_state
        self.is_terminal = is_terminal
        self.is_wall = is_wall

        if self.is_terminal:
            self.name = 't' + str(state_num)

        elif self.is_state:
            self.name = 's' + str(state_num)

        self.reward = reward

    def __str__(self):
        if self.is_state:
            return self.name
        elif self.is_wall:
            return ' X '


# the agent for value iteration or other algorithms (not implemented)
class Agent:
    def __init__(self, gamma, alpha, environment):
        self.gamma = gamma  # personalised gamma
        self.alpha = alpha  # learning rate (not used)

        self.environment = environment  # load into the created grid world
        self.actions = self.environment.actions  # ["N, "S", "W", "E"]

        # initialise state values as 0
        self.state_values = {}
        for i in range(self.environment.rows):
            for j in range(self.environment.cols):
                self.state_values[(i, j)] = 0

        # initialise policy as North
        self.policy = {}
        for i in range(self.environment.rows):
            for j in range(self.environment.cols):
                self.policy[(i, j)] = 'N'

    # value iteration algorithm
    def value_iteration(self):
        tolerance = 0.01  # tolerance
        delta = 1000  # initialise difference to be a large value

        while delta > tolerance:
            delta = 0

            # for each state in the grid if it is a state and not a terminal state
            for i in range(self.environment.rows):
                for j in range(self.environment.cols):
                    if self.environment.board[(i, j)].is_state and not self.environment.board[(i, j)].is_terminal:

                        old_max_value = self.state_values[(i, j)]  # v

                        this_state_values = []  # possible value of the current state for different action

                        for ia in self.actions:  # intended action

                            sum_action_values = 0

                            for aa in self.actions:  # actual action (for different possible next states)

                                next_pos = self.environment.get_next_position((i, j), aa)  # next position
                                trans_prob = self.environment.get_transition_prob(ia, aa)  # transition probability
                                trans_reward = self.environment.get_transition_reward(next_pos)  # transition reward

                                sum_action_values += trans_prob * (trans_reward + self.gamma * self.state_values[next_pos])

                            this_state_values.append(sum_action_values)

                        self.state_values[(i, j)] = max(this_state_values)  # update state value
                        self.policy[(i, j)] = self.actions[np.argmax(this_state_values)]  # update policy

                        delta = max(delta, abs(old_max_value - self.state_values[(i, j)]))  # calculate difference

                        # self.show_values()

    # show state values
    def show_values(self):
        for i in range(self.environment.rows):
            print('-----------------------------------------')
            out = '| '
            for j in range(self.environment.cols):
                if self.environment.board[(i, j)].is_terminal:
                    box = 'T'
                elif self.environment.board[(i, j)].is_wall:
                    box = 'X'
                else:
                    box = "{:.2f}".format(self.state_values[i, j])

                box = box.ljust(7, ' ')
                out += box + ' | '
            print(out)
        print('-----------------------------------------')

    # show optimal policy
    def show_policy(self):
        for i in range(self.environment.rows):
            print('-------------------------')
            out = '| '
            for j in range(self.environment.cols):
                if self.environment.board[(i, j)].is_terminal:
                    box = 'T'
                elif self.environment.board[(i, j)].is_wall:
                    box = 'X'
                else:
                    box = self.policy[(i, j)]

                box = box.ljust(3, ' ')
                out += box + ' | '
            print(out)
        print('-------------------------')


gridworld = Game(num_rows, num_cols, walls, reward_states, penalty_states, None, 0.45)  # initiate gridworld

gridworld.show_board()  # show board

agent = Agent(0.35, None, gridworld)  # initiate the agent

agent.value_iteration()  # value iteration

agent.show_values()  # show state values corresponding to board position

agent.show_policy()  # show optimal policy corresponding to board position

