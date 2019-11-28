import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random


class Pos:
    def __init__(self, r, c):
        self.row = r
        self.col = c

    def __add__(self, o):
        return Pos(self.row + o.row, self.col + o.col)

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, o):
        return self.row == o.row and self.col == o.col

    def __str__(self):
        return "({},{})".format(self.row, self.col)

    def unpack(self):
        return self.row, self.col

    def within(self, shape):
        return self.row >= 0 and self.col >= 0 and self.row < shape[0] and self.col < shape[1]

    @staticmethod
    def iter(shape):
        for r in range(shape[0]):
            for c in range(shape[1]):
                yield Pos(r, c)


class State:
    def __init__(self, player_pos, police_pos):
        self.player_pos = player_pos
        self.police_pos = police_pos

    def __hash__(self):
        return hash((self.player_pos, self.police_pos))

    def __eq__(self, o):
        return self.player_pos == o.player_pos and self.police_pos == o.police_pos

    def __str__(self):
        return "Player: {}, Police: {}".format(self.player_pos, self.police_pos)

    def is_dead(self):
        return self.player_pos == self.police_pos


class City:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    BANK_REWARD = 1
    POLICE_REWARD = -10
    IMPOSSIBLE_REWARD = -100

    def __init__(self, city):
        """ Constructor of the environment City.
        """
        self.city                     = city
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = Pos(0, 0)
        actions[self.MOVE_LEFT]  = Pos(0, -1)
        actions[self.MOVE_RIGHT] = Pos(0, 1)
        actions[self.MOVE_UP]    = Pos(-1, 0)
        actions[self.MOVE_DOWN]  = Pos(1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        s = 0
        for player_pos in Pos.iter(self.city.shape):
            if self.city[player_pos.unpack()] == 1:
                continue

            for police_pos in Pos.iter(self.city.shape):
                new_state = State(player_pos, police_pos)

                states[s] = new_state
                map[new_state] = s

                s += 1
        return states, map

    def get_action(self, s, eps, Q):
        """
        selects an action from the possible actions at state s.
        :param s: current state
        :param eps: exploration rate.
        :param Q: current estimation of rewards (table of dimension S*A)
        :return: selected action a
        """
        rand = np.random.uniform()
        if rand < eps:
            # select an action for exploration.
            # list() because we want to select the keys ( "STAY" instead of Pos obj)
            return random.choice(list(self.actions))
        else:
            # select an action for exploitation
            # TODO write exploitation = pick optimal action using reward estimate table Q
            # this is not needed for a with Q-learning but it is needed for eps-greedy SARSA
            pass

    def __moves(self, state, action):
        """ Makes a step in the city, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return next state index and corresponding transition prob.
        """
        # Compute the future position given current (state, action)
        new_player_pos = state.player_pos + self.actions[action]
        # Is the future position an impossible one ?
        agent_hitting_city_walls = not new_player_pos.within(self.city.shape) or \
                                   self.city[new_player_pos.unpack()] == 1

        # If we can't move, we stay
        if agent_hitting_city_walls:
            new_player_pos = state.player_pos

        police_actions = [Pos(0, -1), Pos(0, 1), Pos(-1, 0), Pos(1, 0)]

        next_states = []
        # keep on picking a new move as long as the chosen move is impossible
        for police_action in police_actions:
            new_police_pos = state.police_pos + police_action
            if new_police_pos.within(self.city.shape):
                next_s = State(new_player_pos, new_police_pos)
                next_states.append(next_s)

        return next_states

    def move(self, state, action):
        reward = 0 # standard reward
        next_s = random.choice(self.__moves(state, action))

        if next_s.player_pos == next_s.police_pos:
            reward += self.POLICE_REWARD
        if self.city[next_s.player_pos.unpack()] == 1:
            reward += self.BANK_REWARD
        agent_stayed = state.player_pos == next_s.player_pos
        if agent_stayed and action != self.STAY:
            reward += self.IMPOSSIBLE_REWARD
        return next_s, reward

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)


def draw_city(city):

    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = city.shape
    colored_city = [[col_map[city[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The City')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = city.shape
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_city,
                            cellLoc='center',
                            loc=(0, 0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def q_learning(env, lambd, eps, player_start, police_start, n_iter):
    """ Learns the Q using Q learning
        :input City env           : The city environment in which we want to learn the Q
        :input float lambd        : the discount factor (survival prob) lambda
        :input float eps          : the exploration rate epsilon
        :input tuple player_start : start coordinates of player (player_x, player_y)
        :input tuple police_start : start coordinates of police (police_x, police_y)
        :input int n_iter         : the number of iterations to run the Q-learning
        :return numpy.array Q     : the learned Q table, dimension S*A
    """
    # The Q learning algorithm requires the knowledge of :
    # - State space
    # - Action space
    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions))
    s = State(Pos(player_start[0], player_start[1]), Pos(police_start[0], police_start[1]))

    for t in range(n_iter):
        a = env.get_action(s, eps, Q)
        s_next, curr_reward = env.move(s, a)
        s_next_index = env.map[s_next]
        s_index = env.map[s]
        lr = compute_lr(t)
        best_action = np.max(Q[s_next_index, :])
        Q[s_index, a] = (1-lr)*Q[s_index, a] + lr*(curr_reward + lambd*best_action)
        s = s_next
    return Q


def sarsa(env, lambd, eps, player_start, police_start, n_iter):
    """ Learns the Q using the SARSA algorithm
        :input City env           : The city environment in which we want to learn the Q
        :input float lambd        : the discount factor (=survival prob) lambda
        :input float eps          : the exploration rate epsilon
        :input tuple player_start : start coordinates of player (player_x, player_y)
        :input tuple police_start : start coordinates of police (police_x, police_y)
        :input int n_iter         : the number of iterations to run the Q-learning
        :return numpy.array Q     : the learned Q table, dimension S*A
    """
    # The SARSA algorithm requires the knowledge of :
    # - State space
    # - Action space
    n_states = env.n_states
    n_actions = env.n_actions

    Q = np.zeros((n_states, n_actions))
    s = State(Pos(player_start[0], player_start[1]), Pos(police_start[0], police_start[1]))
    a = env.get_action(s, eps, Q)

    for t in range(n_iter):
        s_next, curr_reward = env.move(s, a)
        a_next = env.get_action(s_next, eps, Q)
        lr = compute_lr(t)
        s_index = env.map[s]
        s_next_index = env.map[s_next]
        Q[s_index, a] = (1-lr)*Q[s_index, a] + \
                            lr*(curr_reward + lambd*Q[s_next_index, a_next])
        s = s_next
        a = a_next
    return Q


def compute_lr(t):
    """
    A learning rate scheduler that guarantees convergence
    since sum_t(lr_t) = inf & sum_t(lr_t^2) < inf
    :param t: the current iteration of the Q-learning / SARSA algorithm
    :return: the current learning rate
    """
    # start at t+1 to avoid division by 0
    return 1/((t+1)**(2/3))
