import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
import math

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

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
    CAUGHT_REWARD = -10
    REST_REWARD = -100

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
            pass

    def __moves(self, state, action):
        """ Makes a step in the city, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return next state index and corresponding transition prob.
        """
        # Compute the future position given current (state, action)
        print("state:" , state)
        print("state.player_pos:", state.player_pos)
        print("action: ", action)
        print("self.actions[action]: ", self.actions[action])
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
        return random.choice(self.__moves(state, action))

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0;
            s = start;
            # Add the starting position in the city to the path
            path.append(start);
            while t < horizon:
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[self.map[s], t])
                # Add the position in the city corresponding to the next state
                # to the path
                path.append(next_s)
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 0
            s = start

            # Loop while state is not the goal state
            while True:
                path.append(s)

                # if our life is geometrically distributed, check whether we're still alive
                if survival_factor is not None:
                    if random.random() > survival_factor:
                        break

                if s.is_dead():
                    break

                if self.city[s.player_pos.unpack()] == 2:
                    break

                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[self.map[s]])

                # Update time and state for next iteration
                t += 1

                # Update state
                s = next_s

        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)

    def animate_solution(self, path, policy = None):

        # Map a color to each cell in the city
        col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

        # Size of the city
        rows, cols = self.city.shape

        # Create figure of the size of the city
        fig = plt.figure(1, figsize=(cols,rows))

        # Remove the axis ticks and add title title
        ax = plt.gca()
        ax.set_title('Policy simulation')
        ax.set_xticks([])
        ax.set_yticks([])

        # Give a color to each cell
        colored_city = [[col_map[self.city[j,i]] for i in range(cols)] for j in range(rows)]

        # Create figure of the size of the city
        fig = plt.figure(1, figsize=(cols, rows))

        # Create a table to color
        grid = plt.table(cellText=None,
                         cellColours=colored_city,
                         cellLoc='center',
                         loc=(0,0),
                         edges='closed')

        # Modify the height and width of the cells in the table
        tc = grid.properties()['child_artists']
        for cell in tc:
            cell.set_height(1.0/rows)
            cell.set_width(1.0/cols)

        # Update the color at each frame
        out = False

        prev_player_tuple, prev_police_tuple = None, None

        def draw_action_in_cell(pos, action):
            if action == self.STAY:
                return None

            cell = grid.get_celld()[pos]

            arrow_size_x = cell.get_width()*0.33
            arrow_size_y = cell.get_width()*0.33

            cell_mid_x = cell.get_x() + 0.5*cell.get_width()
            cell_mid_y = cell.get_y() + 0.5*cell.get_height()

            dirs = dict()
            dirs[self.MOVE_DOWN] = (0, -arrow_size_y)
            dirs[self.MOVE_UP] = (0, arrow_size_y)
            dirs[self.MOVE_RIGHT] = (arrow_size_x, 0)
            dirs[self.MOVE_LEFT] = (-arrow_size_x, 0)

            dx, dy = dirs[action]

            return plt.arrow(cell_mid_x - dx/2, cell_mid_y - dy/2, dx, dy, width = 0.005)

        arrows = []

        for t, s in enumerate(path):
            player_tuple = s.player_pos.unpack()
            police_tuple = s.police_pos.unpack()

            if prev_player_tuple:
                if not out:
                    # set previous player position back to white
                    grid.get_celld()[prev_player_tuple].set_facecolor(col_map[self.city[prev_player_tuple]])
                    grid.get_celld()[prev_player_tuple].get_text().set_text('')
                if self.city[player_tuple] == 2:
                    grid.get_celld()[player_tuple].set_facecolor(LIGHT_GREEN)
                    grid.get_celld()[player_tuple].get_text().set_text('Player is out')
                    out = True

                # set previous police position back to white
                grid.get_celld()[prev_police_tuple].set_facecolor(col_map[self.city[prev_police_tuple]])
                grid.get_celld()[prev_police_tuple].get_text().set_text('')

            grid.get_celld()[player_tuple].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[player_tuple].get_text().set_text('Player')
            grid.get_celld()[police_tuple].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[police_tuple].get_text().set_text('Police')

            prev_player_tuple, prev_police_tuple = player_tuple, police_tuple

            if policy is not None and (policy.ndim == 1 or t < policy.shape[1]):
                for a in arrows:
                    if a is not None:
                        a.remove()
                arrows = []

                for draw_pos in Pos.iter(self.city.shape):
                    if draw_pos == s.police_pos or self.city[draw_pos.unpack()] != 0:
                        continue

                    draw_state = State(draw_pos, s.police_pos)
                    if policy.ndim == 1:
                        draw_action = policy[self.map[draw_state]]
                    else:
                        draw_action = policy[self.map[draw_state], t]

                    arrow = draw_action_in_cell(draw_pos.unpack(), draw_action)
                    if arrow:
                        arrows.append(arrow)

            # This animation only works in ipython notebook
            display.display(fig)
            display.clear_output(wait=True)
            time.sleep(1)


def draw_city(city):

    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols    = city.shape
    colored_city = [[col_map[city[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The City')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols    = city.shape;
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_city,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def q_learning(env, lambd, eps, n_iter):
    """ Learns the Q using Q learning
        :input City env           : The city environment in which we want to learn the Q
        :input float lambd        : the discount factor (survival prob) lambda
        :input float eps          : the exploration rate epsilon
        :input int n_iter         : the number of iterations to run the Q-learning
        :return numpy.array Q     : the learned Q table, dimension S*A
    """
    # The Q learning algorithm requires the knowledge of :
    # - State space
    # - Action space
    n_states = env.n_states
    n_actions = env.n_actions

    Q = np.zeros((n_states, n_actions))

    #TODO change this magic number
    #State((0,0),(3,3))
    #s = map[state_obj]
    s = env.states[0]
    #s = env.start_state
    for t in range(n_iter):
        a = env.get_action(s, eps, Q)
        s_next, curr_reward = env.move(s, a)
        lr = compute_lr(t)
        Q[s, a] = (1-lr)*Q[s, a] + lr*(curr_reward + lambd*np.max(Q[s_next, :]))
        s = s_next
    return Q


def sarsa(env, lambd, eps, n_iter):
    """ Learns the Q using the SARSA algorithm
        :input City env           : The city environment in which we want to learn the Q
        :input float lambd        : the discount factor (=survival prob) lambda
        :input float eps          : the exploration rate epsilon
        :input int n_iter         : the number of iterations to run the Q-learning
        :return numpy.array Q     : the learned Q table, dimension S*A
    """
    # The SARSA algorithm requires the knowledge of :
    # - State space
    # - Action space
    n_states = env.n_states
    n_actions = env.n_actions

    Q = np.zeros((n_states, n_actions))
    s = env.start_state
    a = env.get_action(s, eps, Q)

    for t in range(n_iter):
        s_next, curr_reward = env.move(s, a)
        a_next = env.get_action(s_next, eps, Q)
        curr_lr = compute_lr(t)
        Q[s, a] = (1-curr_lr)*Q[s, a] + curr_lr*(curr_reward + lambd*Q[s_next, a_next])
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
    return 1/(t**(2/3))
