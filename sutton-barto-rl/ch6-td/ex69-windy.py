import numpy as np
import copy
import matplotlib.pyplot as plt


class WindyGrid():
    def __init__(self, start_state=(0, 3), goal_state=(7, 3)):
        self.start_state = start_state
        self.goal_state = goal_state
        self.mins = [0, 0]
        self.maxs = [9, 6]

        self.col_winds = [0, 0, 0, 1, 1, 1, 2, 2, 1 ,0]

        self.reset()

    def step(self, a):

        # add wind
        self.state[1] -= self.col_winds[self.state[0]]

        if a == 'u':
            self.state[1] -= 1
        elif a == 'ur':
            self.state[1] -= 1
            self.state[0] += 1
        elif a == 'r':
            self.state[0] += 1
        elif a == 'dr':
            self.state[0] += 1
            self.state[1] += 1
        elif a == 'd':
            self.state[1] += 1
        elif a == 'dl':
            self.state[1] += 1
            self.state[0] -= 1
        elif a == 'l':
            self.state[0] -= 1
        elif a == 'ul':
            self.state[0] -= 1
            self.state[1] -= 1

        # edges of grid
        self.state = list(np.clip(self.state, self.mins, self.maxs))

        # check if in goal
        if self.state == list(self.goal_state):
            return self.state, 1


        # if (a == 'u' and self.state[1] == self.y_lim[0]) or (a == 'd' and self.state[1] == self.y_lim[1]) \
        #         or (a == 'l' and self.state[0] == self.x_lim[0]) or (a == 'r' and self.state[0] == self.x_lim[1]):
        #     return self.state, -1

        return self.state, -1


    def reset(self):
        self.state = list(self.start_state)
        return self.state

def get_eps_greedy_a(eps, q):
    # q is an array of q values for all disc actions for particular state
    num_actions = len(q)
    greedy_a = np.argmax(q)
    p = np.ones(num_actions) * eps / num_actions
    p[greedy_a] = 1 - eps + eps / num_actions

    return np.random.choice(num_actions, p=p)


actions = ['u', 'd', 'l', 'r']
# actions = ['u', 'ur', 'r', 'dr', 'd', 'dl', 'l', 'ul']
q = np.zeros([10, 7, len(actions)])
eps = 0.1
alpha = 0.5
eps_completed = 0
num_timesteps = 0

env = WindyGrid()

while eps_completed < 200:
    s = env.reset()
    a_i = get_eps_greedy_a(eps, q[s[0], s[1], :])
    ep_done = False
    while not ep_done:
        ns, rew = env.step(actions[a_i])

        if ns[0] < 0 or ns[1] < 0:
            import ipdb; ipdb.set_trace()

        na_i = get_eps_greedy_a(eps, q[ns[0], ns[1], :])
        q[s[0], s[1], a_i] = q[s[0], s[1], a_i] + alpha * (rew + q[ns[0], ns[1], na_i] - q[s[0], s[1], a_i])
        s = copy.deepcopy(ns)
        a_i = copy.deepcopy(na_i)

        if rew == 1:
            ep_done = True
        num_timesteps += 1
    eps_completed +=1
    print(num_timesteps)
    print(eps_completed)

# see e-greedy policy
s = env.reset()
ep_done = False
greedy_actions = []
while not ep_done:
    a_i = np.argmax(q[s[0], s[1], :])
    s, rew = env.step(actions[a_i])
    greedy_actions.append(actions[a_i])
    if rew == 1:
        ep_done = True

print(greedy_actions)
