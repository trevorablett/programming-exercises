import numpy as np
import matplotlib.pyplot as plt
import itertools


# params
theta = .000001
ph = .75
discount = 0.99

# env info
states = range(1, 100)
values = np.zeros([101])
values[0] = 0
values[100] = 0
values[1:100] = np.random.uniform(0, 1, 99)


def transition_prob(s_new, r, s, a):
    if s_new == s:
        if a == 0:
            if s_new >= 100:
                if r == 1:
                    return 1
                else:
                    return 0
            else:
                if r == 1:
                    return 0
                else:
                    return 1
        else:
            return 0
    elif s_new == s + a:
        if s_new >= 100:
            if r == 1:
                return ph
            else:
                return 0
        else:
            if r == 1:
                return 0
            else:
                return ph
    elif s_new == s - a:
        if r == 1:
            return 0
        else:
            return 1 - ph

# value func
def get_value_and_best_action(s, cur_values):
    actions = range(0, np.min([s, 100 - s]) + 1)
    max_val = 0
    best_action = 0
    for a in actions:
        sum = 0
        if a == 0:
            s_primes = [s]
        else:
            s_primes = [s + a, s - a]
        rews = [1, 0]
        for s_p, r in itertools.product(s_primes, rews):
            sum += transition_prob(s_p, r, s, a) * (r + discount * cur_values[s_p])

        if sum > max_val:
            max_val = sum
            best_action = a

    return max_val, best_action

# value iteration and getting best action
delta = 1e10
pi_star = np.zeros([100])
while delta >= theta:
    delta = 0
    for s in states:
        old_v = values[s]
        values[s], pi_star[s] = get_value_and_best_action(s, values)
        delta = np.max([delta, np.abs(old_v - values[s])])

# plot value estimates and policy
plt.figure()
plt.plot(range(0, 101), values)

plt.figure()
plt.plot(range(0, 100), pi_star)

plt.show()