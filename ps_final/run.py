import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

prob = 0.5



def defineGrid():
    """
    grid: 8x1 grid world
          reward in each cell is -1
          home has 0 reward

    returns: array [0,-1,-1,-1,-1,-1,-1,-1]
    """
    grid = -1 * np.ones(8)
    grid[0] = 0

    return grid


def defineActions():
    """
    actions: up (index +1)
             down (index -1)

    returns: array [up, down]
    """
    actions = np.array([-1, 1])

    return actions


def bellman(state, action, pi):
    """
    args: state
    return: reward
    """
    grid = defineGrid()
    actions = defineActions()

    if state == 0:
        return grid[state]
    elif state == 7:
        return pi[state] + 2*grid[state]

    new_state = state + actions[action]

    reward = grid[state]
    expected_reward = pi[new_state]

    return reward + expected_reward


def main():

    grid = defineGrid()
    actions = defineActions()
    v_pi = np.zeros([len(grid),len(actions)])
    pi = np.zeros(len(v_pi))
    home = 0
    prob = 1

    # for i in range(3):
    # for s in range(len(grid)):
    #     for a in range(len(actions)):
    #         v_pi[s,a] = bellman(s, a, pi)
    #     pi[s] = np.sum(v_pi[s])
    #     print np.sum(pi)
    # print
    # print pi

    # for i in range(3):
        for a in range(len(actions)):
            for s in range(len(grid)):
                v_pi[s,a] = bellman(s, a, pi)
            pi[a] = np.sum(v_pi[a])

    # print pi

if __name__ == '__main__':
    main()
