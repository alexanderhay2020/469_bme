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


# def bellman(state, action, pi):
#     """
#     args: state
#     return: reward
#     """
#     grid = defineGrid()
#     actions = defineActions()
#
#     if state == 0:
#         return grid[state] + pi[0]
#     elif state == 7:
#         return grid[state] + pi[7]
#
#     new_state = state + actions[action]
#
#     reward = grid[state]
#     expected_reward = grid[new_state]
#
#     return reward + expected_reward


def bellman(state, a, v_pi):

	grid = defineGrid()
	actions = defineActions()

	new_state = state + actions[a]

	if state == 0:
        # new_state = state
	    return v_pi[0]
	elif state == 7:
	    new_state = state
	    return v_pi[7]
	#
	#
	reward = grid[state]
	expected_reward = bellman(new_state, a, v_pi)
	# expected_reward = grid[new_state]
	# expected_reward = v_pi[new_state]

	pi = reward + expected_reward

	return pi;



def main():

    grid = defineGrid()
    actions = defineActions()
    pi = np.zeros([len(grid),len(actions)])
    v_pi = np.zeros(len(pi))
    v_old = np.zeros(len(pi))
    home = 0
    prob = 1

    for i in range(1):
        for s in range(len(grid)):
            for a in range(len(actions)):
                pi[s,a] = np.sum((prob*grid[s]) + bellman(s, a, pi))
            v_pi[s] = np.sum(pi[s])

    # for i in range(1):
    #     for s in range(len(grid)):
    #         for a in range(len(actions)):
    #             pi[s,a] = bellman(grid, s, actions, a, v_old)
    #         v_pi[s] = np.sum(pi[s])

    print v_pi

if __name__ == '__main__':
    main()
