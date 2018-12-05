# depth-limited dfs
from state import State
from copy import deepcopy
import numpy as np
import random


N_STATES = 4
N_EPISODES = 20

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

dict_convert = {
    "top": UP,
    "under": DOWN,
    "left": LEFT,
    "right": RIGHT
}

MAX_EPISODE_STEPS = 200

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2

q_table = dict()

def dfs_solver(maze):
    visited = {}
    cellList = []
    neighbors = []
    cellList.append(maze.start)
    parent = {}
    while cellList:
        thisCell = cellList.pop(-1)
        visited[thisCell]=True
        if (thisCell == maze.goal):
            break
        state = State(grid=maze.cells, pos=[thisCell[0], thisCell[1]])
        neighbors = maze.get_neighbors(thisCell)
        neighbors = sorted(neighbors, key=lambda x: q(state)[dict_convert[x[1]]])
        for x in neighbors:
            if (not visited.setdefault(x[0],False)):
                cellList.append(x[0])
                parent[x[0]] = thisCell
    result = []
    current = maze.goal
    result.append(maze.start)
    while (parent[current] != maze.start):
        current = parent[current]
        result.insert(0,current)
    result.append(maze.goal)
    return result


def find_best_way(maze,start,goal, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    actions = []
    maximom = -999999
    action = 0
    # print start
    state = State(grid=maze.cells, pos=[start[0], start[1]])
    neighbors = maze.get_neighbors(start)
    for i in neighbors:
        if i[1] == "top":
            actions.append(0)
        elif i[1] == "under":
            actions.append(1)
        elif i[1] == "right":
            actions.append(3)
        elif i[1] == "left":
            actions.append(2)
    for i in actions:
        queue_state = q(state)[i]
        if queue_state > maximom:
            maximom = queue_state
            action = i
    next_state, reward, done = act(state, action, maze, visited)

    if done:
        visited.append(goal)
    else:
        find_best_way(maze, (next_state.pos[0], next_state.pos[1]), goal, visited)
    return visited

def Q_Learning(maze):
    max_reward = 0
    max_reward=-9999999
    min_visible=[]
    start_state = State(grid=maze.cells, pos=[maze.start[0], maze.start[1]])
    for e in range(N_EPISODES):

        state = start_state
        total_reward = 0
        alpha = alphas[e]
        visited = []
        visible = []
        visible.append(maze.start)
        for _ in range(MAX_EPISODE_STEPS):
            actions=[]
            neighbors = maze.get_neighbors((state.pos[0], state.pos[1]))
            for i in neighbors:
                if i[1] == "top":
                    actions.append(0)
                elif i[1] == "under":
                    actions.append(1)
                elif i[1] == "right":
                    actions.append(3)
                elif i[1] == "left":
                    actions.append(2)
            action = choose_action(state, maze,actions)
            next_state, reward, is_done = act(state, action, maze, visited)
            visited.append((next_state.pos[0], next_state.pos[1]))
            total_reward += reward

            q(state)[action] = q(state, action) + \
                               alpha * (reward + gamma * np.max(q(next_state)) - q(state, action))
            state = next_state
            visible.append((state.pos[0],state.pos[1]))
            if is_done:
                if total_reward > max_reward:
                    max_reward = total_reward
                    visible.append(maze.goal)
                    min_visible = visible
                break

        print('Episode={:d}, total reward={:d}'.format(e + 1, total_reward))

    result = dfs_solver(maze)
    return result


def act(state, action, maze, visited):
    p = new_pos(state, action, maze, visited)
    grid_item = (p[0], p[1])
    is_done = False
    new_grid = deepcopy(state.grid)
    if maze.check_wall((state.pos[0], state.pos[1]), grid_item)[0] != 0:
        reward = -100
    elif grid_item == maze.goal:
        # print "gav"
        reward = 10000
        is_done = True
    elif maze.check_wall((state.pos[0], state.pos[1]), grid_item)[0] == 0:
        # print "yaboo"
        reward = -1
        is_done = False
    else:
        raise ValueError('Unknown grid item={:d}'.format(grid_item))

    return State(grid=new_grid, pos=p), reward, is_done


def new_pos(state, action, maze, visited, max_step=0):
    p = deepcopy(state.pos)
    if action == UP:
        p[0] = max(0, p[0] - 1)
    elif action == DOWN:
        p[0] = min(len(state.grid) - 1, p[0] + 1)
    elif action == LEFT:
        p[1] = max(0, p[1] - 1)
    elif action == RIGHT:
        p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
    else:
        raise ValueError('Unknown action={:d}'.format(action))
    action_temp = []
    for i in range(len(visited)):
        if (p[0], p[1]) == visited[i] and i > len(visited) - 5 and max_step < len(visited):
            for i in ACTIONS:
                if i != action:
                    action_temp.append(i)
            max_step = max_step + 1
            p = new_pos(state, choose_action(state, maze, action_temp), maze, visited, max_step)
    return p


def choose_action(state, maze, action=None):
    if random.uniform(0, 1) < eps:
        if action == None:
            return random.choice(ACTIONS)
        else:
            return random.choice(action)
    else:
        return np.argmax(q(state))


def q(state, action=None):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))

    if action is None:

        return q_table[state]
    return q_table[state][action]