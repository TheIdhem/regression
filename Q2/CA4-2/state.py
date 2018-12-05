
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]
class State:

    def __init__(self, grid, pos):
        self.grid = grid
        self.pos = pos

    def __eq__(self, other):
        return isinstance(other, State) and self.grid == other.grid and self.pos == other.pos

    def __hash__(self):
        return hash(str(self.grid) + str(self.pos))

    def __str__(self):
        return 'State(grid={:d}, pos={:d})'.format(self.grid, self.pos)