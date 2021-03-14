from minimax.state import State

"""
Class representing Game Tree of Wolf and Sheep game

:param wolf_init: initial position of Wolf
:type wolf_init: list
:param root: root of the Game Tree
:type root: State
:param current_state: current state of Game Tree used in traversing
:type current_state: State
"""

class Game_tree():

    def __init__(self, wolf_init) -> None:
        self.wolf_init = wolf_init
        self.root = State(self.wolf_init, [[0, 1], [0, 3], [0, 5], [0, 7]], None)
        self.current_state = self.root

    def visualize(self):
        current_state = self.root
        while current_state is not None:
            print(current_state)
            print("\n")
            current_state = current_state.best_child