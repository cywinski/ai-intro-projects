import random

def minimax(state, depth, max_move, randomly):
    if depth == 0 or state.is_terminal():
        return state.evaluate()

    state.make_successors(max_move)

    if max_move:
        max_eval = float('-inf')
        max_state = None
        for successor in state.get_successors():
            eval_state = minimax(successor, depth - 1, False, randomly)
            if eval_state > max_eval:
                max_eval = eval_state
                max_state = successor
        state.set_best_child(max_state)
        return max_eval

    else:
        min_eval = float('inf')
        min_state = None
        for successor in state.get_successors():
            eval_state = minimax(successor, depth - 1, True, randomly)
            if eval_state < min_eval:
                min_eval = eval_state
                min_state = successor

        if not randomly:
            state.set_best_child(min_state)
        else:
            state.set_best_child(state.get_successors()[random.randint(0, len(state.get_successors()) - 1)])
        return min_eval
