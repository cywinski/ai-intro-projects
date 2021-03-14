def alpha_beta(state, depth, max_move, alpha, beta):
    if depth == 0 or state.is_terminal():
        return state.evaluate()

    state.make_successors(max_move)

    if max_move:
        max_eval = float('-inf')
        max_state = None
        for successor in state.get_successors():
            eval_state = alpha_beta(successor, depth - 1, False, alpha, beta)
            if eval_state > max_eval:
                max_eval = eval_state
                max_state = successor
            alpha = max(alpha, eval_state)
            if beta <= alpha:
                break

        state.set_best_child(max_state)    
        return max_eval
    
    else:
        min_eval = float('inf')
        min_state = None
        for successor in state.get_successors():
            eval_state = alpha_beta(successor, depth - 1, True, alpha, beta)
            if eval_state < min_eval:
                min_eval = eval_state
                min_state = successor
            beta = min(beta, eval_state)
            if beta <= alpha:
                break
            
        state.set_best_child(min_state)
        return min_eval
