states = {0, 1, 2}  # observed states

trace = [0, 0, 1, 0, 0, 2, 2]  # personalised trace
reward = [0, 0, 0, 0, 0, 1, 1]  # personalised reward


# temporal difference algorithm
def td_estimation(t, r, gamma, alpha):

    V = {0: 0, 1: 0, 2: 2}

    for _ in range(1):
        for step in range(len(t) - 1):
            delta = r[step] + gamma * V[t[step + 1]] - V[t[step]]
            V[t[step]] += alpha * delta

    return V


print(td_estimation(trace, reward, 1, 0.5))
