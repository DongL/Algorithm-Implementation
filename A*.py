###########################################
# A* search 
# Author: Dong Liang
# March 26, 2022
###########################################

def solve(cost_function):
    # Initialization
    fringe = PriorityQueue()
    explored = set()
    path_recorder = {}
    costs = {start_city: 0}
    fringe.put((0, start))

    while not fringe.empty():
        _, current_city = fringe.get()
        explored.add(current)

        if is_goal(current_city):
            path = get_path(path_recorder)
            print(len({loc.split(',')[1] for loc in path[4:]}), '===')
            if len({loc.split(',')[1] for loc in path[4:]}) > 30:
                return ' '.join(get_path(path_recorder))

        for next in succ(current):
            new_cost = costs[current] + \
                cost(current, next)
            if next not in explored:
                if next not in costs or new_cost < costs[next]:
                    costs[next] = new_cost
                    priority = new_cost + heuristic(next)
                    fringe.put((priority, next))
                    path_recorder[next] = current
    return False
