###########################################
# Breadth first search
# Author: Dong Liang
# March 26, 2022
###########################################

def BFS(array, node)
  queue = [node]
  while len(queue) > 0:
    current = queue.pop(0)
    array += [node.name]
    queue += node.children
  return array
