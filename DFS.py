###########################################
# Depth first search
# Author: Dong Liang
# March 26, 2022
###########################################

def DFS(array, node)
  queue = [node]
  while len(queue) > 0:
    current = queue.pop(-1)
    array += [node.name]
    queue += node.children
  return array
