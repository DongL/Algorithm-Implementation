#################### Undirected bipartite graph #####################

class Node(object):
    def __init__(self, key):
        self.key = key
        self.neighbors = set()
        self.belongs_to = None

class Adjacency_List(object):
    
    def __init__(self, in_path):
        self.in_path = in_path
        self.existing_nodes = dict()
        self.sets = ["Set1", "Set2"]
        self.adj_list = set()
        self._build_an_adjacency_list_from_file(self.in_path)

    
        
    def _build_individual_node(self, key):
        
        if key not in self.existing_nodes.keys():  # 唯一性
            node = Node(key)
            self.existing_nodes[key] = node
        else:
            node = self.existing_nodes[key]

        return node
        
    
    # Build an adjacency list from file
    def _build_an_adjacency_list_from_file(self, in_path):
        with open(in_path, 'r') as file:
            lines = file.readlines()
            n = int(lines[0].strip())
            
            
            for line in lines[1:]:
                # Construct nodes from file
                key1 = int(line[1].strip())
                node1 = self._build_individual_node(key1)
                key2 = int(line[3].strip())
                node2 = self._build_individual_node(key2)  
                
                # Add edges
                node1.neighbors.add(node2)
                node2.neighbors.add(node1)

                # Add nodes to adj list
                self.adj_list.add(node1)   
                self.adj_list.add(node2)
                    
        return self.adj_list
    
    @property
    def graph(self):
        return [(node.key, [nbs.key for nbs in node.neighbors]) for node in self.adj_list]

    
    def is_bipartite(self):
        first_node = list(self.adj_list)[0]
        queue = [first_node]
        visited = set()
        
        while queue:
            node = queue.pop(0)
            if node.belongs_to == None:
                node.belongs_to = self.sets[0]
                
            for node_nbs in node.neighbors:
                if node_nbs.belongs_to == None:
                    node_nbs.belongs_to = [s for s in self.sets if s != node.belongs_to][0]
                elif node_nbs.belongs_to == node.belongs_to:
                    return "No, it's not bipartite."
                
            visited.add(node)  
            nodes = node.neighbors - visited
            queue.extend(nodes)    
                
        return ([v.key for v in self.adj_list if v.belongs_to == self.sets[0]], 
                [v.key for v in self.adj_list if v.belongs_to == self.sets[1]])