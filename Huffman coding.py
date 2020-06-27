#################### Huffman coding #####################

class Node(object):  
    def __init__(self, char, freq):
        self.left = None
        self.right = None
        self.parent = None
        self.char = char
        self.freq = freq       
        
    @property
    def is_root(self):
    	return self.parent == None

    @property
    def is_left(self):
    	if not self.is_root:
        	return self.parent.left == self
    	else:
    		return False

    @property
    def is_leaf(self):
        return (self.left == None) and (self.right == None)



class Huffman(object):
	def __init__(self, freq_dict):
		self.freq_dict = freq_dict
		self.leaves = [Node(char, freq) for char, freq in self.freq_dict.items()]
		self.Huffman_tree = None


	def _build_Huffman_tree(self):   
	    queue = self.leaves.copy()
	    Huffman_tree = []
	    
	    while len(queue) > 1:
	        queue.sort(key = lambda node: node.freq)

	        left = queue.pop(0)         
	        right = queue.pop(0)          
	        
	        parent = Node(left.char + right.char, left.freq + right.freq)   
	        parent.left = left                     
	        parent.right = right                    
	        
	        left.parent = parent         
	        right.parent = parent        
	        
	        queue.append(parent)

	        Huffman_tree.append(parent)   

	    Huffman_tree.extend(self.leaves)
	    self.Huffman_tree = Huffman_tree

	    return self.Huffman_tree 


	def get_Huffman_code(self):

		self._build_Huffman_tree()

		nodes_leaf = [node for node in self.Huffman_tree if node.is_leaf]
		
		Huffman_codes = {}

		for leaf in nodes_leaf:
			current_node = leaf	
			code = ''
			while not current_node.is_root:
				if current_node.is_left:
					code = '0' + code 
				else: 
					code = '1' + code
				current_node = current_node.parent
			Huffman_codes[leaf.char] = code	

		return Huffman_codes


	def encoding(self, text):
		
		Huffman_codes = self.get_Huffman_code()
		Huffman_text = ''
		
		for char in text:
			Huffman_text = Huffman_text + Huffman_codes[char]

		return Huffman_text		

	
	def decoding(self, Huffman_text):
		self._build_Huffman_tree()

		text = ''
		root = self.Huffman_tree[np.argmax([t.is_root for t in self.Huffman_tree])]
		current_node = root
		
		for code in Huffman_text:
			code = int(code)
			
			if code == 0:
				current_node = current_node.left
			elif code == 1:
				current_node = current_node.right

			if current_node.is_leaf:
				text = text + current_node.char
				current_node = root

		return text