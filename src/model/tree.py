class TreeNode:
    def __init__(self, value):
        
        self.value, self.children, self.parent = value, [], None
        self.time, self.flops, self.params = None, None, None
        self.alloc_mem, self.rsv_mem = None, None
        self.nbytes, self.energy = None, None

    def total_params(self):
        if(self.params is None): 
            self.params = 0
            for c in self.children:
                self.params += c.total_params()
        return self.params
    
    def total_nbytes(self):
        if(self.nbytes is None): 
            self.nbytes = 0
            for c in self.children:
                self.nbytes += c.total_nbytes()
        return self.nbytes

    def total_flops(self):
        if(self.flops is None): 
            self.flops = 0
            for c in self.children:
                self.flops += c.total_flops()
        return self.flops
    
    def max_alloc_mem(self):
        if(self.alloc_mem is None): 
            self.alloc_mem = 0
            for c in self.children:
                if(c.max_alloc_mem() > self.alloc_mem): self.alloc_mem = c.max_alloc_mem()                    
        return self.alloc_mem
    
    def max_rsv_mem(self):
        if(self.rsv_mem is None): 
            self.rsv_mem = 0
            for c in self.children:
                if(c.max_rsv_mem() > self.rsv_mem): self.rsv_mem = c.max_rsv_mem()                    
        return self.rsv_mem
    
    def total_time(self):
        if(self.time is None): 
            self.time = 0
            for c in self.children:
                self.time += c.total_time()
        return self.time


    @staticmethod
    def add_tree_nodes(root, module, module_index, parent=None, count=0):
        
        root.parent = parent
        if(len(list(module.children())) == 0):
            if(count in module_index):
                # print( module_index[count])
                root.flops, root.params, root.alloc_mem, root.rsv_mem, root.time = module_index[count][1:]
            else:
                # unable to analyze this leaf module
                root.alloc_mem, root.time, root.rsv_mem = 0,0,0
                root.flops, root.params, root.nbytes = 0,0,0

        for m in module.children():
            new_node = TreeNode(m._get_name())
            root.children.append(new_node)
            count = TreeNode.add_tree_nodes(new_node, m, module_index
                                            ,parent=root, count= count + 1)

        return count

    @staticmethod
    def print_tree(node, prefix="", level=3, last_child=True):
    
        space_filler = [" " for i in range (21 - len(node.value))]
        str_to_print = prefix + ("└── " if last_child else "├── ") + f"{node.value}" + "".join(space_filler)
        
        if(level ==0 or len(node.children) == 0):
            max_alloc_mem = format(node.max_alloc_mem()/1024/1024, '.2f')
            node_rsv_mem = format(node.max_rsv_mem()/1024/1024, '.2f')
            total_time = format(node.total_time(), '.3f')
            total_flops = format(node.total_flops()/1000/1000/1000, '.3f')
            # total_energy = format(node.total_energy())
            str_to_print = f"{str_to_print}\t[ {max_alloc_mem} MB\t| {node_rsv_mem} MB\t| {total_time} ms\t| {total_flops} G\t]"
            print(str_to_print)
            return
        print(str_to_print)
        
        for i, child in enumerate(node.children):
            is_last = i == len(node.children) - 1
            new_prefix = prefix + ("    " if last_child else "│   ") if not is_last else prefix + "    "
            TreeNode.print_tree(child, new_prefix, level-1, is_last)