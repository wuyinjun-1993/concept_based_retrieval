import torch
class TreeNode:
    def __init__(self, node_id, value, node_type, tgt_count):
        self.node_id = node_id
        self.node_type = node_type
        self.subquery_string = value
        self.tgt_count = tgt_count
        if tgt_count is None:
            self.tgt_count = 1
        self.children = []


    def embed_sub_query(self, model, model_name, processor, device):
        inputs = processor(self.subquery_string)
        if model_name == "default":
            inputs = {key: val.to(device) for key, val in inputs.items()}
            text_features = model.get_text_features(**inputs)
        elif model_name == "blip":
            text_features = model.extract_features({"text_input":inputs}, mode="text").text_embeds_proj[:,0,:].view(1,-1)
        else:
            raise ValueError("Invalid model name")
          
        self.text_features = text_features
    
class Tree:
    def __init__(self, root):
        self.root = root
        
    def check_only_node(self):
        if len(self.root.children) == 1 and len(self.root.children[0].children) <= 0:
            return True
        return False

    def add_child(self, parent, child):
        parent.children.append(child)

    def display(self, node, level=0):
        print(' ' * level * 2 + f"ID: {node.node_id}, Substring: {node.subquery_string}, Type: {node.node_type}, tgt_count: {node.tgt_count}")
        for child in node.children:
            self.display(child, level + 1)
    # def (self):
    def embed_sub_queries(self, model, model_name, processor, device):
      def dfs(node):
          if not node:
              return
          # Append the current node's value to the path
          # path.append(node.value)
          node.embed_sub_query(model, model_name, processor, device)
          
          # If the node has no children, it's a leaf node
          # if not node.children:
          #     paths.append(list(path))
          # else:
              # Explore each child
          for child in node.children:
              dfs(child)
          
          # Backtrack to explore other paths
          # path.pop()
      dfs(self.root)
      # return paths
    
    def obtain_leaf_node_features(self):
        leaf_results = []
        
        def dfs(node):
            if not node:
                return
            
            # Check if the node is a leaf (has no children)
            if not node.children:
                leaf_results.append(node.text_features)
            else:
                # Traverse through the children if not a leaf node
                for child in node.children:
                    dfs(child)

        # Start DFS from the root to collect results
        dfs(self.root)
        return torch.cat(leaf_results)
    
    # def obtain_leaf_node_features(self):
    #     def dfs(node):
    #         if not node:
    #             return
    #         if not node.children:
    #             return node.text_features
    #         else:
    #             for child in node.children:
    #                 return dfs(child)
    #     return dfs(self.root)
        
        
        

def construct_tree(whole_query, aggregate_queries):
  id = 0
  root = TreeNode(id, whole_query, 'root', 1)
  tree = Tree(root)
  for i in range(len(aggregate_queries)):
    singleton_query = aggregate_queries[i][0]
    tgt_count = aggregate_queries[i][1]
    if (tgt_count > 1):
      child = TreeNode(id, singleton_query, 'count', tgt_count)
      id = id + 1
      tree.add_child(root, child)
      id = id + 1
      tree.add_child(child, TreeNode(id, singleton_query, 'plain', 1))
    else:
      id = id + 1
      child = TreeNode(id, singleton_query, 'plain', 1)
      tree.add_child(root, child)
  return tree

def convert_to_trees(whole_query_list, list_of_tuples):
  trees = []
  for i in range(len(whole_query_list)):
    tree = construct_tree(whole_query_list[i], list_of_tuples[i])
    trees.append(tree)
  return trees

def convert_origin_query_to_tree(query):
    root = TreeNode(0, query, 'root', 1)
    tree_11 = Tree(root)
    sub_node = TreeNode(1, query, 'plain', 1)
    tree_11.add_child(root, sub_node)
    return tree_11
    
def analyze_tree(tree):
    def dfs(node, total_nodes, count_nodes, max_tgt_count, total_tgt_count):
        total_nodes += 1

        if node.node_type == 'count':
            count_nodes += 1
            total_tgt_count += node.tgt_count
            max_tgt_count = max(max_tgt_count, node.tgt_count)

        for child in node.children:
            total_nodes, count_nodes, max_tgt_count, total_tgt_count = dfs(
                child, total_nodes, count_nodes, max_tgt_count, total_tgt_count
            )

        return total_nodes, count_nodes, max_tgt_count, total_tgt_count

    return dfs(tree.root, 0, 0, 0, 0)