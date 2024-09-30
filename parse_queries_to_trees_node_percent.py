import re
from agg_query_processing import *

# def parse_nodes(input_string):
#     # Split the input string into individual node strings based on node patterns
#     node_strings = re.split(r"\s+(?=\(\d+\))", input_string.strip())

#     # Initialize a dictionary to hold the parsed results
#     parsed_nodes = {}

#     # Function to clean query: split on -> or [ and take only the first part
#     def clean_query(query):
#         return re.split(r'->|\[', query)[0].strip('. ')

#     # Iterate over each node string and extract the node ID, query, target count, and children
#     for node_string in node_strings:
#         # Extract the node ID from the parentheses
#         node_id_match = re.search(r'\((\d+)\)', node_string)
#         node_id = int(node_id_match.group(1)) if node_id_match else None

#         # Extract the query string from the curly braces
#         query_match = re.search(r'\{(.*?)\}', node_string)
#         query_string = query_match.group(1).strip() if query_match else None

#         # Initialize target count; check if the query starts with "Q ="
#         if query_string.startswith("Q ="):
#             query_string = query_string.replace("Q = ", "")  # Remove "Q =" prefix
#             tgt_count = 1  # Set target count to 1 for root nodes
#         else:
#             # Extract target count from square brackets (if present)
#             tgt_count_match = re.search(r'\[(\d+)\]', node_string)
#             tgt_count = int(tgt_count_match.group(1)) if tgt_count_match else None

#         # Extract children IDs from angle brackets (if present)
#         children_match = re.search(r'<([^>]+)>', node_string)
#         children = [int(c) for c in children_match.group(1).split(',') if c] if children_match else []

#         # Split the query string into subqueries if '|' is present, clean each subquery, and ignore empty subqueries
#         if '|' in query_string:
#             subqueries = [clean_query(subquery.strip()) for subquery in query_string.split('|') if subquery.strip()]
#             query_string = subqueries  # Replace the main query with subqueries
#         else:
#             query_string = clean_query(query_string)  # Clean the main query

#         # Append the extracted data to the parsed_nodes dictionary
#         parsed_nodes[node_id] = {
#             "Query": query_string,
#             "Target Count": tgt_count,
#             "Children": children
#         }

#     return parsed_nodes

def parse_nodes(input_string):
    # Split the input string into individual node strings based on node patterns
    node_strings = re.split(r'\s+(?=\(\d+\))', input_string.strip())

    # Initialize a dictionary to hold the parsed results
    parsed_nodes = {}

    # Function to clean query: split on -> or [ and take only the first part
    def clean_query(query):
        return re.split(r'->|\[', query)[0].strip('. ')

    # Iterate over each node string and extract the node ID, query, target count, and children
    for node_string in node_strings:
        # Extract the node ID from the parentheses
        node_id_match = re.search(r'\((\d+)\)', node_string)
        node_id = int(node_id_match.group(1)) if node_id_match else None
        
        # Extract the query string from the curly braces
        query_match = re.search(r'\{(.*?)\}', node_string)
        query_string = query_match.group(1).strip() if query_match else None

        # Initialize target count; check if the query starts with "Q ="
        if node_id == 0:
            tgt_count = 1  # Set target count to 1 for root nodes
        else:
            # Extract target count from square brackets (if present)
            tgt_count_match = re.search(r'\[(\d+)\]', node_string)
            tgt_count = int(tgt_count_match.group(1)) if tgt_count_match else None

        # Extract children IDs from angle brackets (if present)
        children_match = re.search(r'<([^>]+)>', node_string)
        children = [int(c) for c in children_match.group(1).split(',') if c] if children_match else []

        # Split the query string into subqueries if '|' is present, clean each subquery, and ignore empty subqueries
        subqueries = [clean_query(subquery.strip(' |"')) for subquery in query_string.split('|') if subquery.strip('" |')]

        # Skip the node if no valid subqueries are found
        if len(subqueries) == 0:
            continue

        # If there are multiple subqueries, store them, otherwise store the single query
        if len(subqueries) > 1:
            query_string = subqueries  # Replace the main query with subqueries
        else:
            query_string = subqueries[0]  # Single valid subquery

        # Append the extracted data to the parsed_nodes dictionary
        parsed_nodes[node_id] = {
            "Query": query_string,
            "Target Count": tgt_count,
            "Children": children
        }

    return parsed_nodes

# # Recursive function to build the tree
# def build_tree(parsed_nodes, node_id=0):
#     # Find the node in the parsed_nodes list
#     node_data = parsed_nodes.get(node_id)

#     query = node_data['Query']
#     tgt_count = node_data['Target Count']
#     children_ids = node_data['Children']

#     # Create the current node based on the type
#     nodes = []  # Initialize a list to hold the current node(s)

#     if isinstance(query, list):
#         # Create a plain node for each subquery
#         for subquery in query:
#             subquery_node = TreeNode(node_id, subquery, 'plain', 1)  # Create a plain node for each subquery
#             nodes.append(subquery_node)  # Add to the list of nodes
#     else:
#         # Create the node based on its type
#         if node_id == 0:
#             node = TreeNode(node_id, query, 'root', tgt_count)
#             nodes.append(node)
#         elif tgt_count > 1:
#             node = TreeNode(node_id, query, 'count', tgt_count)
#             #for _ in range(tgt_count):
#             #child_plain_node = TreeNode(node_id, query, 'plain', 1)
#             #node.children.append(child_plain_node)
#             nodes.append(node)
#         elif tgt_count == 1:
#             node = TreeNode(node_id, query, 'plain', 1)
#             nodes.append(node)

#     # Recursively add children nodes
#     for child_id in children_ids:
#         child_nodes = build_tree(parsed_nodes, child_id)
#         if isinstance(child_nodes, TreeNode):
#             # If there is only one child node, append it to the current node
#             node.children.append(child_nodes)
#         else:
#             # Extend the current node's children with returned child nodes
#             node.children.extend(child_nodes)
#     if len(nodes) == 1:
#         return nodes[0]  # Return the single
#     return nodes  # Return the list of nodes


# Recursive function to build the tree
# def build_tree(parsed_nodes, node_id=0, node_counter=None):
#     # Initialize the node counter if it's the first call
#     if node_counter is None:
#         node_counter = [max(parsed_nodes.keys()) + 1]  # Start with the next available node ID

#     # Find the node in the parsed_nodes dictionary
#     node_data = parsed_nodes.get(node_id)

#     query = node_data['Query']
#     tgt_count = node_data['Target Count']
#     children_ids = node_data['Children']

#     # Create a list to hold the nodes for this part of the tree
#     nodes = []

#     if isinstance(query, list):
#         # If the query is a list of subqueries, create a node for each subquery
#         for subquery in query:
#             # Assign a new node ID for each subquery
#             new_node_id = node_counter[0]
#             node_counter[0] += 1  # Increment the global counter
#             subquery_node = TreeNode(new_node_id, subquery, 'plain', 1)  # Create a plain node for each subquery
#             nodes.append(subquery_node)  # Add this subquery node to the list
#     else:
#         # If it's a single query, create the appropriate type of node
#         if node_id == 0:
#             node = TreeNode(node_id, query, 'root', tgt_count)
#             nodes.append(node)
#         elif tgt_count > 1:
#             node = TreeNode(node_id, query, 'count', tgt_count)
#             nodes.append(node)
#         elif tgt_count == 1:
#             node = TreeNode(node_id, query, 'plain', 1)
#             nodes.append(node)

#     # Recursively add child nodes
#     for child_id in children_ids:
#         child_nodes = build_tree(parsed_nodes, child_id, node_counter)  # Pass the global counter
#         if isinstance(child_nodes, TreeNode):
#             # If only one node is returned, append it as a child
#             nodes[0].children.append(child_nodes)
#         else:
#             # If a list of nodes is returned, extend the current node's children with them
#             nodes[0].children.extend(child_nodes)

#     if len(nodes) == 1:
#         return nodes[0]  # Return the single node if that's all there is
#     return nodes  # Return the list of nodes

# input_string = "(0){Q = Several women are gather around a table in a corner surrounded by bookshelves.-><1,2,>} (1) {One woman is gather around a table [2]-><3,>} (2) { | a table in a corner [1]| a table surrounded by bookshelves.[1] } (3) { One woman is gather around a table [1] }"

# input_string="(0){Two young guys with shaggy hair look at their hands while hanging out in the yard.-><1,> } (1){One young guy with shaggy hair looks at his hands while hanging out in the yard. [2]-><2,> } (2) { One young guy with shaggy hair [1]| One young guy looks at his hands [1]| One young guy is hanging out in the yard. [1] }"

# parsed_nodes = parse_nodes(input_string)


# root = build_tree(parsed_nodes, 0)
# tree = Tree(root)
# print()


# def construct_tree_from_string(input_string):
#     parsed_nodes = parse_nodes(input_string)
#     root = build_tree(parsed_nodes, 0)
#     tree = Tree(root)
#     return tree

# def output_tree_list(text_file_path):
#   with open(text_file_path, 'r') as file:
#     txt = file.read()
#   tree_strings = txt.split('\n')
#   tree_strings_cleaned = []
#   for tree_string in tree_strings:
#     tree_string = tree_string.strip()
#     if tree_string:
#         tree_strings_cleaned.append(tree_string)
#   output_tree_list = []
#   for i in range(0, len(tree_strings_cleaned)):
#     print(i)
#     input_string = tree_strings_cleaned[i]
#     print(input_string)
#     tree = construct_tree_from_string(input_string)
#     tree.display(tree.root)
#     output_tree_list.append(tree)
#   return output_tree_list

def build_tree(parsed_nodes, node_id=0, node_counter=None, node_count=0, cnt_node_count=0, max_tgt_count=0, total_tgt_count=0):
    # Initialize the node counter if it's the first call
    if node_counter is None:
        node_counter = max(parsed_nodes.keys()) + 1  # Start with the next available node ID

    # Find the node in the parsed_nodes dictionary
    node_data = parsed_nodes.get(node_id)

    # Check if node_data is None, and handle it
    if node_data is None:
        print(f"Skipping node {node_id} as it was skipped during parsing.")
        return None, node_count, cnt_node_count, max_tgt_count, total_tgt_count  # Skip this node entirely if it was skipped in parsing

    query = node_data['Query']
    tgt_count = node_data['Target Count']
    children_ids = node_data['Children']

    # Create a list to hold the nodes for this part of the tree
    nodes = []

    if isinstance(query, list):
        # If the query is a list of subqueries, process each subquery
        for subquery in query:
            subquery = subquery.strip('"| ')  # Strip leading/trailing | and whitespace
            if subquery:  # If the subquery is not empty after stripping, create a node
                # Assign a new node ID for each valid subquery
                new_node_id = node_counter
                node_counter += 1  # Increment the global counter
                subquery_node = TreeNode(new_node_id, subquery, 'plain', 1)  # Create a plain node for each subquery
                nodes.append(subquery_node)  # Add this subquery node to the list
                node_count += 1  # Increment the total node count
    else:
        query = query.strip('"| ')  # Strip leading/trailing | and whitespace for a single query
        if query:  # Only proceed if the query is not empty after stripping
            # If it's a single query, create the appropriate type of node
            if node_id == 0:
                node = TreeNode(node_id, query, 'root', tgt_count)
                nodes.append(node)
            elif len(children_ids) > 0:
                node = TreeNode(node_id, query, 'count', tgt_count)
                nodes.append(node)
                cnt_node_count += 1  # Increment the count node count
                max_tgt_count = max(max_tgt_count, tgt_count)  # Update max_tgt_count
                total_tgt_count += tgt_count  # Update total_tgt_count
            elif len(children_ids) == 0:
                node = TreeNode(node_id, query, 'plain', 1)
                nodes.append(node)
            node_count += 1  # Increment the total node count

    # Recursively add child nodes
    for child_id in children_ids:
        child_nodes, node_count, cnt_node_count, max_tgt_count, total_tgt_count = build_tree(parsed_nodes, child_id, node_counter, node_count, cnt_node_count, max_tgt_count, total_tgt_count)  # Pass the updated node counter and count
        if child_nodes is None:
            continue  # Skip None children (nodes that were skipped in parsing)
        if isinstance(child_nodes, TreeNode):
            # If only one node is returned, append it as a child
            nodes[0].children.append(child_nodes)
        else:
            # If a list of nodes is returned, extend the current node's children with them
            nodes[0].children.extend(child_nodes)

    if len(nodes) == 1:
        return nodes[0], node_count, cnt_node_count, max_tgt_count, total_tgt_count  # Return the single node and the node count
    return nodes, node_count, cnt_node_count, max_tgt_count, total_tgt_count  # Return the list of nodes and the total node count

def construct_tree_from_string(input_string):
    parsed_nodes = parse_nodes(input_string)
    root, node_count, cnt_node_count, max_tgt_count, total_tgt_count = build_tree(parsed_nodes, 0)
    tree = Tree(root)
    return tree, node_count, cnt_node_count, max_tgt_count, total_tgt_count

def output_tree_list(text_file_path):
  with open(text_file_path, 'r') as file:
    txt = file.read()
  tree_strings = txt.split('\n')
  tree_strings_cleaned = []
  for tree_string in tree_strings:
    tree_string = tree_string.strip()
    if tree_string:
        tree_strings_cleaned.append(tree_string)
  output_tree_list = []
  node_count_list = []
  cnt_node_count_list = []
  max_tgt_count_list = []
  total_tgt_count_list = []
  for i in range(0, len(tree_strings_cleaned)):
    print(i)
    input_string = tree_strings_cleaned[i]
    print(input_string)
    tree, node_count, cnt_node_count, max_tgt_count, total_tgt_count = construct_tree_from_string(input_string)
    tree.display(tree.root)
    output_tree_list.append(tree)
    node_count_list.append(node_count)
    cnt_node_count_list.append(cnt_node_count)
    max_tgt_count_list.append(max_tgt_count)
    total_tgt_count_list.append(total_tgt_count)
  return output_tree_list, node_count_list, cnt_node_count_list, max_tgt_count_list, total_tgt_count_list