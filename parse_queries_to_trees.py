import re
from agg_query_processing import *

def parse_nodes(input_string):
    # Split the input string into individual node strings based on node patterns
    node_strings = re.split(r"\s+(?=\(\d+\))", input_string.strip())

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
        if query_string.startswith("Q ="):
            query_string = query_string.replace("Q = ", "")  # Remove "Q =" prefix
            tgt_count = 1  # Set target count to 1 for root nodes
        else:
            # Extract target count from square brackets (if present)
            tgt_count_match = re.search(r'\[(\d+)\]', node_string)
            tgt_count = int(tgt_count_match.group(1)) if tgt_count_match else None

        # Extract children IDs from angle brackets (if present)
        children_match = re.search(r'<([^>]+)>', node_string)
        children = [int(c) for c in children_match.group(1).split(',') if c] if children_match else []

        # Split the query string into subqueries if '|' is present, clean each subquery, and ignore empty subqueries
        if '|' in query_string:
            subqueries = [clean_query(subquery.strip()) for subquery in query_string.split('|') if subquery.strip()]
            query_string = subqueries  # Replace the main query with subqueries
        else:
            query_string = clean_query(query_string)  # Clean the main query

        # Append the extracted data to the parsed_nodes dictionary
        parsed_nodes[node_id] = {
            "Query": query_string,
            "Target Count": tgt_count,
            "Children": children
        }

    return parsed_nodes



# Recursive function to build the tree
def build_tree(parsed_nodes, node_id=0):
    # Find the node in the parsed_nodes list
    node_data = parsed_nodes.get(node_id)

    query = node_data['Query']
    tgt_count = node_data['Target Count']
    children_ids = node_data['Children']

    # Create the current node based on the type
    nodes = []  # Initialize a list to hold the current node(s)

    if isinstance(query, list):
        # Create a plain node for each subquery
        for subquery in query:
            subquery_node = TreeNode(node_id, subquery, 'plain', 1)  # Create a plain node for each subquery
            nodes.append(subquery_node)  # Add to the list of nodes
    else:
        # Create the node based on its type
        if node_id == 0:
            node = TreeNode(node_id, query, 'root', tgt_count)
            nodes.append(node)
        elif tgt_count > 1:
            node = TreeNode(node_id, query, 'count', tgt_count)
            #for _ in range(tgt_count):
            #child_plain_node = TreeNode(node_id, query, 'plain', 1)
            #node.children.append(child_plain_node)
            nodes.append(node)
        elif tgt_count == 1:
            node = TreeNode(node_id, query, 'plain', 1)
            nodes.append(node)

    # Recursively add children nodes
    for child_id in children_ids:
        child_nodes = build_tree(parsed_nodes, child_id)
        if isinstance(child_nodes, TreeNode):
            # If there is only one child node, append it to the current node
            node.children.append(child_nodes)
        else:
            # Extend the current node's children with returned child nodes
            node.children.extend(child_nodes)
    if len(nodes) == 1:
        return nodes[0]  # Return the single
    return nodes  # Return the list of nodes


input_string = "(0){Q = Several women are gather around a table in a corner surrounded by bookshelves.-><1,2,>} (1) {One woman is gather around a table [2]-><3,>} (2) { | a table in a corner [1]| a table surrounded by bookshelves.[1] } (3) { One woman is gather around a table [1] }"

parsed_nodes = parse_nodes(input_string)


root = build_tree(parsed_nodes, 0)
tree = Tree(root)


