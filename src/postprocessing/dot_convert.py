#!/usr/bin/python3
import networkx as nx
import pydot
from pygraphml import GraphMLParser, Graph

f = open("tree.txt", "r")
dot_graph = [i.rstrip().split(",") for i in f.readlines()]
f.close()
str_conv = lambda a : f"Node{a[1:]}" if a[0] == 'N' else f"Data{a[1:]}"
data = []
for elem in dot_graph:
  a,b = elem
  a_str = ""
  b_str = ""
  color = "pink"
  if b[0] == 'D':
    color = "purple"
  d = (color,str_conv(a),str_conv(b))
  data.append(d)

G = nx.DiGraph()
node_colors = {}
for color, A, B in data:
    # if color == "blue":
    #     node_colors[A] = color
    if color == "purple":
        node_colors[A] = color
        node_colors[B] = color
    else:
        node_colors[A] = color

# Add nodes and edges based on relationships
for color, A, B in data:
    G.add_edge(A, B, color=color)
nx.set_node_attributes(G, node_colors, "color")
# Save the graph in GraphML format
nx.write_graphml(G, "tree_representation.graphml")



# # Parse the DOT graph using pydot
# dot_parser = pydot.parser.DotParser()
# dot_parser.parse_dot_data(dot_graph)
# dot_graph = dot_parser.get_next_object()

# # Create a NetworkX graph from the parsed DOT graph
# G = nx.Graph()
# for edge in dot_graph.get_edges():
#     G.add_edge(edge.get_source(), edge.get_destination())

# # Create a GraphML object
# graphml = Graph()

# # Convert the NetworkX graph to the GraphML format
# for node in G.nodes:
#     graphml.add_node(node)
#     graphml[node].label = node

# for edge in G.edges:
#     graphml.add_edge(edge[0], edge[1])

# # Save the GraphML to a file
# with open("output.graphml", "w") as f:
#     f.write(str(graphml))
