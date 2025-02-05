import networkx as nx
import pandas as pd
import numpy as np
def random_walk(graph:nx.Graph, node:int, steps:int = 4, p:float=1.0, q:float=1.0):
   """  
   perform a Node2vec random walk for a node in a graph 
   p: return parameter, control the likelihood to revisit a node. low value keep local
   q: in-out parameter, inward/outwards node balance; high value local, low value exploration
   """
   if nx.is_isolate(G, node):
        rw = [str(node)]
   else:
       rw = [str(node),]
       start_node = node
       for _ in range(steps):
          weights = []
          neighbors = list(nx.all_neighbors(graph, start_node))
          for neigh in neighbors:
            if str(neigh) == rw[-1]:
                # Control the probability to return to the previous node.
                weights.append(1/ p)
            elif graph.has_edge(neigh,rw[-1]):
                # The probability of visiting a local node.
                weights.append(1)
            else:
                # Control the probability to move forward.
                weights.append(1 / q)

          # we transform the probability to 1
          weight_sum = sum(weights)
          probabilities = [weight / weight_sum for weight in weights]
          walking_node = np.random.choice(neighbors, size=1, p=probabilities)[0]
          rw.append(str(walking_node))
          start_node= walking_node
   return rw

def get_paths(graph:nx.Graph, rws= 10, steps = 4, p=1.0, q=1.0):
   """  
   perform a set of random walks ina graph
   """
   paths = []
   for node in graph.nodes():
     for _ in range(rws):
         paths.append(random_walk(graph, node, steps, p, q))
   return paths

def mapping_rw(rws=None, features=None):
    """mapping clustering labels to a membership matrix
    input
    rws = a list of random walks (as list of list
    features = list of original features
    output
    a panda dataframe where each feature is mapped to the cluster it belongs
    example usage:
    go = mapping_rw(rws=random_walks, features=data.columns.to_list())
    """
    rw_list = [i for i in range(len(rws))]
    A = pd.DataFrame(0, columns=rw_list, index=features)
    for i in range(len(random_walks)):
        rw = list(map(int, rws[i]))
        for j in rw:
            A.loc[features[j],i] = 1
    return A