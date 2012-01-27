#import pygraph.classes.graph
#import pygraph.algorithms.searching

import numpy
import networkx

def all_pairs_sp(m):
    #    # Input: m = adjacency matrix
    #    # Output R = reachability matrix (directed graph)
    #    #        D = distance matrix (directed graph)
    #    
    #    G = pygraph.classes.graph.graph()
    #    G.add_nodes(range(m.shape[0]))
    #    for i in xrange(m.shape[0]):
    #        for j in xrange(m.shape[1]):
    #            if m[i,j] != 0:
    #                G.add_edge((i,j))
    #    
    #    D = pygraph.algorithms.searching.breadth_first_search()
    #    return R,D
    
    G = networkx.Graph()
    G.add_nodes_from(range(m.shape[0]))
    nzx, nzy = numpy.nonzero(m)
    G.add_edges_from([(nzx[i],nzy[i]) for i in range(nzx.size)])
    d = networkx.algorithms.all_pairs_shortest_path(G)
    
    D = numpy.zeros(m.shape)
    R = numpy.zeros(m.shape)
    for i in xrange(D.shape[0]):
        for j in xrange(D.shape[1]):
            if d[i].has_key(j):
                R[i,j] = True
                D[i,j] = len(d[i][j]) - 1
            else:
                R[i,j] = False
                D[i,j] = numpy.Inf
    
    return R,D

def shortest_paths(m, source):
    G = networkx.Graph()
    G.add_nodes_from(range(m.shape[0]))
    nzx, nzy = numpy.nonzero(m)
    G.add_edges_from([(nzx[i],nzy[i]) for i in range(nzx.size)])
    d = networkx.algorithms.single_source_shortest_path_length(G,source)
    
    D = numpy.zeros(m.shape[0])
    for i in xrange(D.size):
        if d.has_key(i):
            D[i] = d[i]
        else:
            D[i] = numpy.Inf
    return D
            
def topological_sort(m):
    G = networkx.DiGraph()
    G.add_nodes_from(range(m.shape[0]))
    nzx, nzy = numpy.nonzero(m)
    G.add_edges_from([(nzx[i],nzy[i]) for i in range(nzx.size)])
    
    return numpy.array(networkx.topological_sort(G))
