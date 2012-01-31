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
    
    #preorder = networkx.dfs_preorder_nodes(G)
    #return numpy.array(networkx.topological_sort(G, preorder))
    return numpy.array(topo_sort_like_bgl(G))

def topo_sort_like_bgl(G):
    """
    Variant of networkx.topological_sort() that returns a topological order
    that should be identical to the one returrned by BGL topological_sort()
    function.
    
    Modifications:
        1. sort the childeren of a visited node (sometimes was not ordered)
        2. reverse the list of new nodes before adding to the big list
    
    """
    if not G.is_directed():
        raise networkx.NetworkXError(
                "Topological sort not defined on undirected graphs.")
    
    # nonrecursive version
    seen={}
    order_explored=[] # provide order and 
    explored={}       # fast search without more general priorityDictionary
    
    #if nbunch is None:
    nbunch = G.nodes_iter() 
    for v in nbunch:     # process all vertices in G
        if v in explored: 
            continue
        fringe=[v]   # nodes yet to look at
        while fringe:
            w=fringe[-1]  # depth first search
            if w in explored: # already looked down this branch
                fringe.pop()
                continue
            seen[w]=1     # mark as seen
            # Check successors for cycles and for new nodes
            new_nodes=[]
            # Nic: sort G[w] because sometimes the children are not in order
            #for n in G[w]:
            for n in sorted(G[w]): 
                if n not in explored:
                    if n in seen: #CYCLE !!
                        raise networkx.NetworkXUnfeasible("Graph contains a cycle.")
                    new_nodes.append(n)
            if new_nodes:   # Add new_nodes to fringe:
                # Nic: reverse here the list of children
                new_nodes.reverse() # Added by Nic
                fringe.extend(new_nodes)
                
            else:           # No new nodes so w is fully explored
                explored[w]=1
                order_explored.insert(0,w) # reverse order explored
                fringe.pop()    # done considering this node
    return order_explored