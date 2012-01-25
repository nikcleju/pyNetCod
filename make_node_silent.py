import numpy
import topsort

def make_node_silent(net,node):
    #MATLAB function newnet = make_node_silent(net, node)
    # Generates a new net structure by removing all outgoing links of node 'node' and then adjusting all following links of the graphs as to preserve
    #  the same replication rate of all nodes
    # The replication rate of all nodes is thus preserved in the new net (except of course node 'node' whose outgoing links are removed)
    #
    # Inputs:
    #   net  =  network configuration structure, with the fields:
    #         capacities  = capacities matrix, in packets (A(i,j) = x means there is an edge from node i to node j of capacity x packets)
    #         errorrates  = error rates matrix
    #         sources     = vector containing the source nodes
    #         helpers     = vector containing the helper nodes (helper = not source and not client)
    #         receivers   = vector containing the client (receiver) nodes
    #   node =  the node whose outgoing links are to be removed
    #
    # Outputs:
    #   newnet = the new network configuration structure, containing the same fields as the input one
    #
    # Nicolae Cleju, EPFL, 2008/2009, TUIASI, 2009/2010
    #==========================================================================
    
    # Create output structure
    #newnet = net;
    newnet = net.copy()
    
    # Compute the original NI and NO
    #NI = sum(net.capacities .* (1-net.errorrates), 1);
    #NO = sum(net.capacities, 2);    
    NI = numpy.sum(net['capacities'] * (1.-net['errorrates']), 0)
    NO = numpy.sum(net['capacities'], 1)
    
    # Compute the original Nrepl
    #Nrepl = NO' - NI;
    #Nrepl(Nrepl < 0) = 0;
    #Nrepl(net.sources) = 0;
    Nrepl = NO - NI
    Nrepl[Nrepl < 0] = 0
    Nrepl[net['sources']] = 0
    
    # Compute topological order
    #order = topological_order(sparse(net.capacities));
    xs, ys = numpy.nonzero(net['capacities'])
    order = topsort.topsort([(xs[i], ys[i]) for i in range(xs.size)])    
    
    # remove node's outgoing links
    #newnet.capacities(node, :) = zeros(1, newnet.nnodes);
    #newnet.errorrates(node, :) = zeros(1, newnet.nnodes);
    newnet['capacities'][node, :] = numpy.zeros(newnet['nnodes'])
    newnet['errorrates'][node, :] = numpy.zeros(newnet['nnodes'])
    
    # For all following nodes in the topo order, adjust outgoing links to maintain the original replication rate
    
    # generate partial list with all nodes following 'node' in the topo order,
    #partial_list = order( (find(order == node, 1) + 1) : numel(order) ); 
    partial_list = order[ (numpy.nonzero(order == node)[0] + 1) : ]
    
    #newNO = NO;
    #newNI = NI;
    newNO = NO.copy()
    newNI = NI.copy()
    
    #for n_index = 1:numel(partial_list)
    for n_index in range(partial_list.size ):
        #n = partial_list(n_index);
        n = partial_list[n_index]
        
        # compute the new NI
        #newNI(n) = sum(newnet.capacities(:,n) .* (1-newnet.errorrates(:,n)));
        newNI[n] = numpy.sum(newnet['capacities'][:,n] * (1-newnet['errorrates'][:,n]))
        
        # if new NI < original NI, adjust outgoing links with the same factor
        #if (newNI(n) < NI(n))
        if newNI[n] < NI[n]:
            #newnet.capacities(n,:) = (newNI(n)/NI(n)) * newnet.capacities(n,:);
            newnet['capacities'][n,:] = (newNI[n]/NI[n]) * newnet['capacities'][n,:]
        #end
        
        # compute the new NO
        #newNO(n) = sum(newnet.capacities(n,:));
        newNO[n] = numpy.sum(newnet['capacities'][n,:])
    #end
    
    # checking
    #f1 = (NI ./ NO');
    f1 = NI / NO
    #f1(isinf(f1)) = 0;
    f1[numpy.isinf(f1)] = 0
    #f1(isnan(f1)) = 0;
    f1[numpy.isnan(f1)] = 0
    #f2 = (newNI ./ newNO');
    f2 = newNI / newNO
    #f2(isinf(f2)) = 0;
    f2[numpy.isinf(f2)] = 0
    #f2(isnan(f2)) = f1(isnan(f2));
    f2[numpy.isnan(f2)] = f1[numpy.isnan(f2)]
    #diff =  f1 - f2;
    diff =  f1 - f2
    #assert(all(diff.^2 < 0.001));
    assert(numpy.all(diff**2 < 0.001))
    