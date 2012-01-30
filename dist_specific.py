import numpy
import graph
import copy

def create_local_network(net, centralnode, p0matrix, ecc):
    # MATLAB function [localnet local2global global2local] = create_local_network(net, centralnode, p0matrix, ecc)
    # Create a local network structure newnet from the neighborhood of radius
    # ecc around node centralnode
    
    # The local network contains the known neighborhood + receivers (they are known)
    # The receivers are added below the bottom descendants of the neighborhood,
    #  creating links between bottom descendants and added receivers.
    
    # Nodes' input capacities
    #b_i = sum(net.capacities .* (1-net.errorrates), 1);
    #b_i = numpy.sum(net['capacities'] * (1-net['errorrates']), 0)
    # Nodes' output capacities
    #b_o = sum(net.capacities, 2)';
    b_o = numpy.sum(net['capacities'], 1)
    
    # Compute the distance matrix between the nodes of the global net (operate
    # on the undirected graph)
    #[R D] = breadthdist(net.capacities + net.capacities');
    R,D = graph.all_pairs_sp(net['capacities'] + net['capacities'].T)
    
    # Find the known subnetwork
    #subnet_nodes  = union ( find( D(:,centralnode) <= ecc ), find( D(centralnode,:) <= ecc ) );
    subnet_nodes  = numpy.union1d ( numpy.nonzero( D[:,centralnode] <= ecc ), numpy.nonzero( D[centralnode,:] <= ecc ) )
    ##subnet_nodes  = union ( subnet_nodes, centralnode);
    
    # make subnet_nodes a row vector, if not already
    #if size(subnet_nodes, 1) > size(subnet_nodes, 2)
    #    subnet_nodes = subnet_nodes';
    #end
    # In numpy, vectors are 1D, so neither row or col
    
    # The known subnetwork matrix
    #subnet_caps        = net.capacities(subnet_nodes, subnet_nodes);
    subnet_caps        = net['capacities'](subnet_nodes, subnet_nodes)
    # The known subnetwork error matrix
    #subnet_errorrates  = net.errorrates(subnet_nodes, subnet_nodes);
    subnet_errorrates  = net['errorrates'](subnet_nodes, subnet_nodes)
    # The top ancestors (sources) of the known subnet = the nodes which have no incoming links in the subnet
    #top_ancestors      = subnet_nodes(find(sum(subnet_caps, 1)==0))'; # nodes with input capacity == 0
    # The bottom descendants (clients) of the known subnet = the nodes which have no outgoing links in the subnet
    #bottom_descendants = subnet_nodes(find(sum(subnet_caps, 2)==0))'; # nodes with output capacity == 0
    bottom_descendants = subnet_nodes(numpy.nonzero(numpy.sum(subnet_caps, 1)==0)) # nodes with output capacity == 0
    
    # Add the real receivers (the ones which are not already in the subnet)
    # Exclude receivers which happen to lie in the known neighborhood
    #remaining_receivers = setdiff(net.receivers, subnet_nodes);
    remaining_receivers = numpy.setdiff1d(net['receivers'], subnet_nodes)
    #numtoadd = numel(remaining_receivers);
    numtoadd = remaining_receivers.size
    
    # Augment subnet matrix with the added receivers
    #subnet_caps = [subnet_caps  zeros(numel(subnet_nodes), numtoadd)];             # pad with zeros to the right
    #subnet_caps = [subnet_caps; zeros(numtoadd, numel(subnet_nodes) + numtoadd)];  # pad with zeros below
    #subnet_errorrates = [subnet_errorrates  zeros(numel(subnet_nodes), numtoadd)];             # pad with zeros to the right
    #subnet_errorrates = [subnet_errorrates; zeros(numtoadd, numel(subnet_nodes) + numtoadd)];  # pad with zeros below    
    subnet_caps = numpy.hstack((subnet_caps, numpy.zeros((subnet_nodes.size, numtoadd))))             # pad with zeros to the right
    subnet_caps = numpy.vstack((subnet_caps, numpy.zeros((numtoadd, subnet_nodes.size + numtoadd))))  # pad with zeros below
    subnet_errorrates = numpy.hstack((subnet_errorrates, numpy.zeros((subnet_nodes.size, numtoadd))))             # pad with zeros to the right
    subnet_errorrates = numpy.vstack((subnet_errorrates, numpy.zeros((numtoadd, subnet_nodes.size + numtoadd))))  # pad with zeros below
    
    # indices of the bottom descendant nodes, relative to the new node numbers of the subnet
    #rel_bottom_descendants = [];  
    rel_bottom_descendants = numpy.array([])
    
    # create links between subnet bottom descendants and the added real receivers
    #for j = 1:numel(bottom_descendants)
    for j in range(bottom_descendants.size):
    
        # index of the bottom descendant node, relative to the subnetwork node numbers
        #descendantnumber = find(subnet_nodes == bottom_descendants(j));
        descendantnumber = numpy.nonzero(subnet_nodes == bottom_descendants[j])
    
        # For each extra receiver we have to add
        #for i = 1:numtoadd
        for i in range(numtoadd):
    
            # real receiver number, relative to the subnetwork node numbers
            #recvnumber = numel(subnet_nodes) + i;
            recvnumber = subnet_nodes.size + i
    
            # Create a link of capacity equal to the output capacity of the
            #  bottom descendant, but with the loss probability equal to p0(node, client)
            # In this way the total number of packets received by the client
            #  from the node stays the same
            # All the subgraph between the node and the client is  modeled by
            #  this single virtual link of capacity b_0 and loss probability p0
            #subnet_caps      (descendantnumber, recvnumber) = b_o(bottom_descendants(j));
            subnet_caps      [descendantnumber, recvnumber] = b_o[bottom_descendants[j]]
            subnet_errorrates[descendantnumber, recvnumber] = p0matrix[bottom_descendants[j], numpy.nonzero(net['receivers'] == remaining_receivers[i])]
        #end
        
        # add the descendant number to the list of bottom descendants relative numbers
        #rel_bottom_descendants = union(rel_bottom_descendants, descendantnumber);
        rel_bottom_descendants = numpy.union1d(rel_bottom_descendants, descendantnumber)
    #end
    
    # Create the local network structure
    localnet = dict() # Python
    #localnet.nnodes     = numel(subnet_nodes) + numtoadd;
    localnet['nnodes']     = subnet_nodes.size + numtoadd
    #localnet.capacities = subnet_caps;
    localnet['capacities'] = subnet_caps.copy()
    #localnet.errorrates = subnet_errorrates;
    localnet['errorrates'] = subnet_errorrates.copy()
    #localnet.sources    = find(sum(subnet_caps .* (1-subnet_errorrates), 1)==0); # nodes with input capacity == 0, relative to the new node numbers
    localnet['sources']    = numpy.nonzero(numpy.sum(subnet_caps * (1.-subnet_errorrates), 0)==0) # nodes with input capacity == 0, relative to the new node numbers
    #localnet.receivers  = (localnet.nnodes - numel(net.receivers) + 1):localnet.nnodes;  # the last nodes
    localnet['receivers']  = numpy.arange((localnet['nnodes'] - net['receivers'].size), localnet['nnodes'])  # the last nodes
    #allnodes = 1:(localnet.nnodes - numel(localnet.receivers));     # excluding receivers
    allnodes = numpy.arange(localnet['nnodes'] - localnet['receivers'].size)     # excluding receivers
    #localnet.helpers    = setdiff (allnodes, localnet.sources);     # excluding sources
    localnet['helpers']    = numpy.setdiff1d (allnodes, localnet['sources'])     # excluding sources
    
    # Create the local to global mapping of node indices
    #local2global = zeros(1, localnet.nnodes);
    local2global = numpy.zeros(localnet['nnodes'])
    #for i = 1:localnet.nnodes
    for i in range(localnet['nnodes']):
        #if i <= numel(subnet_nodes)
        if i <= (subnet_nodes.size - 1):
            #local2global(i) = subnet_nodes(i);
            local2global[i] = subnet_nodes[i]
        else:
            #local2global(i) = remaining_receivers(i-numel(subnet_nodes));
            local2global[i] = remaining_receivers[i-subnet_nodes.size]
        #end
    #end
    
    # Create the global to local mapping of node indices
    #global2local = zeros(1, net.nnodes);
    global2local = numpy.zeros(net['nnodes'])
    #alllocalnodes = union(subnet_nodes, remaining_receivers);
    alllocalnodes = numpy.union1d(subnet_nodes, remaining_receivers)
    #for i = 1:net.nnodes
    for i in range(net['nnodes']):
        #if ~any(alllocalnodes == i)
        if not numpy.any(alllocalnodes == i):
            #global2local(i) = 0; # Node i not in the local network
            global2local[i] = 0 # Node i not in the local network
        else:
            #global2local(i) = find(alllocalnodes == i);
            global2local[i] = numpy.nonzero(alllocalnodes == i)
        #end
    #end
    
    return localnet,local2global,global2local
    


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
    #newnet = net.copy()
    # BUG: dict.copy() is shallow copy, so the underlying arrays were not copied    
    #  use copy.deepcopy() instead
    newnet = copy.deepcopy(net)
    
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
    #xs, ys = numpy.nonzero(net['capacities'])
    #order = graph.topological_sort([(xs[i], ys[i]) for i in range(xs.size)])    
    order = graph.topological_sort(net['capacities'])
    
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
    
    return newnet
    