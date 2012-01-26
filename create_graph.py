    
import scipy.io
import math

import MatlabInputParser    

def create_graph_planetlab3(n_sources, n_helpers, n_receivers, varargin):
    # MATLAB function A = create_graph_planetlab3(n_sources, n_helpers, n_receivers, varargin)
    # Creates a network connection matrix from planetlab data
    # 
    # Input parameters:
    #   n_sources = number of source nodes
    #   n_helpers = number of helper nodes
    #   n_receivers = number of receivers
    #   optional 'paramater', 'value' pairs:
    #       'maxdistance'  => try to find a network with a given maximum distance between two nodes
    #       'maxtries'     => number of times to try to find a network of the specified maximum distance
    #       'removeunnecessary' = specifies whether the nodes which were randomly selected but do not have a connection to any receiver
    #                               or to any source are to be removed (default) or not. This means that in the end the number of helper nodes 
    #                               might be smaller than n_helpers
    #
    # Output parameters
    #   RA = the connection matrix
    #
    # Nicolae Cleju, TUIASI, 2009
    
    # Python
    varargin['n_sources'] = n_sources
    varargin['n_helpers'] = n_helpers
    varargin['n_receivers'] = n_receivers
    
    # parse inputs
    #p = inputParser;   # Create instance of inputParser class.
    p = MatlabInputParser.MatlabInputParser()   # Create instance of inputParser class.
    p.addRequired('n_sources',   lambda x: (numpy.isreal(x) and x > 0));
    p.addRequired('n_helpers',   lambda x: (numpy.isreal(x) and x > 0));
    p.addRequired('n_receivers', lambda x: (numpy.isreal(x) and x > 0));
    #p.addParamValue('maxdistance', -1, lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('maxdistance', -1, lambda x: (numpy.isreal(x)));
    p.addParamValue('maxtries', 50, lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('removeunnecessary', true, lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('minnnodes', 15, lambda x: (isnumeric(x) and x > 0));
    #p.parse(n_sources, n_helpers, n_receivers, varargin{:});
    p.parse(varargin)
    n_sources   = p.Results['n_sources']
    n_helpers   = p.Results['n_helpers']
    n_receivers = p.Results['n_receivers']
    maxdistance = p.Results['maxdistance']
    maxtries    = p.Results['maxtries']
    removeunnecessary = p.Results['removeunnecessary']
    minnnodes   = p.Results['minnnodes']
    
    # Internal parameters
    #num_parents = 4;
    num_parents = 4
    
    # Select parents only from most distant nodes
    #do_distant_parents = 1;
    do_distant_parents = 1
    # Initial percent
    #dp_start = 0.6;
    dp_start = 0.6
    
    #if exist('planetlab_data.mat', 'file')
    #    load 'planetlab_data.mat'
    #else
    try:
        mdict = scipy.io.loadmat('planetlab_data.mat')
        M = mdict['M']
    except e:
        print("Error: couldn't load 'planetlab_data.mat'");
        raise e
        #        fid = fopen('planetlab.txt');
        #        C = textscan(fid, '#s #s #n #n #n #n #n #n', 'Delimiter',',', 'HeaderLines', 1, 'TreatAsEmpty', 'N/A');
        #        fclose(fid);
        #    
        #        B1 = unique(C{1,1});
        #        B2 = unique(C{1,2});
        #        nodesmap = union(B1, B2);
        #        Nmax = numel(nodesmap);
        #        Lmax = numel(C{1,1});
        #        M = zeros(Nmax, Nmax);
        #    
        #        for i = 1:Nmax
        #            # A struct in matlab _is_ a hash table, since you can use variables for addressing field names.
        #            fieldname = [ nodesmap{i};];
        #            fieldname(fieldname == '.') = '_';
        #            fieldname(fieldname == '-') = [];
        #            if ~isletter(fieldname(1))
        #                fieldname = ['n' fieldname];
        #            end
        #            structhash.(fieldname) = i;
        #    
        #        end
        #    
        #        for i = 1:Lmax
        #            fieldname = C{1,1}{i,1};
        #            fieldname(fieldname == '.') = '_';
        #            fieldname(fieldname == '-') = [];
        #            if ~isletter(fieldname(1))
        #                fieldname = ['n' fieldname];
        #            end
        #            node1 = fieldname;
        #    
        #            fieldname = C{1,2}{i,1};
        #            fieldname(fieldname == '.') = '_';
        #            fieldname(fieldname == '-') = [];
        #            if ~isletter(fieldname(1))
        #                fieldname = ['n' fieldname];
        #            end
        #            node2 = fieldname;
        #    
        #            value = C{1,6}(i);          #bottleneck_available_bandwidth_pathchirp(Kbps)
        #            if ~isfinite(value)
        #                value = C{1,7}(i);      #bottleneck_available_bandwidth_spruce(Kbps)
        #                if ~isfinite(value)
        #                    value = C{1,5}(i);  #bottleneck_capacity(Kbps)
        #                    if ~isfinite(value)
        #                        value = 0;      # no data available for this pair of nodes
        #                    end
        #                end
        #            end
        #                
        #            M(structhash.(node1), structhash.(node2)) = value;
        #        end
        #    
        #        clear C
        #        save 'planetlab_data.mat'
        #    end
    
    #M(isnan(M)) = 0;
    M(numpy.isnan(M)) = 0
    
    # assuming a packet of 1024B payload, 8B UDP header, 2*32B NC coeffs header
    ##pktsize = (1024 + 8 + 2*32) * 8; # in bps
    #pktsize = (512 + 8 + 2*32) * 8; # in bps
    pktsize = (512. + 8 + 2*32) * 8; # in bps
    
    # capacity values are in  Kbps
    ##M = (1000*M) ./ (100*pktsize);   # using only 1/100 of available bandwidth
    #M = (1000*M) ./ (200*pktsize);   # using only 1/200 of available bandwidth
    M = (1000.*M) / (200.*pktsize);   # using only 1/200 of available bandwidth
    
    #M = sparse(M);
    
    #nnodes = n_sources + n_helpers + n_receivers;
    nnodes = n_sources + n_helpers + n_receivers
    #G = zeros(nnodes, nnodes);
    G = numpy.zeros((nnodes, nnodes))
    
    #goodradius = 0;
    goodradius = 0
    #goodnnodes = 0;
    goodnnodes = 0
    #ntries = 0;
    ntries = 0
    #while (~goodradius || ~goodnnodes)&& ntries < maxtries
    while (not goodradius or not goodnnodes) and ntries < maxtries:
    
        # choose different sources
        #reachables = [];
        reachables = numpy.array([])
        #while numel(reachables) < (n_helpers + n_receivers)
        while reachables.size < (n_helpers + n_receivers):
            #s = [];
            s = numpy.array([])
            #for i = 1:n_sources
            for i in xrange(n_sources):
                #newsource = ceil(Nmax * rand());
                newsource = math.ceil(Nmax * rng.rand())
                #while ~isempty(find(s == newsource))
                while numpy.nonzero(s == newsource).size != 0:
                    #newsource = ceil(Nmax * rand());
                    newsource = math.ceil(Nmax * rng.rand())
                #end
                
                #s(i) = newsource;
                s[i] = newsource
    
                #[d pred] = shortest_paths(M, s(i));
                d = graph.shortest_paths(M, s[i])
                #curr_reachables = (1:Nmax);
                curr_reachables = numpy.arange(Nmax)
                #curr_reachables = curr_reachables(~isinf(d));
                curr_reachables = curr_reachables[numpy.logical_not(numpy.isinf(d))]
    
                #reachables = union(reachables, curr_reachables);
                reachables = numpy.union1d(reachables, curr_reachables)
            #end
        #end
        
        #allnodes = s;
        allnodes = s.copy()
        #allnodesnotreceivers = s;
        allnodesnotreceivers = s.copy()
    
        # choose helper nodes
        #for i = (n_sources+1):nnodes
        for i in xrange(n_sources,nnodes):
    
            # choose a new node
    
            #newnode = s(1); # init the new node, such that we surely enter the while loop
            newnode = s[0]   # init the new node, such that we surely enter the while loop
            #isreachable = 0;
            isreachable = 0
    
            #while ~isempty(find(allnodes == newnode)) || ~isreachable
            while numpy.nonzero(allnodes == newnode).size !=0 or not isreachable:
                
                # choose random new node
                #newnode = ceil(Nmax * rand());
                newnode = math.ceil(Nmax * rng.rand())
    
                # choose num_parents parents out of the previous nodes
                #parents = [];
                parents = numpy.array([])
                #if numel(allnodesnotreceivers) <= num_parents
                if allnodesnotreceivers.size <= num_parents:
                    #parents = allnodesnotreceivers;
                    parents = allnodesnotreceivers.copy()
                    #parents_indexes = 1:numel(allnodesnotreceivers);
                    parents_indexes = numpy.arage(allnodesnotreceivers)
                else:
                    
                    # Select only parents which are distant from sources ?
                    #if do_distant_parents
                    if do_distant_parents:
                        #parent_distance = Dmin(1:numel(allnodesnotreceivers));
                        parent_distance = Dmin[:allnodesnotreceivers.size]
                        #enough_candidate_parents = 0;
                        enough_candidate_parents = 0
                        #dp = dp_start;
                        dp = dp_start
                        #while ~enough_candidate_parents && dp > 0
                        while not enough_candidate_parents and dp > 0:
                            #candidate_parents_indexes = find(parent_distance >= dp*max(parent_distance));
                            candidate_parents_indexes = numpy.nonzero(parent_distance >= dp*numpy.max(parent_distance))
                            #if numel(candidate_parents_indexes) < num_parents
                            if candidate_parents_indexes.size < num_parents:
                                #dp = dp - 0.1;
                                dp = dp - 0.1
                            else:
                            	  #enough_candidate_parents = 1;
                                enough_candidate_parents = 1
                            #end
                        #end
                        #if ~enough_candidate_parents
                        if not enough_candidate_parents:
                            #candidate_parents_indexes = 1:numel(allnodesnotreceivers);
                            candidate_parents_indexes = numpy.arange(allnodesnotreceivers.size)
                        #end
                    else:
                        #candidate_parents_indexes = 1:numel(allnodesnotreceivers);
                        candidate_parents_indexes = numpy.arange(allnodesnotreceivers.size)
                    #end
                    
                    #perm = randperm(numel(candidate_parents_indexes));
                    perm = rng.shuffle(numpy.arange(candidate_parents_indexes.size))
                    #parents_indexes = candidate_parents_indexes[perm[:num_parents]]
                    parents_indexes = candidate_parents_indexes[perm[:num_parents]]
                    #parents = allnodesnotreceivers(parents_indexes);
                    parents = allnodesnotreceivers[parents_indexes]
    
                #end
    
                # create corresponding column in the overlay graph matrix G
                #newcolumn = zeros(nnodes, 1);
                newcolumn = numpy.zeros((nnodes, 1))
                #for j = 1:numel(parents_indexes)
                for j in xrange(parents_indexes.size):
                    #newcolumn(parents_indexes(j)) = M(parents(j), newnode);
                    newcolumn[parents_indexes[j]] = M[parents[j], newnode]
                #end
    
                # if at least one input connection is non-zero, add node
                #if ~isempty(find(newcolumn))
                if numpy.nonzero(newcolumn).size != 0:
                    #isreachable = true;
                    isreachable = True
                else:
                    #isreachable = false;
                    isreachable = False
                #end
    
            #end
            
            # add node
            #allnodes = [allnodes newnode];
            allnodes = numpy.hstack((allnodesn, newnode))
            #G(:,i) = newcolumn;
            G[:,i] = newcolumn
            #if nnodes - numel(allnodes) >= n_receivers
            if nnodes - allnodes.size >= n_receivers:
                #allnodesnotreceivers = allnodes;
                allnodesnotreceivers = allnodes.copy()
            #end
            
            # Redo distance matrix
            #[R D] = breadthdist(G);
            R,D = graph.all_pairs_sp(G)
            
            # Min distance from a source to the node
            #D(1:(nnodes+1):end) = 0;
            #D[1:(nnodes+1):end] = 0;
            #Dmin = min(D(1:n_sources, :),[],1);
            Dmin = numpy.min(D[:n_sources, :],0)
        #end        
        
        
        # no cycles possible!
        ##A = break_cycles(G);
        ##assert(A == G);
        #A = G;
        A = G.copy()
    
        # remove unnecessary nodes, i.e. nodes that cannot be reached by any
        #  source and nodes that do no reach any client
        #if removeunnecessary
        if removeunnecessary:
            #[R D] = breadthdist(A);
            R,D = graph.all_pairs_sp(A)
            #unnecessary = [];
            unnecessary = numpy.array([])
            
            # not sources or receivers, only helper nodes
            #for i = (n_sources+1):(nnodes-n_receivers)
            for i in xrange(n_sources,nnodes-n_receivers):
                #if ~any(R(1:n_sources,i)) || ~any(R(i,(nnodes-n_receivers+1):nnodes))  # by means of construction, every node is reachable from at least one source
                if (not numpy.any(R[:n_sources,i])) or (not numpy.any(R[i,(nnodes-n_receivers):nnodes])):  # by means of construction, every node is reachable from at least one source
                    #unnecessary = [unnecessary i];
                    unnecessary = numpy.hstack((unnecessary, i))
                #end
            #end
            #A(unnecessary, :) = [];
            numpy.delete(A,unnecessary,0)
            #A(:, unnecessary) = [];
            numpy.delete(A,unnecessary,1)
        #end
        
        # check if number of remaining nodes >= minnodes
        #if size(A,1) >= minnnodes
        if A.shape[0] >= minnnodes:
            #goodnnodes = 1;
            goodnnodes = 1
        else:
            #goodnnodes = 0;
            goodnnodes = 0
        #end
        
        # find max radius (in not-directional network)
        #[R D] = breadthdist(A + A');
        R,D = graph.all_pairs_sp(A + A.T)
        #D(find(D==Inf)) = 0;
        D[D==numpy.Inf] = 0
        #maxradius = max(max(D));
        maxradius = numpy.max(D)
    
        # check if maxdistance == maxradius
        #if maxdistance ~= -1
        if maxdistance != -1:
            #if maxdistance == maxradius
            if maxdistance == maxradius:
                #goodradius = 1;
                goodradius = 1
            else:
                #goodradius = 0;
                goodradius = 0
            #end
        else:
            #goodradius = 1;
            goodradius = 1
        #end
        
        #ntries = ntries + 1;
        ntries = ntries + 1
        #if mod(ntries,10) == 0
        if ntries%10 == 0:
            #disp([num2str(ntries) ' tries...']);
            print(str(ntries) + ' tries...')
        #end
    #end
    
    #if ~goodradius || ~goodnnodes
    if not goodradius or not goodnnodes:
        #error('create_graph_planetlab3:cantcreatenetwork', 'Could not create network with specified max distance and number of nodes');
        print('Could not create network with specified max distance and number of nodes')
        raise
    else:
        #disp(['Created network after ' num2str(ntries) ' tries']);
        print('Created network after ' + str(ntries) + ' tries')
        #disp(['Total number of nodes = ' num2str(size(A,1))]);
        print('Total number of nodes = ' + str(A.shape[0]))
    #end
    
    #end
    
    return A
    
