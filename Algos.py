import numpy
import datetime

#import topsort
import graph

import computings
import updateR
import dist_specific


# Algorithm 1: delay computation for a given network, with given NC nodes

def Algo1_delay_computation(net, sim, ncnodes, prev_estim):
    #MATLAB function tc = Algo1_delay_computation(net, sim, ncnodes, prev_estim)
    
    ##compute_p0_matrix = @compute_p0_matrix;
    ##compute_p0_matrix = @compute_p0_matrix_optimized;
    
    # Precision for iterations
    #precision = 5e-2;
    precision = 5e-2
    
    # Number of nodes
    #N = size(net.capacities,1);
    N = net['capacities'].shape[0]
    
    # Nodes' input capacities
    #b_i = sum(net.capacities .* (1-net.errorrates), 1);
    b_i = numpy.sum(net['capacities'] * (1-net['errorrates']), 0)
    # Nodes' output capacities
    #b_o = sum(net.capacities, 2)';
    b_o = numpy.sum(net['capacities'], 1)
    
    # Generation size
    #gensize = sim.N;
    gensize = sim['N']
    
    # Initialize replication rates
    # Careful to avoid NaN = 0/0
    #R = zeros(1, numel(b_i));
    R = numpy.zeros(b_i.size)
    #R(b_i ~= 0) = b_o(b_i ~= 0) ./ b_i(b_i~=0);
    R[b_i!=0] = b_o[b_i != 0] / b_i[b_i!=0]
    #R = repmat(R, numel(net.receivers), 1);
    R = numpy.tile(R, (net['receivers'].size, 1))
    #Rinit = R;
    Rinit = R.copy()
    
    # Better estimate replication rates if previous delay estimates available
    #if ~isempty(prev_estim)
    #if prev_estim.size != 0:
    if prev_estim:
        #p0matrix_prevestim = compute_p0_matrix(net, prev_estim.ncnodes, R);
        p0matrix_prevestim = computings.compute_p0_matrix(net, prev_estim['ncnodes'], R)
        # Update R
        #R = update_R(R, net, p0matrix_prevestim, prev_estim.ncnodes, prev_estim.tc);
        R = updateR.update_R(R, net, p0matrix_prevestim, prev_estim['ncnodes'], prev_estim['tc'])
    else:
        #p0matrix_prevestim = compute_p0_matrix(net, [], R);
        p0matrix_prevestim = computings.compute_p0_matrix(net, numpy.array([]), R)
    #end
    
    # Nic: Python says: init tc
    tc = numpy.Inf * numpy.ones(net['receivers'].size)
    
    # Repeat until converged
    #converged_tc = false;
    converged_tc = False
    #prev_tc = Inf * ones(1, numel(net.receivers));
    prev_tc = numpy.Inf * numpy.ones(net['receivers'].size)
    #max_tc = zeros(1, numel(net.receivers));
    max_tc = numpy.zeros(net['receivers'].size)
    #niter = 0;
    niter = 0
    #maxiter = 50;
    maxiter = 50
    #while ~converged_tc && niter < maxiter
    while (not converged_tc) and (niter < maxiter):
    
        # Compute p0 matrix for all clients at once
        #p0matrix_allncnodes = compute_p0_matrix(net, ncnodes, R);
        p0matrix_allncnodes = computings.compute_p0_matrix(net, ncnodes, R)
        
        # Repeat for each client r
        #for r_idx = 1:numel(net.receivers)
        for r_idx in range(net['receivers'].size):
            #r = net.receivers(r_idx);
            #r = net['receivers'][r_idx]
            
            #Nc = zeros(1, N);
            Nc = numpy.zeros(N)
            
            #Nc(net.sources) = b_o(net.sources)' .* (1 - p0matrix_allncnodes(net.sources, r_idx));
            Nc[net['sources']] = b_o[net['sources']] * (1 - p0matrix_allncnodes[net['sources'], r_idx])
    
            # For each NC node u
            # Go in topo order
            #order = topological_order(sparse(net.capacities));
            #xs, ys = numpy.nonzero(net['capacities'])
            #order = graph.topological_sort([(xs[i], ys[i]) for i in range(xs.size)])                
            order = graph.topological_sort(net['capacities'])
            # is member returns a logical array of size(order) with 1 on the
            # positions where the node is in ncnodes
            #ncnodes_topo = order(ismember(order, ncnodes));
            ncnodes_topo = order[numpy.in1d(order,ncnodes)]
    
            # For each NC node u
            #for u_idx = 1:numel(ncnodes_topo)
            for u_idx in range(ncnodes_topo.size):
                #u = ncnodes_topo(u_idx);
                u = ncnodes_topo[u_idx]
    
                # Compute p0 matrix with only the previous NC ndoes
                ##p0matrix_prevNC = compute_p0_matrix(net, ncnodes_topo(1:u_idx-1), R);
                #p0matrix_prevNC_single_r = compute_p0_matrix_single_r(net, ncnodes_topo(1:u_idx-1), R, r_idx);
                p0matrix_prevNC_single_r = computings.compute_p0_matrix_single_r(net, ncnodes_topo[:u_idx], R, r_idx)
                
                # Compute t_c(u) with u SF
                ##tc1 = compute_tc(net, sim, ncnodes_topo(1:u_idx-1), Nc, p0matrix_prevNC(:,r_idx));
                #tc1 = compute_tc(net, sim, ncnodes_topo(1:u_idx-1), Nc, p0matrix_prevNC_single_r);
                tc1 = computings.compute_tc(net, sim, ncnodes_topo[:u_idx], Nc, p0matrix_prevNC_single_r)
                
                # Make node u silent
                #silent_net = make_node_silent(net,u);
                silent_net = dist_specific.make_node_silent(net,u)
                
                # Compute p0 matrix for silent network
                ##p0matrix_silent = compute_p0_matrix(silent_net, ncnodes_topo(1:u_idx-1), R);
                #p0matrix_silent_single_r = compute_p0_matrix_single_r(silent_net, ncnodes_topo(1:u_idx-1), R, r_idx);
                p0matrix_silent_single_r = computings.compute_p0_matrix_single_r(silent_net, ncnodes_topo[:u_idx], R, r_idx)
    
                # Compute t_c(u) with u silent
                ##tc2 = compute_tc(silent_net, sim, ncnodes_topo(1:u_idx-1), Nc, p0matrix_silent(:,r_idx));
                #tc2 = compute_tc(silent_net, sim, ncnodes_topo(1:u_idx-1), Nc, p0matrix_silent_single_r);
                tc2 = computings.compute_tc(silent_net, sim, ncnodes_topo[:u_idx], Nc, p0matrix_silent_single_r)
                
                #delta_tc = tc2 - tc1;
                #delta_tc = tc2 - tc1
                #delta_Nc = gensize * (1/tc1 - 1/tc2);
                delta_Nc = gensize * (1./tc1 - 1./tc2)
                 
                #sanity check
                #if delta_Nc < 0
                if delta_Nc < 0:
                    #delta_Nc = 0;
                    delta_Nc = 0.
                #end
                
                # Compute Nc(u) using equation 8
                # sanity check (avoid division by 0)
                #if abs( p0matrix_prevNC(u,r_idx) - 1) < 1e-6
                if abs( p0matrix_prevNC_single_r[u] - 1) < 1e-6:
                    #Nc(u) = 0;
                    Nc[u] = 0
                else:
                    #if b_o(u) > b_i(u)
                    if b_o[u] > b_i[u]:
                        ##Nc(u) = delta_Nc / (1 - p0matrix_prevNC(u,r_idx)^R(r_idx,u));
                        #Nc(u) = delta_Nc / (1 - p0matrix_prevNC_single_r(u)^R(r_idx,u));
                        Nc[u] = delta_Nc / (1. - p0matrix_prevNC_single_r[u]**R[r_idx,u])
                    else:
                        ##Nc(u) = delta_Nc / ( b_o(u) / b_i(u) * (1 - p0matrix_prevNC(u,r_idx)));
                        #Nc(u) = delta_Nc / ( b_o(u) / b_i(u) * (1 - p0matrix_prevNC_single_r(u)));
                        Nc[u] = delta_Nc / ( b_o[u] / b_i[u] * (1. - p0matrix_prevNC_single_r[u]))
                    #end
                #end
                # sanity check
                #if abs(Nc(u)) < 1e-6
                if abs(Nc[u]) < 1e-6:
                    #Nc(u) = 0;
                    Nc[u] = 0
                #end
            #end
            
            # Compute the average decoding delay tc considering all sources and
            # NC nodes simultaneously with Eq. (16)
            #tc(r_idx) = compute_tc(net, sim, ncnodes, Nc, p0matrix_allncnodes(:,r_idx));
            tc[r_idx] = computings.compute_tc(net, sim, ncnodes, Nc, p0matrix_allncnodes[:,r_idx])
        #end
        
        # Update R
        ##R1 = update_R(R, net, p0matrix_allncnodes, ncnodes, tc);
        ##R = update_R_vectorized(R, net, p0matrix_allncnodes, ncnodes, tc);
        ##R = update_R_vectorized(Rinit, net, p0matrix_allncnodes, ncnodes, tc);
        #R = update_R_vectorized(Rinit, net, p0matrix_prevestim, ncnodes, tc);
        R = updateR.update_R_vectorized(Rinit, net, p0matrix_prevestim, ncnodes, tc)
        ##assert(norm(R1 - R2) < 1e-6);
    
        ##disp(['Average tc = ' num2str(mean(tc))]);
        
        # If tc has Inf values (this may happen for distrib algorithm), ignore
        #  them
        #if norm(tc(isfinite(tc)) - prev_tc(isfinite(tc))) <= precision * norm(tc(isfinite(tc)))
        if numpy.linalg.norm(tc[numpy.isfinite(tc)] - prev_tc[numpy.isfinite(tc)]) <= precision * numpy.linalg.norm(tc[numpy.isfinite(tc)]):
            #converged_tc = true;
            converged_tc = True
        #end
        
        #prev_tc = tc;
        prev_tc = tc.copy()
        
        # Save max tc
        #max_tc(tc > max_tc) = tc(tc > max_tc);
        max_tc[tc > max_tc] = tc[tc > max_tc]
        
        #niter = niter + 1;
        niter = niter + 1
    #end
    
    # Check if converged or not
    #if niter == maxiter
    if niter == maxiter:
        #tc = Inf * ones(1, numel(net.receivers)); # not converged
        #If not converged, don't put Inf, put largest values instead
        #tc = max_tc;
        tc = max_tc.copy()
    #end
    
    return tc



# Algorithm 1 (old ICC version): delay computation for a given network, with given NC nodes
def Algo1_delay_computation_NCasS(net, sim, ncnodes, prev_estim):
    # MATLAB function tc = Algo1_delay_computation_NCasS(net, sim, ncnodes, prev_estim)
    
    #compute_p0_matrix = @compute_p0_matrix;
    #compute_p0_matrix = @compute_p0_matrix_optimized;
    
    # Number of nodes
    #N = size(net.capacities,1);
    #N = net['capacities'].shape[0]
    
    # Nodes' input capacities
    #b_i = sum(net.capacities .* (1-net.errorrates), 1);
    b_i = numpy.sum(net['capacities'] * (1-net['errorrates']), 0)
    # Nodes' output capacities
    #b_o = sum(net.capacities, 2)';
    b_o = numpy.sum(net['capacities'], 1)   
    
    # Generation size
    #gensize = sim.N;
    gensize = sim['N']
    
    # Initialize replication rates
    # Careful to avoid NaN = 0/0
    #R = zeros(1, numel(b_i));
    R = numpy.zeros(b_i.size)
    #R(b_i ~= 0) = b_o(b_i ~= 0) ./ b_i(b_i~=0);
    R[b_i!=0] = b_o[b_i!=0] / b_i[b_i!=0]
    #R = repmat(R, numel(net.receivers), 1);
    R = numpy.tile(R, (net['receivers'].size, 1))
    
    # Compute p0 matrix for all clients at once
    #p0matrix_allncnodes = compute_p0_matrix(net, ncnodes, R);
    p0matrix_allncnodes = computings.compute_p0_matrix(net, ncnodes, R)
        
    # Nic: Python says: init I and tc
    I = numpy.zeros(net['receivers'].size)
    tc = numpy.zeros(net['receivers'].size)
    
    # Repeat for each client r
    #for r_idx = 1:numel(net.receivers)
    for r_idx in range(net['receivers'].size):
        #r = net.receivers(r_idx);
        #r = net['receivers'][r_idx]
        
        # Set of sources and NC nodes
        #SandNC = union(net.sources, ncnodes);
        SandNC = numpy.union1d(net['sources'], ncnodes)
        
        # Eq (5): innovative flow
        #I(r_idx) = sum(b_o(SandNC) .* (1 - p0matrix_allncnodes(SandNC,r_idx)'));
        I[r_idx] = numpy.sum(b_o[SandNC] * (1. - p0matrix_allncnodes[SandNC,r_idx]))
        
        # Delay
        #tc(r_idx) = gensize / I(r_idx);
        tc[r_idx] = gensize / I[r_idx]
    #end

    return tc




#Algorithm 2
def Algo2_Centralized_NC_sel(net, sim, runopts, crit):
    # MATLAB function ncnodes = Algo2_Centralized_NC_sel(net, sim, runopts, crit)
    
    # Number of nodes in the network
    #N = size(net.capacities,1);
    N = net['capacities'].shape[0]
    
    # Initialize list of NC nodes
    #ncnodes = [];
    ncnodes = numpy.array([])
    
    # Find initial (no NC nodes) estimates
    #tc_nonc = Algo1_delay_computation(net, sim, [], []);
    tc_nonc = Algo1_delay_computation(net, sim, numpy.array([]), numpy.array([]))
    #prev_estim.ncnodes = [];
    prev_estim = dict()
    prev_estim['ncnodes'] = numpy.array([])
    #prev_estim.tc = tc_nonc;
    prev_estim['tc'] = tc_nonc.copy()
    
    # Select nodes one by one
    #for i = 1:runopts.nNC
    for i in range(runopts['nNC']):
        
        # Find the set of SF nodes
        #SFnodes = setdiff(net.helpers,ncnodes);
        SFnodes = numpy.setdiff1d(net['helpers'],ncnodes)
        
        # Initialize
        #tc_all = Inf * ones(N, numel(net.receivers));
        tc_all = numpy.Inf * numpy.ones((N, net['receivers'].size))
        #tc = Inf * ones(N,1);
        tc = numpy.Inf * numpy.ones(N)
        #fc_all = zeros(N, numel(net.receivers));
        fc_all = numpy.zeros((N, net['receivers'].size))
        #fc = zeros(N,1);
        fc = numpy.zeros(N)
        
        # For each candidate SF node
        #for u_idx = 1:numel(SFnodes)
        for u_idx in range(SFnodes.size):
            #u = SFnodes(u_idx);
            u = SFnodes[u_idx]
            
            # Turn temporarily u into a NC node
            #ncnodes_temp = union(ncnodes, u);
            ncnodes_temp = numpy.union1d(ncnodes, numpy.array([u]))
            
            # Estimate the average decoding delay at the clients tc (using
            # Algorithm 1) with local statistics 
            #if (~runopts.do_old_icc_version)
            if not runopts['do_old_icc_version']:
                # Journal version
                #tc_all(u,:) = Algo1_delay_computation(net, sim, ncnodes_temp, prev_estim);
                tc_all[u,:] = Algo1_delay_computation(net, sim, ncnodes_temp, prev_estim)
            else:
                # Old ICC conf version
                #tc_all(u,:) = Algo1_delay_computation_NCasS(net, sim, ncnodes_temp, prev_estim);
                tc_all[u,:] = Algo1_delay_computation_NCasS(net, sim, ncnodes_temp, prev_estim)
            #end
            #tc(u) = mean(tc_all(u,:));
            tc[u] = numpy.mean(tc_all[u,:])
            
            # Convert delay to flow
            #fc_all(u,:) = sim.N ./ tc_all(u,:);
            fc_all[u,:] = sim['N'] / tc_all[u,:]
            #fc(u) = sum(fc_all(u,:), 2);
            #fc[u] = numpy.sum(fc_all[u,:], 1)
            fc[u] = numpy.sum(fc_all[u,:])
        #end
        
        # Find node which minimizes total delay / maximizes total flow
        #if strcmp(crit,'delay')
        if crit == 'delay':
            #[min_tc, sel_u] = min(tc);
            #min_tc = numpy.min(tc)
            sel_u  = numpy.argmin(tc)
        #elseif strcmp(crit,'flow')
        elif crit == 'flow':
            #[max_fc, sel_u] = max(fc);
            #max_fc = numpy.max(fc)
            sel_u  = numpy.argmax(fc)
        else:
            #error('Not a valid selection criterion');
            print('Not a valid selection criterion')
            raise
        #end
        
        # Save time estimates to use for next Nc node
        #prev_estim.ncnodes = ncnodes;
        prev_estim['ncnodes'] = ncnodes.copy()
        #prev_estim.tc = tc_all(sel_u,:);
        prev_estim['tc'] = tc_all[sel_u,:]
            
        # Add node to NC list
        #ncnodes = [ncnodes sel_u];
        ncnodes = numpy.hstack((ncnodes, sel_u))
        ## disp(['Selected node ' num2str(sel_u)]);
        print "Selected node no.",i,':',sel_u
        f = open('times.txt', 'a')
        f.write(('{:g}   '*net['receivers'].size+'\n').format(*(tuple(tc_all[sel_u,:]))))
        f.close()
    #end
    
    #disp([datestr(now) ': Selected nodes = ' num2str(ncnodes)]);
    print(str(datetime.datetime.now()) + ': Selected nodes = ' + str(ncnodes))
    
    return ncnodes
        
#Algorithm 3
def Algo3_Semidistributed_NC_sel(net, sim, runopts, crit, ecc):
    # MATLAB function ncnodes = Algo3_Semidistributed_NC_sel(net, sim, runopts, crit, ecc)
    
    #compute_p0_matrix = @compute_p0_matrix;
    #compute_p0_matrix = @compute_p0_matrix_optimized;
    
    # Number of nodes in the network
    #N = size(net.capacities,1);
    N = net['capacities'].shape[0]
    
    # Nodes' input capacities
    #b_i = sum(net.capacities .* (1-net.errorrates), 1);
    b_i = numpy.sum(net['capacities'] * (1-net['errorrates']), 0)
    # Nodes' output capacities
    #b_o = sum(net.capacities, 2)';
    b_o = numpy.sum(net['capacities'], 1)
    
    # Initialize list of NC nodes
    #ncnodes = [];
    ncnodes = numpy.array([])
    
    # Find initial (no NC nodes) estimates
    #tc_nonc = Algo1_delay_computation(net, sim, [], []);
    tc_nonc = Algo1_delay_computation(net, sim, numpy.array([]), numpy.array([]))
    prev_estim = dict()
    #prev_estim.ncnodes = [];
    prev_estim['ncnodes'] = numpy.array([])
    #prev_estim.tc = tc_nonc;
    prev_estim['tc'] = tc_nonc.copy()
    
    # Initialize replication rates
    #R = b_o ./ b_i;
    R = b_o / b_i
    #R = repmat(R, numel(net.receivers), 1);
    R = numpy.tile(R, (net['receivers'].size, 1))
    
    # Update replication rates since initial (no NC nodes) estimates are available
    # Needs to compute p0matrix
    #p0matrix_origR = compute_p0_matrix(net, prev_estim.ncnodes, R);
    p0matrix_origR = computings.compute_p0_matrix(net, prev_estim['ncnodes'], R)
    #R = update_R(R, net, p0matrix_origR, prev_estim.ncnodes, prev_estim.tc);
    R = updateR.update_R(R, net, p0matrix_origR, prev_estim['ncnodes'], prev_estim['tc'])
    
    # Select nodes one by one
    #for i = 1:runopts.nNC
    for i in xrange(runopts['nNC']):
        
        # Find the set of SF nodes
        #SFnodes = setdiff(net.helpers,ncnodes);
        SFnodes = numpy.setdiff1d(net['helpers'],ncnodes)
        
        # Initialize
        #tc_all = Inf * ones(N, numel(net.receivers));
        tc_all = numpy.Inf * numpy.ones((N, net['receivers'].size))
        #tc = Inf * ones(N,1);
        tc = numpy.Inf * numpy.ones(N)
        #fc_all = zeros(N, numel(net.receivers));
        fc_all = numpy.zeros((N, net['receivers'].size))
        #fc = zeros(N,1);
        fc = numpy.zeros(N)
    
        # Find the p0matrix that of the global network, using updated R
        #p0matrix_updR = compute_p0_matrix(net, prev_estim.ncnodes, R);
        p0matrix_updR = computings.compute_p0_matrix(net, prev_estim['ncnodes'], R)
        
        # For each candidate SF node
        #for u_idx = 1:numel(SFnodes)
        for u_idx in xrange(SFnodes.size):
            #u = SFnodes(u_idx);
            u = SFnodes[u_idx]
            
            # Turn temporarily u into a NC node
            #ncnodes_temp = union(ncnodes, u);
            ncnodes_temp = numpy.union1d(ncnodes, numpy.array([u]))
            
            # Estimate the average decoding delay at the clients tc (using
            # Algorithm 1)
            # Use the accurate p0matrix computed once above the loop
            # (i)  Create local network from the neighbourhood around node u
            #[localnet l2g g2l] = create_local_network(net, u, p0matrix_updR, ecc);
            localnet,l2g,g2l = dist_specific.create_local_network(net, u, p0matrix_updR, ecc)
            # TODO localnet to globalnet NC node mapping!
            # Don't forget to translate global node indices of NC nodes to local indices
            #prev_estim_g2l = prev_estim;
            prev_estim_g2l = prev_estim.copy()
            #prev_estim_g2l.ncnodes = g2l(prev_estim.ncnodes);
            prev_estim_g2l['ncnodes'] = numpy.atleast_1d(g2l[numpy.int32(prev_estim['ncnodes'])])
            #ncnodes_temp_g2l = g2l(ncnodes_temp);
            ncnodes_temp_g2l = numpy.atleast_1d(g2l[numpy.int32(ncnodes_temp)])
            #ncnodes_temp_g2l(ncnodes_temp_g2l == 0) = []; # remove NC nodes which are not in local network
            #numpy.delete(ncnodes_temp_g2l[ncnodes_temp_g2l == 0]) # remove NC nodes which are not in local network
            numpy.delete(ncnodes_temp_g2l, ncnodes_temp_g2l == 0) # remove NC nodes which are not in local network
            # (ii) Run Algorithm 1 on local network
            #if (~runopts.do_old_icc_version)
            if not runopts['do_old_icc_version']:
                #tc_all(u,:) = Algo1_delay_computation(localnet, sim, ncnodes_temp_g2l, prev_estim_g2l);
                tc_all[u,:] = Algo1_delay_computation(localnet, sim, ncnodes_temp_g2l, prev_estim_g2l)
            else:
                #tc_all(u,:) = Algo1_delay_computation_NCasS(localnet, sim, ncnodes_temp_g2l, prev_estim_g2l);
                tc_all[u,:] = Algo1_delay_computation_NCasS(localnet, sim, ncnodes_temp_g2l, prev_estim_g2l)
            #end
            #tc(u) = mean(tc_all(u,:));
            tc[u] = numpy.mean(tc_all[u,:])
            
            # Convert delay to flow
            #fc_all(u,:) = sim.N ./ tc_all(u,:);
            fc_all[u,:] = sim['N'] / tc_all[u,:]
            #fc(u) = sum(fc_all(u,:), 2);
            fc[u] = numpy.sum(fc_all[u,:])
        #end
        
        # Find node which minimizes total delay / maximizes total flow
        #if strcmp(crit,'delay')
        if crit == 'delay':
            #[min_tc, sel_u] = min(tc);
            #min_tc = numpy.min(tc)
            sel_u = numpy.argmin(tc)
            #elseif strcmp(crit,'flow')
        elif crit == 'flow':
            #[max_fc, sel_u] = max(fc);
            #max_fc = numpy.max(fc)
            sel_u = numpy.argmax(fc)
        else:
            #error('Not a valid selection criterion');
            print('Not a valid selection criterion')
            raise
        #end
        
        # Save time estimates to use for next NC node
        #prev_estim.ncnodes = ncnodes;
        prev_estim['ncnodes'] = ncnodes.copy()
        #prev_estim.tc = tc_all(sel_u,:);
        prev_estim['tc'] = tc_all[sel_u,:]
        
        # Update replication rates using new delay estimates, to use for next NC node
        #R = update_R(R, net, p0matrix_origR, prev_estim.ncnodes, prev_estim.tc);    
        R = updateR.update_R(R, net, p0matrix_origR, prev_estim['ncnodes'], prev_estim['tc']);    
            
        # Add node to NC list
        #ncnodes = [ncnodes sel_u];
        ncnodes = numpy.hstack((ncnodes, sel_u))
        #disp(['Selected node ' num2str(sel_u)]);
    #end
    
    #disp([datestr(now) ': Selected nodes = ' num2str(ncnodes)]);
    print(str(datetime.datetime.now()) + ': Selected nodes = ' + str(ncnodes))
    
    return ncnodes