import math
import numpy
import scipy.stats.distributions

import globalvars

def compute_tc(net, sim, ncnodes, N1s, p0s):
    # MATLAB function tc = compute_tc(net, sim, ncnodes, N1s, p0s)
    #========================================================
    
    # Number of nodes
    #N = size(net.capacities,1);
    N = net['capacities'].shape[0]
    
    # Output capacity
    #b_o = sum(net.capacities,2);
    b_o = numpy.sum(net['capacities'],1)
    
    # Initialize
    #tcu = Inf * ones(1,N);
    tcu = numpy.Inf * numpy.ones(N)
    
    # For each NC node u, compute tcu
    #for u_idx = 1:numel(ncnodes)
    for u_idx in range(ncnodes.size):
        #u = ncnodes(u_idx);
        u = ncnodes[u_idx]
        
        #tcu(u) = compute_tc_u(net, sim, u, N1s(u), p0s(u));
        tcu[u] = compute_tc_u(net, sim, u, N1s[u], p0s[u])
    #end
    
    # For each source, compute tcu
    #for s_idx = 1:numel(net.sources)
    for s_idx in range(net['sources'].size):
        #s = net.sources(s_idx);
        s = net.sources[s_idx]
        
        # sim.N = generation size
        #tcu(s) = sim.N / (b_o(s) * (1 - p0s(s)));
        tcu[s] = sim['N'] / (b_o[s] * (1. - p0s[s]))
    #end
    
    # Eq. (16):
    #tc = 1 / sum( 1./tcu);
    tc = 1. / numpy.sum( 1./tcu)

    return tc
    
    
def compute_tc_u(net, sim, ncnode, N1, p0):
    # MATLAB function tcu = compute_tc_u(net, sim, ncnode, N1, p0)
    #========================================================
    
    global hbuf
    
    # Compute b_o
    #b_o = sum(net.capacities, 2);
    b_o = numpy.sum(net['capacities'],1)
    
    # Maximum number of lines to compute to
    #maxlines = 1000;
    maxlines = 1000
    
    #if (p0 < 1) && (N1 ~= 0)
    if (p0 < 1) and (N1 != 0):
        
        # Compute nu(u)
        #nu = b_o(ncnode) / N1;
        nu = b_o[ncnode] / N1
    
        #if nu >= 1
        if nu >= 1:
            # Node with limited incoming bandwidth
            # Apply eqs (9) - (12)
            
            # Eq (10) and (11): Compute Pc(n,r) for node u
            #t = compute_pcond_table(sim.N, p0, nu, maxlines);
            t = compute_pcond_table(sim['N'], p0, nu, maxlines)
            
            # Eq (12): Compute mathcalPc(n,r) for node u
            #t1 = compute_pcond_firsttime_table(t, sim.N, p0, nu);
            t1 = compute_pcond_firsttime_table(t, sim['N'], p0, nu)
            
            # Eq(13): Expected value
            #exp_N1 = sum( t1(:, sim.N+1 )' .* (0:(size(t1,1)-1)) );
            exp_N1 = numpy.sum( t1[:, sim['N']+1 ] * numpy.arange(t1.shape[0]) )
            
            # safety check: it might be possible to have exp_N1 = 32 - 1e-x
            #if ((32-1e-2) <= exp_N1) && (exp_N1 < 32)
            if ((globalvars.hbuf-1e-2) <= exp_N1) and (exp_N1 < globalvars.hbuf):
                #exp_N1 = 32;
                exp_N1 = globalvars.hbuf
            #end
            # safety check
            #if exp_N1 < sim.N  || t(end) < (1 - 1e-2)
            if exp_N1 < sim['N'] or t[-1] < (1. - 1e-2):
                #exp_N1 = Inf;   # totallines not enough, t1 not complete
                exp_N1 = numpy.Inf;   # totallines not enough, t1 not complete
            #end
            
            # Eq(14): Expected time
            #tcu = exp_N1 / N1;
            tcu = exp_N1 / N1
    
        else:
            # Node over-provisioned in bandwidth
            
            # Eq (15):
            #tcu = sim.N / (b_o(ncnode) * (1 - p0));
            tcu = sim['N'] / (b_o[ncnode] * (1. - p0));
        #end
    else:
        #tcu = Inf;
        tcu = numpy.Inf
    #end
    
    return tcu
    
    
def compute_pcond_table(N, p0, R, totallines):
    # MATLAB function t = compute_pcond_table(N, p0, R, totallines)
    # computes probability table Pc(r,n) where Pc(r,n) = P(r|n) = probability
    # that the client's rank is r after the NC node has received n innovative
    # packets (N1)
    
    # sanity parameter check
    #if R > 1000  # huge replication rate
    if R > 1000:  # huge replication rate
        #if p0^R < 1e-4
        if p0**R < 1e-4:
            # a packet from virtually any time interval will surely reach the
            # client
            #t = eye(N+1);
            t = numpy.eye(N+1)
        else:
            # too much memory needed to compute this
            #t = zeros(N+1);  # will be detected later in compute_tc_u() and the time will be Inf
            t = numpy.zeros((N+1,N+1))  # will be detected later in compute_tc_u() and the time will be Inf
        #end
        return t
    #end
    
    #t = zeros(totallines, N+1);
    t = numpy.zeros((totallines, N+1))
    
    # P(0|0) = 1, i.e. before the NC received its first packet, the client's
    # rank is surely 0
    #t(1,1) = 1;
    t[0,0] = 1
    
    #R_low = floor(R);
    #R_high = floor(R+1);
    #p_low  = floor(R+1) - R;
    #p_high = R - floor(R);
    R_low = math.floor(R)
    R_high = math.floor(R+1)
    p_low  = math.floor(R+1) - R
    p_high = R - math.floor(R)
    
    ## store binomial pdf values in a table, to avoid re-computing them many
    ## times
    ## 'low' is for floor(R), 'high' is for ceil(R)
    ##bino_table_low    = binopdf(1:R_low, R_low, 1-p0);
    ##bino_table_high   = binopdf(1:R_high, R_high, 1-p0);
    #bino_table_zero_low    = binopdf(0:R_low, R_low, 1-p0);
    #bino_table_zero_high   = binopdf(0:R_high, R_high, 1-p0);    
    bino_table_zero_low    = scipy.stats.distributions.binom.pmf(numpy.arange(1,(R_low+1)), R_low, 1.-p0)
    bino_table_zero_high   = scipy.stats.distributions.binom.pmf(numpy.arange(1,(R_high+1)), R_high, 1.-p0)
    
    #bino_table_zero_low_convmtx  = convmtx(bino_table_zero_low,  N + 1);
    #bino_table_zero_low_convmtx_summed  = bino_table_zero_low_convmtx;
    #bino_table_zero_low_convmtx_summed(:,N+1) = sum(bino_table_zero_low_convmtx_summed(:,(N+1):(N + numel(bino_table_zero_low))), 2);
    #bino_table_zero_low_convmtx_summed = bino_table_zero_low_convmtx_summed(:,1:(N+1));
    bino_table_zero_low_convmtx  = convmtx(bino_table_zero_low,  N + 1)
    bino_table_zero_low_convmtx_summed  = bino_table_zero_low_convmtx
    bino_table_zero_low_convmtx_summed[:,N] = numpy.sum(bino_table_zero_low_convmtx_summed[:,N:(N + bino_table_zero_low.size())], 1)
    bino_table_zero_low_convmtx_summed = bino_table_zero_low_convmtx_summed[:,:(N+1)]

    #bino_table_zero_high_convmtx = convmtx(bino_table_zero_high, N + 1);
    #bino_table_zero_high_convmtx_summed  = bino_table_zero_high_convmtx;
    #bino_table_zero_high_convmtx_summed(:,N+1) = sum(bino_table_zero_high_convmtx_summed(:,(N+1):(N + numel(bino_table_zero_high))), 2);
    #bino_table_zero_high_convmtx_summed = bino_table_zero_high_convmtx_summed(:,1:(N+1));
    bino_table_zero_high_convmtx = convmtx(bino_table_zero_high, N + 1)
    bino_table_zero_high_convmtx_summed  = bino_table_zero_high_convmtx;
    bino_table_zero_high_convmtx_summed[:,N] = numpy.sum(bino_table_zero_high_convmtx_summed[:,N:(N + bino_table_zero_high.size())], 1)
    bino_table_zero_high_convmtx_summed = bino_table_zero_high_convmtx_summed[:,:(N+1)]
    
    #bino_table_zero_both_convmtx_summed_weightedsum = p_low * bino_table_zero_low_convmtx_summed + p_high * bino_table_zero_high_convmtx_summed;
    bino_table_zero_both_convmtx_summed_weightedsum = p_low * bino_table_zero_low_convmtx_summed + p_high * bino_table_zero_high_convmtx_summed
    
    ## store the regularized incomplete beta function, i.e. ...
    ## 'low' is for floor(R), 'high' is for ceil(R)
    ##k = 1:R_low;
    ##betainc_table_low  = betainc(p0, R_low-k+1, k, 'upper');
    ##k = 1:R_high;
    ##betainc_table_high = betainc(p0, R_high-k+1, k, 'upper');
    
    #for i = 1:N
    for i in range(N):
        
        #     t(i+1, 1) = p_low * t(i,1) * p0^R_low + p_high * t(i,1) * p0^R_high;
        #     
        #     for j = 1:(i-1)
        #         
        #         # N = i, r = j
        #         limit = min(j, R);
        #         k = 1:limit;
        #         t(i+1,j+1) = p_low  * ( t(i, j+1) * p0^R_low  + sum( t(i+1-1, j+1-k) .*  bino_table_low(k)) ) + ...
        #                      p_high * ( t(i, j+1) * p0^R_high + sum( t(i+1-1, j+1-k) .*  bino_table_high(k)) );
        #     end
        #     
        #     # for j = i, i.e. r = N
        #     j = i;
        #     limit = min(j, R);
        #     k = 1:limit;
        #     t(i+1,j+1) = p_low  * ( t(i, j+1) * p0^R_low  + sum( t(i+1-1, j+1-k) .*  betainc_table_low(k)) ) + ...
        #                  p_high * ( t(i, j+1) * p0^R_high + sum( t(i+1-1, j+1-k) .*  betainc_table_high(k)) );
        
        #t2_1 = conv(t(i,:), bino_table_zero_low);
        #t2_1(i+1) = sum(t2_1((i+1):numel(t2_1)));
        #t2_1 = t2_1(1:(i+1));
        ##t2_1 = padarray(t2_1, [0 33 - i - 1] , 0,'post');
        #t2_1 = [t2_1 zeros(1, N - i)];
        t2_1 = numpy.convolve(t[i,:], bino_table_zero_low)
        t2_1[i+1] = numpy.sum(t2_1[(i+1):])
        t2_1 = t2_1[:(i+2)]
        t2_1 = numpy.concatenate((t2_1, numpy.zeros(N - i)))

        #t2_2 = conv(t(i,:), bino_table_zero_high);
        #t2_2(i+1) = sum(t2_2((i+1):numel(t2_2)));
        #t2_2 = t2_2(1:(i+1));
        ##t2_2 = padarray(t2_2, [0 33 - i - 1] , 0,'post');
        #t2_2 = [t2_2 zeros(1, N - i)];
        t2_2 = numpy.convolve(t[i,:], bino_table_zero_high)
        t2_2[i+1] = numpy.sum(t2_2[(i+1):])
        t2_2 = t2_2[:(i+2)]
        t2_2 = numpy.concatenate((t2_2, numpy.zeros( N - i)))
    
        #t2 = p_low * t2_1 + p_high * t2_2;
        t2 = p_low * t2_1 + p_high * t2_2
        #t(i+1,:) = t2;
        t[i+1,:] = t2
    
        #     if ~all(t(i+1, :) == t2)
        #         assert(all(abs(t(i+1, :) - t2) < 1e-6));
        #     end
    #end
    
    
    #for i = (N+1):(totallines-1)
    for i in range(N+1,totallines):
        #     for j = 1:(N-1)
        #         limit = min(j, R);
        #         k = 1:limit;
        #         t(i+1,j+1) = p_low  * ( t(i, j+1) * p0^R_low  + sum( t(i+1-1, j+1-k) .*  bino_table_low(k)) )+ ...
        #                      p_high * ( t(i, j+1) * p0^R_high + sum( t(i+1-1, j+1-k) .*  bino_table_high(k)) );
        #     end
        #     
        #     # for j = i
        #     j = N;
        #     limit = min(j, R);
        #     k = 1:limit;
        #     # t(i+1,j+1) = t(i, j+1) * 1 + ...
        #     t(i+1,j+1) = p_low  * ( t(i, j+1) + sum( t(i+1-1, j+1-k) .*  betainc_table_low(k)) ) + ...
        #                  p_high * ( t(i, j+1) + sum( t(i+1-1, j+1-k) .*  betainc_table_high(k)) );
        #              
                     
        #     #t2_1 = conv(t(i,:), bino_table_zero_low);
        #     t2_1 = t(i,:) * bino_table_zero_low_convmtx_summed;
        #     #t2_1(N+1) = sum(t2_1((N+1):numel(t2_1)));
        #     #t2_1 = t2_1(1:(N+1));
        #     
        #     #t2_2 = conv(t(i,:), bino_table_zero_high);
        #     t2_2 = t(i,:) * bino_table_zero_high_convmtx_summed;
        #     #t2_2(N+1) = sum(t2_2((N+1):numel(t2_2)));
        #     #t2_2 = t2_2(1:(N+1));
        # 
        #     t2 = p_low * t2_1 + p_high * t2_2;
        #     t(i+1,:) = t2;
        #t(i+1,:) = t(i,:) * bino_table_zero_both_convmtx_summed_weightedsum;
        t[i+1,:] = numpy.dot(t[i,:] , bino_table_zero_both_convmtx_summed_weightedsum)
        
        #     if ~all(t(i+1, :) == t2)
        #         assert(all(abs(t(i+1, :) - t2) < 1e-6));
        #     end
        
        # check if not almost finished, to reduce computing
        #if mod(i+1,20) == 0 # check every 20 lines
        if (i+1)%20 == 0: # check every 20 lines
            #if t(i+1, N+1) > (1-1e-3)
            if t[i+1, N+1] > (1-1e-3):
                break
            #end
        #end
    #end
    
    #t = t(1:(i+1),:);
    t = t[:(i+2),:]
    
    # if ~all( abs(sum(t, 2) - ones(totallines,1)) < 1e-4)
    #     assert(all( abs(sum(t, 2) - ones(totallines,1)) < 1e-4));
    # end
    
    return t
    
def compute_pcond_firsttime_table(t, N, p0, R):
    # MATLAB function t1 = compute_pcond_firsttime_table(t, N, p0, R)
    # computes probability table t1(n,r) where t1(r,n) = probability that the
    # client arrives for_the_first_time at rank r after the NC node has
    # received n packets (N1)
    
    # parameter checking
    # it is possible to get p0 = 1 + 1e-16, and then betainc() throws an error
    #if p0 > 1
    if p0 > 1:
        #if abs(p0 - 1) < 1e-6
        if abs(p0 - 1) < 1e-6:
            #p0 = 1;
            p0 = 1
        #end
    #end
    
    # sanity parameter check
    #if R > 1000  # huge replication rate
    if R > 1000:  # huge replication rate
        
        #if p0^R < 1e-4
        if p0**R < 1e-4:
            # a packet from virtually any time interval will surely reach the
            # client
            # t is eye(N+1)
            #t1 = eye(N+1);
            t1 = numpy.eye(N+1)
            return t1
        else:
            #if R > 20000
            if R > 20000:
                # too much memory & time needed to compute this
                #t1 = zeros(N+1);  # will be detected later in compute_tc_u() and the time will be Inf
                t1 = numpy.zeros((N+1,N+1))  # will be detected later in compute_tc_u() and the time will be Inf
                return t1
            #else:
                # can still handle it, do nothing
            #end
        #end
    #end
    
    #totallines = size(t, 1);
    totallines = t.shape[0]
    #t1 = zeros(totallines,N+1);
    t1 = numpy.zeros((totallines,N+1))
    
    # t1(0,0) = 1, i.e. before the NC received its first packet, the client's
    # rank is surely 0
    #t1(1,1) = 1;
    t1[0,0] = 1
    
    #R_low = floor(R);
    #R_high = floor(R+1);
    #p_low  = floor(R+1) - R;
    #p_high = R - floor(R);
    R_low = math.floor(R)
    R_high = math.floor(R+1)
    p_low  = math.floor(R+1) - R
    p_high = R - math.floor(R)
    ## store binomial pdf values in a table, to avoid re-computing them many
    ## times
    ## 'low' is for floor(R), 'high' is for ceil(R)
    ##bino_table_low    = binopdf(1:R_low, R_low, 1-p0);
    ##bino_table_high   = binopdf(1:R_high, R_high, 1-p0);
    
    # store the regularized incomplete beta function, i.e. ...
    # 'low' is for floor(R), 'high' is for ceil(R)
    #k = 1:R_low;
    #betainc_table_low  = betainc(p0, R_low-k+1, k, 'upper');
    #k = 1:R_high;
    #betainc_table_high = betainc(p0, R_high-k+1, k, 'upper');
    k = numpy.arange(1,R_low+1)
    # betainc(... 'upper') = 1 - betainc(...)
    betainc_table_low  = 1 - scipy.special.betainc(p0, R_low-k+1, k)
    k = numpy.arange(1,R_high+1)
    betainc_table_high = 1 - scipy.special.betainc(p0, R_high-k+1, k)
    
    # If numel(betainc_table_low) > 2000, do row by row processing (slower)
    # Otherwise compute convolution matrix (faster, more moemory)
    
    #if numel(betainc_table_low) > 2000
    if betainc_table_low.size > 2000:
        #-------------------------------------------------
        # Implement convolutions using row by row processing
        #-------------------------------------------------
        #for i = 1:N
        for i in range(N):
        
            #     t1(i+1, 1) = 0;
            #     
            #     for j = 1:i
            #         
            # #         # N = i, r = j
            #         limit = min(j, R);
            #         k = 1:limit;
            #         # Replace bino_table with beta_inc table:
            #         # probab that at step 2 we get the first packet is p0^R * probab
            #         # that _at_least_ 1 packet arrives, not exactly 1 packet 
            #         t1(i+1,j+1) = sum( t(i+1-1, j+1-k) .*  ( p_low * betainc_table_low(k) + p_high * betainc_table_high(k) ));
            #     end
        
            #t2_1 = [0 conv(t(i, 1:i), betainc_table_low)];
            t2_1 = numpy.hstack((0, numpy.convolute(t[i, :(i+1)], betainc_table_low)))
            #t2_1 = t2_1(1:(i+1));
            t2_1 = t2_1[:(i+2)]
            #t2_1 = padarray(t2_1, [0 33 - i - 1] , 0,'post');
            t2_1 = numpy.hstack((t2_1, numpy.zeros(t2_1.shape[0], 33-i-2)))
    
            #t2_2 = [0 conv(t(i, 1:i), betainc_table_high)];
            t2_2 = numpy.hstack((0, numpy.convolute(t[i, :(i+1)], betainc_table_high)))
            #t2_2 = t2_2(1:(i+1));
            t2_2 = t2_2[:(i+2)]
            #t2_2 = padarray(t2_2, [0 33 - i - 1] , 0,'post');
            t2_2 = numpy.hstack((t2_2, numpy.zeros(t2_2.shape[0], 33-i-2)))
    
            #t2 = p_low * t2_1 + p_high * t2_2;
            t2 = p_low * t2_1 + p_high * t2_2
            #t1(i+1,:) = t2;
            t1[i+1,:] = t2
            
            #     if ~all(t1(i+1, :) == t2)
            #         assert(all(abs(t1(i+1, :) - t2) < 1e-6));
            #     end
        #end
    
        #for i = (N+1):(totallines-1)
        for i in range((N+1),totallines-1):
            #     for j = 1:N
            #         limit = min(j, R);
            #         k = 1:limit;
            #         t1(i+1,j+1) = sum( t(i+1-1, j+1-k) .*  ( p_low * betainc_table_low(k) + p_high * betainc_table_high(k) ));
            #     end
            
            #t2_1 = [0 conv(t(i, 1:N), betainc_table_low)];
            t2_1 = numpy.hstack((0, numpy.convolute(t[i, :N], betainc_table_low)))
            #t2_1 = t2_1(1:(N+1));
            t2_1 = t2_1[:(N+1)]
    
            #t2_2 = [0 conv(t(i, 1:N), betainc_table_high)];
            t2_2 = numpy.hstack((0, numpy.convolute(t[i, :N], betainc_table_high)))
            #t2_2 = t2_2(1:(N+1));
            t2_2 = t2_2[:(N+1)]
    
            #t2 = p_low * t2_1 + p_high * t2_2;
            t2 = p_low * t2_1 + p_high * t2_2
            #t1(i+1,:) = t2;
            t1[i+1,:] = t2
            
            #     if mod(i+1,20) == 0
            #         if t1(i+1,N+1) < 1e-6
            #             break
            #         end
            #     end
            
        #end
    
    else:
        #-------------------------------------------------
        # Implement convolutions using convolution matrix
        #-------------------------------------------------
        
        # create convolution matrix
        # for a row vector r, r * cm = conv(r, betainc_table_low)
        #cm_low  = convmtx(betainc_table_low, N);
        cm_low  = convmtx(betainc_table_low, N)
        #cm_high = convmtx(betainc_table_high, N);
        cm_high = convmtx(betainc_table_high, N)
        
        
        # ge lower diagonal part of t, without the last column
        #tl = tril(t(:,1:N), 0);
        tl = numpy.tril(t[:,:N], 0)
    
        # convolve t1 with betainc_table_low by multipliying with conv matrix
        #res_low = tl * cm_low;
        res_low = numpy.dot(tl , cm_low)
        #res_low = [zeros(totallines,1)  res_low(:,1:N)];
        res_low = numpy.hstack((numpy.zeros(totallines,1), res_low[:,:N]))
    
        # convolve t1 with betainc_table_high by multipliying with conv matrix
        #res_high = tl * cm_high;
        res_high = numpy.dot(tl , cm_high)
        #res_high = [zeros(totallines,1)  res_high(:,1:N)];
        res_high = numpy.hstack((numpy.zeros(totallines,1), res_high[:,:N]))
    
        # add
        #res = p_low * res_low + p_high * res_high;
        res = p_low * res_low + p_high * res_high
    
        # add one row on top, remove last row
        #res = [1 zeros(1,N); res(1:totallines-1,:)];
        res = numpy.vstack((numpy.hstack((1, numpy.zeros(N))) , res[:totallines-1,:]))

        # remove excess values from convolution
        #res = tril(res);
        res = numpy.tril(res)
    
        #t1 = res;
        t1 = res
    #end
    
    #-------------------------------------------------
    # Checking
    #-------------------------------------------------
    # if ~all( all( abs(t1 - res) < 1e-6))
    #     assert(all( all( abs(t1 - res) < 1e-6)));
    # end
    
    # if ~all( abs(sum(t1, 1) - ones(1, N+1)) < 1e-3)
    #     assert(all( abs(sum(t1, 1) - ones(1, N+1)) < 1e-3));
    # end    

    return t1
    
def convmtx(h,N):
    # Returns 
    
    # fliplr: numpy.fliplr() also exists
    #hinv = h[::-1]
    return scipy.linalg.toeplitz(numpy.hstack((h[0],numpy.zeros(N-1))), numpy.hstack((h, numpy.zeros(N-1))))
    