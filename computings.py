import math
import numpy
import scipy.stats.distributions
import scipy.weave

import os
os.environ['PATH'] = r'C:\MinGW\bin;' + os.environ['PATH']

#import topsort
import graph
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
        s = net['sources'][s_idx]
        
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
            
            #-------------------------------------------------
            # First version, like Matlab original
            # ------------------------------------------------
            
            # Eq (10) and (11): Compute Pc(n,r) for node u
            #t = compute_pcond_table(sim.N, p0, nu, maxlines);
            t = compute_pcond_table(sim['N'], p0, nu, maxlines)
            
            # Eq (12): Compute mathcalPc(n,r) for node u
            #t1 = compute_pcond_firsttime_table(t, sim.N, p0, nu);
            t1 = compute_pcond_firsttime_table(t, sim['N'], p0, nu)
            
            # Eq(13): Expected value
            #exp_N1 = sum( t1(:, sim.N+1 )' .* (0:(size(t1,1)-1)) );
            exp_N1 = numpy.sum( t1[:, sim['N']] * numpy.arange(t1.shape[0]) )
            
            # safety check: it might be possible to have exp_N1 = 32 - 1e-x
            #if ((32-1e-2) <= exp_N1) && (exp_N1 < 32)
            if ((globalvars.hbuf-1e-2) <= exp_N1) and (exp_N1 < globalvars.hbuf):
                #exp_N1 = 32;
                exp_N1 = globalvars.hbuf
            #end
            
            # safety check
            #if exp_N1 < sim.N  || t(end) < (1 - 1e-2)
            if exp_N1 < sim['N'] or t[-1,-1] < (1. - 1e-2):
                #exp_N1 = Inf;   # totallines not enough, t1 not complete
                exp_N1 = numpy.Inf;   # totallines not enough, t1 not complete
            #end
            
            #-------------------------------------------------
            # Second version, optimized
            # Replaced with a single function compute_exp_N1(), no more tables
            #-------------------------------------------------

            #exp_N1_v2, debugargs = compute_exp_N1(sim['N'], p0, nu, maxlines)
            exp_N1_v2 = compute_exp_N1(sim['N'], p0, nu, maxlines)

            # safety check: it might be possible to have exp_N1 = 32 - 1e-x
            #if ((32-1e-2) <= exp_N1) && (exp_N1 < 32)
            if ((globalvars.hbuf-1e-2) <= exp_N1_v2) and (exp_N1_v2 < globalvars.hbuf):
                #exp_N1 = 32;
                exp_N1_v2 = globalvars.hbuf
            #end

            # safety check
            #if exp_N1 < sim.N  || t(end) < (1 - 1e-2)
            #if exp_N1 < sim['N'] or t[-1,-1] < (1. - 1e-2):
            if exp_N1_v2 < sim['N']:
                #exp_N1 = Inf;   # totallines not enough, t1 not complete
                exp_N1_v2 = numpy.Inf;   # totallines not enough, t1 not complete
            #end

            #-------------------------------------------------
            # Third version, optimized
            # Implemented single function compute_exp_N1() in C
            #-------------------------------------------------

            exp_N1_v3 = compute_exp_N1_C(sim['N'], p0, nu, maxlines)

            # safety check: it might be possible to have exp_N1 = 32 - 1e-x
            #if ((32-1e-2) <= exp_N1) && (exp_N1 < 32)
            if ((globalvars.hbuf-1e-2) <= exp_N1_v3) and (exp_N1_v3 < globalvars.hbuf):
                #exp_N1 = 32;
                exp_N1_v3 = globalvars.hbuf
            #end

            # safety check
            #if exp_N1 < sim.N  || t(end) < (1 - 1e-2)
            #if exp_N1 < sim['N'] or t[-1,-1] < (1. - 1e-2):
            if exp_N1_v3 < sim['N']:
                #exp_N1 = Inf;   # totallines not enough, t1 not complete
                exp_N1_v3 = numpy.Inf;   # totallines not enough, t1 not complete
            #end

            #-------------------------------------------------
            # Checking
            #-------------------------------------------------
            if (exp_N1 == numpy.Inf and exp_N1_v2 != numpy.Inf) or \
               (exp_N1 == numpy.Inf and exp_N1_v3 != numpy.Inf) or \
               (exp_N1 != numpy.Inf and exp_N1_v2 == numpy.Inf) or \
               (exp_N1 != numpy.Inf and exp_N1_v3 == numpy.Inf) or \
               (exp_N1_v2 == numpy.Inf and exp_N1_v3 != numpy.Inf) or \
               (exp_N1_v2 != numpy.Inf and exp_N1_v3 == numpy.Inf):
                   assert(False)
            elif exp_N1 == numpy.Inf and exp_N1_v2 == numpy.Inf and exp_N1_v3 == numpy.Inf:
                assert(True)
            else:
                assert(abs(exp_N1 - exp_N1_v2) / exp_N1 < 1e-10)
                assert(abs(exp_N1_v2 - exp_N1_v3) < 1e-15)

            #-------------------------------------------------
            
            exp_N1 = exp_N1_v3
            
            # Eq(14): Expected time
            #tcu = exp_N1 / N1;
            tcu = exp_N1 / N1

            #if exp_N1_v2 < sim['N']:
            #    exp_N1_v2 = numpy.Inf;   # totallines not enough, t1 not complete
            
            #tcu_v2 = exp_N1_v2 / N1
            #if tcu != numpy.Inf and tcu_v2 != numpy.Inf :
            #    assert(abs(tcu - tcu_v2) / tcu < 1e-10)
            #if tcu ==numpy.Inf and tcu_v2 != numpy.Inf:
            #    assert(False)
            #if tcu !=numpy.Inf and tcu_v2 == numpy.Inf:
            #    assert(False)
                
    
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
    bino_table_zero_low    = scipy.stats.distributions.binom.pmf(numpy.arange(0,(R_low+1)), R_low, 1.-p0)
    bino_table_zero_high   = scipy.stats.distributions.binom.pmf(numpy.arange(0,(R_high+1)), R_high, 1.-p0)
    
    #bino_table_zero_low_convmtx  = convmtx(bino_table_zero_low,  N + 1);
    #bino_table_zero_low_convmtx_summed  = bino_table_zero_low_convmtx;
    #bino_table_zero_low_convmtx_summed(:,N+1) = sum(bino_table_zero_low_convmtx_summed(:,(N+1):(N + numel(bino_table_zero_low))), 2);
    #bino_table_zero_low_convmtx_summed = bino_table_zero_low_convmtx_summed(:,1:(N+1));
    bino_table_zero_low_convmtx  = convmtx(bino_table_zero_low,  N + 1)
    bino_table_zero_low_convmtx_summed  = bino_table_zero_low_convmtx
    bino_table_zero_low_convmtx_summed[:,N] = numpy.sum(bino_table_zero_low_convmtx_summed[:,N:(N + bino_table_zero_low.size)], 1)
    bino_table_zero_low_convmtx_summed = bino_table_zero_low_convmtx_summed[:,:(N+1)]

    #bino_table_zero_high_convmtx = convmtx(bino_table_zero_high, N + 1);
    #bino_table_zero_high_convmtx_summed  = bino_table_zero_high_convmtx;
    #bino_table_zero_high_convmtx_summed(:,N+1) = sum(bino_table_zero_high_convmtx_summed(:,(N+1):(N + numel(bino_table_zero_high))), 2);
    #bino_table_zero_high_convmtx_summed = bino_table_zero_high_convmtx_summed(:,1:(N+1));
    bino_table_zero_high_convmtx = convmtx(bino_table_zero_high, N + 1)
    bino_table_zero_high_convmtx_summed  = bino_table_zero_high_convmtx;
    bino_table_zero_high_convmtx_summed[:,N] = numpy.sum(bino_table_zero_high_convmtx_summed[:,N:(N + bino_table_zero_high.size)], 1)
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
        t2_1 = numpy.concatenate((t2_1, numpy.zeros(N - i - 1)))

        #t2_2 = conv(t(i,:), bino_table_zero_high);
        #t2_2(i+1) = sum(t2_2((i+1):numel(t2_2)));
        #t2_2 = t2_2(1:(i+1));
        ##t2_2 = padarray(t2_2, [0 33 - i - 1] , 0,'post');
        #t2_2 = [t2_2 zeros(1, N - i)];
        t2_2 = numpy.convolve(t[i,:], bino_table_zero_high)
        t2_2[i+1] = numpy.sum(t2_2[(i+1):])
        t2_2 = t2_2[:(i+2)]
        t2_2 = numpy.concatenate((t2_2, numpy.zeros( N - i - 1)))
    
        #t2 = p_low * t2_1 + p_high * t2_2;
        t2 = p_low * t2_1 + p_high * t2_2
        #t(i+1,:) = t2;
        t[i+1,:] = t2
    
        #     if ~all(t(i+1, :) == t2)
        #         assert(all(abs(t(i+1, :) - t2) < 1e-6));
        #     end
    #end
    
    
    #for i = (N+1):(totallines-1)
    for i in range(N,totallines-1):
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
        k = numpy.arange(1,R_low+1)
        # betainc(... 'upper') = 1 - betainc(...)
        # Scipy betainc() takes p as the last argument!
        betainc_table_low  = 1 - scipy.special.betainc(R_low-k+1, k, p0)
        k = numpy.arange(1,R_high+1)
        betainc_table_high = 1 - scipy.special.betainc(R_high-k+1, k, p0)
    
        cm_low  = convmtx(numpy.hstack((betainc_table_low,numpy.array([0]))), N)
        cm_high = convmtx(betainc_table_high, N)
        cm_all_w = p_low * cm_low + p_high * cm_high #t2_2 = conv(t(i,:), bino_table_zero_high);
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
        # BUG: Because Matlab is 1-based, while it checks at lines 20, 40 etc,
        #  this actually means in Python 19, 39, etc. So check (i+2)%20 instead
        #if (i+1)%20 == 0: # check every 20 lines
        if (i+2)%20 == 0: # check every 20 lines
            #if t(i+1, N+1) > (1-1e-3)
            if t[i+1, N] > (1-1e-3):
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
    # Scipy betainc() takes p as the last argument!
    betainc_table_low  = 1 - scipy.special.betainc(R_low-k+1, k, p0)
    k = numpy.arange(1,R_high+1)
    betainc_table_high = 1 - scipy.special.betainc(R_high-k+1, k, p0)
    
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
            t2_1 = numpy.hstack((t2_1, numpy.zeros((t2_1.shape[0], 33-i-2))))
    
            #t2_2 = [0 conv(t(i, 1:i), betainc_table_high)];
            t2_2 = numpy.hstack((0, numpy.convolute(t[i, :(i+1)], betainc_table_high)))
            #t2_2 = t2_2(1:(i+1));
            t2_2 = t2_2[:(i+2)]
            #t2_2 = padarray(t2_2, [0 33 - i - 1] , 0,'post');
            t2_2 = numpy.hstack((t2_2, numpy.zeros((t2_2.shape[0], 33-i-2))))
    
            #t2 = p_low * t2_1 + p_high * t2_2;
            t2 = p_low * t2_1 + p_high * t2_2
            #t1(i+1,:) = t2;
            t1[i+1,:] = t2
            
            #     if ~all(t1(i+1, :) == t2)
            #         assert(all(abs(t1(i+1, :) - t2) < 1e-6));
            #     end
        #end
    
        #for i = (N+1):(totallines-1)
        for i in range(N,totallines-1):
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
        res_low = numpy.hstack((numpy.zeros((totallines,1)), res_low[:,:N]))
    
        # convolve t1 with betainc_table_high by multipliying with conv matrix
        #res_high = tl * cm_high;
        res_high = numpy.dot(tl , cm_high)
        #res_high = [zeros(totallines,1)  res_high(:,1:N)];
        res_high = numpy.hstack((numpy.zeros((totallines,1)), res_high[:,:N]))
    
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
    
    
    

def compute_p0_matrix(net, ncnodes, R):
  N = net['capacities'].shape[0]
  p0matrix = numpy.zeros((N, net['receivers'].size))
  for r_index in range(net['receivers'].size):
    p0matrix[:,r_index] = compute_p0_matrix_single_r(net, ncnodes, R, r_index)
  return p0matrix

def compute_p0_matrix_single_r(net, ncnodes, R, r_index):
  # MATLAB: function p0matrix = compute_p0_matrix(net, ncnodes, R)
  # Computes the loss probability p0 for every node and for every receiver
  #  p0(node, receiver) = the probability that a packet sent by node 'node' will not reach receiver 'receiver', even if it is duplicated along the way
  #  (i.e. neither the sent packet nor any of its possible duplicates won't reach the receiver)
  #
  # Inputs:
  #       net
  #       ncnodes      = vector of NC nodes (to exclude form normal
  #                          computations)
  # Outputs:
  #     p0matrix = array holding the p0 value of every node, for every
  #     receiver
  #
  # Nicolae Cleju, EPFL, 2008/2009, TUIASI, 2009/2010
  #==========================================================================
  
  #-------------------------------------------
  # Init
  #-------------------------------------------
  
  # Total number of nodes
  #N = size(net.capacities,1);
  N = net['capacities'].shape[0]

  # Number of input and output packets
  #b_i = sum(net.capacities .* (1-net.errorrates), 1);
  #b_o = sum(net.capacities, 2)';
  b_i = numpy.sum(net['capacities'] * (1 - net['errorrates']), 0)
  b_o = numpy.sum(net['capacities'], 1)

  # Hold the transpose of capacities for faster access
  capT = net['capacities'].T

  # The topological order
  #order = topological_order(sparse(net.capacities));
  #xs, ys = numpy.nonzero(net['capacities'])
  #order = graph.topological_sort([(xs[i], ys[i]) for i in range(xs.size)])
  order = graph.topological_sort(net['capacities'])

  # Init output matrix
  #p0matrix = zeros(N, numel(net.receivers));
  #p0matrix = zeros(N, 1);
  #p0matrix = numpy.zeros(N, net['rceivers'].size)
  p0matrix = numpy.zeros(N)

  # Holds floor(replication rate)
  #M = zeros(1,N);
  M = numpy.zeros(N)

  # Holds the probability that an input packet is replicated M times in node n
  #p_rep_M = zeros(1,N);
  p_rep_M = numpy.zeros(N)

  # Holds the probability that an input packet is not lost in a buffer overflow in node n
  #p_ov = zeros(1,N);
  p_ov = numpy.zeros(N)

  # for each client c seperately
  #for r_index = 1:numel(net.receivers)
  #r = net.receivers(r_index);
  #for r_index in range(net['receivers'].size):
  r = net['receivers'][r_index]

  # Prepare data
  # For every node in topo order
  #for n = order'
  for n in order:

      # If not source
      #if b_i(n) ~= 0
      if b_i[n] != 0:
          
          # M = floor(replication rate)
          #M(n) = floor(R(r_index,n));
          M[n] = math.floor(R[r_index,n])

          # Probability of M(n) - for linear interpolation
          #p_rep_M(n) = 1 - (  R(r_index,n) - M(n)  );
          p_rep_M[n] = 1. - (  R[r_index,n] - M[n]  )

          # Probability of not buffer overflow
          #if b_o(n) > b_i(n)
          if b_o[n] > b_i[n]:
              #p_ov(n) = 0;       # No overflow
              p_ov[n] = 0.
          else:
              #p_ov(n) = 1 - b_o(n) /b_i(n) ;    # Buffer overflow
              p_ov[n] = 1. - b_o[n] / b_i[n]
          #end
      #end
  #end
  
  # restrict list only to nodes ahead of the receiver (without the receiver itself)
  #partial_limit = find(order == r, 1) - 1;
  partial_limit = numpy.nonzero(order == r)[0] - 1;
  
  # for the receiver, initialize p0 to 0
  #p0matrix(r, r_index) = 0; #0# losses, 100# one copy
  #p0matrix(r) = 0; %0% losses, 100% one copy
  #p0matrix[r, r_index] = 0.
  p0matrix[r] = 0. #0% losses, 100% one copy
  
  # for all nodes after the receiver (which cannot reach the receiver) set the p0 to 1 (100# loss)
  #for n_index = (find(order == r, 1) + 1) : N
  for n_index in range((numpy.nonzero(order == r)[0] + 1), N):
      #n = order(n_index);
      n = order[n_index]
      #p0matrix(n, r_index) = 1;
      #p0matrix(n) = 1;
      #p0matrix[n, r_index] = 1.
      p0matrix[n] = 1.
  #end
  
  # go only from the receiver r backwards
  #for u_index = partial_limit:-1:1
  for u_index in range(partial_limit,-1,-1):  # need to include the end, 0
      #u = order(u_index);
      u = order[u_index]
      
      # find child nodes of node u
      #children = find(net.capacities(u,:));
      #children = find(capT(:,u));
      #children = numpy.nonzero(net['capacities'][u,:])
      children = numpy.nonzero(capT[:,u])[0]
      
      # if u has no children, set p0 to 1 (100# loss)
      #  this may happen if u is another receiver sitting before the
      #  current one in the topo order
      #if (isempty(children))
      if children.size == 0:
          #p0matrix(u,r_index) = 1;
          #p0matrix(u) = 1;
          #p0matrix[u,r_index] = 1.
          p0matrix[u] = 1.
      else:
          
          # initialize sum of children probabilities
          #p0_partsum = 0;
          p0_partsum = 0.
          
          # for each child
          #for child_index = 1:numel(children)
          for child_index in range(children.size):
             #child = children(child_index);
             child = children[child_index]
            
             # initialize p0 of u through this child
             #p0_child = 0;
             p0_child = 0.

             # probability that a packet from node 'u' is sent on the link to child 'child'
             #rho = net.capacities(u,child) / b_o(u);
             rho = net['capacities'][u,child] / b_o[u]
             
             #------------------------------------------------------------------
             # if child is receiver, loss prob = rho * errorrate
             #if child == r
             if child == r:
                 #p0_child = rho * net.errorrates(u,child);
                 p0_child = rho * net['errorrates'][u,child]

             else:
                 # child is not the receiver                   
                 
                 # make sure child is not a NC node (not in ncnodes)
                 #if any(ncnodes == child)
                 #if child == ncnodes
                 #if numpy.any(ncnodes == child):
                 #if ncnodes.contains(child):
                 if child in ncnodes:
                     
                     # child is NC node, make p0 equal to 1, because we exclude paths which go through NC nodes
                     #p0_child = rho * 1; # 100# losses
                     p0_child = rho * 1. # 100# losses

                 else:  
                     # child is not a NC node, process it
                     # p0 = rho * (probab. lost on the link from u to child + 
                     #              + not lost on the link. but lost in a buffer overflow in child
                     #              + not lost on the link, survives buffer overflow, is replicated M times and all M+1 copies are lost after the child node
                     #                      or is replicated M+1 times and all M+2 copies are lost after the chlid node)

                      # if is replicating node, no chance of buffer
                      # overflow
  #                         if new_p_ov(child) == 1
  #                             p0_child  = rho * (...
  #                                             errorrates(u,child) + ...
  #                                             (1-errorrates(u,child)) * ( new_p_rep_M(child)*p0(child, r_index)^(new_M(child)) + (1-new_p_rep_M(child))*p0(child, r_index)^(new_M(child) + 1)));
  #                         else
  #                             p0_child  = rho * (...
  #                                             errorrates(u,child) + ...
  #                                             (1-errorrates(u,child))*(1-new_p_ov(child)) + ...
  #                                             (1-errorrates(u,child)) * new_p_ov(child) * p0(child, r_index));
  #                         end          
                      #pi = net.errorrates(u,child);
                      pi = net['errorrates'][u,child]
                      #Beta = p_ov(child);
                      Beta = p_ov[child]
                      #exponent = R(r_index, child);
                      exponent = R[r_index, child]
                      #if exponent < 1 exponent = 1; end
                      if exponent < 1:
                        exponent = 1
                      #p0_child = rho * (pi + (1-pi)*Beta + (1-pi)*(1-Beta)* p0matrix(child,r_index)^exponent);
                      #p0_child = rho * (pi + (1-pi)*Beta + (1-pi)*(1-Beta)* p0matrix(child)^exponent);
                      #p0_child = rho * (pi + (1-pi)*Beta + (1-pi)*(1-Beta)* p0matrix[child,r_index]**exponent)
                      p0_child = rho * (pi + (1-pi)*Beta + (1-pi)*(1-Beta)* p0matrix[child]**exponent)

                      # Sanity check
                      # Comment this to increase speed
                      #assert(~isnan(p0_child));
                 #end

             #end

             # add p0 to partial sum
             #p0_partsum = p0_partsum + p0_child;
             p0_partsum = p0_partsum + p0_child
          #end

          # check 0 < p0 < 1, allow for some margin of numerical error
          #if p0_partsum < -1e-6 || p0_partsum > (1 + 1e-6)
          if p0_partsum < -1e-6 or p0_partsum > (1 + 1e-6):
              #disp(['p0 = ' num2str(p0_partsum)]);
              print('p0 = ' + str(p0_partsum))
              #error('Error: p0 not ok!');
              raise
          #end
          
          # round small numerical errors
          #if -1e-6 < p0_partsum && p0_partsum < 0
          if -1e-6 < p0_partsum and p0_partsum < 0:
              #p0_partsum = 0;
              p0_partsum = 0.
          #end
          #if 1 < p0_partsum && p0_partsum < (1+(1e-6))
          if 1 < p0_partsum and p0_partsum < (1+(1e-6)):
              #p0_partsum = 1;
              p0_partsum = 1.
          #end
          
          # Save final probability density function
          #p0matrix(u,r_index) = p0_partsum;
          #p0matrix(u) = p0_partsum;
          #p0matrix[u,r_index] = p0_partsum
          p0matrix[u] = p0_partsum
      #end
  #end
  #end
  
  return p0matrix
  
def compute_exp_N1(N, p0, R, totallines):

    # sanity parameter check
    #if R > 1000  # huge replication rate
    if R > 1000:  # huge replication rate
        #if p0^R < 1e-4
        if p0**R < 1e-4:
            # a packet from virtually any time interval will surely reach the
            # client
            #t = eye(N+1);
            #t = numpy.eye(N+1)
            #t1 = numpy.eye(N+1)
            return N
        else:
            # too much memory needed to compute this
            #t = zeros(N+1);  # will be detected later in compute_tc_u() and the time will be Inf
            #t = numpy.zeros((N+1,N+1))  # will be detected later in compute_tc_u() and the time will be Inf
            #if R > 20000:
                # too much memory & time needed to compute this
                #t1 = zeros(N+1);  # will be detected later in compute_tc_u() and the time will be Inf
                #t1 = numpy.zeros((N+1,N+1))  # will be detected later in compute_tc_u() and the time will be Inf
            return 0            
            # TODO: sort this mess
        #end
        #return t
    #end
    
    # P(0|0) = 1, i.e. before the NC received its first packet, the client's
    # rank is surely 0
    t_line = numpy.zeros(N+1)
    t_line[0] = 1
    
    R_low = math.floor(R)
    R_high = math.floor(R+1)
    p_low  = math.floor(R+1) - R
    p_high = R - math.floor(R)
    
    ##############
    # pcond stuff    
    ##############
    
    ## store binomial pdf values in a table, to avoid re-computing them many
    ## times
    ## 'low' is for floor(R), 'high' is for ceil(R)
    bino_table_zero_low    = scipy.stats.distributions.binom.pmf(numpy.arange(0,(R_low+1)), R_low, 1.-p0)
    bino_table_zero_high   = scipy.stats.distributions.binom.pmf(numpy.arange(0,(R_high+1)), R_high, 1.-p0)
    
    bino_table_zero_low_convmtx  = convmtx(bino_table_zero_low,  N + 1)
    bino_table_zero_low_convmtx_summed  = bino_table_zero_low_convmtx
    bino_table_zero_low_convmtx_summed[:,N] = numpy.sum(bino_table_zero_low_convmtx_summed[:,N:(N + bino_table_zero_low.size)], 1)
    bino_table_zero_low_convmtx_summed = bino_table_zero_low_convmtx_summed[:,:(N+1)]

    bino_table_zero_high_convmtx = convmtx(bino_table_zero_high, N + 1)
    bino_table_zero_high_convmtx_summed  = bino_table_zero_high_convmtx;
    bino_table_zero_high_convmtx_summed[:,N] = numpy.sum(bino_table_zero_high_convmtx_summed[:,N:(N + bino_table_zero_high.size)], 1)
    bino_table_zero_high_convmtx_summed = bino_table_zero_high_convmtx_summed[:,:(N+1)]
    
    bino_table_zero_both_convmtx_summed_weightedsum = p_low * bino_table_zero_low_convmtx_summed + p_high * bino_table_zero_high_convmtx_summed
    
    ##############
    # pcond_first stuff    
    ##############
    
    k = numpy.arange(1,R_low+1)
    # betainc(... 'upper') = 1 - betainc(...)
    # Scipy betainc() takes p as the last argument!
    betainc_table_low  = 1 - scipy.special.betainc(R_low-k+1, k, p0)
    k = numpy.arange(1,R_high+1)
    betainc_table_high = 1 - scipy.special.betainc(R_high-k+1, k, p0)

    cm_low  = convmtx(numpy.hstack((betainc_table_low,numpy.array([0]))), N)
    cm_high = convmtx(betainc_table_high, N)
    cm_all_w = p_low * cm_low + p_high * cm_high    
    
    ##############
    
    #bino_table_zero_both_summed_weighted = p_low * numpy.hstack((bino_table_zero_low,numpy.array([0]))) + p_high * bino_table_zero_high
    exp_N1 = 0.0
    # DEBUG:
    #debugargs = numpy.zeros((totallines+2,3))
    
    for i in range(totallines-2):
        t_line = numpy.dot(t_line, bino_table_zero_both_convmtx_summed_weightedsum)
        if i < N:
            t_line[i+1] = numpy.sum(t_line[(i+1):])
            t_line[(i+2):] = 0
        
        if p0 > 1:
            if abs(p0 - 1) < 1e-6:
                p0 = 1
    
        #t1line = numpy.zeros(N+1)
        #t1line[0] = 1
    
        #    # ge lower diagonal part of t, without the last column
        #    #tl = tril(t(:,1:N), 0);
        #    tl = numpy.tril(t[:,:N], 0)
        #
        #    # convolve t1 with betainc_table_low by multipliying with conv matrix
        #    res_low = numpy.dot(tl , cm_low)
        #    res_low = numpy.hstack((numpy.zeros((totallines,1)), res_low[:,:N]))
        #
        #    # convolve t1 with betainc_table_high by multipliying with conv matrix
        #    res_high = numpy.dot(tl , cm_high)
        #    res_high = numpy.hstack((numpy.zeros((totallines,1)), res_high[:,:N]))
        #
        #    # add
        #    res = p_low * res_low + p_high * res_high
    
        tfirst_line = numpy.dot(t_line[:N] , cm_all_w)
        
        ## add one row on top, remove last row
        ##res = [1 zeros(1,N); res(1:totallines-1,:)];
        #res = numpy.vstack((numpy.hstack((1, numpy.zeros(N))) , res[:totallines-1,:]))
    
        ## remove excess values from convolution
        ##res = tril(res);
        #res = numpy.tril(res)

        # BUG: Because Matlab is 1-based, while it checks at lines 20, 40 etc,
        #  this actually means in Python 19, 39, etc. So check (i+2)%20 instead
        #if (i+1)%20 == 0: # check every 20 lines
        if (i+2)%20 == 0: # check every 20 lines
            if t_line[N] > (1-1e-3):
                break      

        #exp_N1 = numpy.sum( t1[:, sim['N']] * numpy.arange(t1.shape[0]) )
        # BUG: move this after test, because in the original function we added a
        #  line on top of the table
        #  e.g. if t had 40 lines, 
        if i+2 >= N:
          exp_N1 += (i+2) * tfirst_line[N-1]
          #debugargs[i+2,0] = i+2
          #debugargs[i+2,1] = tfirst_line[N-1]
          #debugargs[i+2,2] = t_line[N]

    # Implement t[-1,-1] < (1. - 1e-2) in compute_tc_u()
    if t_line[-1] < (1. - 1e-2):
        exp_N1 = 0.0 # this will force tc = Inf in compute_tc_u()

    #debugargs = debugargs[:i+3,:]
    #return exp_N1, debugargs
    return exp_N1
        
      
def compute_exp_N1_C(N, p0, R, totallines):
    """
    Implement compute_pcond_table(), compute_pcond_firsttime_table() and the
    calculation formula of exp_N1 as a single C function
    """

    #print N, p0, R, totallines

    # sanity parameter check
    #if R > 1000  # huge replication rate
    if R > 1000:  # huge replication rate
        #if p0^R < 1e-4
        if p0**R < 1e-4:
            # a packet from virtually any time interval will surely reach the
            # client
            #t = eye(N+1);
            #t = numpy.eye(N+1)
            #t1 = numpy.eye(N+1)
            return N
        else:
            # too much memory needed to compute this
            #t = zeros(N+1);  # will be detected later in compute_tc_u() and the time will be Inf
            #t = numpy.zeros((N+1,N+1))  # will be detected later in compute_tc_u() and the time will be Inf
            #if R > 20000:
                # too much memory & time needed to compute this
                #t1 = zeros(N+1);  # will be detected later in compute_tc_u() and the time will be Inf
                #t1 = numpy.zeros((N+1,N+1))  # will be detected later in compute_tc_u() and the time will be Inf
            return 0            
            # TODO: sort this mess
        #end
        #return t
    #end
    
    # P(0|0) = 1, i.e. before the NC received its first packet, the client's
    # rank is surely 0
    t_line = numpy.zeros(N+1)
    t_line[0] = 1.
    
    R_low = math.floor(R)
    R_high = math.floor(R+1)
    p_low  = math.floor(R+1) - R
    p_high = R - math.floor(R)
    
    ##############
    # pcond stuff    
    ##############
    
    ## store binomial pdf values in a table, to avoid re-computing them many
    ## times
    ## 'low' is for floor(R), 'high' is for ceil(R)
    bino_table_zero_low    = scipy.stats.distributions.binom.pmf(numpy.arange(0,(R_low+1)), R_low, 1.-p0)
    bino_table_zero_high   = scipy.stats.distributions.binom.pmf(numpy.arange(0,(R_high+1)), R_high, 1.-p0)
    
    bino_table_zero_low_convmtx  = convmtx(bino_table_zero_low,  N + 1)
    bino_table_zero_low_convmtx_summed  = bino_table_zero_low_convmtx
    bino_table_zero_low_convmtx_summed[:,N] = numpy.sum(bino_table_zero_low_convmtx_summed[:,N:(N + bino_table_zero_low.size)], 1)
    bino_table_zero_low_convmtx_summed = bino_table_zero_low_convmtx_summed[:,:(N+1)]

    bino_table_zero_high_convmtx = convmtx(bino_table_zero_high, N + 1)
    bino_table_zero_high_convmtx_summed  = bino_table_zero_high_convmtx;
    bino_table_zero_high_convmtx_summed[:,N] = numpy.sum(bino_table_zero_high_convmtx_summed[:,N:(N + bino_table_zero_high.size)], 1)
    bino_table_zero_high_convmtx_summed = bino_table_zero_high_convmtx_summed[:,:(N+1)]
    
    bino_table_zero_both_convmtx_summed_weightedsum = p_low * bino_table_zero_low_convmtx_summed + p_high * bino_table_zero_high_convmtx_summed
    
    ##############
    # pcond_first stuff    
    ##############
    
    k = numpy.arange(1,R_low+1)
    # betainc(... 'upper') = 1 - betainc(...)
    # Scipy betainc() takes p as the last argument!
    betainc_table_low  = 1 - scipy.special.betainc(R_low-k+1, k, p0)
    k = numpy.arange(1,R_high+1)
    betainc_table_high = 1 - scipy.special.betainc(R_high-k+1, k, p0)

    cm_low  = convmtx(numpy.hstack((betainc_table_low,numpy.array([0]))), N)
    cm_high = convmtx(betainc_table_high, N)
    cm_all_w = p_low * cm_low + p_high * cm_high    
    
    ##############
    
    #bino_table_zero_both_summed_weighted = p_low * numpy.hstack((bino_table_zero_low,numpy.array([0]))) + p_high * bino_table_zero_high
    exp_N1 = 0.0
    # DEBUG:
    #debugargs = numpy.zeros((totallines+2,3))
    
    # For C++
    t_line_comp = numpy.zeros_like(t_line)
    tfirst_line = numpy.zeros(N+1)
    nrows1 = bino_table_zero_both_convmtx_summed_weightedsum.shape[0]
    nrows2 = cm_all_w.shape[0]
    #fp0 = 
    
    #    code = r"""
    #    #for i in range(totallines-2):
    #    for (int i = 0; i < totallines-2; i++)
    #    {
    #        #t_line = numpy.dot(t_line, bino_table_zero_both_convmtx_summed_weightedsum)
    #        for (int c = 0; c < N+1; c++):
    #            t_line_comp[c] = t_line[0] * bino_table_zero_both_convmtx_summed_weightedsum[0,c]
    #        for (int r = 1; r < nrows1; r++):
    #            for (int c = 0; c < N+1; c++):
    #                t_line_comp[c] += t_line[r] * bino_table_zero_both_convmtx_summed_weightedsum[r,c]
    #        // copy back:
    #        for (int c = 0; c < N+1; c++):
    #            t_line[c] = t_line_comp[c]
    #        
    #        #if i < N:
    #        #    t_line[i+1] = numpy.sum(t_line[(i+1):])
    #        #    t_line[(i+2):] = 0
    #        if i < N
    #            for (int c = i+2; c < N+1; c++)
    #            {
    #                t_line[i+1] += t_line[c];
    #                t_line[c] = 0.0;
    #            }
    #        
    #        #if p0 > 1:
    #        #    if abs(p0 - 1) < 1e-6:
    #        #        p0 = 1
    #        if p0 > 1
    #            if (p0-1) < 0.000001
    #                p0 = 1.0;
    #    
    #        #t1line = numpy.zeros(N+1)
    #        #t1line[0] = 1
    #    
    #        #tfirst_line = numpy.dot(t_line[:N] , cm_all_w)
    #        for (int c = 0; c < N; c++):
    #            tfirst_line_comp[c] = t_line[0] * cm_all_w[0,c]
    #        for (int r = 1; r < nrows2; r++):
    #            for (int c = 0; c < N; c++):
    #                tfirst_line_comp[c] += t_line[r] * cm_all_w[r,c]
    #        // copy back:
    #        for (int c = 0; c < N; c++)
    #            tfirst_line[c] = tfirst_line_comp[c]        
    #        
    #        #if (i+2)%20 == 0: # check every 20 lines
    #        #    if t_line[N] > (1-1e-3):
    #        #        break      
    #        if (i+2)%20 == 0
    #            if t_line[N] > (1-1e-3)
    #                break;
    #
    #        #if i+2 >= N:
    #        #  exp_N1 += (i+2) * tfirst_line[N-1]
    #        #  #debugargs[i+2,0] = i+2
    #        #  #debugargs[i+2,1] = tfirst_line[N-1]
    #        #  #debugargs[i+2,2] = t_line[N]
    #        if (i+2) >= N
    #            exp_N1 += (i+2) * tfirst_line[N-1]
    #    }
    #    # Implement t[-1,-1] < (1. - 1e-2) in compute_tc_u()
    #    #if t_line[-1] < (1. - 1e-2):
    #    #    exp_N1 = 0.0 # this will force tc = Inf in compute_tc_u()
    #    if t_line[N] < (1. - 1e-2)
    #        exp_N1 = 0.;
    #
    #    #debugargs = debugargs[:i+3,:]
    #    #return exp_N1, debugargs
    #    
    #    return_val = exp_N1
    #    """

   
    code = r"""
    int i, r, c;
    for (i = 0; i < totallines-2; i++)
    {
        
        for (c = 0; c < N+1; c++) {
            t_line_comp(c) = t_line(0) * bino_table_zero_both_convmtx_summed_weightedsum(0,c);
        }
        for (r = 1; r < nrows1; r++) {
            for (c = 0; c < N+1; c++) {
                t_line_comp(c) += t_line(r) * bino_table_zero_both_convmtx_summed_weightedsum(r,c);
            }
        }
        for (c = 0; c < N+1; c++) {    
            t_line(c) = t_line_comp(c);
        }
        
        if (i < N) {
            for (c = i+2; c < N+1; c++)
            {
                t_line(i+1) += t_line(c);
                t_line(c) = 0.;
            }
        }
        //printf("i = %d, t_line(i,N) = %g\n",i,t_line(N));
        //printf("t_line");
        //for (c = 0; c < N+1; c++) {    
        //    printf("%g ", t_line(c));
        //}
        //printf("\n");
        
        
        if (p0 > 1) {
            if ((double(p0) - 1) < 0.000001) {
                p0 = 1.0;
            }
        }
    
        for (c = 0; c < N; c++) {
            tfirst_line(c) = t_line(0) * cm_all_w(0,c);
        }
        for (r = 1; r < nrows2; r++) {
            for (c = 0; c < N; c++) {
                tfirst_line(c) += t_line(r) * cm_all_w(r,c);
            }
        }
        
        //printf("i = %d, tfirst_line(i,N-1) = %g\n",i,tfirst_line(N-1));
        //printf("tfirst_line: ");
        //for (c = 0; c < N+1; c++) {    
        //    printf("%g ", tfirst_line(c));
        //}
        //printf("\n");        
        
        if ((i+2)%20 == 0) {
            if (t_line(N) > (1-1e-3)) {
                break;
            }
        }

        if ((i+2) >= N) {
            exp_N1 += (i+2) * tfirst_line(N-1);
        }
        
    }
    
    if (t_line(N) < (1. - 1e-2)) {
        exp_N1 = 0.;
    }
    

    return_val = exp_N1;
    """        
    
    exp_N1 = scipy.weave.inline(code,['t_line', 't_line_comp', 'tfirst_line', 'bino_table_zero_both_convmtx_summed_weightedsum', 'cm_all_w', 'totallines', 'p0', 'N', 'nrows1', 'nrows2', 'exp_N1'],type_converters=scipy.weave.converters.blitz, compiler = 'gcc', verbose = 2)
    
    return exp_N1
    