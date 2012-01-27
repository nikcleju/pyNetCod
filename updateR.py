import numpy
#import scipy.io
import math

import globalvars

    
def update_R(R, net, p0matrix, ncnodes, tc):
  # MATLAB: function newR = update_R(R, net, p0matrix, ncnodes, tc)
  # Update R
  
  # Nodes' input capacities
  #b_i = sum(net.capacities .* (1-net.errorrates), 1);
  b_i = numpy.sum(net['capacities'] * (1 - net['errorrates']), 0)
  # Nodes' output capacities
  #b_o = sum(net.capacities, 2)';
  b_o = numpy.sum(net['capacities'], 1)
  
  #newR = R;
  newR = R.copy()
  
  #for r_idx = 1:numel(net.receivers)
  for r_idx in range(net['receivers'].size):
      # for each SF node u
      #SFnodes = setdiff(net.helpers, ncnodes);
      SFnodes = numpy.setdiff1d(net['helpers'], ncnodes)
      #for u_idx = 1:numel(SFnodes)
      for u_idx in range(SFnodes.size):
          #u = SFnodes(u_idx);
          u = SFnodes[u_idx]
          
          # Only for replicating nodes
          #if (b_o(u) > b_i(u))
          if b_o[u] > b_i[u]:
              # #Requiv = compute_Requiv(tc(r_idx), p0matrix(u, r_idx), b_i(u), b_o(u) / b_i(u));        
              #if R(r_idx, u) > 1
              if R[r_idx, u] > 1:
                  #Requiv = compute_Requiv(tc(r_idx), p0matrix(u, r_idx), b_i(u), R(r_idx, u));
                  Requiv = compute_Requiv(tc[r_idx], p0matrix[u, r_idx], b_i[u], R[r_idx, u])
              else:
                  #Requiv = R(r_idx, u);
                  Requiv = R[r_idx, u]
              #end
              
              #if Requiv >= 0
              if Requiv >= 0:
                  #newR(r_idx, u) = Requiv;
                  newR[r_idx, u] = Requiv
              #else
                  # #error('');
              #end
          #end
      #end
  #end
  
  return newR
  
def compute_Requiv(estim_delay, p0, NI, Rinitial):
    #MATLAB function Requiv = compute_Requiv(estim_delay, p0, NI, Rinitial)
     
    #global hbuf;
    #global sum_inverses_E1;
    #global sum_inverses_E2;
    #global sum_inverses_E1E2;
    
    ##if isempty(sum_inverses_E1)
    #if globalvars.sum_inverses_E1.size == 0:
    #    #load suminv_hbuf32;
    #    mdict = scipy.io.loadmat('suminv_hbuf32.mat')
    #    globalvars.sum_inverses_E1 = mdict['sum_inverses_E1']
    #    globalvars.sum_inverses_E2 = mdict['sum_inverses_E2']
    #    globalvars.sum_inverses_E1E2 = mdict['sum_inverses_E1E2']
    ##end
    
    #if exist('estim_delay', 'var') && ~isempty(estim_delay) && p0 < 1 && estim_delay < 1e4
    if p0 < 1 and estim_delay < 1e4:
        
        #estim_input_packets = estim_delay * NI;
        estim_input_packets = estim_delay * NI
        
        # interpolate
        #estim_input_packets_low  = floor(estim_input_packets);
        estim_input_packets_low  = numpy.floor(estim_input_packets)
        #estim_input_packets_high = estim_input_packets_low + 1;
        estim_input_packets_high = estim_input_packets_low + 1
        #p_low  = estim_input_packets_high - estim_input_packets;
        p_low  = estim_input_packets_high - estim_input_packets
        #p_high = 1 - p_low;
        p_high = 1 - p_low
    
        
        # low
        #if estim_input_packets_low <= 0
        if estim_input_packets_low <= 0:
            #Requiv_low = Rinitial;
            Requiv_low = Rinitial
        #else if estim_input_packets_low >= 2*hbuf
        else:
            if estim_input_packets_low >= 2*globalvars.hbuf:
                #Ri = 1+(Rinitial-1)*[sum_inverses_E1   ones(1, estim_input_packets_low-2*hbuf)   sum_inverses_E2];
                Ri = 1+(Rinitial-1)*numpy.concatenate((globalvars.sum_inverses_E1, numpy.ones(estim_input_packets_low-2*globalvars.hbuf), globalvars.sum_inverses_E2))
            else:
                #Ri = 1+(Rinitial-1)*sum_inverses_E1E2{estim_input_packets_low};
                Ri = 1+(Rinitial-1)*globalvars.sum_inverses_E1E2[estim_input_packets_low]
                # DEBUG
                print globalvars.sum_inverses_E1E2[estim_input_packets_low]
                print Rinitial
                #end
            #Requiv_low = log( sum(p0.^Ri) / estim_input_packets_low ) / log(p0);
            Requiv_low = math.log( numpy.sum(p0**Ri) / estim_input_packets_low ) / math.log(p0)
            # DEBUG:
            print estim_input_packets_low
            print numpy.sum(p0**Ri)
            print math.log(p0)
            print p0
            
        #end    
        
        # high
        #if estim_input_packets_high <= 0
        if estim_input_packets_high <= 0:
            #Requiv_high = Rinitial;
            Requiv_high = Rinitial
        #else if estim_input_packets_high >= 2*hbuf
        else:
            if estim_input_packets_high >= 2*globalvars.hbuf:
                #Ri = 1+(Rinitial-1)*[sum_inverses_E1   ones(1, estim_input_packets_high-2*hbuf)   sum_inverses_E2];
                Ri = 1+(Rinitial-1)*numpy.concatenate((globalvars.sum_inverses_E1, numpy.ones(estim_input_packets_high-2*globalvars.hbuf), globalvars.sum_inverses_E2))
            else:
                #Ri = 1+(Rinitial-1)*sum_inverses_E1E2{estim_input_packets_high};
                Ri = 1+(Rinitial-1)*globalvars.sum_inverses_E1E2[estim_input_packets_high]
            #end
            #Requiv_high = log( 1 - sum(1 - p0.^Ri) / estim_input_packets_high ) / log(p0);
            Requiv_high = math.log( numpy.sum(p0**Ri) / estim_input_packets_high ) / math.log(p0)
        #end
        
        #Requiv = p_low * Requiv_low + p_high * Requiv_high;
        Requiv = p_low * Requiv_low + p_high * Requiv_high
        #if isnan(Requiv)
        if math.isnan(Requiv):
            #Requiv = Rinitial;
            Requiv = Rinitial
        #end
        #if isinf(Requiv)    # because, for example, Ri is too large
        if math.isinf(Requiv):    # because, for example, Ri is too large
            #Requiv =Rinitial;
            Requiv =Rinitial
        #end
            
        #if ( Requiv > Rinitial);
        #   # assert(abs(Rech - Rinitial) < 1e-6);
        #end
        
        #if (Requiv < 0)
        if (Requiv < 0):
            #assert(abs(Requiv) < 1e-6);
            print Requiv # DEBUG
            print p_low, p_high
            print Requiv_low, Requiv_high
            assert(abs(Requiv) < 1e-6)
            #Requiv = 0;
            Requiv = 0
        #end
        
    else:
        # Something is not right!
        # Don't change anything
        #Requiv = Rinitial;
        Requiv = Rinitial
    #end
    
    return Requiv
    

def update_R_vectorized(R, net, p0matrix, ncnodes, tc):
    # TODO: compare with update_R
    #  update_R() works for every NC node, this is vectorized
    #
    # MATLAB function newR = update_R_vectorized(R, net, p0matrix, ncnodes, tc)
    # Update R
    
    # Nodes' input capacities
    #b_i = sum(net.capacities .* (1-net.errorrates), 1);
    b_i = numpy.sum(net['capacities'] * (1 - net['errorrates']), 0)
    # Nodes' output capacities
    #b_o = sum(net.capacities, 2)';
    b_o = numpy.sum(net['capacities'], 1)    
    
    #newR = R;
    newR = R.copy()
    
    #persistent hbuf;
    #persistent sum_inverses_E1;
    #persistent sum_inverses_E2;
    #persistent sum_inverses_E1E2;
    #persistent sum_inverses_E1E2_mat;
    
    #if isempty(sum_inverses_E1)
    #    load suminv_hbuf32;
    #end
    
    #for r_idx = 1:numel(net.receivers)
    for r_idx in range(net['receivers'].size):
        
        #estim_delay = tc(r_idx);
        estim_delay = tc[r_idx]
        #if estim_delay < 1e4
        if estim_delay < 1e4:
            
            #estim_input_packets = estim_delay * b_i;
            estim_input_packets = estim_delay * b_i
        
            # interpolate
            #estim_input_packets_low  = floor(estim_input_packets);
            #estim_input_packets_high = estim_input_packets_low + 1;
            #p_low  = estim_input_packets_high - estim_input_packets;
            #p_high = 1 - p_low;
            estim_input_packets_low  = numpy.int32(numpy.floor(estim_input_packets))
            estim_input_packets_high = estim_input_packets_low + 1
            p_low  = estim_input_packets_high - estim_input_packets
            p_high = 1 - p_low            
        
            #=================
            # low
            #=================
            #Requiv_low = R(r_idx, :);
            Requiv_low = R[r_idx, :]
    
            #indices1 = ~ismember(1:net.nnodes, ncnodes) & (estim_input_packets_low>0) & (estim_input_packets_low >= 2*hbuf) & (R(r_idx,:) > 1) & (p0matrix(:,r_idx) < 1)' & (b_o(:) > b_i(:))';
            indices1 = numpy.logical_not(numpy.in1d(numpy.arange(net['nnodes']), ncnodes))
            indices1 = numpy.logical_and(indices1, estim_input_packets_low>0)
            indices1 = numpy.logical_and(indices1, estim_input_packets_low >= 2*globalvars.hbuf)
            indices1 = numpy.logical_and(indices1, R[r_idx,:] > 1)
            indices1 = numpy.logical_and(indices1, p0matrix[:,r_idx] < 1)
            indices1 = numpy.logical_and(indices1, b_o > b_i)
            #p0 = p0matrix(indices1,r_idx);
            p0 = p0matrix[indices1,r_idx]
            #p0_2D = repmat(p0matrix(indices1, r_idx), 1, 2*hbuf);
            p0_2D = numpy.tile(numpy.reshape(p0matrix[indices1, r_idx],(-1,1)), (1, 2*globalvars.hbuf))
            #Rinitial = R(r_idx, indices1);
            Rinitial = R[r_idx, indices1]
            #exp_2D = (Rinitial-1)'*[sum_inverses_E1 sum_inverses_E2];
            exp_2D = numpy.outer((Rinitial-1), numpy.concatenate((globalvars.sum_inverses_E1, globalvars.sum_inverses_E2)))
            
            #sum_p0_Ri = p0 .* (sum(p0_2D.^exp_2D, 2) + (estim_input_packets_low(indices1)-2*hbuf)'.*p0.^((Rinitial-1)'));
            sum_p0_Ri = p0 * (numpy.sum(p0_2D**exp_2D, 1) + (estim_input_packets_low[indices1]-2*globalvars.hbuf) * p0**(Rinitial-1))
            ##sum_p0_Ri = p0 .* (sum(p0.*((Ri-1)*[sum_inverses_E1 sum_inverses_E2])) + (estim_input_packets_low-2*hbuf)*p0.*(Ri-1));
            
            #Requiv_low(indices1) = log( sum_p0_Ri ./ estim_input_packets_low(indices1)' ) ./ log(p0);
            Requiv_low[indices1] = numpy.log( sum_p0_Ri / estim_input_packets_low[indices1] ) / numpy.log(p0)
            
            
            #indices2 = ~ismember(1:net.nnodes, ncnodes) & (estim_input_packets_low>0) & (estim_input_packets_low < 2*hbuf) & (R(r_idx,:) > 1) & (p0matrix(:,r_idx) < 1)' & (b_o(:) > b_i(:))';
            indices2 = numpy.logical_not(numpy.in1d(numpy.arange(net['nnodes']), ncnodes))
            indices2 = numpy.logical_and(indices2, estim_input_packets_low>0)
            indices2 = numpy.logical_and(indices2, estim_input_packets_low < 2*globalvars.hbuf)
            indices2 = numpy.logical_and(indices2, R[r_idx,:] > 1)
            indices2 = numpy.logical_and(indices2, p0matrix[:,r_idx] < 1)
            indices2 = numpy.logical_and(indices2, b_o > b_i)            
            #p0 = p0matrix(indices2,r_idx);
            p0 = p0matrix[indices2,r_idx]
            #p0_2D = repmat(p0matrix(indices2, r_idx), 1, 2*hbuf-1);
            p0_2D = numpy.tile(numpy.reshape(p0matrix[indices2, r_idx],(-1,1)), (1, 2*globalvars.hbuf-1))
            #Rinitial = R(r_idx, indices2);
            Rinitial = R[r_idx, indices2]
            #exp_2D = repmat( (Rinitial-1)',1, size(sum_inverses_E1E2_mat,2) ).*sum_inverses_E1E2_mat(estim_input_packets_low(indices2),:);
            exp_2D = numpy.tile( numpy.atleast_2d(Rinitial-1).T, (1, globalvars.sum_inverses_E1E2_mat.shape[1] )) * globalvars.sum_inverses_E1E2_mat[estim_input_packets_low[indices2],:]
    
            #sum_p0_Ri = p0 .* sum(p0_2D.^exp_2D, 2);
            sum_p0_Ri = p0 * numpy.sum(p0_2D**exp_2D, 1)
            #Requiv_low(indices2) = log( sum_p0_Ri ./ estim_input_packets_low(indices2)' ) ./ log(p0);
            Requiv_low[indices2] = numpy.log( sum_p0_Ri / estim_input_packets_low[indices2] ) / numpy.log(p0)
            
            #=================
            # high
            #=================
            #Requiv_high = R(r_idx, :);
            Requiv_high = R[r_idx, :]
    
            #indices1 = ~ismember(1:net.nnodes, ncnodes) & (estim_input_packets_high>0) & (estim_input_packets_high >= 2*hbuf) & (R(r_idx,:) > 1) & (p0matrix(:,r_idx) < 1)' & (b_o(:) > b_i(:))';
            indices1 = numpy.logical_not(numpy.in1d(numpy.arange(net['nnodes']), ncnodes))
            indices1 = numpy.logical_and(indices1, estim_input_packets_high>0)
            indices1 = numpy.logical_and(indices1, estim_input_packets_high >= 2*globalvars.hbuf)
            indices1 = numpy.logical_and(indices1, R[r_idx,:] > 1)
            indices1 = numpy.logical_and(indices1, p0matrix[:,r_idx] < 1)
            indices1 = numpy.logical_and(indices1, b_o > b_i)            
            #p0 = p0matrix(indices1,r_idx);
            p0 = p0matrix[indices1,r_idx]
            #p0_2D = repmat(p0matrix(indices1, r_idx), 1, 2*hbuf);
            p0_2D = numpy.tile(numpy.reshape(p0matrix[indices1, r_idx],(-1,1)), (1, 2*globalvars.hbuf))
            #Rinitial = R(r_idx, indices1);
            Rinitial = R[r_idx, indices1]
            #exp_2D = (Rinitial-1)'*[sum_inverses_E1 sum_inverses_E2];
            exp_2D = numpy.outer((Rinitial-1), numpy.concatenate((globalvars.sum_inverses_E1, globalvars.sum_inverses_E2)))
            
            #sum_p0_Ri = p0 .* (sum(p0_2D.^exp_2D, 2) + (estim_input_packets_high(indices1)-2*hbuf)'.*p0.^((Rinitial-1)'));
            sum_p0_Ri = p0 * (numpy.sum(p0_2D**exp_2D, 1) + (estim_input_packets_high[indices1]-2*globalvars.hbuf) * p0**(Rinitial-1))
            ##sum_p0_Ri = p0 .* (sum(p0.*((Ri-1)*[sum_inverses_E1 sum_inverses_E2])) + (estim_input_packets_high-2*hbuf)*p0.*(Ri-1));
            
            #Requiv_high(indices1) = log( sum_p0_Ri ./ estim_input_packets_high(indices1)' ) ./ log(p0);
            Requiv_high[indices1] = numpy.log( sum_p0_Ri / estim_input_packets_high[indices1] ) / numpy.log(p0)
            
            #indices2 = ~ismember(1:net.nnodes, ncnodes) & (estim_input_packets_high>0) & (estim_input_packets_high < 2*hbuf) & (R(r_idx,:) > 1) & (p0matrix(:,r_idx) < 1)' & (b_o(:) > b_i(:))';
            indices2 = numpy.logical_not(numpy.in1d(numpy.arange(net['nnodes']), ncnodes))
            indices2 = numpy.logical_and(indices2, estim_input_packets_high>0)
            indices2 = numpy.logical_and(indices2, estim_input_packets_high < 2*globalvars.hbuf)
            indices2 = numpy.logical_and(indices2, R[r_idx,:] > 1)
            indices2 = numpy.logical_and(indices2, p0matrix[:,r_idx] < 1)
            indices2 = numpy.logical_and(indices2, b_o > b_i)
            #p0 = p0matrix(indices2,r_idx);
            p0 = p0matrix[indices2,r_idx]
            #p0_2D = repmat(p0matrix(indices2, r_idx), 1, 2*hbuf-1);
            p0_2D = numpy.tile(numpy.reshape(p0matrix[indices2, r_idx],(-1,1)), (1, 2*globalvars.hbuf-1))
            #Rinitial = R(r_idx, indices2);
            Rinitial = R[r_idx, indices2]
            #exp_2D = repmat( (Rinitial-1)',1, size(sum_inverses_E1E2_mat,2) ).*sum_inverses_E1E2_mat(estim_input_packets_high(indices2),:);
            exp_2D = numpy.tile( numpy.atleast_2d(Rinitial-1).T, (1, globalvars.sum_inverses_E1E2_mat.shape[1] )) * globalvars.sum_inverses_E1E2_mat[estim_input_packets_high[indices2],:]
            
            #sum_p0_Ri = p0 .* sum(p0_2D.^exp_2D, 2);
            sum_p0_Ri = p0 * numpy.sum(p0_2D**exp_2D, 1)
            #Requiv_high(indices2) = log( sum_p0_Ri ./ estim_input_packets_high(indices2)' ) ./ log(p0);
            Requiv_high[indices2] = numpy.log( sum_p0_Ri / estim_input_packets_high[indices2] ) / numpy.log(p0)
            
            #allRequiv = p_low .* Requiv_low + p_high .* Requiv_high;
            allRequiv = p_low * Requiv_low + p_high * Requiv_high
            #newR(r_idx, indices1 | indices2) = allRequiv(indices1 | indices2);
            newR[r_idx, numpy.logical_or(indices1 , indices2)] = allRequiv[numpy.logical_or(indices1 , indices2)]
            
        #end
        
    ##     # for each SF node u
    ##     SFnodes = setdiff(net.helpers, ncnodes);
    ##     for u_idx = 1:numel(SFnodes)
    ##         u = SFnodes(u_idx);
    ## 
    ##         # Only for replicating nodes
    ##         if (b_o(u) > b_i(u))
    ##             #Requiv = compute_Requiv(tc(r_idx), p0matrix(u, r_idx), b_i(u), b_o(u) / b_i(u));        
    ##             if R(r_idx, u) > 1
    ##                 Requiv = compute_Requiv(tc(r_idx), p0matrix(u, r_idx), b_i(u), R(r_idx, u));
    ##             else
    ##                 Requiv = R(r_idx, u);
    ##             end
    ## 
    ##             if Requiv >= 0
    ##                 newR(r_idx, u) = Requiv;
    ##             else
    ##                 #error('');
    ##             end
    ##         end
    ##     end
    #end
    return newR
    
#def ismember(a, b):
#    # from:
#    #http://stackoverflow.com/questions/7448554/replicating-the-indices-result-of-matlabs-ismember-function-in-numpy
#    # tf = np.in1d(a,b) # for newer versions of numpy
#    tf = np.array([i in b for i in a])
#    u = np.unique(a[tf])
#    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
#    return tf, index    
