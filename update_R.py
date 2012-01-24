import numpy
import scipy.io
import math

hbuf = 0
sum_inverses_E1 = numpy.array()
sum_inverses_E2 = numpy.array()
sum_inverses_E1E2 = numpy.array()
    
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
    
    global hbuf;
    global sum_inverses_E1;
    global sum_inverses_E2;
    global sum_inverses_E1E2;
    
    #if isempty(sum_inverses_E1)
    if sum_inverses_E1.size == 0:
        #load suminv_hbuf32;
        mdict = scipy.io.loadmat('suminv_hbuf32.mat')
        sum_inverses_E1 = mdict['sum_inverses_E1']
        sum_inverses_E2 = mdict['sum_inverses_E2']
        sum_inverses_E1E2 = mdict['sum_inverses_E1E2']
    #end
    
    #if exist('estim_delay', 'var') && ~isempty(estim_delay) && p0 < 1 && estim_delay < 1e4
    if p0 < 1 and estim_delay < 1e4:
        
        #estim_input_packets = estim_delay * NI;
        estim_input_packets = estim_delay * NI
        
        # interpolate
        #estim_input_packets_low  = floor(estim_input_packets);
        estim_input_packets_low  = math.floor(estim_input_packets)
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
            if estim_input_packets_low >= 2*hbuf:
                #Ri = 1+(Rinitial-1)*[sum_inverses_E1   ones(1, estim_input_packets_low-2*hbuf)   sum_inverses_E2];
                Ri = 1+(Rinitial-1)*numpy.concatenate((sum_inverses_E1, numpy.ones(estim_input_packets_low-2*hbuf), sum_inverses_E2))
            else:
                #Ri = 1+(Rinitial-1)*sum_inverses_E1E2{estim_input_packets_low};
                Ri = 1+(Rinitial-1)*sum_inverses_E1E2[estim_input_packets_low]
                #end
            #Requiv_low = log( sum(p0.^Ri) / estim_input_packets_low ) / log(p0);
            Requiv_low = math.log( numpy.sum(p0**Ri) / estim_input_packets_low ) / math.log(p0)
        #end    
        
        # high
        #if estim_input_packets_high <= 0
        if estim_input_packets_high <= 0:
            #Requiv_high = Rinitial;
            Requiv_high = Rinitial
        #else if estim_input_packets_high >= 2*hbuf
        else:
            if estim_input_packets_high >= 2*hbuf:
                #Ri = 1+(Rinitial-1)*[sum_inverses_E1   ones(1, estim_input_packets_high-2*hbuf)   sum_inverses_E2];
                Ri = 1+(Rinitial-1)*numpy.concatenate((sum_inverses_E1, numpy.ones(estim_input_packets_high-2*hbuf), sum_inverses_E2))
            else:
                #Ri = 1+(Rinitial-1)*sum_inverses_E1E2{estim_input_packets_high};
                Ri = 1+(Rinitial-1)*sum_inverses_E1E2[estim_input_packets_high]
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