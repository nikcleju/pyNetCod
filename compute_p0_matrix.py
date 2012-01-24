import numpy
import math
import topsort

def compute_p0_matrix(net, ncnodes, R):
  N = net['capacities'].shape[0]
  p0matrix = numpy.zeros(N, net['rceivers'].size)
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
  xs, ys = numpy.nonzero(net['capacities'])
  order = topsort.topsort([(xs[i], ys[i]) for i in range(xs.size)])

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
      children = numpy.nonzero(capT[:,u])
      
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
                 if ncnodes.contains(child):
                     
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
  