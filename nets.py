
import time
import os.path
import shutil
import numpy
import scipy.io

import MatlabInputParser
import create_graph

from numpy.random import RandomState
rng = RandomState()

def generate_example(folder, varargin):
    # MATLAB function [folder, randstate, net, sim, runopts] = generate_example(folder, varargin)
    #============================================
    ## Description
    #============================================
    # Generates an example network
    #
    # Nicolae Cleju, EPFL, 2008/2009,
    #                TUIASI, 2009/2010
    
    #============================================
    ## Parse inputs & default parameter values
    #============================================
    varargin['folder'] = folder   # Python
    
    #p = inputParser;   # Create instance of inputParser class.
    p = MatlabInputParser.MatlabInputParser()   # Create instance of inputParser class.
    
    # Scenario folder
    p.addRequired('folder',  lambda x: (isinstance(x,str)));
    
    # Random state generator
    p.addParamValue('randstate', sum(100*time.time()), lambda x: (numpy.isreal(x)));
    
    # Nodes
    p.addParamValue('n_sources',     1,      lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('n_helpers',     30,     lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('n_receivers',   3,      lambda x: (numpy.isreal(x) and x > 0));
    
    # Packets
    p.addParamValue('n_packets',     32,     lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('gf_dim',        16,     lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('payload',       512,    lambda x: (numpy.isreal(x) and x > 0));
    
    # Simulation options
    p.addParamValue('nruns',         100,    lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('rndnruns',      1000,   lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('replication',   'any',  lambda x: (isinstance(x,str)));              # to add more checking here
    p.addParamValue('stoptime',      50,     lambda x: (numpy.isreal(x) and x > 0));
    
    # Graph generation
    p.addParamValue('maxdistance',   -1,     lambda x: (numpy.isreal(x) and x > 0));   # will be passed to create_graph_planetlab2()
    p.addParamValue('maxtries',      100,     lambda x: (numpy.isreal(x) and x > 0));     # will be passed to create_graph_planetlab2()
    #p.addParamValue('removeunnecessary', true,     lambda x: (numpy.isreal(x) and x > 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('minnnodes',      15,    lambda x: (numpy.isreal(x) and x > 0));     # will be passed to create_graph_planetlab2()
    
    # Run options
    p.addParamValue('do_global_delay',      1,    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('do_global_flow',       1,    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('do_dist_delay',        1,    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('do_dist_flow',         1,    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('nNC',                 10,    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('do_old_icc_version',   0,    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    
    # Radii for distributed algorithm
    p.addParamValue('rmin',          1,      lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('rstep',         1,      lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('rmax',          numpy.Inf,    lambda x: (numpy.isreal(x) and x >= 0));
    
    # Number of nodes per layer for plotgraph()
    p.addParamValue('plotK',         3,      lambda x: (numpy.isreal(x) and x > 0));
    
    # Overwrite automatically?
    p.addParamValue('auto_option',        '-',      lambda x: (isinstance(x,str)) );
    
    # ==== Parse ====
    #p.parse(folder, varargin{:});
    p.parse(varargin);
    
    # ==== Get results ====
    folder      = p.Results['folder']
    randstate   = p.Results['randstate']
    n_sources   = p.Results['n_sources']
    n_helpers   = p.Results['n_helpers']
    n_receivers = p.Results['n_receivers']
    N        = p.Results['n_packets']
    gf_dim   = p.Results['gf_dim']
    payload  = p.Results['payload']
    nruns    = p.Results['nruns']
    rndnruns = p.Results['rndnruns']
    replication = p.Results['replication']
    stoptime = p.Results['stoptime']
    do_global_delay = p.Results['do_global_delay']
    do_global_flow = p.Results['do_global_flow']
    do_dist_delay = p.Results['do_dist_delay']
    do_dist_flow = p.Results['do_dist_flow']
    nNC = p.Results['nNC']
    do_old_icc_version = p.Results['do_old_icc_version']
    rmin  = p.Results['rmin']
    rstep = p.Results['rstep']
    rmax  = p.Results['rmax']
    plotK = p.Results['plotK']
    auto_option = p.Results['auto_option']
    
    # Python:
    if auto_option == 'o':
        auto_overwrite = True
    
    #============================================
    ## INITIALIZATION
    #============================================
    #close all;
    
    # Initialize rand to a different state each time: 
    # randstate = sum(100*clock);
    #rand('state',randstate);
    rng.seed(randstate)
    
    # Size of packets
    # it is used to generate the link capacities in bps in the configuration files
    # payload = useful packet data, 8 = UDP header size, N = number of NC
    # coefficients, gf_dim = size of a coefficient (in bytes)
    #pktsize = 8 * (payload + 8 + N*(gf_dim/8));
    pktsize = 8 * (payload + 8 + N*(gf_dim/8))
    ##pktsize = 8 * (payload + 4 + N*(gf_dim/8) + 18);
    
    #============================================
    ## Generate network
    #============================================
    #disp('Creating randomized connectivity matrix ...');
    print('Creating randomized connectivity matrix ...')
    #clear RA;
    #clear PError;
    
    # # create randomized network from a regular hop structure
    # # [RA,PError] = makerandhopsCIJ_dir(K, L, K, 0.16, 0.1, 0.05, 'uniform', struct('min',5,'max',20));
    # if any(strcmpi(p.UsingDefaults, 'maxdistance'))
    #     RA = create_graph_randhops(1, 4, 8, 3, 'linkdev', 0.16, 'edgeremovalprob', 0.1, 'caprandtype', 'uniform', 'caprandparams', struct('min',5,'max',20));
    # else
    #     RA = create_graph_randhops(1, 4, 8, 3, 'linkdev', 0.16, 'edgeremovalprob', 0.1, 'caprandtype', 'uniform', 'caprandparams', struct('min',5,'max',20),...
    #             'maxdistance', p.Results.maxdistance, 'maxtries', p.Results.maxtries);
    # end
    
    # create network from PlanetLab data
    #RA = create_graph_planetlab3(n_sources, n_helpers, n_receivers, 'maxdistance', p.Results.maxdistance, 'maxtries', p.Results.maxtries, 'minnnodes', p.Results.minnnodes);
    RA = create_graph.create_graph_planetlab3(n_sources, n_helpers, n_receivers, {'maxdistance':p.Results['maxdistance'], 'maxtries':p.Results['maxtries'], 'minnnodes':p.Results['minnnodes']})
    
    #PError = 0.05 * (RA ~= 0);
    PError = 0.05 * (RA != 0)
    #sources = 1:n_sources;
    sources = numpy.arange(n_sources)
    
    # update receivers, number of nodes
    #nnodes = size(RA,1);
    nnodes = RA.shape[0]
    #receivers = (nnodes-n_receivers+1):nnodes;
    receivers = numpy.arange(nnodes-n_receivers, nnodes)
    #helpers = (n_sources + 1):(nnodes - n_receivers);
    helpers = numpy.arange(n_sources, nnodes - n_receivers)
    
    # ensure receivers have no output links
    #RA(receivers, :) = zeros(numel(receivers), nnodes);
    RA[receivers, :] = numpy.zeros(receivers.size, nnodes)
    
    #caps_pkts = RA;
    caps_pkts = RA.copy()
    #errorrates = PError;
    errorrates = PError.copy()
    #capacities = pktsize .* caps_pkts;
    
    # ===============================
    ## Create configuration structures
    # ===============================
    
    net = dict();
    sim = dict();
    runopts = dict();
    
    # network structure
    net['nnodes'] = nnodes
    net['capacities'] = caps_pkts
    net['errorrates'] = errorrates
    net['sources'] = sources
    net['helpers'] = helpers
    net['receivers'] = receivers
    
    # simulation options
    sim['N'] = N
    sim['pktsize'] = pktsize
    sim['nruns'] = nruns
    sim['rndnruns'] = rndnruns
    sim['stoptime'] = stoptime
    sim['replication'] = replication
    sim['gf_dim'] = gf_dim
    
    # run options
    runopts['do_global_delay'] = do_global_delay
    runopts['do_global_flow'] = do_global_flow
    runopts['do_dist_delay'] = do_dist_delay
    runopts['do_dist_flow'] = do_dist_flow
    runopts['nNC'] = nNC
    runopts['do_old_icc_version'] = do_old_icc_version
    runopts['rmin'] = rmin
    runopts['rstep'] = rstep
    runopts['rmax'] = rmax
    
    # plot graph
    #figure;
    #plotgraph(plotK, net);
    
    ## Directories
    
    # if folder can be converted to a number 
    #if ~isempty(str2num(folder))
        #graphn = str2num(folder);
    try:
        caught = False
        graphn = int(folder)
    except:
        caught = True
        
    if not caught:
        #while exist(['../scenarios2/' num2str(graphn)], 'dir')
        while os.path.isdir('../scenarios2/'+ str(graphn)):
            #repeat = 1;
            repeat = 1
            #while (repeat)
            while repeat:
                #prompt = sprintf('Folder ''#d'' exists, remove it or increment number? (''rem''/''[inc]'')', graphn);
                prompt = 'Folder ' + str(graphn) + ' exists, remove it or increment number? (''rem''/''[inc]'')'
                #if auto_overwrite
                if auto_overwrite:
                    #reply = 'rem';
                    reply = 'rem'
                    #disp('rem (auto_overwrite enabled)');
                    print('rem (auto_overwrite enabled)')
                else:
                    #reply = input(prompt, 's');
                    reply = raw_input(prompt)
                #end
                #if strcmp(reply, 'rem')
                if reply == 'rem':
                    #rmdir(['../scenarios2/' num2str(graphn)], 's');
                    shutil.rmtree('../scenarios2/' + str(graphn))
                    #repeat = 0;
                    repeat = 0
                    #elseif strcmp(reply, 'inc') || isempty(reply)
                elif reply == 'inc' or reply == '':
                    #graphn = graphn + 1;
                    graphn = graphn + 1
                    #repeat = 0;
                    repeat = 0
                #end
            #end
        #end
        #folder = num2str(graphn);
        folder = str(graphn)
    
    else:
        #if exist(folder, 'dir')
        if os.path.isdir(folder):
            #repeat = 1;
            repeat = 1
            #while (repeat)
            while repeat:
                #prompt = ['Folder ''' folder ''' exists, remove it, overwrite or abort? (''r''/''o''/''[a]'')'];
                prompt = "Folder '" + folder + "' exists, remove it, overwrite or abort? ('r'/'o'/'[a]')"
                #if auto_option ~= '-'
                if auto_option != '-':
                    #reply = auto_option;
                    reply = auto_option
                    #if reply == 'r'
                    if reply == 'r':
                        #disp('auto remove enabled');
                        print('auto remove enabled')
                        #elseif reply == 'o'
                    elif reply == 'o':
                        #disp('auto overwrite enabled');
                        print('auto overwrite enabled')
                        #elseif reply == 'a'
                    elif reply == 'a':
                        #disp('auto abort enabled');
                        print('auto abort enabled')
                    #end
                else:
                    #reply = input(prompt, 's');
                    reply = raw_input(prompt)
                #end
                #if strcmp(reply, 'r')
                if reply == 'r':
                    #disp('Existing folder will be removed')
                    print('Existing folder will be removed')
                    #rmdir(folder, 's');
                    shutil.rmtree(folder)
                    #repeat = 0;
                    repeat = 0
                    #elseif reply == 'o'
                elif reply == 'o':
                    #disp('Existing folder will be overwritten')
                    print('Existing folder will be overwritten')
                    #repeat = 0;
                    repeat = 0
                    #elseif reply == 'a' || isempty(reply)
                elif reply == 'a' or reply == '':
                    #error('Aborting');
                    print('Aborting')
                    raise
                #end
            #end
        #end
    #end
    
    #disp(['Folder = ' folder]);
    print('Folder = ' + folder)
    
    # make directory
    #dirname = [folder];
    dirname = folder
    #if ~exist(dirname, 'dir')
    if not os.path.isdir(dirname):
        #mkdir(dirname);
        os.path.mkdir(dirname)
    #end
    #if ~exist([dirname '/config'], 'dir')
    #    mkdir([dirname '/config']);
    #end
    if not os.path.isdir(dirname+'/config'):
        os.path.mkdir(dirname+'/config')
    #if ~exist([dirname '/results'], 'dir')
    #    mkdir([dirname '/results']);
    #end
    if not os.path.isdir(dirname+'/results'):
        os.path.mkdir(dirname+'/results')
    #if ~exist([dirname '/figs'], 'dir')
    #    mkdir([dirname '/figs']);
    #end
    if not os.path.isdir(dirname+'/figs'):
        os.path.mkdir(dirname+'/figs')
    #if ~exist([dirname '/scripts'], 'dir')
    #    mkdir([dirname '/scripts']);
    #end
    if not os.path.isdir(dirname+'/scripts'):
        os.path.mkdir(dirname+'/scripts')
    
    # copy linux script files for simulation
    #copyfile('ppss/ppss.sh', dirname);
    shutil.copyfile('ppss/ppss.sh', dirname)
    #copyfile('ppss/runppss.py', dirname);
    shutil.copyfile('ppss/runppss.py', dirname)
    
    # copy linux scripts for archieving
    #copyfile('scripts/*', [dirname '/scripts']);
    shutil.copyfile('scripts/*', dirname + '/scripts')
    
    ## Save
    # save graph plot
    #saveas(gcf, [dirname '/figs/graph.png']);
    
    # Save in matlab.mat
    #save([dirname '/matlab.mat'], 'folder', 'randstate', 'net', 'sim', 'runopts', 'p');
    mdict = {'folder':folder, 'randstate':randstate, 'net':net, 'sim':sim, 'runopts':runopts, 'p':p}
    scipy.io.savemat(dirname + '/matlab.mat', mdict)
    
    
    return folder, randstate, net, sim, runopts