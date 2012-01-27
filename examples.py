
import time
import os
import os.path
import shutil
import numpy
import scipy.io

import MatlabInputParser
import create_graph
import Algos
import graph
import files
import misc

from numpy.random import RandomState
rng = RandomState()

class ExceptionGenerateExample:
    pass

def generate_example(folder, varargin=None):
    # MATLAB function [folder, randstate, net, sim, runopts] = generate_example(folder, varargin)
    #============================================
    ## Description
    #============================================
    # Generates an example network
    #
    # Nicolae Cleju, EPFL, 2008/2009,
    #                TUIASI, 2009/2010
    
    if varargin == None:
        varargin = {}
        
    #============================================
    ## Parse inputs & default parameter values
    #============================================
    varargin['folder'] = folder   # Python
    
    #p = inputParser;   # Create instance of inputParser class.
    p = MatlabInputParser.MatlabInputParser()   # Create instance of inputParser class.
    
    # Scenario folder
    p.addRequired('folder',  lambda x: (isinstance(x,str)));
    
    # Random state generator
    p.addParamValue('randstate', time.time(), lambda x: (numpy.isreal(x)));
    
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
    #plotK = p.Results['plotK']
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
    rng.seed(int(randstate))
    
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
    RA[receivers, :] = numpy.zeros((receivers.size, nnodes))
    
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
                    raise ExceptionGenerateExample
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
        os.mkdir(dirname)
    #end
    #if ~exist([dirname '/config'], 'dir')
    #    mkdir([dirname '/config']);
    #end
    if not os.path.isdir(dirname+'/config'):
        os.mkdir(dirname+'/config')
    #if ~exist([dirname '/results'], 'dir')
    #    mkdir([dirname '/results']);
    #end
    if not os.path.isdir(dirname+'/results'):
        os.mkdir(dirname+'/results')
    #if ~exist([dirname '/figs'], 'dir')
    #    mkdir([dirname '/figs']);
    #end
    if not os.path.isdir(dirname+'/figs'):
        os.mkdir(dirname+'/figs')
    #if ~exist([dirname '/scripts'], 'dir')
    #    mkdir([dirname '/scripts']);
    #end
    if not os.path.isdir(dirname+'/scripts'):
        os.mkdir(dirname+'/scripts')
    
    # copy linux script files for simulation
    #copyfile('ppss/ppss.sh', dirname);
    shutil.copy('ppss/ppss.sh', dirname)
    #copyfile('ppss/runppss.py', dirname);
    shutil.copy('ppss/runppss.py', dirname)
    
    # copy linux scripts for archieving
    #copyfile('scripts/*', [dirname '/scripts']);
    #shutil.copy('scripts/*', dirname + '/scripts')
    # copy all files
    src_files = os.listdir('scripts')
    for file_name in src_files:
        full_file_name = os.path.join('scripts', file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dirname + '/scripts')

    
    ## Save
    # save graph plot
    #saveas(gcf, [dirname '/figs/graph.png']);
    
    # Save in matlab.mat
    #save([dirname '/matlab.mat'], 'folder', 'randstate', 'net', 'sim', 'runopts', 'p');
    #mdict = {'folder':folder, 'randstate':randstate, 'net':net, 'sim':sim, 'runopts':runopts, 'p':p}
    # Don't save p, because of the lambda functions    
    mdict = {'folder':folder, 'randstate':randstate, 'net':net, 'sim':sim, 'runopts':runopts}
    scipy.io.savemat(dirname + '/matlab.mat', mdict)
    
    return folder, randstate, net, sim, runopts
    
    
def generate_batch(dirs):
    # MATLAB function generate_batch(dirs)
    
    # set paths
    #addpath('E:\Master\MatlabNew\code2');
    #addpath('E:\Master\MatlabNew\code2\routeradjacency');
    #addpath('E:\Master\MatlabNew\code2\results'); 
    
    n_helpers = 160
    minnnodes = 100
    rmin = 1
    #rmax = Inf;
    rmax = numpy.Inf
    auto_option = 'a'
    maxtries = 20
    
    #generate_example('E:\Master\MatlabNew\scenarios2\141', 'n_helpers', 140, 'minnnodes', 90, 'rmin', 1, 'rmax', Inf, 'auto_option', 'r', 'maxtries', 10)
    
    fulldirs = dict()
    #if isnumeric(dirs)
    if numpy.all(numpy.isreal(dirs)):
        basepath = 'E:\\Master\\MatlabNew\\scenarios2\\'
        
        #for i=1:numel(dirs)
        for i in xrange(dirs.size):
            #fulldirs{i} = [basepath num2str(dirs(i))];
            fulldirs[i] = basepath + str(dirs[i])
        #end
    else:
        fulldirs = dirs
    #end
    
    #for i=1:numel(fulldirs)
    for i in xrange(len(fulldirs)):
        try:
            #generate_example(fulldirs{i}, 'n_helpers', n_helpers, 'minnnodes', minnnodes, 'rmin', rmin, 'rmax', rmax, 'auto_option', auto_option, 'maxtries', maxtries);
            generate_example(fulldirs[i], {'n_helpers': n_helpers, 'minnnodes': minnnodes, 'rmin': rmin, 'rmax': rmax, 'auto_option': auto_option, 'maxtries': maxtries})
        #catch exception
        except ExceptionGenerateExample:
            #disp(['Caught exception: ' exception.identifier ' ' exception.message]);
            print('Caught exception: ExceptionGenerateExample')
            #if (strcmp(exception.identifier, 'create_graph_planetlab3:cantcreatenetwork'))
            #disp(['Could not create scenario ' fulldirs{i}]);
            print('Could not create scenario ' + fulldirs[i])
            raise 
            #else
            #    # Unknown exception, let it propagate
            #    rethrow(exception);
            #end
        #end
    #end
    
        
    # n_helpers = 100;
    # n_helpers_min = 100;
    # n_helpers_max = 200;
    # 
    # minnodes = 30;
    # maxdistance  = 5;
    # 
    # dirnames = {};
    # 
    # #dirs = 180:189;
    # #dirs = 181;
    # for n_idx = 1:numel(dirs)
    #     n = dirs(n_idx);
    # 
    #     found_network = 0;
    #     while ~found_network
    #         caught_exception = 0;
    # 
    #         try
    #             disp(['Trying n = ' num2str(n) ', n_helpers = ' num2str(n_helpers)]);
    #             run_example(num2str(n), 'n_helpers', n_helpers, 'maxdistance', maxdistance, 'minnnodes', minnodes, 'rmin', 1, 'rmax', Inf, 'method', [5], 'auto_overwrite', 1 )
    # 
    #         catch exception
    # 
    #             disp(['Caught exception: ' exception.identifier ' ' exception.message]);
    #             if (strcmp(exception.identifier, 'create_graph_planetlab3:cantcreatenetwork'))
    #                 caught_exception = 1;
    #                 n_helpers = n_helpers + 10;
    #             elseif (strcmp(exception.identifier, 'node_selection:initdelaytoolarge'))
    #                 caught_exception = 1;
    #             else
    #                # Unknown error. Just let it propagate.
    #                rethrow(exception);
    #             end
    #         end
    #     
    #         global global_times_p0_skipped;
    #         global global_times_p0_done;
    #         disp(['global_times_p0_skipped = ' num2str(global_times_p0_skipped)])
    #         disp(['global_times_p0_done = ' num2str(global_times_p0_done)])
    #         global global_times_innov_skipped;
    #         global global_times_innov_done;
    #         disp(['global_times_innov_skipped = ' num2str(global_times_innov_skipped)])
    #         disp(['global_times_innov_done = ' num2str(global_times_innov_done)])
    #         global global_hash_p0matrix_keys;
    #         global global_hash_p0matrix_values;
    #         global global_hash_innov_keys;
    #         global global_hash_innov_values;
    #         s1 = whos('global_hash_p0matrix_keys');
    #         s2 = whos('global_hash_p0matrix_values');
    #         s3 = whos('global_hash_innov_keys');
    #         s4 = whos('global_hash_innov_values');
    #         disp(['Sizes = ' num2str(s1.bytes) ', ' num2str(s2.bytes) ', ' num2str(s3.bytes) ', ' num2str(s4.bytes), ' bytes']);
    #         disp(['Sizes = ' num2str(s1.bytes/1024/1024) ', ' num2str(s2.bytes/1024/1024) ', ' num2str(s3.bytes/1024/1024) ', ' num2str(s4.bytes/1024/1024), ' MB']);
    #         disp(['Total size = '  num2str(s1.bytes + s2.bytes + s3.bytes + s4.bytes) ' bytes']);
    #         disp(['Total size = '  num2str((s1.bytes + s2.bytes + s3.bytes + s4.bytes)/1024/1024) ' MB']);
    #         
    #         # Exception or not, clear global variables from global workspace and then pack memory
    #         clear global *;
    #         #clear_global_variables();
    #         #pack;
    #         
    #         if ~caught_exception
    #             if n_helpers > n_helpers_min
    #                 n_helpers = n_helpers - 10;
    #             end
    #             found_network = 1;
    #             disp('Exception not caught, found network');
    #             dirnames = [dirnames, num2str(n)];
    #         end
    #         
    #         if n_helpers == n_helpers_max
    #             n_helpers = n_helpers_min;
    #             disp('n_helpers = max, re-setting to min');
    #         end
    #     end
    # end
    # 
    # #dirnames = {'160', '165'};
    # 
    # # write dir to file
    # fid = fopen(['../scenarios/dirs' num2str(id) '.txt'],'w'); 
    # for i = 1:numel(dirnames)
    #     fprintf(fid, '#s\n', dirnames{i});
    # end
    # fclose(fid);
    # 
    # # create finished file
    # fid = fopen(['../scenarios/finished' num2str(id)],'w'); 
    # fprintf(fid, '#s', datestr(clock));
    # fclose(fid);


def generate_example_and_run(folder, varargin=None):
    # MATLAB function generate_example_and_run(folder, varargin)
    #============================================
    ## Description
    #============================================
    # Generates an example network and runs the node selection algorithm
    #
    # Nicolae Cleju, EPFL, 2008/2009,
    #                TUIASI, 2009/2010
    
    #============================================
    
    if varargin == None:
        varargin = {}
        
    #disp('Welcome');
    print('Welcome')
    
    # Generate example
    #disp('Generating example ...');
    print('Generating example ...')
    #if isempty(varargin)
    if not varargin:
        #[folder, randstate, net, sim, runopts] = generate_example(folder);
        folder, randstate, net, sim, runopts = generate_example(folder)
    else:
        #[folder, randstate, net, sim, runopts] = generate_example(folder, varargin{:});
        folder, randstate, net, sim, runopts = generate_example(folder, varargin)
    #end
    
    # Run example
    #disp('Running example');
    print('Running example')
    #run_example(folder, randstate, net, sim, runopts);
    run_example(folder, randstate, net, sim, runopts)
    
    ## 
    #disp('Finished.');
    print('Finished.')
    

def load_example(folder, varargin=None):
    # MATLAB function [folder, randstate, net, sim, runopts] = load_example(folder, varargin)
    #============================================
    ## Description
    #============================================
    # Loads an existing scenario (with optional overwriting of parameters)
    #
    # Nicolae Cleju, EPFL, 2008/2009,
    #                TUIASI, 2009/2011
    
    #============================================
    
    if varargin == None:
        varargin = {}
    
    #============================================
    ## Load saved data
    #============================================
    #orig_folder = folder;
    #orig_folder = folder.copy()
    orig_folder = folder
    # This overwrites the variable 'folder'
    ##load([folder '\32\matlab.mat']);
    #load([folder '/matlab.mat']);
    #mdict = scipy.io.loadmat(folder + '/matlab.mat', squeeze_me=True)
    folder, randstate, net, sim, runopts = misc.myloadmat(folder + '/matlab.mat')
    
    #folder = mdict['folder'][0]
    #randstate = mdict['randstate'][0,0]
    #net = mdict['net'][0,0]
    #sim = mdict['sim'][0,0]
    #runopts = mdict['runopts'][0,0]
    #p = mdict['p']
    
    # Squeeze arrays after loading from mat file!
    #for key in net:
    #    for key in net.dtype.names:
    #        #if isinstance(net[key], numpy.array):
    #            net[key] = numpy.squeeze(net[key])
    #            if net[key].shape == ():
    #                print key # DEBUG
    #                while not numpy.isscalar(net[key]):
    #                    #net[key] = numpy.array([net[key][()]])
    #                    net[key] = net[key][()]
    #                net[key] = numpy.array([net[key]])
    #    for key in sim.dtype.names:
    #        #if isinstance(sim[key], numpy.array):
    #            sim[key] = numpy.squeeze(sim[key])
    #            #if sim[key].shape == ():
    #            while not numpy.isscalar(sim[key]):
    #                #sim[key] = numpy.array([sim[key][()]])
    #                sim[key] = sim[key][()]
    #    for key in runopts.dtype.names:
    #        #if isinstance(runopts[key], numpy.array):
    #            runopts[key] = numpy.squeeze(runopts[key])
    #            #if runopts[key].shape == ():
    #            while not numpy.isscalar(runopts[key]):
    #                #runopts[key] = numpy.array([runopts[key][()]])
    #                runopts[key] = runopts[key][()]
    
    #folder = orig_folder;
    folder = orig_folder;
    #clear orig_folder;
    #disp(['Loaded data from ''' folder '/matlab.mat''']);
    print("Loaded data from '" + folder + "/matlab.mat'")
    
    #============================================
    ## Parse inputs & default parameter values
    #============================================
    varargin['folder'] = folder   # Python
    
    #p = inputParser;   # Create instance of inputParser class.
    p = MatlabInputParser.MatlabInputParser()
    
    # Scenario folder
    p.addRequired('folder',  lambda x: (isinstance(x,str)));
    
    # Random state generator
    p.addParamValue('randstate', randstate, lambda x: (numpy.isreal(x)));
    
    # Nodes
    # p.addParamValue('n_sources',     ,      lambda x: (numpy.isreal(x) and x > 0));
    # p.addParamValue('n_helpers',     30,     lambda x: (numpy.isreal(x) and x > 0));
    # p.addParamValue('n_receivers',   3,      lambda x: (numpy.isreal(x) and x > 0));
    
    # Packets
    p.addParamValue('n_packets',     sim['N'],     lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('gf_dim',        sim['gf_dim'],     lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('payload',       sim['pktsize']/8. - 8- sim['N']*(sim['gf_dim']/8.),    lambda x: (numpy.isreal(x) and x > 0));
    
    # Simulation options
    p.addParamValue('nruns',         sim['nruns'],    lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('rndnruns',      sim['rndnruns'],   lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('replication',   sim['replication'],  lambda x: (isinstance(x,str)));              # to add more checking here
    p.addParamValue('stoptime',      sim['stoptime'],     lambda x: (numpy.isreal(x) and x > 0));
    
    # Graph generation
    # p.addParamValue('maxdistance',   -1,     lambda x: (numpy.isreal(x) and x > 0));   # will be passed to create_graph_planetlab2()
    # p.addParamValue('maxtries',      100,     lambda x: (numpy.isreal(x) and x > 0));     # will be passed to create_graph_planetlab2()
    # #p.addParamValue('removeunnecessary', true,     lambda x: (numpy.isreal(x) and x > 0));     # will be passed to create_graph_planetlab2()
    # p.addParamValue('minnnodes',      15,    lambda x: (numpy.isreal(x) and x > 0));     # will be passed to create_graph_planetlab2()
    
    # Run options
    p.addParamValue('do_global_delay',     runopts['do_global_delay'],    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('do_global_flow',      runopts['do_global_flow'],    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('do_dist_delay',       runopts['do_dist_delay'],    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('do_dist_flow',        runopts['do_dist_flow'],    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    p.addParamValue('nNC',                 runopts['nNC'],    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    #if isfield('runopts','do_old_icc_version')
    if 'do_old_icc_version' in runopts:
        #p.addParamValue('do_old_icc_version',  runopts['do_old_icc_version'],    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
        p.addParamValue('do_old_icc_version',  runopts['do_old_icc_version'],    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    else:
        p.addParamValue('do_old_icc_version',  0,    lambda x: (numpy.isreal(x) and x >= 0));     # will be passed to create_graph_planetlab2()
    #end
    
    # Radii for distributed algorithm
    p.addParamValue('rmin',          runopts['rmin'],      lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('rstep',         runopts['rstep'],      lambda x: (numpy.isreal(x) and x > 0));
    p.addParamValue('rmax',          runopts['rmax'],    lambda x: (numpy.isreal(x) and x >= 0));
    
    # Number of nodes per layer for plotgraph()
    #p.addParamValue('plotK',         3,      lambda x: (numpy.isreal(x) and x > 0));
    
    # Overwrite automatically?
    #p.addParamValue('auto_overwrite',        0,      lambda x: (numpy.isreal(x) and x >= 0));
    
    # ==== Parse ====
    #p.parse(folder, varargin{:});
    p.parse(varargin);
    
    # ==== Get results ====
    folder      = p.Results['folder']
    randstate   = p.Results['randstate']
    #n_sources   = p.Results.n_sources']
    #n_helpers   = p.Results.n_helpers']
    #n_receivers = p.Results.n_receivers']
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
    #plotK = p.Results.plotK;
    #auto_overwrite = p.Results.auto_overwrite;
    
    ## Init
    #rand('state',randstate);
    rng.seed(int(randstate))
    
    # Size of packets
    # it is used to generate the link capacities in bps in the configuration files
    # payload = useful packet data, 8 = UDP header size, N = number of NC
    # coefficients, gf_dim = size of a coefficient (in bytes)
    pktsize = 8 * (payload + 8 + N*(gf_dim/8))
    
    # ===============================
    ## Create configuration structures
    # ===============================
    
    # simulation options
    sim['N'] = N;
    sim['pktsize'] = pktsize;
    sim['nruns'] = nruns;
    sim['rndnruns'] = rndnruns;
    sim['stoptime'] = stoptime;
    sim['replication'] = replication;
    sim['gf_dim'] = gf_dim;
    
    # run options
    runopts['do_global_delay'] = do_global_delay;
    runopts['do_global_flow'] = do_global_flow;
    runopts['do_dist_delay'] = do_dist_delay;
    runopts['do_dist_flow'] = do_dist_flow;
    runopts['nNC'] = nNC;
    runopts['do_old_icc_version'] = do_old_icc_version;
    runopts['rmin'] = rmin;
    runopts['rstep'] = rstep;
    runopts['rmax'] = rmax;
    
    return folder, randstate, net, sim, runopts
   
   
   
def load_batch_and_run(dirs, varargin=None):
    # MATLAB function load_batch_and_run(dirs, varargin)

    if varargin == None:
        varargin = {}
        
    fulldirs = dict()
    
    if numpy.all(numpy.isreal(dirs)):
        ##basepath = 'E:\Master\MatlabNew\scenarios2\';
        #basepath = 'E:\Master\Results2\manuscript_iccversion\';
        basepath = 'E:\\Master\\Results2\\manuscript_iccversion\\'
        ##basepath = '/home/scs/ncleju/ns3/matlabcode/scenarios2/';
        ##basepath = '/home/scs/ncleju/ns3/matlabcode/scenarios2/r_a/';

        #for i=1:numel(dirs)
        for i in xrange(dirs.size):
            #fulldirs{i} = [basepath num2str(dirs(i))];
            fulldirs[i] = basepath + str(dirs[i])
        #end
    else:
        fulldirs = dirs
    #end
    
    # TODO: parallel
    #parfor i=1:numel(fulldirs)
    for i in xrange(fulldirs.size):
        try:
            #load_example_and_run(fulldirs{i}, varargin{:});
            load_example_and_run(fulldirs[i], varargin)
        #catch exception
        except:
            #disp(['Caught exception: ' exception.identifier ' ' exception.message]);
            print('Caught exception')
            #disp(['Could not run scenario ' fulldirs{i}]);
            print('Could not run scenario ' + fulldirs[i])
            raise
        #end
    #end



def load_example_and_run(folder, varargin=None):
    # MATLAB function load_example_and_run(folder, varargin)
    #============================================
    ## Description
    #============================================
    # Generates an example network and runs the node selection algorithm
    #
    # Nicolae Cleju, EPFL, 2008/2009,
    #                TUIASI, 2009/2010
    
    #============================================
    
    if varargin == None:
        varargin = {}
        
    # Generate example
    #disp('Loading example ...');
    print('Loading example ...')
    #[folder, randstate, net, sim, runopts] = load_example(folder, varargin{:});
    folder, randstate, net, sim, runopts = load_example(folder, varargin)
    
    # Run example
    #disp('Running example');
    print('Running example')
    #run_example(folder, randstate, net, sim, runopts);
    run_example(folder, randstate, net, sim, runopts)



    
def run_example(folder, randstate, net, sim, runopts):
    # MATLAB function run_example(folder, randstate, net, sim, runopts)
    #============================================
    ## Description
    #============================================
    # Runs a loaded scenario
    #
    # Nicolae Cleju, EPFL, 2008/2009,
    #                TUIASI, 2009/2011
    #============================================
    
    # Init winners
    winners = {}
    #winners.global_delay = [];
    winners['global_delay'] = numpy.array([])
    winners['global_flow']  = numpy.array([])
    #winners.dist_delay   = {};
    winners['dist_delay']   = {}
    winners['dist_flow']    = {}
    #winners.r = [];
    winners['r'] = numpy.array([])
    
    # First create basic file (nonc, allnodesnc, random)
    #output_scenario_folder(folder, net, sim, runopts, winners);
    files.output_scenario_folder(folder, net, sim, runopts, winners)
    
    # Run Algorithm 2
    #if runopts.do_global_delay
    if runopts['do_global_delay']:
        #disp('Global, delay:');
        print('Global, delay:')
        #winners.global_delay = Algo2_Centralized_NC_sel(net,sim,runopts,'delay');
        winners['global_delay'] = Algos.Algo2_Centralized_NC_sel(net,sim,runopts,'delay')
    #end
    # Save winners
    #save([folder '/matlab.mat'], 'winners', '-append');
    mdict = scipy.io.loadmat(folder + '/matlab.mat')
    mdict['winners'] = winners  # append
    scipy.io.savemat(folder + '/matlab.mat', mdict)
    
    #if runopts.do_global_flow
    if runopts['do_global_flow']:
        #disp('Global, flow:');
        print('Global, flow:')
        #winners.global_flow = Algo2_Centralized_NC_sel(net,sim,runopts,'flow');
        winners['global_flow'] = Algos.Algo2_Centralized_NC_sel(net,sim,runopts,'flow')
    #end
    # Save winners
    #save([folder '/matlab.mat'], 'winners', '-append');
    mdict = scipy.io.loadmat(folder + '/matlab.mat')
    mdict['winners'] = winners  # append
    scipy.io.savemat(folder + '/matlab.mat', mdict)
    
    # Run Algorithm 3
    # Set maximum eccentricity (radius)
    #if runopts.rmax == Inf
    if runopts['rmax'] == numpy.Inf:
        #[R D] = breadthdist(net.capacities + net.capacities');
        R,D = graph.all_pairs_sp(net.capacities + net.capacities.T)
        #runopts.rmax = max(max(D));
        runopts.rmax = numpy.max(D)
    #end
    
    #if runopts.do_dist_delay || runopts.do_dist_flow
    if runopts['do_dist_delay'] or runopts['do_dist_flow']:
        #winners.r = runopts.rmin:runopts.rstep:runopts.rmax;
        winners['r'] = numpy.arange(runopts['rmin'], runopts['rmax']+1, runopts['rstep'])
    #end
    #if runopts.do_dist_delay
    if runopts['do_dist_delay']:
        #for r = runopts.rmin:runopts.rstep:runopts.rmax
        for r in winners['r']:
            #disp(['Distributed, delay, r = ' num2str(r) ' :']);
            print('Distributed, delay, r = ' + str(r) + ' :')
            #winners.dist_delay{r} = Algo3_Semidistributed_NC_sel(net,sim,runopts,'delay',r);
            winners['dist_delay'][r] = Algos.Algo3_Semidistributed_NC_sel(net,sim,runopts,'delay',r)
            # Save winners
            #save([folder '/matlab.mat'], 'winners', '-append');
            mdict = scipy.io.loadmat(folder + '/matlab.mat')
            mdict['winners'] = winners  # append
            scipy.io.savemat(folder + '/matlab.mat', mdict)
        #end
    #end
    
    #if runopts.do_dist_flow
    if runopts['do_dist_flow']:
        #for r = runopts.rmin:runopts.rstep:runopts.rmax
        for r in winners['r']:
            #disp(['Distributed, flow, r = ' num2str(r) ' :']);
            print('Distributed, flow, r = ' + str(r) + ' :')
            #winners.dist_flow{r} = Algo3_Semidistributed_NC_sel(net,sim,runopts,'flow',r);
            winners['dist_flow'][r] = Algos.Algo3_Semidistributed_NC_sel(net,sim,runopts,'flow',r)
            # Save winners
            #save([folder '/matlab.mat'], 'winners', '-append');
            mdict = scipy.io.loadmat(folder + '/matlab.mat')
            mdict['winners'] = winners  # append
            scipy.io.savemat(folder + '/matlab.mat', mdict)
        #end
    #end
    
    # Create files
    #output_scenario_folder(folder, net, sim, runopts, winners);
    files.output_scenario_folder(folder, net, sim, runopts, winners)