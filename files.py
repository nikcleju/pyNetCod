
import numpy
import math

def create_config_files(configfilename, logfilename, resultsfilename, ncnodes, net, sim):
    # MATLAB function create_config_files(configfilename, logfilename, resultsfilename, ncnodes, net, sim)
    
    nnodes = net['nnodes']
    capacities = net['capacities']
    errorrates = net['errorrates']
    sources = net['sources']
    #helpers = net.helpers;
    receivers = net['receivers']
    
    N = sim['N']
    pktsize = sim['pktsize']
    nruns = sim['nruns']
    #rndnruns = sim.rndnruns;
    stoptime = sim['stoptime']
    replication = sim['replication']
    gf_dim = sim['gf_dim']
    
    #[file_id message] = fopen(configfilename,'w');
    #if (file_id == -1)
    #    disp('file open error');
    #end
    try:
        file_id = open(configfilename,'w')
    except IOError:
        print('file open error')
    
    file_id.write('//-----------------------------------\r\n')
    file_id.write('// NetCoding Configuration File\r\n')
    file_id.write('//-----------------------------------\r\n')
    file_id.write('\r\n')
    
    file_id.write('files:\r\n')
    file_id.write('{\r\n')
    file_id.write('\tlog = ' + logfilename+ ';\r\n')
    file_id.write('\tmeantimes = ' + resultsfilename + ';\r\n')
    file_id.write('};\r\n')
    
    file_id.write('channel:\r\n')
    file_id.write('{\r\n')
    file_id.write('\tdatarate = "5Mbps";\r\n')
    file_id.write('\tdelay = "10us";\r\n')
    file_id.write('};\r\n')
    
    file_id.write('graph:\r\n')
    file_id.write('{\r\n')
    file_id.write('\tnnodes = ' + str(nnodes) + ';\r\n')
    file_id.write('\tlinks = (\r\n')
    #for i = 1:nnodes
    for i in range(nnodes):
        for j in range(nnodes):
            #if capacities(i,j) ~= 0
            if capacities[i,j] != 0:
                #file_id.write('\t\t{src=#d; dest=#d; cap="#dbps"; err=#f;},\r\n', i-1, j-1, round(capacities(i,j)*pktsize), errorrates(i,j));
                file_id.write('\t\t{src='+str(i)+'; dest='+j+'; cap="'+math.round(capacities[i,j]*pktsize)+'bps"; err='+errorrates(i,j)+';},\r\n')
            #end
        #end
    #end
    # delete last comma
    #fseek(file_id, 0, 'bof');  # bogus seek to avoid matlab 6.5.0 fseek bug
    file_id.seek(0, 0)  # bogus seek to avoid matlab 6.5.0 fseek bug
    #fseek(file_id, -3, 'eof');
    file_id.seek( -3, 2)
    #file_id.write('\r\n');
    file_id.write('\r\n');
    
    file_id.write('\t);\r\n');
    file_id.write('};\r\n');
    
    ### nodes
    
    file_id.write('nodes:\r\n');
    file_id.write('{\r\n');
    
    file_id.write('\tsources = [ ');
    #for i = 1:numel(sources)
    for i in range(sources.size):
        #file_id.write('#d, ', sources(i)-1);
        file_id.write(str(sources[i]) + ', ')
    #end
    file_id.seek( -2, 2)
    file_id.write(' ];\r\n')
    
    file_id.write('\treceivers = [ ')
    #for i = 1:numel(receivers)
    for i in range(receivers.size):
        file_id.write(str(receivers[i]) + ', ');
    #end
    #fseek(file_id, -2, 'eof');
    file_id.seek( -2, 2)
    file_id.write(' ];\r\n')
    
    # 	onlyfw = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ];
    
    file_id.write('\tonlyfw = [ ');
    haswritten = 0
    #for i = 1:nnodes
    for i in range(nnodes):
        #if ( sum(sources == i) == 0 && sum(receivers == i) == 0 && sum(ncnodes == i) == 0 )
        if ( numpy.sum(sources == i) == 0 and numpy.sum(receivers == i) == 0 and numpy.sum(ncnodes == i) == 0 ):
            #file_id.write('#d, ', i-1);
            file_id.write(str(i) + ', ')
            haswritten = 1
        #end
    #end
    #if(haswritten)
    if(haswritten):
        #fseek(file_id, -2, 'eof');
        file_id.seek( -2, 2)
    #end
    file_id.write(' ];\r\n');
    
    file_id.write('\tdatarate = "400kbps"; # this value is overwritten by the graph.links individual values\r\n');
    file_id.write('\tpacketsize = 512;\r\n');
    file_id.write('\thelperbuffersize = '+ str(N) +';\r\n');
    file_id.write('\treceiverbuffersize = 1000;\r\n');
    file_id.write('\r\n');
    
    #if ~isempty(find(errorrates > 0, 1))
    if numpy.nonzero(errorrates > 0)[0].size != 0:
        #realerrorrate = errorrates(find(errorrates > 0, 1));
        xytup = numpy.nonzero(errorrates > 0)
        realerrorrate = errorrates[xytup[0][0], xytup[1][0]]
    else:
        #realerrorrate = 0;
        realerrorrate = 0
    #end
    
    file_id.write('\tpkterrorrate = '+str(realerrorrate)+'; # this value is NOT YET overwritten by the graph.links individual values\r\n')
    file_id.write('\r\n');
    file_id.write('\tjitterbits_sigma = 25;\r\n');
    file_id.write('\tjitterbits_bound = 250;\r\n');
    file_id.write('};\r\n');
    
    # netcoding
    file_id.write('netcoding:\r\n');
    file_id.write('{\r\n');
    file_id.write('\tnpackets = '+str(N)+';\r\n');
    file_id.write('\tnsourcepackets = '+str(N)+';\r\n');
    file_id.write('\r\n');
    file_id.write('\tgf_dimension = '+str(gf_dim)+'; // GF(2^dimension)\r\n');
    file_id.write('};\r\n');
    
    # simulation
    file_id.write('simulation:\r\n');
    file_id.write('{\r\n');
    file_id.write('\tscale = 1;\r\n');
    file_id.write('\r\n');
    file_id.write('\tnruns = '+str(nruns)+';\r\n');
    file_id.write('\tstarttime = 0; 	// seconds\r\n');
    file_id.write('\tstoptime = '+str(stoptime)+'; 	// seconds\r\n');
    file_id.write('\treplication = "'+replication+'"; \r\n');
    file_id.write('};\r\n');
    
    file_id.write('### eof\r\n');
    
    #file_status=fclose(file_id);
    file_id.close()
    
    
    
    # //----------------------------
    # // NetCoding Configuration File
    # #---------------------------
    # //
    # //	0	1	2
    # //	3	4	5
    # //	6	7	8
    # //	9	10	11
    # //	12	13	14
    # //	15	16	17
    # 
    # // all hops fully connected (no connection intra-hop) 
    # // sources = 0, 1, 2 
    # 
    # files:
    # {
    # 	log = "clog_raptor_6hops.txt";
    # 	meantimes = "meantimes_raptor_6hops.txt";
    # };
    # 
    # channel:
    # {
    #       datarate = "5Mbps";
    #       delay = "10us";
    # };
    # 
    # graph:
    # {
    # 	nnodes = 18;
    # 	links = (	{src=0; dest=3; cap="400kbps";},
    # 				{src=0; dest=4; cap="400kbps";},
    # 				{src=0; dest=5; cap="400kbps";},
    # 				{src=1; dest=3; cap="400kbps";},
    # 				{src=1; dest=4; cap="400kbps";},
    # 				{src=1; dest=5; cap="400kbps";},
    # 				{src=2; dest=3; cap="400kbps";},
    # 				{src=2; dest=4; cap="400kbps";},
    # 				{src=2; dest=5; cap="400kbps";},
    # 				
    # 				{src=3; dest=6; cap="400kbps";},
    # 				{src=3; dest=7; cap="400kbps";},
    # 				{src=3;	dest=8; cap="400kbps";},
    # 				{src=4; dest=6; cap="400kbps";},
    # 				{src=4; dest=7; cap="400kbps";},
    # 				{src=4; dest=8; cap="400kbps";},
    # 				{src=5; dest=6; cap="400kbps";},
    # 				{src=5; dest=7; cap="400kbps";},
    # 				{src=5; dest=8; cap="400kbps";},
    # 				
    # 				{src=6; dest=9; cap="400kbps";},
    # 				{src=6; dest=10; cap="400kbps";},
    # 				{src=6; dest=11; cap="400kbps";},
    # 				{src=7; dest=9; cap="400kbps";},
    # 				{src=7; dest=10; cap="400kbps";},
    # 				{src=7; dest=11; cap="400kbps";},
    # 				{src=8; dest=9; cap="400kbps";},
    # 				{src=8; dest=10; cap="400kbps";},
    # 				{src=8; dest=11; cap="400kbps";},
    # 
    # 				{src=9; dest=12; cap="400kbps";},
    # 				{src=9; dest=13; cap="400kbps";},
    # 				{src=9; dest=14; cap="400kbps";},
    # 				{src=10; dest=12; cap="400kbps";},
    # 				{src=10; dest=13; cap="400kbps";},
    # 				{src=10; dest=14; cap="400kbps";},
    # 				{src=11; dest=12; cap="400kbps";},
    # 				{src=11; dest=13; cap="400kbps";},
    # 				{src=11; dest=14; cap="400kbps";},
    # 
    # 				{src=12; dest=15; cap="400kbps";},
    # 				{src=12; dest=16; cap="400kbps";},
    # 				{src=12; dest=17; cap="400kbps";},
    # 				{src=13; dest=15; cap="400kbps";},
    # 				{src=13; dest=16; cap="400kbps";},
    # 				{src=13; dest=17; cap="400kbps";},
    # 				{src=14; dest=15; cap="400kbps";},
    # 				{src=14; dest=16; cap="400kbps";},
    # 				{src=14; dest=17; cap="400kbps";}
    # 			);
    # };
    # 
    # nodes:
    # {
    # 	sources = [ 0, 1, 2 ];
    # 	receivers = [15, 16, 17]; 
    # 	onlyfw = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ];
    # 	
    # 	datarate = "400kbps"; # this value is overwritten by the graph.links individual values
    # 	packetsize = 512;
    # 	helperbuffersize = 32;
    # 	receiverbuffersize = 1000;
    # 	
    # 	pkterrorrate = 0.043;
    # 	
    # 	jitterbits_sigma = 50;
    # 	jitterbits_bound = 1000;
    # };
    # 
    # netcoding:
    # {
    # 	npackets = 100;
    # 	nsourcepackets = 100;
    # 	
    # 	gf_dimension = 8; // GF(2^dimension)
    # };
    # 
    # raptor:
    # {
    # 	K = 239;
    # 	N = 270;
    # };
    # 
    # simulation:
    # {
    # 	scale = 1;
    # 
    # 	nruns = 5;
    # 	starttime = 0; 	// seconds
    # 	stoptime = 2; 	// seconds
    # };
    # 
    # ### eof 

def create_random_config_files(configfilename, logfilename, resultsfilename, numberofncnodes, net, sim):
    # MATLAB function create_random_config_files(configfilename, logfilename, resultsfilename, numberofncnodes, net, sim)
    
    nnodes = net['nnodes']
    capacities = net['capacities']
    errorrates = net['errorrates']
    sources = net['sources']
    helpers = net.helpers;
    receivers = net['receivers']
    
    N = sim['N']
    pktsize = sim['pktsize']
    #nruns = sim['nruns']
    rndnruns = sim.rndnruns;
    stoptime = sim['stoptime']
    replication = sim['replication']
    gf_dim = sim['gf_dim']    
    
    #helpers = 1:nnodes;
    #helpers(receivers) = [];
    #helpers(sources) = [];
    
    #[file_id message] = fopen(configfilename,'w');
    #if (file_id == -1)
    #    disp('file open error');
    #end
    try:
        file_id = open(configfilename,'w')
    except IOError:
        print('file open error')
    
    file_id.write('//-----------------------------------\r\n')
    file_id.write('// NetCoding Configuration File\r\n')
    file_id.write('//-----------------------------------\r\n')
    file_id.write('\r\n')
    
    file_id.write('files:\r\n')
    file_id.write('{\r\n')
    file_id.write('\tlog = ' + logfilename+ ';\r\n')
    file_id.write('\tmeantimes = ' + resultsfilename + ';\r\n')
    file_id.write('};\r\n')
    
    file_id.write('channel:\r\n')
    file_id.write('{\r\n')
    file_id.write('\tdatarate = "5Mbps";\r\n')
    file_id.write('\tdelay = "10us";\r\n')
    file_id.write('};\r\n')
    
    file_id.write('graph:\r\n')
    file_id.write('{\r\n')
    file_id.write('\tnnodes = ' + str(nnodes) + ';\r\n')
    file_id.write('\tlinks = (\r\n')
    #for i = 1:nnodes
    for i in range(nnodes):
        for j in range(nnodes):
            #if capacities(i,j) ~= 0
            if capacities[i,j] != 0:
                #file_id.write('\t\t{src=#d; dest=#d; cap="#dbps"; err=#f;},\r\n', i-1, j-1, round(capacities(i,j)*pktsize), errorrates(i,j));
                file_id.write('\t\t{src='+str(i)+'; dest='+j+'; cap="'+math.round(capacities[i,j]*pktsize)+'bps"; err='+errorrates(i,j)+';},\r\n')
            #end
        #end
    #end
    # delete last comma
    #fseek(file_id, 0, 'bof');  # bogus seek to avoid matlab 6.5.0 fseek bug
    file_id.seek(0, 0)  # bogus seek to avoid matlab 6.5.0 fseek bug
    #fseek(file_id, -3, 'eof');
    file_id.seek( -3, 2)
    #file_id.write('\r\n');
    file_id.write('\r\n');
    
    file_id.write('\t);\r\n');
    file_id.write('};\r\n');
    
    ### nodes
    
    file_id.write('nodes:\r\n');
    file_id.write('{\r\n');
    
    file_id.write('\tsources = [ ');
    #for i = 1:numel(sources)
    for i in range(sources.size):
        #file_id.write('#d, ', sources(i)-1);
        file_id.write(str(sources[i]) + ', ')
    #end
    file_id.seek( -2, 2)
    file_id.write(' ];\r\n')
    
    file_id.write('\treceivers = [ ')
    #for i = 1:numel(receivers)
    for i in range(receivers.size):
        file_id.write(str(receivers[i]) + ', ');
    #end
    #fseek(file_id, -2, 'eof');
    file_id.seek( -2, 2)
    file_id.write(' ];\r\n')
    
    # 	onlyfw = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ];
    
    file_id.write('\tonlyfw = [ ');
    
    #for i = helpers
    for i in helpers:
    ##for i = 1:nnodes
        ##if ( sum(sources == i) == 0 && sum(receivers == i) == 0 && sum(apriori_nc == i) == 0 && sum(ncnode == i) == 0)
            file_id.write(str(i) + ', ')
            haswritten = 1
        ##end
    #end
    #if(haswritten)
    if(haswritten):
        #fseek(file_id, -2, 'eof');
        file_id.seek( -2, 2)
    #end
    file_id.write(' ];\r\n');
    
    #---------------------------
    # Here random nodes:
    if numberofncnodes > 0:
        file_id.write('\trandom:	// make some random nodes every simulation\r\n');
        file_id.write('\t{\r\n');
        file_id.write('\t\tnnodes = '+numberofncnodes+';\r\n');
    
        file_id.write('\t\tpossiblenodes = [ ');
        for i in helpers:
            file_id.write(str(i) + ', ')
        #end
        file_id.seek( -2, 2)
        file_id.write(' ];\r\n');
    
        file_id.write('\t\ttype = "netcoding"; # how to set the node: "onlyfw" or "netcoding"\r\n');
        file_id.write('\t};\r\n');
    #end
    #---------------------------
    
    file_id.write('\tdatarate = "400kbps"; # this value is overwritten by the graph.links individual values\r\n');
    file_id.write('\tpacketsize = 512;\r\n');
    file_id.write('\thelperbuffersize = '+ str(N) +';\r\n');
    file_id.write('\treceiverbuffersize = 1000;\r\n');
    file_id.write('\r\n');
    
    #if ~isempty(find(errorrates > 0, 1))
    if numpy.nonzero(errorrates > 0)[0].size != 0:
        #realerrorrate = errorrates(find(errorrates > 0, 1));
        xytup = numpy.nonzero(errorrates > 0)
        realerrorrate = errorrates[xytup[0][0], xytup[1][0]]
    else:
        #realerrorrate = 0;
        realerrorrate = 0
    #end
    
    file_id.write('\tpkterrorrate = '+str(realerrorrate)+'; # this value is overwritten by the graph.links individual values\r\n' );
    file_id.write('\r\n');
    file_id.write('\tjitterbits_sigma = 25;\r\n');
    file_id.write('\tjitterbits_bound = 250;\r\n');
    file_id.write('};\r\n');
    
    # netcoding
    file_id.write('netcoding:\r\n');
    file_id.write('{\r\n');
    file_id.write('\tnpackets = '+str(N)+';\r\n');
    file_id.write('\tnsourcepackets = '+str(N)+';\r\n');
    file_id.write('\r\n');
    file_id.write('\tgf_dimension = '+str(gf_dim)+'; // GF(2^dimension)\r\n');
    file_id.write('};\r\n');
    
    # simulation
    file_id.write('simulation:\r\n');
    file_id.write('{\r\n');
    file_id.write('\tscale = 1;\r\n');
    file_id.write('\r\n');
    file_id.write('\tnruns = '+str(rndnruns)+';\r\n', ); # rndnruns, not nruns
    file_id.write('\tstarttime = 0; 	// seconds\r\n');
    file_id.write('\tstoptime = '+str(stoptime)+'; 	// seconds\r\n');
    file_id.write('\treplication = "'+replication+'"; \r\n');
    file_id.write('};\r\n');
    
    file_id.write('### eof\r\n');
    
    #file_status=fclose(file_id);
    file_id.close()

def write_config_files(dirname,fileprefix,winners, net, sim):
    #MATLAB function write_config_files(dirname,fileprefix,winners, net, sim)
    
    #ncnodes = [];    # needed for create_config_files
    ncnodes = numpy.array([])    # needed for create_config_files
    
    # Create nonc file
    #configfilename = [ dirname fileprefix '_nonc.cfg'];
    #logfilename = ['clog_' fileprefix '_nonc.txt'];
    #resultsfilename = ['meantimes_' fileprefix '_nonc.txt'];
    configfilename = dirname + fileprefix + '_nonc.cfg'
    logfilename = 'clog_' + fileprefix + '_nonc.txt'
    resultsfilename = 'meantimes_' + fileprefix + '_nonc.txt'

    #run create_config_files;
    #function create_config_files(configfilename, logfilename, resultsfilename, nnodes, capacities, errorrates, sources, receivers, ncnodes, scaleN, nruns, stoptime, dontreplicate)
    create_config_files(configfilename, logfilename, resultsfilename, ncnodes, net, sim);
    
    #filename = fileprefix;
    filename = fileprefix

    # Create the NC files
    #for i = winners
    for i in winners:
        # (i-1) because we want to write nodes 0-based, but they are 1-based in
        # Matlab
        #filename = [filename '_' num2str(i-1)];
        #configfilename = [dirname filename '.cfg'];
        #logfilename = ['clog_' filename '.txt'];
        #resultsfilename = ['meantimes_' filename '.txt'];
        filename = filename + '_' + str(i)
        configfilename = dirname + filename + '.cfg'
        logfilename = 'clog_' + filename + '.txt'
        resultsfilename = 'meantimes_' + filename + '.txt'

        #ncnodes = [ncnodes i];
        ncnodes = numpy.append(ncnodes, i)
        #run create_config_files;
        create_config_files(configfilename, logfilename, resultsfilename, ncnodes, net, sim);
    #end


def create_config_dat(filepath, data, headers):
    # MATLAB function create_config_dat(filepath, data, headers)
    
    #lenmax = numel(data{1});
    lenmax = len(data[0])
    #for n = 2:numel(data)
    for n in xrange(1, len(data)):
        #if numel(data{n}) > lenmax
        if data[n].size > lenmax:
            #lenmax = numel(data{n});
            lenmax = data[n].size
        #end
    #end
    
    #fid = fopen(filepath, 'wt');
    fid = open(filepath, 'wt')
    
    # print headers
    #strheaders = [];
    strheaders = ''
    #strheadersarg = [];
    strheadersarg = ''
    #for k = 1:length(headers)
    for k in xrange(len(headers)):
        #strheaders = [strheaders '#12s'];
        strheaders = strheaders + '{:12s}'
        #strheadersarg = [strheadersarg ', headers{' int2str(k) '}'];
        strheadersarg = strheadersarg + 'headers[' + str(k) + '],'
    #end
    strheadersarg = strheadersarg[:-1]   # remove last comma
    #strcommand = ['fprintf(fid, [''#'' strheaders ''\n'']' strheadersarg ');' ];
    strcommand = "fid.write(('#' + strheaders + '\\n').format(" + strheadersarg + "));"
    # Python: need the following
    #  fid.write(('#' + strheaders + '\\n').format(headers[0],headers[1],headers[2]));
    #eval(strcommand);
    eval(strcommand)
    
    # print data
    #for i = 1:lenmax
    for i in xrange(lenmax):
        #for k = 1:length(data)
        for k in xrange(len(data)):
            #if i <= numel(data{k})
            if i <= data[k].size:
                #fprintf(fid, '#12d', data{k}(i)-1);
                fid.write( '#12d', data[k][i])
            else:
                #fprintf(fid, '        ');
                fid.write( '        ')
            #end
        #end
        #fprintf(fid, '\n');
        fid.write('\n')
    #end
    
    #fclose(fid);
    fid.close()


def output_scenario_folder(dirname, net, sim, runopts, winners):
    # MATLAB function output_scenario_folder(dirname, net, sim, runopts, winners)
    
    # Creating nonc file
    #disp('Writing nonc config file ...');
    #write_config_files([dirname '/config/'], 'config',[], net, sim);
    write_config_files(dirname + '/config/', 'config',numpy.array([]), net, sim)
    
    # Creating all nodes nc file
    #disp('Writing all nodes nc config file ...');
    filename = 'config_allnodes'
    #configfilename = [dirname '/config/' filename '.cfg'];
    configfilename = dirname + '/config/' + filename + '.cfg'
    #logfilename = ['clog_' filename '.txt'];
    logfilename = 'clog_' + filename + '.txt'
    #resultsfilename = ['meantimes_' filename '.txt'];
    resultsfilename = 'meantimes_' + filename + '.txt'
    #create_config_files(configfilename, logfilename, resultsfilename, net.helpers, net, sim);
    create_config_files(configfilename, logfilename, resultsfilename, net['helpers'], net, sim)
    
    # Create random NC config files
    #disp('Writing random config files ...');
    #for apriori_random_count = 0:runopts.nNC
    for apriori_random_count in xrange(runopts['nNC']+1):
        #filename = ['config_rnd_' num2str(apriori_random_count)];
        filename = 'config_rnd_' + str(apriori_random_count)
        #configfilename = [dirname '/config/' filename '.cfg'];
        configfilename = dirname + '/config/' + filename + '.cfg'
        #logfilename = ['clog_' filename '.txt'];
        logfilename = 'clog_' + filename + '.txt'
        #resultsfilename = ['meantimes_' filename '.txt'];
        resultsfilename = 'meantimes_' + filename + '.txt'
        #create_random_config_files(configfilename, logfilename, resultsfilename, apriori_random_count, net, sim);
        create_random_config_files(configfilename, logfilename, resultsfilename, apriori_random_count, net, sim)
    #end
    
    ## Global
    
    # Init tables
    #global_table = {};
    global_table = dict()
    #global_table_header = {};
    global_table_header = dict()
    #global_index = 1;
    global_index = 0
    # Helpers
    #global_table{global_index} = net.helpers;
    global_table[global_index] = net['helpers']
    #global_table_header{global_index} = 'HELPERS';
    global_table_header[global_index] = 'HELPERS'
    #global_index = global_index + 1;
    global_index = global_index + 1
    
    #if runopts.do_global_delay
    if runopts['do_global_delay']:
        # Global, delay
        #disp('Writing global delay config files ...');
        #write_config_files([dirname '/config/'], 'config',winners.global_delay, net, sim);
        write_config_files(dirname + '/config/', 'config', winners['global_delay'], net, sim)
        #global_table{global_index} = winners.global_delay;
        global_table[global_index] = winners['global_delay']
        global_table_header[global_index] = 'GLOBDELAY'
        #global_index = global_index+1;
        global_index = global_index+1
    #end
    #if runopts.do_global_flow
    if runopts.do_global_flow:
        # Global, flow
        #disp('Writing global flow config files ...');
        #write_config_files([dirname '/config/'], 'config',winners.global_flow, net, sim);
        write_config_files(dirname + '/config/', 'config', winners['global_flow'], net, sim)
        #global_table{global_index} = winners.global_flow;
        global_table[global_index] = winners['global_flow']
        global_table_header[global_index] = 'GLOBFLOW'
        global_index = global_index+1
    #end
    # Create global config.dat file
    #disp('Writing config_glob.dat ...');
    #create_config_dat([dirname '/config_glob.dat'], global_table, global_table_header);
    create_config_dat(dirname + '/config_glob.dat', global_table, global_table_header)
    
    
    ## Distributed
    
    # Distributed, delay
    #disp('Writing distributed delay config files ...');
    #for r_idx = 1:numel(winners.r)
    for r_idx in xrange(winners['r'].size):
        #r = winners.r(r_idx);
        r = winners['r'][r_idx]
        #write_config_files([dirname '/config/'], 'config',winners.dist_delay{r}, net, sim);
        write_config_files(dirname+'/config/', 'config', winners['dist_delay'][r], net, sim)
    #end
    
    # Distributed, flow
    #disp('Writing distributed flow config files ...');
    #for r_idx = 1:numel(winners.r)
    for r_idx in xrange(winners['r'].size):
        #r = winners.r(r_idx);
        r = winners['r'][r_idx]
        #write_config_files([dirname '/config/'], 'config',winners.dist_flow{r}, net, sim);
        write_config_files(dirname+'/config/', 'config', winners['dist_flow'][r], net, sim)
    #end
    
    # Create dist delay config.dat file
    #dist_delay_table{1} = net.helpers;
    dist_delay_table = {0:net['helpers']}
    #dist_table_header{1} = 'HELPERS';
    dist_table_header = {0:'HELPERS'}
    #for r_idx = 1:numel(winners.r)
    for r_idx in xrange(winners['r'].size):
        #dist_table_header{1 + r_idx} = num2str(winners.r(r_idx));
        dist_table_header[1 + r_idx] = str(winners['r'][r_idx])
        #dist_delay_table{1 + r_idx} = winners.dist_delay{winners.r(r_idx)};
        dist_delay_table[1 + r_idx] = winners['dist_delay'][winners['r'][r_idx]]
    #end
    #disp('Writing config_dist_delay.dat ...');
    #create_config_dat([dirname '/config_dist_delay.dat'], dist_delay_table, dist_table_header);
    create_config_dat(dirname +'/config_dist_delay.dat', dist_delay_table, dist_table_header)
    
    # Create dist flow config.dat file
    #dist_flow_table{1} = net.helpers;
    dist_flow_table = {0: net['helpers']}
    #dist_table_header{1} = 'HELPERS';
    dist_table_header = {0: 'HELPERS'}
    #for r_idx = 1:numel(winners.r)
    for r_idx in xrange(winners['r'].size):
        #dist_table_header{1 + r_idx} = num2str(winners.r(r_idx));
        dist_table_header[1 + r_idx] = str(winners['r'][r_idx])
        #dist_flow_table{1 + r_idx} = winners.dist_flow{winners.r(r_idx)};
        dist_flow_table[1 + r_idx] = winners['dist_flow'][winners['r'][r_idx]]
    #end
    #disp('Writing config_dist_flow.dat ...');
    #create_config_dat([dirname '/config_dist_flow.dat'], dist_flow_table, dist_table_header);
    create_config_dat(dirname +'/config_dist_flow.dat', dist_flow_table, dist_table_header)
    
    #disp(['Written output files to ' dirname]);
    print('Written output files to '+ dirname)