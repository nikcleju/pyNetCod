# misc

def fixmatlabload(mdict):
    
    folder = mdict['folder'][0]
    randstate = mdict['randstate'][0,0]

    net = dict();
    sim = dict();
    runopts = dict();
    
    # network structure
    net['nnodes'] = mdict['net']['nnodes'][0,0][0,0]
    net['capacities'] = mdict['net']['capacities'][0,0]
    net['errorrates'] = mdict['net']['errorrates'][0,0]
    net['sources'] = mdict['net']['sources'][0,0][:,0]
    net['helpers'] = mdict['net']['helpers'][0,0][:,0]
    net['receivers'] = mdict['net']['receivers'][0,0][:,0]
    
    # simulation options
    sim['N'] = mdict['sim']['N'][0,0][0,0]
    sim['pktsize'] = mdict['sim']['pktsize'][0,0][0,0]
    sim['nruns'] = mdict['sim']['nruns'][0,0][0,0]
    sim['rndnruns'] = mdict['sim']['rndnruns'][0,0][0,0]
    sim['stoptime'] = mdict['sim']['stoptime'][0,0][0,0]
    sim['replication'] = mdict['sim']['replication'][0,0][0]
    sim['gf_dim'] = mdict['sim']['gf_dim'][0,0][0,0]
    
    # run options
    runopts['do_global_delay'] = mdict['runopts']['do_global_delay'][0,0][0,0]
    runopts['do_global_flow'] = mdict['runopts']['do_global_flow'][0,0][0,0]
    runopts['do_dist_delay'] = mdict['runopts']['do_dist_delay'][0,0][0,0]
    runopts['do_dist_flow'] = mdict['runopts']['do_dist_flow'][0,0][0,0]
    runopts['nNC'] = mdict['runopts']['nNC'][0,0][0,0]
    runopts['do_old_icc_version'] = mdict['runopts']['do_old_icc_version'][0,0][0,0]
    runopts['rmin'] = mdict['runopts']['rmin'][0,0][0,0]
    runopts['rstep'] = mdict['runopts']['rstep'][0,0][0,0]
    runopts['rmax'] = mdict['runopts']['rmax'][0,0][0,0]
    
    # Convert from MATLAB 1-based node numbering to Python 0-based numbering
    net['sources'] = net['sources'] - 1
    net['helpers'] = net['helpers'] - 1
    net['receivers'] = net['receivers'] - 1    
    
    # Convert to ints
    sim['N'] = int(sim['N'])
    
    return folder, randstate, net, sim, runopts
    
def fixmatlabsave(mdict):
    # Convert to MATLAB 1-based node numbering    
    mdict['net']['sources'] = mdict['net']['sources'] + 1
    mdict['net']['helpers'] = mdict['net']['helpers'] + 1
    mdict['net']['receivers'] = mdict['net']['receivers'] + 1
    
    # Convert to doubles
    mdict['sim']['N'] = float(mdict['sim']['N'])
