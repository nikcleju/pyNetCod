import os
#import pp
import re
import time
import shutil
from subprocess import call
import datetime
 
""" Run NS-3 simulation """
import sys
import getopt


# function to execute when parameters are not ok
# should display help
def usage():
    print __doc__


# Creates a file with all the config files to execute:
#   out of the given list of all config files, selects the ones matching a given pattern, and writes them in the output file
# Input parameters:
#   outputfilename  = the name of the output file, in which the function writes all matching config files
#   filedir         = directory of config files
#   filelist        = the list of all config files
#   pattern         = the pattern (Unix style matching); only files with names matching this pattern are kept
def createfileslist(outputfilename, filedir, filelist, pattern):
    from fnmatch import fnmatch
    
    try:
        f = open(outputfilename, "w")
    except:
        print "Cannot open file", outputfilename, "for writing"
        raise
    
    for filename in filelist:
        if fnmatch(filename, pattern):
            pathtoconfig = os.path.join(filedir, filename)
            f.write(pathtoconfig)
            f.write('\n')

# Runs ppss.sh with all the parameters
# Input parameters:
#   fileslistfile     = file containing all the config files to run
#   pathtoexecutable  = path to executable
#   pathtoresultsdir  = path to results directory (parameter of the executable)
#   numproc           = number of processors (or cores)
def runppss(fileslistfile, pathtoexecutable, pathtoresultsdir, numproc):
    #print "Now running command:"
    #print "./ppss.sh "+"-f "+fileslistfile+" -c "+"'"+pathtoexecutable+" "+"\"$ITEM\""+" "+pathtoresultsdir+"' "
    folder = os.path.join(os.path.basename(os.path.dirname(os.getcwd()))  ,   os.path.basename(os.getcwd()))
    timestr = datetime.datetime.now().strftime("%b %d %X - ")
    print timestr + "Running ppss in folder " + folder
    returncode = os.system("./ppss.sh "+"-p "+str(numproc)+" -f "+fileslistfile+" -c "+"'"+pathtoexecutable+" "+"\"$ITEM\""+" "+pathtoresultsdir+"' ")


# Attempts to open the given file and read the helper nodes from it
# Input parameters:
#   configfile  = the config file to look in
# Return:
#   helpers     = the list of helper nodes
def readhelpers(configfile):
    timestr = datetime.datetime.now().strftime("%b %d %X - ")
    print timestr + "Attempt to open file \"", configfile, "\" to read helper nodes"
    helpers = []
    lines = []
    try:
        f = open(configfile, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            #words = line.split()
            words = re.split('\W+', line)
            #print "Line = ", line
            #print "Words = ", words
            #time.sleep(1)
            if len(words) > 0:
                if words[1].startswith("onlyfw"):
                    for word in words[2:]:
                        try:
                            n = int(word)
                            helpers.append(n)
                            #print n, ", node number"
                        except ValueError:
                            #print "\"", word, "\", not a node number"
                            pass
                    break
    except IOError:
        print "Cannot open \"", configfile,"\" for reading"
        return
    # now 'helpers' contains all the helpers
    timestr = datetime.datetime.now().strftime("%b %d %X - ")
    print timestr + "Helpers = ", helpers
    return helpers

    
# Creates a new configuration file by modifying some lines from an existing one
# Input:
#   templateconfigfile  = full path to the template config file used
#   pathtoconfigdir     = path to the directory where the new config file is written (can be different from the path of the template file)
#   strncnodes          = list of NC nodes, as strings, not numbers
#   stronlyfw           = list of non-NC (i.e. FW) helpers, as strings, not numbers
def createnewconfigfilefromexisting(templateconfigfile, pathtoconfigdir, strncnodes, stronlyfw):
    
    # Attempt to read config file
    #print "Attempt to open file \"", templateconfigfile, "\" to use as a template"
    lines = []
    try:
        f = open(templateconfigfile, 'r')
        lines = f.readlines()
        f.close()
    except IOError:
        print "Cannot open \"", templateconfigfile,"\" for reading"
        return

    # Create the ending string
    filending = '_'.join(strncnodes)
    
    # Create the new filenames
    newconfigfilename   = ''.join(["config_grd_", filending, ".cfg"])
    newresultsfilename  = ''.join(["meantimes_config_grd_", filending, ".txt"])
    
    # Prepare the modified 'log', 'meantimes' and 'onlyfw' lines
    newlog        = ''.join(["\tlog = \"clog_config_grd_", filending, ".txt\";\r\n"])
    newmeantimes  = ''.join(["\tmeantimes = \"meantimes_config_grd_", filending, ".txt\";\r\n"])
    commaonlyfw = ", ".join(stronlyfw)
    newonlyfw     = ''.join(["\tonlyfw = [ ", commaonlyfw, " ];\r\n"])
    
    # Write the new config file
    newlines = []
    for line in lines:
        if line.startswith("\tlog"):
            newlines.append(newlog)             # replace 'log' line with the new one
        elif line.startswith("\tmeantimes"):
            newlines.append(newmeantimes)       # replace 'meantimes' line with the new one
        elif line.startswith("\tonlyfw"):
            newlines.append(newonlyfw)          # replace 'onlyfw' line with the new one
        else:
            newlines.append(line)               # don't replace the line, use the one from the template
    
    # The actual writing
    try:
        f = open(os.path.join(pathtoconfigdir, newconfigfilename), "w")
    except:
        print "Cannot open file \"", os.path.join(pathtoconfigdir, newconfigfilename), "\" for writing"
        return 
    for ln in newlines:
        f.write(ln)
    f.close()
    
    return newconfigfilename, newresultsfilename


# Read a result file
# Input parameters:
#   resultfile = the result file (full path)
# Return:
#   times,flows = two lists: times = list of times, flows = list of flows
def readresults(resultfile):
    try:
        f = open(resultfile, "r")
    except IOError:
        print "Cannot open file \"", resultfile, "\" for reading"
        return
    
    lines = f.readlines()
    
    times = []
    flows = []
    for line in lines:
        words = line.split()
        #print words
        #time.sleep(1)
        if len(words) > 1:
            if words[1] != "0":   # word[1] = second word = time
                try:
                    times.append(float(words[1]))
                    #print "Node[",words[0],"] =", long(float(words[1]))
                except ValueError:
                    pass
            try:
                flows.append(float(words[7]))
                #print "Node[",words[0],"] =", long(float(words[1]))
            except ValueError:
                pass
                    
    return times,flows
    
def printconfiggreedytxt(path, ncnodes):
    try:
        f = open(path, "w")
    except IOError:
        print "Cannot open file", path, "for writing"
        return
    
    f.write("# GREEDY\n")
    for n in ncnodes:
        f.write(''.join(["     ", str(n), "\n"]))
    f.close()

    
# main function
def main(argv):   
    
    # read parameters
    try:                                
        opts, args = getopt.getopt(argv, "j:m:", ["dogreedydelay", "dogreedyflow", "maxdelay="])
    except getopt.GetoptError:          
        usage()                         
        sys.exit(2)       

    # process parameters
    numproc = 4
    pattern = "*"
    dogreedydelay = 0
    dogreedyflow = 0
    skiplongsims = 0
    skiplongsimsmaxdelay = 0
    do_rnd_separately = 0
    rndnumproc = 1    
    
    print
    print "-----------------------------"
    print "Simulation parameters:"
    print "-----------------------------"
    for opt, arg in opts:               
        if opt == "-j":
            if arg in ("1", "2", "3", "4"):
                numproc = int(arg)
            else:
                usage()                         
                sys.exit(2)
            print "Number of processors = ", arg
        elif opt == "-r":
            if arg in ("1", "2", "3", "4"):
                rndnumproc = int(arg)
                do_rnd_separately = 1
            else:
                rndnumproc = numproc
            print "Number of processors for random simulations = ", arg
        elif opt == "-m":
            pattern = arg
            print "Processing only files which match", pattern, " (Unix-style matching)"
        elif opt == "--dogreedydelay":
            dogreedydelay = 1
            print "Will do greedy search for minimum delay"
        elif opt == "--dogreedyflow":
            dogreedyflow = 1
            print "Will do greedy search for maximum flow"
        elif opt == "--maxdelay":
            skiplongsims = 1
            skiplongsimsmaxdelay = float(arg)
            print "Will skip simulations longer than " + arg + " seconds"
    print "-----------------------------"
    print "End of simulation parameters"
    print "-----------------------------"
    print
    
    
    # setup paths
    pathtoconfigdir     = os.path.join(os.getcwd(), "config")
    pathtoresultsdir    = os.path.join(os.getcwd(), "results")
    #pathtoexecutable    = "/home/cleju/ns-3.2/build/optimized/scratch/multiple-sources/multiple-sources"
    pathtoexecutable    = "/home/nic/ns3/ns-3.2/build/optimized/scratch/multiple-sources/multiple-sources"
    fileslistfile       = "fileslist.txt"
    
    #do_rnd_separately   = 0
    #rndnumproc          = 1
    rndfileslistfile    = "fileslistrnd.txt"
    
    templateconfigfile  = "config_nonc.cfg"
    skiplongconfigfile  = "config_nonc.cfg"
    skiplongresultfile  = "meantimes_config_nonc.txt"
    
    # Test simulation length
    simok = 1
    if skiplongsims:
        #returncode = os.system(pathtoexecutable + " " + pathtoconfigdir + "/" + skiplongconfigfile + " " + pathtoresultsdir)
        
        # create a file containing only the skiplongconfigfile, run it and read results
        filelist = os.listdir(pathtoconfigdir)
        createfileslist(fileslistfile, pathtoconfigdir, filelist, skiplongconfigfile)  # pattern = skiplongconfigfile
        
        timestr = datetime.datetime.now().strftime("%b %d %X - ")
        print timestr + "Checking simulation length: running file " + skiplongconfigfile
        runppss(fileslistfile, pathtoexecutable, pathtoresultsdir, numproc)
        times,flows = readresults(os.path.join(pathtoresultsdir, skiplongresultfile))

        for t in times:
            if t > skiplongsimsmaxdelay:
                simok = 0
                break
                
    if simok == 0:
        timestr = datetime.datetime.now().strftime("%b %d %X - ")
        print timestr + "Simulation too long, exiting"
        return
    else:
        timestr = datetime.datetime.now().strftime("%b %d %X - ")
        print timestr + "Simulation length small, going on with the rest of it"
        print
        
    
    # select the config files matching the pattern
    filelist = os.listdir(pathtoconfigdir)
    run_rnd_separately = 0
    if do_rnd_separately and pattern == "*":
        # no pattern given 
        # separate rnd simulations from the others, in order to run them with on a single core
        createfileslist(fileslistfile, pathtoconfigdir, filelist, "config_[!r]*")    # all config files which don't start with "config_r"
        createfileslist(rndfileslistfile, pathtoconfigdir, filelist, "config_rnd*") # all config files that start with "config_rnd"
        run_rnd_separately = 1
    else:
        createfileslist(fileslistfile, pathtoconfigdir, filelist, pattern)
        run_rnd_separately = 0

    # run ppss for the list of selected config files
    timestr = datetime.datetime.now().strftime("%b %d %X - ")
    if run_rnd_separately == 0:
        print timestr + "Running all the simulations in the config directory"
        runppss(fileslistfile, pathtoexecutable, pathtoresultsdir, numproc)
    else:
        print timestr + "Running all the non-random simulations in the config directory"
        runppss(fileslistfile, pathtoexecutable, pathtoresultsdir, numproc)
        print
        print timestr + "Running random simulations separately on " + str(rndnumproc) + " processors"
        runppss(rndfileslistfile, pathtoexecutable, pathtoresultsdir, rndnumproc)
    print
    
    # Greedy processing
    if dogreedydelay or dogreedyflow:
        #print "Starting greedy search"
        
        # Setup parameters:
        N = 10                   # Number of NC nodes to find (iterations of the algorithm)
        stepstoskip_delay = 0    # doesn't run the simulations for these first nodes, assumes results are already created
                                 #   useful to resume greedy search after an interruption
        stepstoskip_flow  = 0     
        
        # Attempt to read the helper nodes from a configuration file
        helpers = readhelpers(os.path.join(pathtoconfigdir, templateconfigfile))
        
        # Save a copy
        orighelpers = list(helpers)   # not just =, this would make both names point to the same list
    
        # Greedy delay search
        if dogreedydelay:
            timestr = datetime.datetime.now().strftime("%b %d %X - ")
            print timestr + "Starting greedy search for minimum average delay"
            ncnodes = []                        # the list of NC nodes found
            strncnodes = []                     # the list of NC nodes as strings, not numbers; used to build the file ending string, i.e. the list of NC nodes joined by '_'
            remainingonlyfw = list(orighelpers) # the list of remaining FW (i.e. non-NC) nodes
                                                #   not just =, this would make both names point to the same list
            strremainingonlyfw = []             # the list of remaining FW (i.e. non-NC) nodes, as strings
            for n in remainingonlyfw:
              strremainingonlyfw.append(str(n));
            
            while len(ncnodes) < N:
                configfilelist  = []    # list of new config files built in this iteration step
                resultsfilelist = []    # list of result files, which is read in the end to select the best NC node
                
                # Create current config files
                for currentncnode in remainingonlyfw:
                    
                    # add the current node to the list of nc nodes
                    tempstrncnodes = list(strncnodes)
                    tempstrncnodes.append(str(currentncnode))
                    
                    # remove the current node from the list of fw nodes
                    tempstrremainingonlyfw = list(strremainingonlyfw)
                    tempstrremainingonlyfw.remove(str(currentncnode))
                    
                    # create the new file
                    newconfigfilename, newresultsfilename = createnewconfigfilefromexisting(os.path.join(pathtoconfigdir, templateconfigfile), pathtoconfigdir, tempstrncnodes, tempstrremainingonlyfw)
                    
                    # add the new flenames to the lists
                    configfilelist.append(newconfigfilename)
                    resultsfilelist.append(newresultsfilename)

                # print resultsfilelist            
                # run current config files
                
                # Run ppss
                if stepstoskip_delay == 0:
                    # create a file with the list of the new config files, to be fed to ppss
                    createfileslist(fileslistfile, pathtoconfigdir, configfilelist, "*")
                    # run ppss
                    timestr = datetime.datetime.now().strftime("%b %d %X - ")
                    print
                    print timestr + "Greedy search for min average delay, step " + str(len(ncnodes)+1) + " of " + str(N)
                    runppss(fileslistfile, pathtoexecutable, pathtoresultsdir, numproc)
                else:
                    # skip simulation
                    timestr = datetime.datetime.now().strftime("%b %d %X - ")
                    print
                    print timestr + "Skipping simulations, remaining steps to skip = ", stepstoskip_delay
                    stepstoskip_delay = stepstoskip_delay - 1

                # Read results, find the best NC node
                inf = 1e100
                averages = []   # list of average delays, for each NC node tested

                # compute average delays              
                for mfile in resultsfilelist:
                    
                    # read
                    times,flows = readresults(os.path.join(pathtoresultsdir,mfile))
                    
                    # compute average
                    count = 0
                    sum = 0
                    for t in times:
                        sum = sum + t
                        count  = count + 1
                    if count > 0:
                        averages.append(sum / count)
                    else:
                        averages.append(inf)
                
                # find minimum average delay
                minpos = 0                  # node of minimum delay
                minavg = averages[0]        # minimum delay
                i = 1                       # current node
                for avg in averages[1:]:
                    if avg < minavg:
                        minavg = avg
                        minpos = i
                    i = i + 1
                
                # save delays in a file
                #with open("averagedelays.txt", "a") as f:
                #  f.write(' '.join(averages));
                try:
                  f = open("averagedelays.txt", "a")
                except:
                  print "Cannot open file \"", "averagedelays.txt", "\" for writing"
                  raise
                straverages = []
                for avg in averages:
                  straverages.append(str(avg))
                f.write(' '.join(straverages) + "\n")
                f.close()
        
                # this is the best NC node
                #newncnode = helpers[minpos]
                newncnode = remainingonlyfw[minpos]
                
                # add the best NC node to list
                ncnodes.append(newncnode)
                strncnodes.append(str(newncnode))
                remainingonlyfw.remove(newncnode)
                strremainingonlyfw.remove(str(newncnode))
                
                timestr = datetime.datetime.now().strftime("%b %d %X - ")
                print timestr + "NC nodes = ", ncnodes
                
                # delete all other meantimes files
                #resultsfilelist.pop(minpos)    # exclude winner
                #for m in resultsfilelist:
                #    os.remove(m)
                
            printconfiggreedytxt("configgreedy.txt", ncnodes)
            timestr = datetime.datetime.now().strftime("%b %d %X - ")
            print timestr + "Finished greedy search for minimum average delay"
            print
        
        if dogreedyflow:
            timestr = datetime.datetime.now().strftime("%b %d %X - ")
            print timestr + "Starting greedy search for maximum flow"
            ncnodes = []                        # the list of NC nodes found
            strncnodes = []                     # the list of NC nodes as strings, not numbers; used to build the file ending string, i.e. the list of NC nodes joined by '_'
            remainingonlyfw = list(orighelpers) # the list of remaining FW (i.e. non-NC) nodes
                                                #   not just =, this would make both names point to the same list
            strremainingonlyfw = []             # the list of remaining FW (i.e. non-NC) nodes, as strings
            for n in remainingonlyfw:
              strremainingonlyfw.append(str(n));

            while len(ncnodes) < N:
                configfilelist  = []    # list of new config files built in this iteration step
                resultsfilelist = []    # list of result files, which is read in the end to select the best NC node
                
                # Create current config files
                for currentncnode in remainingonlyfw:
                
                    # add the current node to the list of nc nodes
                    tempstrncnodes = list(strncnodes)
                    tempstrncnodes.append(str(currentncnode))
                    
                    # remove the current node from the list of fw nodes
                    tempstrremainingonlyfw = list(strremainingonlyfw)
                    tempstrremainingonlyfw.remove(str(currentncnode))
                    
                    # create the new file
                    newconfigfilename, newresultsfilename = createnewconfigfilefromexisting(os.path.join(pathtoconfigdir, templateconfigfile), pathtoconfigdir, tempstrncnodes, tempstrremainingonlyfw)
                    
                    # add the new flenames to the lists
                    configfilelist.append(newconfigfilename)
                    resultsfilelist.append(newresultsfilename)

                # print resultsfilelist            
                # run current config files
                
                # Run ppss
                if stepstoskip_flow == 0:
                    # create a file with the list of the new config files, to be fed to ppss
                    createfileslist(fileslistfile, pathtoconfigdir, configfilelist, "*")
                    # run ppss
                    timestr = datetime.datetime.now().strftime("%b %d %X - ")
                    print
                    print timestr + "Greedy search for max total flow, step " + str(len(ncnodes)+1) + " of " + str(N)
                    runppss(fileslistfile, pathtoexecutable, pathtoresultsdir, numproc)
                else:
                    # skip simulation
                    timestr = datetime.datetime.now().strftime("%b %d %X - ")
                    print
                    print timestr + "Skipping simulations, remaining steps to skip = ", stepstoskip_flow
                    stepstoskip_flow = stepstoskip_flow - 1


                # Read results, find the best NC node
                totalflows = []                 # list of total flows, for each NC node tested
                
                # compute total flows  
                for mfile in resultsfilelist:
                
                    # read
                    times,flows = readresults(os.path.join(pathtoresultsdir,mfile))
                    
                    # compute total
                    sum = 0
                    for f in flows:
                        sum = sum + f
                    totalflows.append(sum)

                # save flows in a file
                #with open("maximumflows.txt", "a") as f:
                #  f.write(' '.join(totalflows));                    
                try:
                  f = open("totalflows.txt", "a")
                except:
                  print "Cannot open file \"", "totalflows.txt", "\" for writing"
                  raise
                strtotals = []
                for total in totalflows:
                  strtotals.append(str(total))
                f.write(' '.join(strtotals) + "\n")
                f.close()
                
                # find maximum total flow
                maxpos = 0                  # node of maximum flow
                maxval = totalflows[0]      # maximum flow
                i = 1                       # current node
                for val in totalflows[1:]:
                    if val > maxval:
                        maxval = val
                        maxpos = i
                    i = i + 1
                  
                # this is the best NC node                
                #newncnode = helpers[maxpos]
                newncnode = remainingonlyfw[maxpos]
                
                # add the best NC node to list
                ncnodes.append(newncnode)
                strncnodes.append(str(newncnode))
                remainingonlyfw.remove(newncnode)
                strremainingonlyfw.remove(str(newncnode))
                
                timestr = datetime.datetime.now().strftime("%b %d %X - ")
                print timestr + "NC nodes =", ncnodes
                
                # delete all other meantimes files
                #resultsfilelist.pop(minpos)    # exclude winner
                #for m in resultsfilelist:
                #    os.remove(m)
                
            printconfiggreedytxt("configgreedymaxflow.txt", ncnodes)
            timestr = datetime.datetime.now().strftime("%b %d %X - ")
            print timestr + "Finished greedy search for maximum flow"
            print
            
        #print "Finished GREEDY"        
    timestr = datetime.datetime.now().strftime("%b %d %X - ")
    print timestr + "Finished everything, exiting"
    
if __name__ == "__main__":
    main(sys.argv[1:])
