
import numpy

import examples

def testCreate():
    n_helpers = 50
    minnnodes = 30
    rmin = 1
    rmax = numpy.Inf
    auto_option = 'a'
    maxtries = 100
    
    examples.generate_example('D:\\Nic\\Dev\\ExNC\\1', {'n_helpers': n_helpers, 'minnnodes': minnnodes, 'rmin': rmin, 'rmax': rmax, 'auto_option': auto_option, 'maxtries': maxtries})    

def testLoadRun():
    examples.load_example_and_run('..\\ExNC\\1')

if __name__ == "__main__":
    test2()