
import numpy

import examples

#import unittest

def testCreate():
    n_helpers = 25
    minnnodes = 20
    rmin = 1
    rmax = numpy.Inf
    auto_option = 'a'
    maxtries = 100
    
    examples.generate_example('D:\\Nic\\Dev\\ExNC\\2', {'n_helpers': n_helpers, 'minnnodes': minnnodes, 'rmin': rmin, 'rmax': rmax, 'auto_option': auto_option, 'maxtries': maxtries})    

def testLoadRun_globaldelay():
    examples.load_example_and_run('..\\ExNC\\2', {'do_global_delay':1,
                                                  'do_global_flow':0,
                                                  'do_dist_delay':0,
                                                  'do_dist_flow':0})

def testLoadRun_globalflow():
    examples.load_example_and_run('..\\ExNC\\2', {'do_global_delay':0,
                                                  'do_global_flow':1,
                                                  'do_dist_delay':0,
                                                  'do_dist_flow':0})
                                                  
def testLoadRun_distdelay():
    examples.load_example_and_run('..\\ExNC\\2', {'do_global_delay':0,
                                                  'do_global_flow':0,
                                                  'do_dist_delay':1,
                                                  'do_dist_flow':0})                                                  
def testLoadRun_distflow():
    examples.load_example_and_run('..\\ExNC\\2', {'do_global_delay':0,
                                                  'do_global_flow':0,
                                                  'do_dist_delay':0,
                                                  'do_dist_flow':1})                                                  

def testLoadRun_distdelay_r1():
    examples.load_example_and_run('..\\ExNC\\2', {'do_global_delay':0,
                                                  'do_global_flow':0,
                                                  'do_dist_delay':1,
                                                  'do_dist_flow':0,
                                                  'rmin':1,
                                                  'rmax':1})                                                  

def profileLoadRun_globaldelay():
    
    import cProfile
    cProfile.run('import tests; tests.testLoadRun_globaldelay()', 'profile_testLoadRun_globaldelay')


#class testComputings(unittest.TestCase):
#    
#def setUp(self):
#    self.


if __name__ == "__main__":
    testLoadRun_globaldelay()