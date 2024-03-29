
import numpy
import scipy.io

hbuf = 0
sum_inverses_E1 = numpy.array([])
sum_inverses_E2 = numpy.array([])
sum_inverses_E1E2 = numpy.array([])
sum_inverses_E1E2_mat = numpy.array([])

if sum_inverses_E1.size == 0:
    mdict = scipy.io.loadmat('suminv_hbuf32.mat', squeeze_me=True)
    hbuf = mdict['hbuf'][()]
    sum_inverses_E1 = mdict['sum_inverses_E1']
    sum_inverses_E2 = mdict['sum_inverses_E2']
    sum_inverses_E1E2 = mdict['sum_inverses_E1E2']   
    sum_inverses_E1E2_mat = mdict['sum_inverses_E1E2_mat']   