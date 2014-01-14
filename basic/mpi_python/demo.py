import numpy as np
from mpi4py import MPI

def myrank():
    return MPI.COMM_WORLD.Get_rank()

def psum(a):
    locsum = np.sum(a)
    rcvBuf = np.array(0.0, 'd')
    MPI.COMM_WORLD.Allreduce([locsum, MPI.DOUBLE], 
        [rcvBuf, MPI.DOUBLE], op=MPI.SUM)
    return rcvBuf

rank = myrank()
a = np.arange(rank*5, (rank+1)*5, dtype='float')
sum = psum(a)

print "At node %d: local sum = %d" % (rank, np.sum(a))
print "At node %d: total sum = %d" % (rank, sum)
