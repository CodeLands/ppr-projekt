
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Rank {rank} on {os.uname().nodename}: ulimit -n = {os.popen('ulimit -n').read().strip()}")
