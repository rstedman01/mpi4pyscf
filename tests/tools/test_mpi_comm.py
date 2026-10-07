#!/usr/bin/env python
'''
Tests for the communication helpers in mpi4pyscf.tools.mpi.
Run as for the other tests, with pytest on rank 0 only (see runtests.sh):
    mpiexec -np 3 tests/runtests.sh
Each check is a function that is executed on all ranks through mpi.pool.apply.
The functions are shipped to the workers as bare code objects, so they must
import everything they need locally and must not refer to module-level names.
mpi.BLKSIZE is made tiny so that the chunked communication loops (which protect
against the 2**31-1 element limit of MPI-3 counts) are really exercised.
'''
import pytest
from mpi4pyscf.tools import mpi

def _run(fn, *args):
    return mpi.pool.apply(fn, args, args)

def _bcast(n,m,dtype):
    import numpy as np
    from mpi4pyscf.tools import mpi
    
    old_blksize, mpi.BLKSIZE = mpi.BLKSIZE, 7
    ref = np.arange(n*m).reshape(n,m) * np.ones(1,dtype=dtype)
    ref = ref * (1+1j if np.dtype(dtype).kind == 'c' else 1)
    res = mpi.bcast(ref if mpi.rank == 0 else None)
    return bool(mpi.comm.allreduce(np.array_equal(res, ref), op=mpi.MPI.LAND))

def _reduce(n,m,dtype):
    import numpy
    from mpi4pyscf.tools import mpi
    old_blksize, mpi.BLKSIZE = mpi.BLKSIZE, 7
    size = mpi.pool.size
    base = numpy.arange(n*m).reshape(n, m).astype(dtype)
    ref = base * (size*(size+1)//2)
    res = mpi.allreduce(base * (mpi.rank+1))
    ok = numpy.allclose(res, ref)
    res = mpi.reduce(base * (mpi.rank+1))
    if mpi.rank == 0:
        ok = ok and numpy.allclose(res, ref)
    return bool(mpi.comm.allreduce(ok, op=mpi.MPI.LAND))


def _gather(n, m):
    import numpy
    from mpi4pyscf.tools import mpi
    old_blksize, mpi.BLKSIZE = mpi.BLKSIZE, 7
    size = mpi.pool.size
    parts = [numpy.full((n+k, m), k+1.) for k in range(size)]
    mine = parts[mpi.rank]
    ok = True
    res = mpi.allgather(mine)
    ok = ok and numpy.array_equal(res, numpy.vstack(parts))
    res = mpi.allgather(mine, split_recvbuf=True)
    ok = ok and all(numpy.array_equal(a, b) for a, b in zip(res, parts))
    res = mpi.gather(mine)
    if mpi.rank == 0:
        ok = ok and numpy.array_equal(res, numpy.vstack(parts))
        res = None
    res = mpi.gather(mine, split_recvbuf=True)
    if mpi.rank == 0:
        ok = ok and all(numpy.array_equal(a, b) for a, b in zip(res, parts))
    return bool(mpi.comm.allreduce(ok, op=mpi.MPI.LAND))


def _scatter(n, m):
    import numpy
    from mpi4pyscf.tools import mpi
    old_blksize, mpi.BLKSIZE = mpi.BLKSIZE, 7
    size = mpi.pool.size
    parts = [numpy.random.RandomState(k).random_sample((n+k, m-k))
             for k in range(size)]
    res = mpi.scatter(parts if mpi.rank == 0 else None)
    return bool(mpi.comm.allreduce(numpy.array_equal(res, parts[mpi.rank]),
                                   op=mpi.MPI.LAND))


def _alltoall(n, m):
    import numpy
    from mpi4pyscf.tools import mpi
    old_blksize, mpi.BLKSIZE = mpi.BLKSIZE, 7
    size, rank = mpi.pool.size, mpi.rank
    sendbuf = [numpy.full((n+i-rank, m-i+rank), 10.*rank+i) for i in range(size)]
    res = mpi.alltoall(sendbuf, split_recvbuf=True)
    ok = all(numpy.array_equal(res[r], numpy.full((n+rank-r, m-rank+r), 10.*r+rank))
             for r in range(size))
    res = mpi.alltoall(sendbuf)
    ref = numpy.hstack([numpy.full((n+rank-r)*(m-rank+r), 10.*r+rank)
                        for r in range(size)])
    ok = ok and numpy.array_equal(res, ref)
    return bool(mpi.comm.allreduce(ok, op=mpi.MPI.LAND))


def _rotate(n, m, blocking):
    import numpy
    from mpi4pyscf.tools import mpi
    old_blksize, mpi.BLKSIZE = mpi.BLKSIZE, 7
    size, rank = mpi.pool.size, mpi.rank
    sendbuf = numpy.full((n, m), float(rank))
    res = mpi.rotate(sendbuf, blocking=blocking)
    return bool(mpi.comm.allreduce(
        numpy.array_equal(res, numpy.full((n, m), float((rank+1) % size))),
        op=mpi.MPI.LAND))


def _work_partition(ntasks, kind):
    '''Every task is handled exactly once, and the iteration terminates.'''
    from mpi4pyscf.tools import mpi
    tasks = list(range(ntasks))
    if kind == 'share':
        mine = list(mpi.work_share_partition(tasks, interval=.01))
    elif kind == 'steal':
        mine = list(mpi.work_stealing_partition(tasks, interval=.01))
    else:
        mine = list(mpi.static_partition(tasks))
    done = mpi.comm.gather(mine)
    if mpi.rank == 0:
        return sorted(x for d in done for x in d) == tasks
    return True


@pytest.mark.parametrize('dtype', ['f8', 'c16'])
def test_bcast(dtype):
    assert _run(_bcast, 11, 5, dtype)


@pytest.mark.parametrize('dtype', ['f8', 'c16'])
def test_reduce(dtype):
    assert _run(_reduce, 9, 4, dtype)


def test_gather_allgather():
    assert _run(_gather, 6, 5)


def test_scatter():
    assert _run(_scatter, 12, 9)


def test_alltoall():
    assert _run(_alltoall, 12, 9)


@pytest.mark.parametrize('blocking', [True, False])
def test_rotate(blocking):
    assert _run(_rotate, 10, 7, blocking)


@pytest.mark.parametrize('kind', ['static', 'share', 'steal'])
def test_work_partition(kind):
    assert _run(_work_partition, 40, kind)