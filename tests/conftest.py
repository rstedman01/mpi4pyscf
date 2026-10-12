'''
Allow running the tests with pytest started on every MPI rank:

    mpiexec -np 2 pytest tests

mpi4pyscf suspends all ranks except rank 0 inside ``import mpi4pyscf`` and
exits them once rank 0 finishes.  Without this hook the workers would import
mpi4pyscf in the middle of test collection, and that exit would surface as a
pytest INTERNALERROR (SystemExit).  Instead, park the workers here, before any
test module is collected; only rank 0 runs pytest.  tests/runtests.sh, which
starts pytest on rank 0 only, keeps working as well.
'''


def pytest_cmdline_main(config):
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() != 0:
        import mpi4pyscf  # noqa: F401  Does not return; exits with status 0.
