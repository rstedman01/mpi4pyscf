def pytest_cmdline_main(config):
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() != 0:
        import mpi4pyscf