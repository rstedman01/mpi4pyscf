#!/bin/bash

# mpiexec -np 2 tests/runtests.sh
#
# pytest runs on rank 0 only.  The other ranks import mpi4pyscf, which parks
# them in the worker loop until rank 0 sends work.  The rank is read from the
# environment of whichever launcher is in use (Open MPI, Intel MPI/MPICH
# Hydra, PMIx, Slurm).

# https://www.python.org/dev/peps/pep-0565/
#export PYTHONWARNINGS=once # Warn once per Python process
#export PYTHONWARNINGS=ignore # Never warn

RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${SLURM_PROCID:-0}}}}

if [ "$RANK" == 0 ]; then
  if python -c 'import pytest_cov' 2>/dev/null; then
    COV="--cov-config=.coveragerc --cov-branch --cov-report=term-missing --cov=mpi4pyscf"
  fi
  pytest $COV --showlocals tests
else
  python -c 'import mpi4pyscf'
fi
