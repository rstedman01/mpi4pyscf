#!/usr/bin/env python

'''
Parallel RCCSD and RCCSD(T) against the serial PySCF implementation.

    mpiexec -np 2 tests/runtests.sh
'''

import pytest
from pyscf import gto, scf, cc
from mpi4pyscf import cc as mpicc


@pytest.fixture(scope='module')
def get_mf():
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='cc-pvdz', verbose=0)
    return scf.RHF(mol).run()


@pytest.mark.parametrize('frozen', [0, 1])
def test_ccsd_and_ccsd_t(get_mf, frozen):
    mf = get_mf
    ref = cc.CCSD(mf, frozen=frozen)
    ref.conv_tol = 1e-10
    ref.conv_tol_normt = 1e-8
    ref.kernel()
    et_ref = ref.ccsd_t()

    mycc = mpicc.RCCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    et = mycc.ccsd_t()
    assert mycc.converged
    assert abs(mycc.e_corr - ref.e_corr) < 1e-8
    assert abs(et - et_ref) < 1e-8
