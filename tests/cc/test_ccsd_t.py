#!/usr/bin/env python
'''
Parallel test of CCSD and CCSD(T) using RHF reference against serial PySCF implementations.

    mpiexec -np 2 tests/runtests.sh
'''

import pytest 
import numpy
from pyscf import gto, scf, cc
from mpi4pyscf import cc as mpicc

@pytest.fixture(scope='module')
def get_mf():
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='ccpvtz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf.run()

@pytest.mark.parametrize('frozen', [0,1])
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
    
    numpy.testing.assert_allclose(mycc.e_corr, ref.e_corr, atol=1e-9, rtol=0)
    numpy.testing.assert_allclose(et, et_ref, atol=1e-9, rtol=0)
