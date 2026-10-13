#!/usr/bin/env python
import sys

import numpy
from pyscf import gto, dft
import mpi4pyscf
from mpi4pyscf import dft as mpi_dft

import pytest

@pytest.fixture
def get_mol(scope='module'):
    mol = gto.M(atom='''
O 0 0     0
H 0 -.757 .587
H 0  .757 .587''',
                basis='cc-pvdz')
    return mol


def test_veff(get_mol):
    mol = get_mol
    nao = mol.nao
    numpy.random.seed(1)
    dm = numpy.random.random((2,nao,nao))
    dm = dm + dm.transpose(0,2,1)
    dm *= .01
    dm += mol.UKS(xc='blyp').get_init_guess()

    mf = mpi_dft.RKS(mol)
    mf.xc = 'b3lyp'
    vxc = mf.get_veff(mol, dm)
    mf0 = mol.RKS(xc='b3lyp')
    vxc0 = mf0.get_veff(mol, dm)
    assert abs(vxc0-vxc).max() < 1e-9

    mol1 = mol.copy()
    mol1.spin = 2
    mf = mpi_dft.UKS(mol1)
    mf.xc = 'b3lyp'
    vxc = mf.get_veff(mol1, dm)
    mf0 = mol1.UKS(xc='b3lyp')
    vxc0 = mf0.get_veff(mol1, dm)
    assert abs(vxc0-vxc).max() < 1e-9

def test_mpi_uks(get_mol):
    mol = get_mol
    
    mf = mpi_dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.direct_scf_tol = 1e-9
    empi = mf.kernel()
    
    mf0 = mol.UKS(xc='b3lyp')
    mf0.direct_scf_tol = 1e-9
    eref = mf0.kernel()
    
    assert mf.converged
    assert mf0.converged
    numpy.testing.assert_allclose(empi, eref, rtol=0, atol=1e-9)

