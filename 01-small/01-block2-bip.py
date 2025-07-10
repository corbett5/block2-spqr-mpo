
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import numpy as np

data = np.load('00-ints.npz')

ncas = int(data['ncas'])
n_elec = int(data['n_elec'])
spin = int(data['spin'])
h1e = data['h1e']
g2e = data['g2e']
ecore = float(data['ecore'])

driver = DMRGDriver(scratch="/tmp", symm_type=SymmetryTypes.SZ, stack_mem=200 << 30, n_threads=64)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)

b = driver.expr_builder()

for i in range(ncas):
    for j in range(ncas):
        if abs(h1e[i, j]) >= 1E-14:
            b.add_term('cd', [i, j], h1e[i, j])
            b.add_term('CD', [i, j], h1e[i, j])
for i in range(ncas):
    for j in range(ncas):
        for k in range(ncas):
            for l in range(ncas):
                if abs(g2e[i, j, k, l]) >= 1E-14:
                    if i != k and j != l:
                        b.add_term('ccdd', [i, k, l, j], 0.5 * g2e[i, j, k, l])
                        b.add_term('CCDD', [i, k, l, j], 0.5 * g2e[i, j, k, l])
                    b.add_term('cCDd', [i, k, l, j], 0.5 * g2e[i, j, k, l])
                    b.add_term('CcdD', [i, k, l, j], 0.5 * g2e[i, j, k, l])
b.add_term('', [], float(ecore))

mpo = driver.get_mpo(b.finalize(), algo_type=MPOAlgorithmTypes.FastBipartite, iprint=2)

nnz, size, max_bond = 0, 0, 0
for k, t in enumerate(mpo.tensors):
    xnnz = len(t.lmat.data)
    xsize = t.lmat.m * t.lmat.n
    max_bond = max(max_bond, t.lmat.m, t.lmat.n)
    nnz += xnnz
    size += xsize

print('sparsity = %.4f' % ((size - nnz) / size), 'max_bond =', max_bond, 'nnz =', nnz, 'size =', size)

bond_dims = [500] * 4 + [1000] * 4 + [2000] * 4 + [3000] * 4
noises = [1E-5] * (len(bond_dims) - 2) + [0] * 2
thrds = [1E-7] * len(bond_dims)
n_sweeps = len(bond_dims)

ket = driver.get_random_mps(tag="KET", bond_dim=500, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds, tol=1E-12, cutoff=1E-24, iprint=2)
print('DMRG energy = %20.15f' % energy)
