
import numpy as np, json
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2.algebra.io import MPOTools

data = np.load('00-ints.npz')

ncas = int(data['ncas'])
n_elec = int(data['n_elec'])
spin = int(data['spin'])
h1e = data['h1e']
g2e = data['g2e']
ecore = float(data['ecore'])

driver = DMRGDriver(scratch="/tmp", symm_type=SymmetryTypes.SAny, stack_mem=200 << 30, n_threads=64)
driver.set_symmetry_groups("U1", "U1")
Q = driver.bw.SX

driver.initialize_system(n_sites=ncas, vacuum=Q(0, 0), target=Q(n_elec, spin), hamil_init=False)
site_basis = [[(Q(0, 0), 1), (Q(1, 1), 1), (Q(1, -1), 1), (Q(2, 0), 1)]] * ncas
site_ops = [{"": np.identity(4)}] * ncas
driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)

print('loading mpo ...')

with open('03-itensor-spqr-mpo.json', 'r') as file:
    mpoJSON = json.load(file)

totalNNZ = 0
for dict in mpoJSON:
    for (tag, shape, arr) in dict["D"]:
        totalNNZ += len(arr)
        
print(f"nnz from json = {totalNNZ}")

pympo = MPOTools.from_itensor(mpoJSON)
mpo = MPOTools.to_block2(pympo, driver.basis, add_ident=True)

nnz, size, max_bond = 0, 0, 0
for k, t in enumerate(mpo.tensors):
    xnnz = len(t.lmat.data)
    xsize = t.lmat.m * t.lmat.n
    max_bond = max(max_bond, t.lmat.m, t.lmat.n)
    nnz += xnnz
    size += xsize

print('sparsity = %.4f' % ((size - nnz) / size), 'max_bond =', max_bond, 'nnz =', nnz, 'size =', size)

# bond_dims = [500] * 4 + [1000] * 4 + [2000] * 4 + [3000] * 4
# noises = [1E-5] * (len(bond_dims) - 2) + [0] * 2
# thrds = [1E-7] * len(bond_dims)
# n_sweeps = len(bond_dims)

# ket = driver.get_random_mps(tag="KET", bond_dim=500, nroots=1)
# energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds, tol=1E-12, cutoff=1E-24, iprint=2)
# print('DMRG energy = %20.15f' % energy)
