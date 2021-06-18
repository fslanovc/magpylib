"""
Test computation of Cylinder implementation
"""

import numpy as np
import _run_analytic_paper_final

# observer positions (cylinder CS) r,phi,z, units: [m], [rad]
obs_pos = np.array([(0,.6,3), (1,np.pi,4), (2,2*np.pi,5)])

# cylinder dimensions (cylinder CS) r1,r2,phi1,phi2,z1,z2, units: [m], [rad]
dim = np.array([
    (0, 2, .6, np.pi, 3, 5),
    (1, 3, .1, 4.5, 4, 6),
    (3, 5, 0, 2*np.pi, 6, 10),
    (0, .1, .1, .2, 1, 2)])

# magnetization vectors (spherical CS) [A/m], [rad]
mag = np.array([
    (.7, .7, .3),
    (.8, .8, .4),
    (.9, .9, .5),
    (1, 1, .6)])

results = _run_analytic_paper_final.H_total_final(obs_pos, dim, mag)
print(results)
#np.save('_data_test_results',np.nan_to_num(results))
test = np.load('_data_test_results.npy')
assert np.allclose(np.nan_to_num(results), test)
#print(results)

obs_pos = np.array([(0,0,2)])
dim = np.array([(0,1,0,2*np.pi,-1,1)])
mag = np.array([(1,0,np.pi/2)])
results = _run_analytic_paper_final.H_total_final(obs_pos, dim, mag)
print(results)

import magpylib as mag3
cyl = mag3.magnet.Cylinder((1,0,0), (2,2))
print(cyl.getB(0,0,2))


