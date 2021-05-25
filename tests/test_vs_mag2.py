import pickle
import os
import numpy as np
import magpylib as mag3

# GENERATE TESTDATA ---------------------------------------
# import pickle
# import magpylib as magpy

# # linear motionfrom (0,0,0) to (3,-3,3) in 100 steps
# pm = magpy.source.magnet.Box(mag=(111,222,333), dim=(1,2,3))
# B1 = np.array([pm.getB((i,-i,i)) for i in np.linspace(0,3,100)])

# # rotation (pos_obs around magnet) from 0 to 444 deg, starting pos_obs at (0,3,0) about 'z'
# pm = magpy.source.magnet.Box(mag=(111,222,333), dim=(1,2,3))
# possis = [(3*np.sin(t/180*np.pi),3*np.cos(t/180*np.pi),0) for t in np.linspace(0,444,100)]
# B2 = np.array([pm.getB(p) for p in possis])

# # spiral (magnet around pos_obs=0) from 0 to 297 deg, about 'z' in 100 steps
# pm = magpy.source.magnet.Box(mag=(111,222,333), dim=(1,2,3), pos=(3,0,0))
# B = []
# for i in range(100):
#     B += [pm.getB((0,0,0))]
#     pm.rotate(3,(0,0,1),anchor=(0,0,0))
#     pm.move((0,0,.1))
# B3 = np.array(B)

# B = np.array([B1,B2,B3])
# pickle.dump(B, open('testdata_vs_mag2.p', 'wb'))
# -------------------------------------------------------------

def test_vs_mag2_linear():
    """ test against margpylib v2
    """
    data = pickle.load(open(os.path.abspath('tests/testdata/testdata_vs_mag2.p'),'rb'))[0]
    poso = [(t,-t,t) for t in np.linspace(0,3,100)]
    pm = mag3.magnet.Box(mag=(111,222,333), dim=(1,2,3))

    B = mag3.getB(pm, poso)
    assert np.allclose(B, data), 'vs mag2 - linear'


def test_vs_mag2_rotation():
    """ test against margpylib v2
    """
    data = pickle.load(open(os.path.abspath('tests/testdata/testdata_vs_mag2.p'),'rb'))[1]
    pm = mag3.magnet.Box(mag=(111,222,333), dim=(1,2,3))
    possis = [(3*np.sin(t/180*np.pi),3*np.cos(t/180*np.pi),0) for t in np.linspace(0,444,100)]
    B = pm.getB(possis)
    assert np.allclose(B, data), 'vs mag2 - rot'


def test_vs_mag2_spiral():
    """ test against margpylib v2
    """
    data = pickle.load(open(os.path.abspath('tests/testdata/testdata_vs_mag2.p'),'rb'))[2]
    pm = mag3.magnet.Box(mag=(111,222,333), dim=(1,2,3), pos=(3,0,0))

    angs = np.linspace(0,297,100)
    pm.rotate_from_angax(angs, 'z', anchor=0, start=0)
    pm.move([(0,0,.1)]*99, start=1, increment=True)
    B = pm.getB((0,0,0))
    assert np.allclose(B, data), 'vs mag2 - rot'