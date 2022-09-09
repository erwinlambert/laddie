import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from geometry import Geometry
from forcing import Forcing
from layer import LayerModel

from plotfunctions import prettyplot

import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

N = 2

geom = Geometry('FRIS')
geom.coarsen(N=N)
geom = geom.create()

forc = Forcing(geom).linear(S1=34.8,T1=-2.3)
layer = LayerModel(forc)

layer.dt = 216

layer.minD = 4.4

layer.Ah = 25*N
layer.Kh = 25*N

layer.utide = .1

layer.saveday = 10
layer.restday = 10

ds = layer.compute(days=380,restartfile='linear_S134.8_T1-2.3_340')

