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

N = 1
years = np.arange(1980,2012)

geom = Geometry('CrossDots')
geom.coarsen(N=N)
geom = geom.create()

for yy in years:
    forc = Forcing(geom).mitgcm(startyear=yy,endyear=yy)
    layer = LayerModel(forc)
    layer.dt = 144
    layer.Ah = 20
    layer.Kh = 20
    layer.Cdtop = 1.23e-3
    layer.minD = 3.2
    layer.saveday = 5
    layer.restday = 100
    ds = layer.compute(days=20)
