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
years = [1973,1986]

geom = Geometry('CrossDots')
geom.coarsen(N=N)
geom = geom.create()

for yy in years:
    forc = Forcing(geom).mitgcm(startyear=yy,endyear=yy)
    layer = LayerModel(forc)
    layer.dt = 20
    layer.Ah = 6
    layer.Kh = 1
    layer.Cdtop = .000525
    layer.saveday = 5
    layer.restday = 100
    ds = layer.compute(days=15)
