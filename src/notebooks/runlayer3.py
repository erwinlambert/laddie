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

N = 4
years = np.arange(1986,1991)

geom = Geometry('CrossDots')
geom.coarsen(N=N)
geom = geom.create()

for yy in years:
    forc = Forcing(geom).mitgcm(startyear=yy,endyear=yy)
    layer = LayerModel(forc)
    layer.dt = 70
    layer.Ah = 60
    layer.Kh = 10
    layer.Cdtop = .000675
    layer.saveday = 5
    layer.restday = 100
    ds = layer.compute(days=20)
