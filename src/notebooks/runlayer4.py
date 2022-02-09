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
years = np.arange(2012,2017)
#years = np.arange(1979,2012)

geom = Geometry('CrossDots')
geom.coarsen(N=N)
geom = geom.create()

for yy in years:
    forc = Forcing(geom).mitgcm(startyear=yy,endyear=yy)
    layer = LayerModel(forc)
    layer.dt = 720
    layer.Ah = 80
    layer.Kh = 80
    layer.Cdtop = 1.07e-3
    layer.minD = 4.8
    layer.saveday = 5
    layer.restday = 100
    ds = layer.compute(days=25)
