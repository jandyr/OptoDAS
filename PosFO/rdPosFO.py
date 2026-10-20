import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from pyproj import CRS, Transformer
import pandas as pd
#import csv
#\__________Local functions__________/
#           +-> path[0] is reserved for script path
import utils as u
import plot as p
#\_____________________________________________/
#
# -------------- Main   ---------------------
#------------ Read S FO position file
data = pd.read_csv("../Data/CableS.xyz", sep='\t', names=['lat','lon','depth'], header=None,comment='>')   #, delim_whitespace=True 
data = np.column_stack((data['lat'], data['lon'], data['depth']))
#    lat, lon, z = data[:,0], data[:,1], data[:,2]
#---  Limits distance to the first repetition station
dummy = u.repSt()
# dummy = (lat, lon)
dummy = np.array(dummy)
dist = np.linalg.norm(data[:, :2] - dummy, axis=1)
idx  = np.argmin(dist)
data = data[:idx+1]
#
print(f">> First repetition station at ({dummy[0]:.5f}°N,{dummy[1]:.5f}°W)")
print(f"   FO is limited to: {data[idx,0]:.5f}°N, {data[idx,1]:.5f}°W, {data[idx,2]:.0f}m")
#
#------------ Transform to a Cartesian frame
#---  Initialize pyproj.crs.CRS class
#      UTM WGS84 10N        EPSG EPSG:32610 
crs_geod = CRS.from_epsg(4326)
crs_proj = CRS.from_proj4("+proj=utm +zone=10 +north +datum=WGS84")
transformer = Transformer.from_crs(crs_geod, crs_proj, always_xy=True)
#---  Transform coordinates to utm
x, y = transformer.transform(data[:,1], data[:,0])
#                                lon        lat
#    Y = -Y  # flip latitude
#---  Transform to a new origin at the beach
x = x - x[0]
y = y - y[0]
#---  Reconstruct matrix data
data = np.column_stack((x, y, data[:, 2]))
#
#------------ Limits distance along FO
#    dist = 0
dist = [0]
for i in range(1, data.shape[0]):
    dist.append(np.linalg.norm(data[i] - data[i-1]))
#        dist += np.linalg.norm(data[i] - data[i-1])
dist  = np.array(dist)
dummy = np.cumsum(dist)
print(f"\n>> The total distance of {dist.shape[0]} of the OptoDAS mapped points is {dummy[-1]:.1f}m")
dist  = dummy
#
#------------ Find the best few intial points that better fit a 3-D line
# Initialize variables for finding the best fit
minRes = float('inf')
best_k = -1
# Residuals for different k
res = [] 
#---  Maximum number of initial points to check. Cap it at 10
k_max = min(data.shape[0], 10)
print(f"\n>> Checking collinearity for the first {k_max} points")
#---  Iterate from k=3 up to k_max
for k in range(3, k_max + 1):
    subset = data[0:k]
    totSqRes = u.fit_res(subset)
#---  Calculate residual per point as a metric for comparison
    pointRes = totSqRes / k
#
    res.append({
        'k': k,
        'total_squared_residuals': totSqRes,
        'pointRes': pointRes
    })
#    print(f"  First {k} points: Total squared residuals = {totSqRes:.4e}, Residual per point = {pointRes:.4e}")
#---  Update best_k if a better fit (lower residual per point) is found
    if pointRes < minRes:
        minRes = pointRes
        best_k = k
#
if best_k != -1:
    best = data[0:best_k]
    d = [0]
    for i in range(1, best.shape[0]):
        d.append(np.linalg.norm(best[i] - best[i-1]))
    d = np.array(d)
    d = np.cumsum(d)    
    print(f"\n>> The 1st {best_k} FO points along {d[-1]:.0f}m have the best linear fit.")
else:
    print("Could not determine a best fit")
#
#------------ Visualize
while True:
    ent = input(f'<< The FO is {dist[-1] / 1000.:.1f}km. Limit to D[km] (full = dflt/rtn, <0=stop):')
    ent = None if not ent else float(ent.rstrip())
#---  limits
    if ent is not None and ent > 0:
        ent = ent * 1000.
        idx = np.argmin(np.abs(dist - ent))
        p.pltFO(data[:idx+1])
    elif ent is not None and ent < 0:
#---  Break free
        break
    else:
        p.pltFO(data)
#
#------------ 

