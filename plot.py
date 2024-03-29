#/usr/bin/python

### Plots position estimates relative to "current location" or location that the images were taken from
### Current plot is set for Hawaii

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel(r'C:\Users\kkepa\Desktop\hokulele\starnav\positions.xlsx','estimated_positions_papa')
#cf = pd.read_excel(r'C:\Users\kkepa\Desktop\hokulele\starnav\positions.xlsx','img_centers_papa2')

fig = plt.figure(figsize=(12,9))

m = Basemap(projection='mill',
           llcrnrlat = 18,
           urcrnrlat = 25,
           llcrnrlon = -163,
           urcrnrlon = -153,
           resolution = 'i')

#m = Basemap(projection='mill', llcrnrlat = -90, urcrnrlat = 90, llcrnrlon = -180, urcrnrlon = 180, resolution = 'i')

#m.bluemarble()
m.drawcoastlines()
m.shadedrelief()

m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])


est_pos_lat_y = df['latitude'].tolist()
est_pos_lon_x = df['longitude'].tolist()

#cen_lat_y = cf['latitude'].tolist()
#cen_lon_x = cf['longitude'].tolist()

m.scatter(est_pos_lon_x, est_pos_lat_y, latlon = True, s = 10, c = 'red', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)
#m.scatter(cen_lon_x, cen_lat_y, latlon = True, s = 50, c = 'blue', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)
m.scatter(-157.8581, 21.3099, latlon = True, s = 50, c = 'green', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)

plt.title('StarNav Position Estimate', fontsize=20)

plt.show()

