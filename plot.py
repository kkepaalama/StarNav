#/usr/bin/python

### Plots position estimates relative to "current location" or location that the images were taken from
### Current plot is set for Hawaii

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel(r'C:\Users\kaiak\Desktop\hokulele\starnav\centroids1.xlsx','Sheet1')
#cf = pd.read_excel(r'C:\Users\kkepa\Desktop\hokulele\starnav\positions.xlsx','img_centers_papa2')

fig = plt.figure(figsize=(12,9))

m = Basemap(projection='mill',
           llcrnrlat = 18,
           urcrnrlat = 26,
           llcrnrlon = -160,
           urcrnrlon = -150,
           resolution = 'i')

#m = Basemap(projection='mill', llcrnrlat = -90, urcrnrlat = 90, llcrnrlon = -180, urcrnrlon = 180, resolution = 'i')

#m.bluemarble()
m.drawcoastlines()
m.shadedrelief()

m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])


pos_lat_y = df['latitude'].tolist()
pos_lon_x = df['longitude'].tolist()

#cen_lat_y = cf['latitude'].tolist()
#cen_lon_x = cf['longitude'].tolist()

#m.scatter(pos_lon_x, pos_lat_y, latlon = True, s = 10, c = 'red', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)
#m.scatter(cen_lon_x, cen_lat_y, latlon = True, s = 50, c = 'blue', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)
m.scatter(-157.8581, 21.3099, latlon = True, s = 50, c = 'green', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)
m.scatter(-157.4324935122459, 21.989404363888454, latlon = True, s = 50, c = 'b', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)
m.scatter(-157.37439052661568, 22.002703758654306, latlon = True, s = 50, c = 'b', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)
m.scatter(-158.6618406134087, 21.511097894412615, latlon = True, s = 50, c = 'b', marker = 'o', alpha = 1, edgecolor = 'k', linewidth = 1, zorder = 2)

plt.title('Global Position', fontsize=20)

plt.show()

