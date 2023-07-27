#usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

df = pd.read_excel(r'C:\Users\kkepa\Desktop\hokulele\starnav\latLon_vs_rollPitch.xlsx')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

latitude_error = df['lat_error']
longitude_error = df['lon_error']
plt.plot(longitude_error, latitude_error, 'o', label = 'Error')
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Latitude (degrees)')
ax.set_title('Coordinate Error')


latitude = 0
longitude = 0
plt.plot(longitude, latitude, 'o', c = 'green', label = 'Actual')
plt.legend()

plt.show()
