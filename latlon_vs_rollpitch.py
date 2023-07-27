#usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
#latitude = np.array([22.30006652, 22.26184008, 22.18251118, 22.01234732, 22.21244253, 22.07986719, 22.68104832, 21.85034543, 22.07429846, 22.21935624, 22.45302453, 22.65584406, 22.20992682, 21.56527863, 21.78656881, 21.87672805, 21.0730975, 21.98690266, 21.91204009, 21.811202, 21.88633561, 21.94283975, 21.75900804, 21.84191872, 21.64412811, 22.02471482, 22.13565317, 21.90269206, 21.94208585, 22.07866022, 21.8814611, 22.11892694, 22.01960898])
#longitude = np.array([-159.2900442, -159.3369935, -159.3109512, -159.1590225, -159.2862804, -159.2837478, -159.1343074, -159.1888043, -159.3006886, -159.3251747, -159.2669824, -159.3381097, -159.2004542, -158.8406872, -159.0906407, -159.125117, -157.4209439, -159.1210579, -159.2635101, -159.2137806, -159.1546028, -159.0589357, -159.2487443, -158.9427703, -159.0379061, -159.2287246, -159.3165537, -159.2229679, -159.126432, -159.2672091, -159.3001824, -159.0943018, -159.3142514])
#roll = np.array([0.06161488, 0.060681908, 0.05953222, 0.057906205, 0.060259497, 0.058298702, 0.068824302, 0.05510514, 0.057991775, 0.059890519, 0.062596178, 0.058811718, 0.060632254, 0.375331264, 0.376835452, 0.378779722, 0.378736781, 0.380152567, 0.378555679, 0.37689549, 0.378779722, 0.379579552, 0.376228376, 0.379753832, 0.375896685, 0.295193476, 0.29610358, 0.293371759, 0.296018158, 0.295569835, 0.292415983, 0.297692438, 0.29436636])
#pitch = np.array([0.033877646, 0.034438217, 0.033513569, 0.030113758, 0.033418451, 0.032541805, 0.034838711, 0.029066929, 0.032871173, 0.034155746, 0.033161109, 0.034966874, 0.032652942, 0.058054058, 0.063744324, 0.064284275, 0.063998795, 0.065165002, 0.066036481, 0.064648014, 0.064521706, 0.063610049, 0.064747358, 0.06095149, 0.060791207, 0.078102709, 0.080208081, 0.076962765, 0.076904524, 0.079274883, 0.078047433, 0.077347295, 0.079481394])

#df = pd.read_excel(r'C:\Users\kkepa\Desktop\hokulele\starnav\latLon_vs_rollPitch.xlsx')

'''fig, ax = plt.subplots()

ax.scatter(x = df['latitude'], y = df['roll'])
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel("roll (radians)")


fig, ax = plt.subplots()

ax.scatter(x = df['latitude'], y = df['pitch'])
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel("pitch (radians)")


fig, ax = plt.subplots()

ax.scatter(x = df['longitude'], y = df['roll'])
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("roll (radians)")


fig, ax = plt.subplots()

ax.scatter(x = df['longitude'], y = df['pitch'])
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("pitch (radians)")'''



df = pd.read_excel(r'C:\Users\kkepa\Desktop\hokulele\starnav\latLon_vs_rollPitch.xlsx')

fig, ax = plt.subplots(2)
ax[0].scatter(y = df['latitude'], x = df['roll'])
ax[0].set_ylabel("Latitude (degrees)")
ax[0].set_xlabel("Roll (radians)")
ax[0].errorbar(x = df['roll'], y = df['latitude'], yerr = df['lat_error'], fmt = 'o', elinewidth = 0.5, capsize = 1)
#ax[0].set_title('Latitude vs. Roll')

ax[1].scatter(y = df['latitude'], x = df['pitch'])
ax[1].set_ylabel("Latitude (degrees)")
ax[1].set_xlabel("Pitch (radians)")
ax[1].errorbar(x = df['pitch'], y = df['latitude'], yerr = df['lat_error'], fmt = 'o', elinewidth = 0.5, capsize = 1)
#ax[1].set_title('Latitude vs. Pitch')


fig, ax = plt.subplots(2)
ax[0].scatter(y = df['longitude'], x = df['roll'])
ax[0].set_ylabel("Longitude (degrees)")
ax[0].set_xlabel("Roll (radians)")
ax[0].errorbar(x = df['roll'], y = df['longitude'], yerr = df['lon_error'], fmt = 'o', elinewidth = 0.5, capsize = 1)
#ax[0].set_title('Longitude vs. Roll')

ax[1].scatter(y = df['longitude'], x = df['pitch'])
ax[1].set_ylabel("Longitude (degrees)")
ax[1].set_xlabel("Pitch (radians)")
ax[1].errorbar(x = df['pitch'], y = df['longitude'], yerr = df['lon_error'], fmt = 'o', elinewidth = 0.5, capsize = 1)
#ax[1].set_title('Longitude vs. Pitch')

plt.show()
