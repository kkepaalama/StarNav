import numpy as np
import random as ran
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
import main
import frame_trans as f

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

def R(tilt):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(tilt[0])), -np.sin(tilt[0])],
                      [0, np.sin(tilt[0]), np.cos(tilt[0])]])
    y_rot = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])],
                      [0, 1, 0],
                      [-np.sin(tilt[1]), 0, np.cos(tilt[1])]])
    z_rot = np.array([[np.cos(tilt[2]), -np.sin(tilt[2]), 0],
                      [np.sin(tilt[2]), np.cos(tilt[2]), 0],
                      [0, 0, 1]])
    R = np.dot(z_rot, np.dot(y_rot, x_rot))
    return R

#earth frame
'''
The following shows how vectors are rotated around a specific origin
'''
oE = np.array([0, 0, 0]) 
xE = np.array([1, 0, 0])
yE = np.array([0, 1, 0])
zE = np.array([0, 0, 1])

ax.quiver(oE[0], oE[1], oE[2], xE[0], xE[1], xE[2], color = 'red')
ax.quiver(oE[0], oE[1], oE[2], yE[0], yE[1], yE[2], color = 'blue')
ax.quiver(oE[0], oE[1], oE[2], zE[0], zE[1], zE[2], color = 'green')

'''tilt = np.radians([0, 0, 0])
R = Rotation_XYZ(tilt)

u = np.array([0, 1, 0])
v = np.array([-1, 0, 0])
w = zE

u, v, w = np.dot(R, u), np.dot(R, v), np.dot(R, w)'''

E_EoS0 = f.E_EoS0
E_EoS1 = f.E_EoS1
E_EoS2 = f.E_EoS2

ax.quiver(oE[0], oE[1], oE[2], E_EoS0[0], E_EoS0[1], E_EoS0[2], color = 'm')
ax.quiver(oE[0], oE[1], oE[2], E_EoS1[0], E_EoS1[1], E_EoS1[2], color = 'm')
ax.quiver(oE[0], oE[1], oE[2], E_EoS2[0], E_EoS2[1], E_EoS2[2], color = 'm')
#ax.quiver(zE[0], zE[1], zE[2], w[0], w[1], w[2], color = 'y')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.view_init(elev=30, azim=50)

plt.show()