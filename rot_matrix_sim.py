#usr/bin/python

###Visual simulation for euler rotaion sequences. The origin frame is fixed to show the progression of the rotating frame.

import matplotlib.pyplot as plt
import numpy as np
from main import Rot, R, R_inverse

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
#ax.plot_surface(x, y, z, color = 'w')

# if i dont set these, the plot is all zoomed in
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
origo= np.array([0,0,0])
ax.view_init(30, 45)

#Local frame: this frame is fixed
xL, yL, zL = [1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]
ax.quiver(origo[0],origo[1],origo[2], xL[0], xL[1], xL[2],  color="r", alpha = 0.5)#,normalize=True) #x-axis
ax.quiver(origo[0],origo[1],origo[2], yL[0], yL[1], yL[2], color="g", alpha = 0.5)#,normalize=True) #y-axis
ax.quiver(origo[0],origo[1],origo[2], zL[0], zL[1], zL[2], color="b", alpha = 0.5)#,normalize=True) #z-axis or normal vector
#ax.quiver(origo[0],origo[1],origo[2], 0, 0, -1, color="orange", alpha = 0.5)#,normalize=True) #gravity vector 

# Body frame: this frame rotates
rad = np.pi/180
tilt = [30*rad, 10*rad, 10*rad]
R = R(tilt)
x_B, y_B, z_B = [1, 0, 0], [0, 1, 0], [0, 0, 1]
Rx, Ry, Rz = np.dot(R, x_B), np.dot(R, y_B), np.dot(R, z_B)
ax.quiver(origo[0],origo[1],origo[2], Rx[0], Rx[1], Rx[2], color="c")#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], Ry[0], Ry[1], Ry[2], color="y")#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], Rz[0], Rz[1], Rz[2], color="m")#,normalize=True)


# verification1: shows counter rotation
# determine how to rotate dashed quivers to match r, g, b quivers
rad = np.pi/180
#tilt = [30*rad, 10*rad, 10*rad]

R_in = R_inverse(tilt)

xB, yB, zB = np.dot(R_in, Rx)*1.2, np.dot(R_in, Ry)*1.2, np.dot(R_in, Rz)*1.2

'''xb = Rot(Rx, xL)
yb = Rot(Ry, yL)
zb = Rot(Rz, zL)

xB = np.dot(xb, Rx)*1.2
yB = np.dot(yb, Ry)*1.2
zB = np.dot(zb, Rz)*1.2'''

ax.quiver(origo[0],origo[1],origo[2], xB[0], xB[1], xB[2], color="c", linestyle = 'dashed')#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], yB[0], yB[1], yB[2], color="y", linestyle = 'dashed')#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], zB[0], zB[1], zB[2], color="m", linestyle = 'dashed')#,normalize=True)

#ax.quiver(origo[0],origo[1],origo[2], Rx[0], Rx[1], Rx[2], color="c", linestyle = 'dashed')#,normalize=True)
#ax.quiver(origo[0],origo[1],origo[2], Ry[0], Ry[1], Ry[2], color="y", linestyle = 'dashed')#,normalize=True)
#ax.quiver(origo[0],origo[1],origo[2], Rz[0], Rz[1], Rz[2], color="m", linestyle = 'dashed')#,normalize=True)

# right pseudo inverse
#np.linalg.pinv(a)


# axis label placement
ax.text(0.1, 0.0, -0.2, r'$0$')
ax.text(1.3, 0, 0, r'$x$')
ax.text(0, 1.3, 0, r'$y$')
ax.text(0, 0, 1.3, r'$z$')

# Set an equal aspect ratio
ax.set_aspect('auto')


plt.show()
