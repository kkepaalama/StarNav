import matplotlib.pyplot as plt
import numpy as np
from main import raw_cent, cent, cel2ecef, B, K, q, n, imu2cam, pos, R, Rot
from coords_papakolea import s1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

raw_cent = raw_cent(s1.time, s1.radec)
cent = cent(s1.time, s1.radec)
cam_tilt = np.array([0, 0, 0])
star0 = cel2ecef(s1.time, s1.cel[0])
star = pos(cam_tilt, star0)
body0 = cel2ecef(s1.time, s1.body[0])
body = pos(cam_tilt, s1.body[0])
centroid = pos(cam_tilt, cent)

# Plot the surface
#ax.plot_surface(x, y, z, color = 'w')

# if i dont set these, the plot is all zoomed in
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
origo=[0,0,0]
ax.view_init(30, 45)

# Coordinate system axis
ax.quiver(origo[0],origo[1],origo[2], 1, 0, 0, color="r", linestyle = 'dashed')#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], 0, 1, 0, color="g", linestyle = 'dashed')#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], 0, 0, 1, color="b", linestyle = 'dashed')#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], -0.886493, -0.340148, 0.313735, color="orange")#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], s1.body[0, 0], s1.body[0, 1], s1.body[0, 2], color="purple",normalize=True)
ax.quiver(origo[0],origo[1],origo[2], s1.cel[0, 0], s1.cel[0, 1], s1.cel[0, 2], color="c", normalize=True)
ax.quiver(origo[0],origo[1],origo[2], body0[0], body0[1], body0[2], color="olive", normalize=True)
ax.quiver(origo[0],origo[1],origo[2], star0[0], star0[1], star0[2], color="pink", normalize=True)


rad = np.pi/180
tilt = [0, 0, 30*rad]
R = R(tilt)


# axis label placement
ax.text(0.1, 0.0, -0.2, r'$0$')
ax.text(1.3, 0, 0, r'$x$')
ax.text(0, 1.3, 0, r'$y$')
ax.text(0, 0, 1.3, r'$z$')
ax.text(-0.486493/2, -0.340148/2, 0.313735/2, r'$r$')

# Set an equal aspect ratio
ax.set_aspect('auto')


plt.show()

'''rad = np.pi/180
center = [-0.886493, -0.340148, 0.313735]
a = np.array([0, 0, 1])
b = np.array([-0.886493, -0.340148, 0.313735])
Rot = Rot(a, b)
x0, y0, z0 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
Rotx, Roty, Rotz = np.dot(Rot, x0), np.dot(Rot, y0), np.dot(Rot, z0)
tilt = [0, 0, 30*rad]
R = R(tilt)
Rx, Ry, Rz = np.dot(R, Rotx), np.dot(R, Roty), np.dot(R, Rotz)
ax.quiver(center[0],center[1],center[2], Rx[0], Rx[1], Rx[2], color="b")#,normalize=True) #z
ax.quiver(center[0],center[1],center[2], Ry[0], Ry[1], Ry[2], color="g")#,normalize=True) #x
ax.quiver(center[0],center[1],center[2], Rz[0], Rz[1], Rz[2], color="r")#,normalize=True) #y'''
