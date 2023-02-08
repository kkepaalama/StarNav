import matplotlib.pyplot as plt
import numpy as np
from main import raw_cent, cent, cel2ecef, B, K, q, n, imu2cam, pos, R, Rot, R_inverse
from coords_papakolea import s1


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
origo=[0,0,0]
ax.view_init(30, 45)

# Coordinate system axis
ax.quiver(origo[0],origo[1],origo[2], 1, 0, 0, color="r", alpha = 0.5, linestyle = 'dashed')#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], 0, 1, 0, color="g", alpha = 0.5, linestyle = 'dashed')#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], 0, 0, 1, color="b", alpha = 0.5, linestyle = 'dashed')#,normalize=True)


rad = np.pi/180
tilt = [30*rad, 45*rad, 30*rad]
R = R(tilt)
x, y, z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
Rx, Ry, Rz = np.dot(R, x), np.dot(R, y), np.dot(R, z)
ax.quiver(origo[0],origo[1],origo[2], Rx[0], Rx[1], Rx[2], color="r")#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], Ry[0], Ry[1], Ry[2], color="g")#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], Rz[0], Rz[1], Rz[2], color="b")#,normalize=True)



rad = np.pi/180
tilt = [30*rad, 45*rad, 30*rad]
R_in = R_inverse(tilt)
#x, y, z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
Rx, Ry, Rz = np.dot(R_in, Rx), np.dot(R_in, Ry), np.dot(R_in, Rz)
ax.quiver(origo[0],origo[1],origo[2], Rx[0], Rx[1], Rx[2], color="c")#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], Ry[0], Ry[1], Ry[2], color="y")#,normalize=True)
ax.quiver(origo[0],origo[1],origo[2], Rz[0], Rz[1], Rz[2], color="m")#,normalize=True)


# axis label placement
ax.text(0.1, 0.0, -0.2, r'$0$')
ax.text(1.3, 0, 0, r'$x$')
ax.text(0, 1.3, 0, r'$y$')
ax.text(0, 0, 1.3, r'$z$')

# Set an equal aspect ratio
ax.set_aspect('auto')


plt.show()

