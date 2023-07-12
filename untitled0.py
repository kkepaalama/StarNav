#usr/bin/python

###Visual simulation for euler rotaion sequences. The origin frame is fixed to show the progression of the rotating frame.

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ground_truth = 21.295084, -157.811978
gtc = np.array([-0.57062386,  0.51307648,  0.64120273])


ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
o = np.array([0,0,0])
ax.view_init(30, 45)

#Local frame: this frame is fixed
x, y, z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
ax.quiver(o[0],o[1],o[2], x[0], x[1], x[2],  color="r", alpha = 0.5)#,normalize=True) #x-axis
ax.quiver(o[0],o[1],o[2], y[0], y[1], y[2], color="g", alpha = 0.5)#,normalize=True) #y-axis
ax.quiver(o[0],o[1],o[2], z[0], z[1], z[2], color="b", alpha = 0.5)#,normalize=True) #z-axis or normal vector
#ax.quiver(o[0],o[1],o[2], 0, 0, -1, color="orange", alpha = 0.5)#,normalize=True) #gravity vector 

# Body frame: this frame rotates
theta = np.radians(30)
phi = np.radians(10)
psi = np.radians(0)
rx = main.Rx(theta)
ry = main.Ry(phi)
rz = main.Rz(psi)
ryx = np.dot(ry, rx)
rxyz = np.dot(rz, np.dot(ry, rx))
zp = np.dot(rxyz, z) #rotated vector
ax.quiver(o[0],o[1],o[2], zp[0], zp[1], zp[2], color="m")#,normalize=True)


zpp = z + zp #resultant vector
ax.quiver(o[0],o[1],o[2], zpp[0], zpp[1], zpp[2], color="c")#,normalize=True) #resultant vector with origin at [0, 0, 0]
#ax.quiver(z[0],z[1], z[2], zp[0], zp[1], zp[2], color="y")#,normalize=True) #translated vector zp with origin at end of z

# axis label placement
ax.text(0.1, 0.0, -0.2, r'$0$')
ax.text(1.3, 0, 0, r'$x$')
ax.text(0, 1.3, 0, r'$y$')
ax.text(0, 0, 1.3, r'$z$')

# Set an equal aspect ratio
ax.set_aspect('auto')

zpp_norm = zpp/np.linalg.norm(zpp)
Rxyz = np.dot(main.Rz(psi/2), np.dot(main.Ry(phi/2), main.Rx(theta/2)))
z1 = np.dot(np.transpose(Rxyz), zpp_norm)
z1 = z1/np.linalg.norm(z1)

print('zp: ', zp)
print('zpp_norm: ', zpp_norm)
print('z1: ', z1)

z_sph = main.car2sph(z)
z1_sph = main.car2sph(z1)

coords_1 = (z_sph[0], z_sph[1])
coords_2 = (z1_sph[0], z1_sph[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()
