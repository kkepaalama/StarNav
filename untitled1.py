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
theta = np.radians(10)
phi = np.radians(20)
#theta = 0.06161487984311333
#phi = 0.03387764611236473
psi = -0.4118977034706618
rx = main.Rx(theta)
ry = main.Ry(phi)
rz = main.Rz(psi)
ryx = np.dot(ry, rx)
gtc_p = np.dot(ryx, gtc)
ax.quiver(o[0],o[1],o[2], gtc_p[0], gtc_p[1], gtc_p[2], color="m")#,normalize=True)

gtc_pp = gtc + gtc_p
ax.quiver(o[0],o[1],o[2], gtc_pp[0], gtc_pp[1], gtc_pp[2], color="c")#,normalize=True)

ax.quiver(gtc[0], gtc[1], gtc[2], gtc_p[0], gtc_p[1], gtc_p[2], color="y")#,normalize=True)

# axis label placement
ax.text(0.1, 0.0, -0.2, r'$0$')
ax.text(1.3, 0, 0, r'$x$')
ax.text(0, 1.3, 0, r'$y$')
ax.text(0, 0, 1.3, r'$z$')

# Set an equal aspect ratio
ax.set_aspect('auto')

gtc_pp_norm = gtc_pp/np.linalg.norm(gtc_pp)
Rxy = np.dot(main.Ry(phi/2), main.Rx(theta/2))
gtc_1 = np.dot(np.transpose(Rxy), gtc_pp)
gtc_1 = gtc_1/np.linalg.norm(gtc_1)

print('gtc_p: ', gtc_p)
print('gtc_pp_norm: ', gtc_pp_norm)
print('gtc_pp', main.car2sph(gtc_pp_norm))
print('gtc_1: ', gtc_1)

gtc__sph = main.car2sph(gtc)
gtc_1_sph = main.car2sph(gtc_1)

coords_1 = (gtc__sph[0], gtc__sph[1])
coords_2 = (gtc_1_sph[0], gtc_1_sph[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()
