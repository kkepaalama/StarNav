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
z_axis = np.array([0, 0, 1])


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
ax.quiver(o[0], o[1], o[2], x[0], x[1], x[2],  color="r", alpha = 0.5) #x-axis
ax.quiver(o[0], o[1], o[2], y[0], y[1], y[2], color="g", alpha = 0.5) #y-axis
ax.quiver(o[0], o[1], o[2], z[0], z[1], z[2], color="b", alpha = 0.5) #z-axis or normal vector
#ax.quiver(o[0],o[1],o[2], 0, 0, -1, color="orange", alpha = 0.5) #gravity vector 

# Body frame: this frame rotates
theta = np.radians(30)
phi = np.radians(30)
psi = np.radians(10)
rx = main.Rx(theta)
ry = main.Ry(phi)
rz = main.Rz(psi)
ryx = np.dot(ry, rx)
rxyz = np.dot(rz, np.dot(ry, rx))
zp = np.dot(rxyz, z) #rotated vector
ax.quiver(o[0], o[1], o[2], zp[0], zp[1], zp[2], color="purple")
ax.quiver(z[0], z[1], z[2], zp[0], zp[1], zp[2], color="blueviolet")

z_result = z + zp
ax.quiver(o[0], o[1], o[2], z_result[0], z_result[1], z_result[2], color="dodgerblue")

z_result_norm = z_result/np.linalg.norm(z_result)
zz = np.dot(2, z_axis) - z_result_norm
ax.quiver(z_result_norm[0], z_result_norm[1], z_result_norm[2], zz[0], zz[1], zz[2], color="forestgreen")


ax.quiver(o[0], o[1], o[2], gtc[0], gtc[1], gtc[2], color="orange")

B = main.B(z_axis, gtc, 1)
K = main.K(B)
q = main.q(K)
rotation_matrix = main.q2R(q)

zp_new = np.dot(rotation_matrix, zp)
ax.quiver(o[0], o[1], o[2], zp_new[0], zp_new[1], zp_new[2], color="pink")
ax.quiver(gtc[0], gtc[1], gtc[2], zp_new[0], zp_new[1], zp_new[2], color="lightpink")

zpp_new = gtc + zp_new
ax.quiver(o[0], o[1], o[2], zpp_new[0], zpp_new[1], zpp_new[2], color="deepskyblue")

zz1 = np.dot(rotation_matrix, zz)
zpp_new_norm = zpp_new/np.linalg.norm(zpp_new)
ax.quiver(zpp_new_norm[0], zpp_new_norm[1], zpp_new_norm[2], zz1[0], zz1[1], zz1[2], color="olivedrab")

zpp = z + zp #resultant vector
#ax.quiver(o[0],o[1],o[2], zpp[0], zpp[1], zpp[2], color="c") #resultant vector with origin at [0, 0, 0]
#ax.quiver(z[0],z[1], z[2], zp[0], zp[1], zp[2], color="y") #translated vector zp with origin at end of z

z_final = zpp_new_norm + zz1
z_final_norm = z_final/np.linalg.norm(z_final)


zpp_norm = zpp/np.linalg.norm(zpp)
Rxyz = np.dot(main.Rz(psi/2), np.dot(main.Ry(phi/2), main.Rx(theta/2)))
z1 = np.dot(np.transpose(Rxyz), zpp_norm)
z1 = z1/np.linalg.norm(z1)

'''print('zp: ', zp)
print('zpp_norm: ', zpp_norm)
print('z1: ', z1)'''

z_sph = main.car2sph(gtc)
z1_sph = main.car2sph(z_final_norm)

coords_1 = (z_sph[0], z_sph[1])
coords_2 = (z1_sph[0], z1_sph[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')


# axis label placement
ax.text(0.1, 0.0, -0.2, r'$0$')
ax.text(1.3, 0, 0, r'$x$')
ax.text(0, 1.3, 0, r'$y$')
ax.text(0, 0, 1.3, r'$z$')

# Set an equal aspect ratio
ax.set_aspect('auto')
plt.show()