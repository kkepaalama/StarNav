#usr/bin/python

###Visual simulation for euler rotaion sequences. The origin frame is fixed to show the progression of the rotating frame.

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
from ost_output_papakolea import s1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ground_truth = 21.295084, -157.811978
ground_truth_radians = np.radians([21.295084, -157.811978])
gtc = np.array([-0.86272793, -0.35186237,  0.36317129])
z_axis = np.array([0, 0, 1])
z_axis_double = np.dot(2, z_axis)


ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
o = np.array([0,0,0])
ax.view_init(30, 45)

# axis label placement
ax.text(0.1, 0.0, -0.2, r'$0$')
ax.text(1.3, 0, 0, r'$x$')
ax.text(0, 1.3, 0, r'$y$')
ax.text(0, 0, 1.3, r'$z$')

# Set an equal aspect ratio
ax.set_aspect('auto')

#Local frame: this frame is fixed
x, y, z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
ax.quiver(o[0], o[1], o[2], x[0], x[1], x[2],  color="r", alpha = 0.5) #linestyle = 'dashed')#,normalize=True) #x-axis
ax.quiver(o[0], o[1], o[2], y[0], y[1], y[2], color="g", alpha = 0.5) #linestyle = 'dashed')#,normalize=True) #y-axis
ax.quiver(o[0], o[1], o[2], z[0], z[1], z[2], color="b", alpha = 0.5) #linestyle = 'dashed')#,normalize=True) #z-axis or normal vector
ax.quiver(o[0], o[1], o[2], gtc[0], gtc[1], gtc[2], color="dodgerblue", alpha = 0.5)

# Body frame: this frame rotates
phi = 0.06161487984311333 #roll
theta = 0.03387764611236473 #pitch
psi = -0.4118977034706618
rx = main.Rx(phi)
ry = main.Ry(theta)
ryx = np.dot(ry, rx)
rz = main.Rz(-psi)
rzyx = np.dot(rx, np.dot(ry, rz))

i = np.array([-0.89138523 , -0.39121153, 0.22887966]) #main.cel2ecef(s1.time, s1.cel[0], s1.radec, 'car')
i1 = np.array([-0.85606642, -0.47075679, 0.21339712]) #main.cel2ecef(s1.time, s1.cel[1], s1.radec, 'car')
i2 = np.array([-0.89898379, -0.31911886, 0.29998550]) #main.cel2ecef(s1.time, s1.cel[2], s1.radec, 'car')
i3 = np.array([-0.91261350, -0.22635306, 0.34044220]) #main.cel2ecef(s1.time, s1.cel[3], s1.radec, 'car')
i4 = np.array([-0.89116869, -0.34124666, 0.29894663]) #main.cel2ecef(s1.time, s1.cel[4], s1.radec, 'car')
i5 = np.array([-0.89768473, -0.28909588, 0.33254428]) #main.cel2ecef(s1.time, s1.cel[5], s1.radec, 'car')
i6 = np.array([-0.87582264, -0.34158169, 0.34096430]) #main.cel2ecef(s1.time, s1.cel[6], s1.radec, 'car')
i7 = np.array([-0.85185478, -0.40241655, 0.33526759]) #main.cel2ecef(s1.time, s1.cel[7], s1.radec, 'car')
i8 = np.array([-0.88039207, -0.25063391, 0.40260706]) #main.cel2ecef(s1.time, s1.cel[8], s1.radec, 'car')
i9 = np.array([-0.86503257, -0.31358988, 0.39163764]) #main.cel2ecef(s1.time, s1.cel[9], s1.radec, 'car')
i10 = np.array([-0.83881418, -0.39783868, 0.37163848]) #main.cel2ecef(s1.time, s1.cel[10], s1.radec, 'car')
i11 = np.array([-0.84867601, -0.30162074, 0.43448125]) #main.cel2ecef(s1.time, s1.cel[11], s1.radec, 'car')

v = np.array([-0.88649268, -0.34014846, 0.31373516]) #boresight vector

idx = [1, 2, 0]
b = s1.body[0][idx]
b1 = s1.body[1][idx]
b2 = s1.body[2][idx]
b3 = s1.body[3][idx]
b4 = s1.body[4][idx]
b5 = s1.body[5][idx]
b6 = s1.body[6][idx]
b7 = s1.body[7][idx]
b8 = s1.body[8][idx]
b9 = s1.body[9][idx]
b10 = s1.body[10][idx]
b11 = s1.body[11][idx]

i = i
b = b

B = main.B(z_axis, i, 1)
K = main.K(B)
q = main.q(K)
rotation_matrix = main.q2R(q) #iRb rotation from body to inertial 

ip = np.dot(np.transpose(rotation_matrix), i)

ax.quiver(o[0], o[1], o[2], i[0], i[1], i[2], color="indigo")
ax.quiver(o[0], o[1], o[2], b[0], b[1], b[2], color="pink")

br = np.dot(np.transpose(ryx), b)
br = np.dot(rz, br)
ax.quiver(o[0], o[1], o[2], br[0], br[1], br[2], color="m")
#ax.quiver(o[0], o[1], o[2], ip[0], ip[1], ip[2], color="orange")

brr = np.dot(rotation_matrix, br)
ax.quiver(o[0], o[1], o[2], brr[0], brr[1], brr[2], color="m")

b_test = np.dot(rotation_matrix, b)
ax.quiver(o[0], o[1], o[2], b_test[0], b_test[1], b_test[2], color="pink")
gtc_test = np.dot(np.transpose(rotation_matrix), gtc)
ax.quiver(o[0], o[1], o[2], gtc_test[0], gtc_test[1], gtc_test[2], color="dodgerblue", alpha = 0.5)

ipp = ip + b
#ax.quiver(o[0], o[1], o[2], ipp[0], ipp[1], ipp[2], color="y")

bp = np.dot(rotation_matrix, b)
bpp = i + bp
bpp_norm = bpp/np.linalg.norm(bpp)
#ax.quiver(o[0], o[1], o[2], bpp_norm[0], bpp_norm[1], bpp_norm[2], color="y")

true_position = main.car2sph(gtc)
#estimated_position = main.car2sph(bpp_norm)
estimated_position_average = 22.790702626103695, -156.38232688929745

print('true position: ', true_position)
#print('estimate position: ', estimated_position)
print('estimate position averaged: ', estimated_position_average)


coords_1 = (true_position[0], true_position[1])
#coords_2 = (estimated_position[0], estimated_position[1])
coords_2 = (estimated_position_average[0], estimated_position_average[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()