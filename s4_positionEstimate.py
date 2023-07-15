#usr/bin/python

'''
File returns position estimate for image belonging to s13 class
'''

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
from ost_output_papakolea import s13

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
ax.quiver(o[0], o[1], o[2], z[0], z[1], z[2], color="b") #alpha = 0.5) #linestyle = 'dashed')#,normalize=True) #z-axis or normal vector
ax.quiver(o[0], o[1], o[2], gtc[0], gtc[1], gtc[2], color="dodgerblue", alpha = 0.5)

# Body frame: this frame rotates
phi = s13.imu_tilt[0] #0.06068190843512727 #roll
theta = s13.imu_tilt[1] #0.03443821713316293 #pitch
psi = s13.imu_tilt[2]
rx = main.Rx(phi)
ry = main.Ry(theta)
ryx = np.dot(ry, rx)
rz = main.Rz(-psi)
rzyx = np.dot(rx, np.dot(ry, rz))

e = main.cel2ecef(s13.time, s13.cel[0], s13.radec, 'car')
e1 = main.cel2ecef(s13.time, s13.cel[1], s13.radec, 'car')
e2 = main.cel2ecef(s13.time, s13.cel[2], s13.radec, 'car')
e3 = main.cel2ecef(s13.time, s13.cel[3], s13.radec, 'car')
e4 = main.cel2ecef(s13.time, s13.cel[4], s13.radec, 'car')
e5 = main.cel2ecef(s13.time, s13.cel[5], s13.radec, 'car')
e6 = main.cel2ecef(s13.time, s13.cel[6], s13.radec, 'car')
e7 = main.cel2ecef(s13.time, s13.cel[7], s13.radec, 'car')
e8 = main.cel2ecef(s13.time, s13.cel[8], s13.radec, 'car')
e9 = main.cel2ecef(s13.time, s13.cel[9], s13.radec, 'car')
e10 = main.cel2ecef(s13.time, s13.cel[10], s13.radec, 'car')


i = e
i1 = e1
i2 = e2
i3 = e3
i4 = e4
i5 = e5
i6 = e6
i7 = e7
i8 = e8
i9 = e9
i10 = e10


idx = [1, 2, 0]
b = s13.body[0][idx]
b1 = s13.body[1][idx]
b2 = s13.body[2][idx]
b3 = s13.body[3][idx]
b4 = s13.body[4][idx]
b5 = s13.body[5][idx]
b6 = s13.body[6][idx]
b7 = s13.body[7][idx]
b8 = s13.body[8][idx]
b9 = s13.body[9][idx]
b10 = s13.body[10][idx]

B0 = main.B(b, i, 1)
B1 = main.B(b1, i1, 1)
B2 = main.B(b2, i2, 1)
B3 = main.B(b3, i3, 1)
B4 = main.B(b4, i4, 1)
B5 = main.B(b5, i5, 1)
B6 = main.B(b6, i6, 1)
B7 = main.B(b7, i7, 1)
B8 = main.B(b8, i8, 1)
B9 = main.B(b9, i9, 1)
B10 = main.B(b10, i10, 1)

B = B0 + B1 +B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9 + B10
K = main.K(B)
q = main.q(K)
iRb = main.q2R(q) #iRb rotation from body to inertial


z_axis_new = np.dot(iRb, z_axis)
ax.quiver([0], o[1], o[2], z_axis_new[0], z_axis_new[1], z_axis_new[2], color="b")

t = np.dot(np.transpose(ryx), z_axis)
ax.quiver(o[0], o[1], o[2], t[0], t[1], t[2], color="green")

tp = np.dot(iRb, t)
ax.quiver([0], o[1], o[2], tp[0], tp[1], tp[2], color="green")

tpp = np.dot(main.Rz(np.pi), t)
#ax.quiver([0], o[1], o[2], tpp[0], tpp[1], tpp[2], color="seagreen")

tpp1 = np.dot(main.Rz(psi), tpp)
ax.quiver([0], o[1], o[2], tpp1[0], tpp1[1], tpp1[2], color="limegreen")

tpp1_r = np.dot(iRb, tpp1)
ax.quiver([0], o[1], o[2], tpp1_r[0], tpp1_r[1], tpp1_r[2], color="limegreen")


true_position = main.car2sph(gtc)
estimated_position = main.car2sph(tpp1_r)

print('true position: ', true_position)
print('estimate position: ', estimated_position)


coords_1 = (true_position[0], true_position[1])
coords_2 = (estimated_position[0], estimated_position[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()