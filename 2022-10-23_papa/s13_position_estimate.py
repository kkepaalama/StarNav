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
psi = s13.imu_tilt[2] #-0.443027483275695
rx = main.Rx(phi)
ry = main.Ry(theta)
ryx = np.dot(ry, rx)
rz = main.Rz(-psi)
rzyx = np.dot(rx, np.dot(ry, rz))

e = main.cel2ecef(s13.time, s13.cel[0], s13.radec, 'gcrs')
e1 = main.cel2ecef(s13.time, s13.cel[1], s13.radec, 'gcrs')
e2 = main.cel2ecef(s13.time, s13.cel[2], s13.radec, 'gcrs')
e3 = main.cel2ecef(s13.time, s13.cel[3], s13.radec, 'gcrs')
e4 = main.cel2ecef(s13.time, s13.cel[4], s13.radec, 'gcrs')
e5 = main.cel2ecef(s13.time, s13.cel[5], s13.radec, 'gcrs')
e6 = main.cel2ecef(s13.time, s13.cel[6], s13.radec, 'gcrs')
e7 = main.cel2ecef(s13.time, s13.cel[7], s13.radec, 'gcrs')
e8 = main.cel2ecef(s13.time, s13.cel[8], s13.radec, 'gcrs')
e9 = main.cel2ecef(s13.time, s13.cel[9], s13.radec, 'gcrs')
e10 = main.cel2ecef(s13.time, s13.cel[10], s13.radec, 'gcrs')
ev = main.cel2ecef(s13.time, s13.cel[10], s13.radec, 'radec2car')


i = np.array([-0.87364813, -0.43728868,  0.21334824])
i1 = np.array([-0.91067012, -0.28411608,  0.29992993])
i2 = np.array([-0.92070424, -0.19089963,  0.34038356])
i3 = np.array([-0.90371705, -0.30652793,  0.29889137])
i4 = np.array([-0.90821315, -0.25416816,  0.33248671])
i5 = np.array([-0.88648763, -0.31572136,  0.33831884])
i6 = np.array([-0.8668032 , -0.36916859,  0.33521148])
i7 = np.array([-0.88945202, -0.2164066 ,  0.402546  ])
i8 = np.array([-0.87653859, -0.2799051 ,  0.39157786])
i9 = np.array([-0.85973542, -0.26857845,  0.4344199 ])
i10 = np.array([-0.86231167, -0.20062163,  0.46494047])

v = np.array([-0.88635613, -0.34064836,  0.31357855])

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


b_n_i = np.dot(iRb, z_axis) #normal vector of body rotated to fixed frame
ax.quiver([0], o[1], o[2], b_n_i[0], b_n_i[1], b_n_i[2], color="b")

i_gt_b = np.dot(np.transpose(iRb), gtc) #ground truth in the body frame
ax.quiver([0], o[1], o[2], i_gt_b[0], i_gt_b[1], i_gt_b[2], color="dodgerblue")

i_v_b = np.dot(np.transpose(iRb), v) #bosesight vector from i frame rotated to body frame
#ax.quiver([0], o[1], o[2], i_v_b[0], i_v_b[1], i_v_b[2], color="violet")

'''z_axis_new = np.dot(iRb, z_axis)
y_axis_new = np.dot(iRb, y_axis)
x_axis_new = np.dot(iRb, x_axis)
ax.quiver([0], o[1], o[2], z_axis_new[0], z_axis_new[1], z_axis_new[2], color="b")
ax.quiver([0], o[1], o[2], y_axis_new[0], y_axis_new[1], y_axis_new[2], color="g")
ax.quiver([0], o[1], o[2], x_axis_new[0], x_axis_new[1], x_axis_new[2], color="r")'''

BB = main.B(z_axis, i_gt_b, 1)
KK = main.K(BB)
qq = main.q(KK)
gRz = main.q2R(qq)


xyz_euler = main.rotation2euler(gRz, 'XYZ')
xzy_euler = main.rotation2euler(gRz, 'XZY')
yzx_euler = main.rotation2euler(gRz, 'YZX')
yxz_euler = main.rotation2euler(gRz, 'YXZ')
zxy_euler = main.rotation2euler(gRz, 'ZXY')
zyx_euler = main.rotation2euler(gRz, 'ZYX')

zyx = main.euler(phi, theta, psi, 'ZYX')
#zyx1 = main.euler(zyx_euler[0], zyx_euler[1], zyx_euler[2], 'ZYX')

i_v_br = np.dot(zyx, i_v_b)
#i_v_br = np.dot(main.Rz(-s15.imu_tilt[2]), i_v_br1)
ax.quiver([0], o[1], o[2], i_v_br[0], i_v_br[1], i_v_br[2], color="violet")

v_i = np.dot(iRb, i_v_br)
ax.quiver([0], o[1], o[2], v_i[0], v_i[1], v_i[2], color="violet")

print(phi, theta, psi)
#print('xyz:', xzy_euler)
#print('xzy:', xzy_euler)
#print('yzx:', yzx_euler)
#print('yxz:', yxz_euler)
#print('zxy:', zxy_euler)
print('zyx:', zyx_euler)

true_position = main.car2sph(gtc)
estimated_position = main.car2sph(v_i)

print('true position_XYZ: ', true_position)
print('estimate position_XYZ: ', estimated_position)


coords_1 = (true_position[0], true_position[1])
coords_2 = (estimated_position[0], estimated_position[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()
