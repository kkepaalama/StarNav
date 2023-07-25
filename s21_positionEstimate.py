#usr/bin/python

'''
File returns position estimate for image belonging to s21 class
'''

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
from ost_output_papakolea import s21


plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ground_truth = 21.295084, -157.811978 #ground truth of Honolulu, HI <latitude, longitude> in decimal degrees
ground_truth_radians = np.radians([21.295084, -157.811978]) #ground truth of Honolulu, HI <latitude, longitude> in radians
gtc = np.array([-0.86272793, -0.35186237,  0.36317129]) #ground truth of Honolulu, HI <x, y, z> as a unit vector in cartesian coordinates
z_axis = np.array([0, 0, 1])
y_axis = np.array([0, 1, 0])
x_axis = np.array([1, 0, 0])
o = np.array([0,0,0])

#matplotlib plot set up
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

#
x, y, z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
ax.quiver(o[0], o[1], o[2], x[0], x[1], x[2],  color="r", alpha = 0.5) #linestyle = 'dashed')#,normalize=True) #x-axis
ax.quiver(o[0], o[1], o[2], y[0], y[1], y[2], color="g", alpha = 0.5) #linestyle = 'dashed')#,normalize=True) #y-axis
ax.quiver(o[0], o[1], o[2], z[0], z[1], z[2], color="b") #alpha = 0.5) #linestyle = 'dashed')#,normalize=True) #z-axis or normal vector
ax.quiver(o[0], o[1], o[2], gtc[0], gtc[1], gtc[2], color="dodgerblue", alpha = 0.5)

#tilt measurements from Berry IMU
phi = s21.imu_tilt[0] #roll
theta = s21.imu_tilt[1] #pitch
psi = s21.imu_tilt[2] #yaw

#converts star coordinates in GCRS (which is a ECI frame) to ITRS (which is a ECEF frame)
'''e = main.cel2ecef(s21.time, s21.cel[0], s21.radec, 'gcrs')
e1 = main.cel2ecef(s21.time, s21.cel[1], s21.radec, 'gcrs')
e2 = main.cel2ecef(s21.time, s21.cel[2], s21.radec, 'gcrs')
e3 = main.cel2ecef(s21.time, s21.cel[3], s21.radec, 'gcrs')
e4 = main.cel2ecef(s21.time, s21.cel[4], s21.radec, 'gcrs')
e5 = main.cel2ecef(s21.time, s21.cel[5], s21.radec, 'gcrs')
e6 = main.cel2ecef(s21.time, s21.cel[6], s21.radec, 'gcrs')
e7 = main.cel2ecef(s21.time, s21.cel[7], s21.radec, 'gcrs')
e8 = main.cel2ecef(s21.time, s21.cel[8], s21.radec, 'gcrs')
e9 = main.cel2ecef(s21.time, s21.cel[9], s21.radec, 'gcrs')
e10 = main.cel2ecef(s21.time, s21.cel[10], s21.radec, 'gcrs')
e11 = main.cel2ecef(s21.time, s21.cel[11], s21.radec, 'gcrs')
e12 = main.cel2ecef(s21.time, s21.cel[12], s21.radec, 'gcrs')
e13 = main.cel2ecef(s21.time, s21.cel[13], s21.radec, 'gcrs')
e14 = main.cel2ecef(s21.time, s21.cel[14], s21.radec, 'gcrs')
e15 = main.cel2ecef(s21.time, s21.cel[15], s21.radec, 'gcrs')
e16 = main.cel2ecef(s21.time, s21.cel[16], s21.radec, 'gcrs')
e17 = main.cel2ecef(s21.time, s21.cel[17], s21.radec, 'gcrs')
e18 = main.cel2ecef(s21.time, s21.cel[18], s21.radec, 'gcrs')'''


#corresponds to the following above; same array just faster to access. Dont have to run astropy everytime (which takes several minutes)
i = np.array([-0.97959712, -0.19169769, -0.0603447 ]) #main.cel2ecef(s21.time, s21.cel[0], s21.radec, 'gcrs')
i1 = np.array([-0.96983058, -0.23418565, -0.06771802])
i2 = np.array([-0.93037007, -0.34918968, -0.11170543])
i3 = np.array([-0.94694094, -0.31307492, -0.0727114 ])
i4 = np.array([-0.93560047, -0.35124988, -0.03571119])
i5 = np.array([-0.96433335, -0.2635803 , 0.02422014])
i6 = np.array([-0.96675552, -0.25240618, 0.04092535])
i7 = np.array([-0.93383178, -0.35744188, -0.01391052])
i8 = np.array([-0.93236179, -0.36150796, -0.00367147])
i9 = np.array([-0.93951902, -0.34227064,  0.01244262])
i10 = np.array([-0.95742727, -0.28454042, 0.04868026])
i11 = np.array([-0.90488738, -0.42507007, -0.02223197])
i12 = np.array([-0.97624367, -0.18023666, 0.1202624 ])
i13 = np.array([-0.97647579, -0.17228726, 0.1296616 ])
i14 = np.array([-0.96361974, -0.24691403, 0.10232522])
i15 = np.array([-0.95963119, -0.26259484, 0.10075677])
i16 = np.array([-0.97610962, -0.12533066, 0.17748871])
i17 = np.array([-0.97405992, -0.14299939, 0.17538085])
i18 = np.array([-0.89799432, -0.43923634, 0.02603169])



v = np.array([-0.96975442, -0.24353585, 0.01632937]) #boresight vector of the camera
ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color="violet")

#re-indexes body coordinates
# same as applying rotation from startracker "s" to body "b" --> bRs
idx = [1, 2, 0]
b = s21.body[0][idx]
b1 = s21.body[1][idx]
b2 = s21.body[2][idx]
b3 = s21.body[3][idx]
b4 = s21.body[4][idx]
b5 = s21.body[5][idx]
b6 = s21.body[6][idx]
b7 = s21.body[7][idx]
b8 = s21.body[8][idx]
b9 = s21.body[9][idx]
b10 = s21.body[10][idx]
b11 = s21.body[11][idx]
b12 = s21.body[12][idx]
b13 = s21.body[13][idx]
b14 = s21.body[14][idx]
b15 = s21.body[15][idx]
b16 = s21.body[16][idx]
b17 = s21.body[17][idx]
b18 = s21.body[18][idx]

#davenport q-Method set up
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
B11 = main.B(b11, i11, 1)
B12 = main.B(b12, i12, 1)
B13 = main.B(b13, i13, 1)
B14 = main.B(b14, i14, 1)
B15 = main.B(b15, i15, 1)
B16 = main.B(b16, i16, 1)
B17 = main.B(b17, i17, 1)
B18 = main.B(b18, i18, 1)

'''
#star vectors in body frame plotted in body frame
ax.quiver(o[0], o[1], o[2], b[0], b[1], b[2], color="orange")
ax.quiver(o[0], o[1], o[2], b1[0], b1[1], b1[2], color="orange")
ax.quiver(o[0], o[1], o[2], b2[0], b2[1], b2[2], color="orange")
ax.quiver(o[0], o[1], o[2], b3[0], b3[1], b3[2], color="orange")
ax.quiver(o[0], o[1], o[2], b4[0], b4[1], b4[2], color="orange")
ax.quiver(o[0], o[1], o[2], b5[0], b5[1], b5[2], color="orange")
ax.quiver(o[0], o[1], o[2], b6[0], b6[1], b6[2], color="orange")
ax.quiver(o[0], o[1], o[2], b7[0], b7[1], b7[2], color="orange")
ax.quiver(o[0], o[1], o[2], b8[0], b8[1], b8[2], color="orange")
ax.quiver(o[0], o[1], o[2], b9[0], b9[1], b9[2], color="orange")
ax.quiver(o[0], o[1], o[2], b10[0], b10[1], b10[2], color="orange")
ax.quiver(o[0], o[1], o[2], b11[0], b11[1], b11[2], color="orange")
ax.quiver(o[0], o[1], o[2], b12[0], b12[1], b12[2], color="orange")
ax.quiver(o[0], o[1], o[2], b13[0], b13[1], b13[2], color="orange")
ax.quiver(o[0], o[1], o[2], b14[0], b14[1], b14[2], color="orange")
ax.quiver(o[0], o[1], o[2], b15[0], b15[1], b15[2], color="orange")
ax.quiver(o[0], o[1], o[2], b16[0], b16[1], b16[2], color="orange")
ax.quiver(o[0], o[1], o[2], b17[0], b17[1], b17[2], color="orange")
ax.quiver(o[0], o[1], o[2], b18[0], b18[1], b18[2], color="orange")
ax.quiver(o[0], o[1], o[2], b19[0], b19[1], b19[2], color="orange")
ax.quiver(o[0], o[1], o[2], b20[0], b20[1], b20[2], color="orange")
#star vectors in inertial frame plotted in inertial frame
ax.quiver(o[0], o[1], o[2], i[0], i[1], i[2], color="coral")
ax.quiver(o[0], o[1], o[2], i1[0], i1[1], i1[2], color="coral")
ax.quiver(o[0], o[1], o[2], i2[0], i2[1], i2[2], color="coral")
ax.quiver(o[0], o[1], o[2], i3[0], i3[1], i3[2], color="coral")
ax.quiver(o[0], o[1], o[2], i4[0], i4[1], i4[2], color="coral")
ax.quiver(o[0], o[1], o[2], i5[0], i5[1], i5[2], color="coral")
ax.quiver(o[0], o[1], o[2], i6[0], i6[1], i6[2], color="coral")
ax.quiver(o[0], o[1], o[2], i7[0], i7[1], i7[2], color="coral")
ax.quiver(o[0], o[1], o[2], i8[0], i8[1], i8[2], color="coral")
ax.quiver(o[0], o[1], o[2], i9[0], i9[1], i9[2], color="coral")
ax.quiver(o[0], o[1], o[2], i10[0], i10[1], i10[2], color="coral")
ax.quiver(o[0], o[1], o[2], i11[0], i11[1], i11[2], color="coral")
ax.quiver(o[0], o[1], o[2], i12[0], i12[1], i12[2], color="coral")
ax.quiver(o[0], o[1], o[2], i13[0], i13[1], i13[2], color="coral")
ax.quiver(o[0], o[1], o[2], i14[0], i14[1], i14[2], color="coral")
ax.quiver(o[0], o[1], o[2], i15[0], i15[1], i15[2], color="coral")
ax.quiver(o[0], o[1], o[2], i16[0], i16[1], i16[2], color="coral")
ax.quiver(o[0], o[1], o[2], i17[0], i17[1], i17[2], color="coral")
ax.quiver(o[0], o[1], o[2], i18[0], i18[1], i18[2], color="coral")
ax.quiver(o[0], o[1], o[2], i19[0], i19[1], i19[2], color="coral")
ax.quiver(o[0], o[1], o[2], i20[0], i20[1], i20[2], color="coral")'''

B = B0 + B1 +B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9 + B10 + B11 +B12 + B13 + B14 + B15 + B16 + B17 + B18
K = main.K(B)
q = main.q(K)
iRb = main.q2R(q) #rotation from body to inertial

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
#i_v_br = np.dot(main.Rz(np.radians(-s21.imu_tilt[2])), i_v_br1)
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



'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
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
ax.quiver(o[0], o[1], o[2], x[0], x[1], x[2],  color="r", alpha = 0.5, linestyle = 'dashed')#,normalize=True) #x-axis
ax.quiver(o[0], o[1], o[2], y[0], y[1], y[2], color="g", alpha = 0.5, linestyle = 'dashed')#,normalize=True) #y-axis
ax.quiver(o[0], o[1], o[2], z[0], z[1], z[2], color="b", alpha = 0.5, linestyle = 'dashed')#,normalize=True) #z-axis or normal vector
#ax.quiver(o[0], o[1], o[2], gtc[0], gtc[1], gtc[2], color="dodgerblue", alpha = 0.5)

vr = np.dot(np.transpose(iRb), v)
ax.quiver(o[0], o[1], o[2], vr[0], vr[1], vr[2], color="violet")

gtcr = np.dot(np.transpose(iRb), gtc)
ax.quiver([0], o[1], o[2], gtcr[0], gtcr[1], gtcr[2], color="dodgerblue")

BB = main.B(gtcr, z_axis, 1)
KK = main.K(BB)
qq = main.q(KK)
gRz = main.q2R(qq)


y_axis_new1 = np.dot(np.transpose(gRz), y_axis)
x_axis_new1 = np.dot(np.transpose(gRz), x_axis)
ax.quiver([0], o[1], o[2], y_axis_new1[0], y_axis_new1[1], y_axis_new1[2], color="limegreen")
ax.quiver([0], o[1], o[2], x_axis_new1[0], x_axis_new1[1], x_axis_new1[2], color="coral")

xyz_euler = main.rotation2euler(gRz, 'XYZ')
xzy_euler = main.rotation2euler(gRz, 'XZY')
yzx_euler = main.rotation2euler(gRz, 'YZX')
yxz_euler = main.rotation2euler(gRz, 'YXZ')
zxy_euler = main.rotation2euler(gRz, 'ZXY')
zyx_euler = main.rotation2euler(gRz, 'ZYX')

xyz = main.euler(xyz_euler[0], xyz_euler[1], xyz_euler[2], 'XYZ')
xzy = main.euler(xzy_euler[0], xzy_euler[1], xzy_euler[2], 'XZY')
#yzx = main.euler(phi, theta, psi, 'YZX')
#yxz = main.euler(phi, theta, psi, 'YXZ')
#zxy = main.euler(phi, theta, psi, 'ZXY')
zyx = main.euler(zyx_euler[0], zyx_euler[1], zyx_euler[2], 'ZYX')
zyx_test = main.euler(phi, theta, psi, 'ZYX')

gtc_test = np.dot(zyx_test, z_axis)
gtc_test1 = np.dot(iRb, gtc_test)


print(phi, theta, psi)
print('xyz:', xzy_euler)
print('xzy:', xzy_euler)
print('yzx:', yzx_euler)
print('yxz:', yxz_euler)
print('zxy:', zxy_euler)
print('zyx:', zyx_euler)

true_position = main.car2sph(gtc)
estimated_position = main.car2sph(gtc_test1)

print('true position_XYZ: ', true_position)
print('estimate position_XYZ: ', estimated_position)


coords_1 = (true_position[0], true_position[1])
coords_2 = (estimated_position[0], estimated_position[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')'''

plt.show()