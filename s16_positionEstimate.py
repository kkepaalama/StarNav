#usr/bin/python

'''
File returns position estimate for image belonging to s16 class
'''

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
from ost_output_papakolea import s1, s16


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
phi = s16.imu_tilt[0] #roll
theta = s16.imu_tilt[1] #pitch
psi = s16.imu_tilt[2] #yaw

#converts star coordinates in GCRS (which is a ECI frame) to ITRS (which is a ECEF frame)
'''e = main.cel2ecef(s16.time, s16.cel[0], s16.radec, 'gcrs')
e1 = main.cel2ecef(s16.time, s16.cel[1], s16.radec, 'gcrs')
e2 = main.cel2ecef(s16.time, s16.cel[2], s16.radec, 'gcrs')
e3 = main.cel2ecef(s16.time, s16.cel[3], s16.radec, 'gcrs')
e4 = main.cel2ecef(s16.time, s16.cel[4], s16.radec, 'gcrs')
e5 = main.cel2ecef(s16.time, s16.cel[5], s16.radec, 'gcrs')
e6 = main.cel2ecef(s16.time, s16.cel[6], s16.radec, 'gcrs')
e7 = main.cel2ecef(s16.time, s16.cel[7], s16.radec, 'gcrs')
e8 = main.cel2ecef(s16.time, s16.cel[8], s16.radec, 'gcrs')
e9 = main.cel2ecef(s16.time, s16.cel[9], s16.radec, 'gcrs')
e10 = main.cel2ecef(s16.time, s16.cel[10], s16.radec, 'gcrs')
e11 = main.cel2ecef(s16.time, s16.cel[11], s16.radec, 'gcrs')
e12 = main.cel2ecef(s16.time, s16.cel[12], s16.radec, 'gcrs')
e13 = main.cel2ecef(s16.time, s16.cel[13], s16.radec, 'gcrs')
e14 = main.cel2ecef(s16.time, s16.cel[14], s16.radec, 'gcrs')
e15 = main.cel2ecef(s16.time, s16.cel[15], s16.radec, 'gcrs')
e16 = main.cel2ecef(s16.time, s16.cel[16], s16.radec, 'gcrs')
e17 = main.cel2ecef(s16.time, s16.cel[17], s16.radec, 'gcrs')
e18 = main.cel2ecef(s16.time, s16.cel[18], s16.radec, 'gcrs')
e19 = main.cel2ecef(s16.time, s16.cel[19], s16.radec, 'gcrs')
e20 = main.cel2ecef(s16.time, s16.cel[20], s16.radec, 'gcrs')
ev = main.cel2ecef(s16.time, s16.cel[20], s16.radec, 'radec2car')'''

#corresponds to the following above; same array just faster to access. Dont have to run astropy everytime (which takes several minutes)
i = np.array([-0.97883206, -0.18928355, -0.07784306]) #main.cel2ecef(s16.time, s16.cel[0], s16.radec, 'gcrs')
i1 = np.array([-0.96766537, -0.23351755, -0.09535886])
i2 = np.array([-0.97638069, -0.20745946, -0.06034373])
i3 = np.array([-0.96593067, -0.2497845 , -0.06771704])
i4 = np.array([-0.92462202, -0.3641377 , -0.11170445])
i5 = np.array([-0.9417727 , -0.32829466, -0.07271041])
i6 = np.array([-0.92981849, -0.36628191, -0.0357102 ])
i7 = np.array([-0.98822733, -0.11794265,  0.09744921])
i8 = np.array([-0.95996041, -0.27908672,  0.02422112])
i9 = np.array([-0.96256234, -0.26795309,  0.04092634])
i10 = np.array([-0.92795017, -0.37244457, -0.01390953])
i11 = np.array([-0.98166125, -0.16635629,  0.09309579])
i12 = np.array([-0.92641487, -0.37648644, -0.00367048])
i13 = np.array([-0.93388112, -0.35736694,  0.01244361])
i14 = np.array([-0.95271743, -0.29993282,  0.04868125])
i15 = np.array([-0.97321218, -0.1959458 ,  0.12026337])
i16 = np.array([-0.97357233, -0.18800117,  0.12966256])
i17 = np.array([-0.95951539, -0.26241109,  0.1023262 ])
i18 = np.array([-0.95527463, -0.27802557,  0.10075774])
i19 = np.array([-0.97396291, -0.14104475,  0.17748965])
i20 = np.array([-0.97162881, -0.15867816,  0.1753818 ]) 



v = np.array([-0.9694715 , -0.24468602,  0.01593007]) #boresight vector of the camera
ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color="violet")

#re-indexes body coordinates
# same as applying rotation from startracker "s" to body "b" --> bRs
idx = [1, 2, 0]
b = s16.body[0][idx]
b1 = s16.body[1][idx]
b2 = s16.body[2][idx]
b3 = s16.body[3][idx]
b4 = s16.body[4][idx]
b5 = s16.body[5][idx]
b6 = s16.body[6][idx]
b7 = s16.body[7][idx]
b8 = s16.body[8][idx]
b9 = s16.body[9][idx]
b10 = s16.body[10][idx]
b11 = s16.body[11][idx]
b12 = s16.body[12][idx]
b13 = s16.body[13][idx]
b14 = s16.body[14][idx]
b15 = s16.body[15][idx]
b16 = s16.body[16][idx]
b17 = s16.body[17][idx]
b18 = s16.body[18][idx]
b19 = s16.body[19][idx]
b20 = s16.body[20][idx]

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
B19 = main.B(b19, i19, 1)
B20 = main.B(b20, i20, 1)


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

B = B0 + B1 +B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9 + B10 + B11 +B12 + B13 + B14 + B15 + B16 + B17 + B18 + B19 + B20
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
#i_v_br = np.dot(main.Rz(np.radians(-s16.imu_tilt[2])), i_v_br1)
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