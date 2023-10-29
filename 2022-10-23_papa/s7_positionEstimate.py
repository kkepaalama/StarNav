#usr/bin/python

'''
File returns position estimate for image belonging to s7 class
'''

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
from ost_output_papakolea import s7

plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ground_truth = 21.295084, -157.811978 #ground truth of Honolulu, HI <latitude, longitude> in decimal degrees
ground_truth_radians = np.radians([21.295084, -157.811978]) #ground truth of Honolulu, HI <latitude, longitude> in radians
gtc = np.array([-0.86272793, -0.35186237,  0.36317129]) #ground truth of Honolulu, HI <x, y, z> as a unit vector in cartesian coordinates
z_axis = np.array([0, 0, 1])

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
phi = s7.imu_tilt[0] #0.06161487984311333 #roll
theta = s7.imu_tilt[1] #0.03387764611236473 #pitch
psi = s7.imu_tilt[2] #yaw #-0.4118977034706618 #yaw collected from thrid party source (OST) NOT RETRIEVED FROM BERRY IMU

#converts star coordinates in GCRS (which is a ECI frame) to ITRS (which is a ECEF frame)
e = main.cel2ecef(s7.time, s7.cel[0], s7.radec, 'gcrs')
e1 = main.cel2ecef(s7.time, s7.cel[1], s7.radec, 'gcrs')
e2 = main.cel2ecef(s7.time, s7.cel[2], s7.radec, 'gcrs')
e3 = main.cel2ecef(s7.time, s7.cel[3], s7.radec, 'gcrs')
e4 = main.cel2ecef(s7.time, s7.cel[4], s7.radec, 'gcrs')
e5 = main.cel2ecef(s7.time, s7.cel[5], s7.radec, 'gcrs')
e6 = main.cel2ecef(s7.time, s7.cel[6], s7.radec, 'gcrs')
e7 = main.cel2ecef(s7.time, s7.cel[7], s7.radec, 'gcrs')
e8 = main.cel2ecef(s7.time, s7.cel[8], s7.radec, 'gcrs')
e9 = main.cel2ecef(s7.time, s7.cel[9], s7.radec, 'gcrs')
e10 = main.cel2ecef(s7.time, s7.cel[10], s7.radec, 'gcrs')
e11 = main.cel2ecef(s7.time, s7.cel[11], s7.radec, 'gcrs')
ev = main.cel2ecef(s7.time, s7.cel[11], s7.radec, 'radec2car')

#corresponds to the following above; same array just faster to access. Dont have to run astropy everytime (which takes several minutes)
i = np.array([-0.9091501 , -0.35903504,  0.21104485])
i1 = np.array([-0.89880487, -0.37388054,  0.22882993])
i2 = np.array([-0.86503533, -0.45408846,  0.21334822])
i3 = np.array([-0.90501015, -0.30165983,  0.29992991])
i4 = np.array([-0.91684361, -0.20865467,  0.34038354])
i5 = np.array([-0.89762532, -0.32393314,  0.29889135])
i6 = np.array([-0.90313233, -0.27167003,  0.33248668])
i7 = np.array([-0.88022148, -0.33279194,  0.33831881])
i8 = np.array([-0.88510436, -0.233553  ,  0.40254597])
i9 = np.array([-0.87096636, -0.29679012,  0.39157784])
i10 = np.array([-0.8543852 , -0.28514089,  0.43441987])
i11 = np.array([-0.85827409, -0.21724654,  0.46494045])

v = np.array([-0.88652983, -0.34033606,  0.3134266 ]) #boresight vector of the camera

#re-indexes body coordinates
# same as applying rotation from startracker "s" to body "b" --> bRs
idx = [1, 2, 0]
b = s7.body[0][idx]
b1 = s7.body[1][idx]
b2 = s7.body[2][idx]
b3 = s7.body[3][idx]
b4 = s7.body[4][idx]
b5 = s7.body[5][idx]
b6 = s7.body[6][idx]
b7 = s7.body[7][idx]
b8 = s7.body[8][idx]
b9 = s7.body[9][idx]
b10 = s7.body[10][idx]
b11 = s7.body[11][idx]


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

B = B0 + B1 +B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9 + B10 + B11
K = main.K(B)
q = main.q(K)
iRb = main.q2R(q) #rotation from body to inertial


b_n_i = np.dot(iRb, z_axis) #normal vector of body rotated to fixed frame
ax.quiver([0], o[1], o[2], b_n_i[0], b_n_i[1], b_n_i[2], color="b")

i_gt_b = np.dot(np.transpose(iRb), gtc) #ground truth in the body frame
ax.quiver([0], o[1], o[2], i_gt_b[0], i_gt_b[1], i_gt_b[2], color="dodgerblue")

i_v_b = np.dot(np.transpose(iRb), v) #bosesight vector from i frame rotated to body frame
#ax.quiver([0], o[1], o[2], i_v_b[0], i_v_b[1], i_v_b[2], color="violet")


BB = main.B(z_axis, i_gt_b, 1)
KK = main.K(BB)
qq = main.q(KK)
gRv = main.q2R(qq) #rotation between ground truth in body frame and boresight vector in body frame

#BB = main.B(v, gtc, 1)
#KK = main.K(BB)
#qq = main.q(KK)
#gRv = main.q2R(qq) #rotation between ground truth in body frame and boresight vector in body frame

#extracts euler angles from gRz rotation matrix
xyz_euler = main.rotation2euler(gRv, 'XYZ')
xzy_euler = main.rotation2euler(gRv, 'XZY')
yzx_euler = main.rotation2euler(gRv, 'YZX')
yxz_euler = main.rotation2euler(gRv, 'YXZ')
zxy_euler = main.rotation2euler(gRv, 'ZXY')
zyx_euler = main.rotation2euler(gRv, 'ZYX')

#comparing euler angles extracted from gRz and pitch and roll in a ZYX sequence
zyx = main.euler(phi, theta, psi, 'ZYX')
#zyx1 = main.euler(zyx_euler[0], zyx_euler[1], zyx_euler[2], 'ZYX')

#print('zyx: ', zyx)
#print('zyx1: ',zyx1)

i_v_br = np.dot(zyx, i_v_b)
#i_v_br = np.dot(main.Rz(s7.imu_tilt[2]), i_v_br1)
ax.quiver([0], o[1], o[2], i_v_br[0], i_v_br[1], i_v_br[2], color="violet")

v_i = np.dot(iRb, i_v_br)
ax.quiver([0], o[1], o[2], v_i[0], v_i[1], v_i[2], color="violet")


#print(phi, theta, psi)
#print('xyz:', xzy_euler)
#print('xzy:', xzy_euler)
#print('yzx:', yzx_euler)
#print('yxz:', yxz_euler)
#print('zxy:', zxy_euler)
#print('zyx:', zyx_euler)


true_position = main.car2sph(gtc)
estimated_position = main.car2sph(v_i)
#estimated_position_average = 22.790702626103695, -156.38232688929745

print('true position: ', true_position)
print('estimate position: ', estimated_position)
#print('estimate position averaged: ', estimated_position_average)


coords_1 = (true_position[0], true_position[1])
coords_2 = (estimated_position[0], estimated_position[1])
#coords_1 = (18.292083690572113, -159.01179766742428) #difference between ost boresight roll and in-house boresight roll
#coords_2 = (18.284473438960504, -159.00812327836573)
#coords_2 = (estimated_position_average[0], estimated_position_average[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()
