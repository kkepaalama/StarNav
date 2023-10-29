#usr/bin/python

'''
File returns position estimate for image belonging to s10 class
'''

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
from ost_output_papakolea import s10


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
phi = s10.imu_tilt[0] #roll
theta = s10.imu_tilt[1] #pitch
psi = s10.imu_tilt[2] #yaw

#converts star coordinates in GCRS (which is a ECI frame) to ITRS (which is a ECEF frame)
e = main.cel2ecef(s10.time, s10.cel[0], s10.radec, 'gcrs')
e1 = main.cel2ecef(s10.time, s10.cel[1], s10.radec, 'gcrs')
e2 = main.cel2ecef(s10.time, s10.cel[2], s10.radec, 'gcrs')
e3 = main.cel2ecef(s10.time, s10.cel[3], s10.radec, 'gcrs')
e4 = main.cel2ecef(s10.time, s10.cel[4], s10.radec, 'gcrs')
e5 = main.cel2ecef(s10.time, s10.cel[5], s10.radec, 'gcrs')
e6 = main.cel2ecef(s10.time, s10.cel[6], s10.radec, 'gcrs')
e7 = main.cel2ecef(s10.time, s10.cel[7], s10.radec, 'gcrs')
e8 = main.cel2ecef(s10.time, s10.cel[8], s10.radec, 'gcrs')
e9 = main.cel2ecef(s10.time, s10.cel[9], s10.radec, 'gcrs')
e10 = main.cel2ecef(s10.time, s10.cel[10], s10.radec, 'gcrs')
e11 = main.cel2ecef(s10.time, s10.cel[11], s10.radec, 'gcrs')
ev = main.cel2ecef(s10.time, s10.cel[11], s10.radec, 'radec2car')

#corresponds to the following above; same array just faster to access. Dont have to run astropy everytime (which takes several minutes)
i = np.array([-0.902362  , -0.36521182,  0.22882994])
i1 = np.array([-0.86936606, -0.44574107,  0.21334823])
i2 = np.array([-0.90787183, -0.29293473,  0.29992992])
i3 = np.array([-0.91880953, -0.19981998,  0.34038355])
i4 = np.array([-0.90070174, -0.31527809,  0.29889136])
i5 = np.array([-0.90570543, -0.26296439,  0.3324867 ])
i6 = np.array([-0.88537547, -0.31605803,  0.34090699])
i7 = np.array([-0.88338396, -0.32430399,  0.33831882])
i8 = np.array([-0.8873114 , -0.22502266,  0.40254599])
i9 = np.array([-0.87378275, -0.28839293,  0.39157785])
i10 = np.array([-0.85709022, -0.27690384,  0.43441989])
i11 = np.array([-0.86032542, -0.20897521,  0.46494046])


v = np.array([-0.88646827, -0.34027765,  0.31366403]) #boresight vector of the camera
ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color="violet")

#re-indexes body coordinates
# same as applying rotation from startracker "s" to body "b" --> bRs
idx = [1, 2, 0]
b = s10.body[0][idx]
b1 = s10.body[1][idx]
b2 = s10.body[2][idx]
b3 = s10.body[3][idx]
b4 = s10.body[4][idx]
b5 = s10.body[5][idx]
b6 = s10.body[6][idx]
b7 = s10.body[7][idx]
b8 = s10.body[8][idx]
b9 = s10.body[9][idx]
b10 = s10.body[10][idx]
b11 = s10.body[11][idx]

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
#i_v_br = np.dot(main.Rz(np.radians(-s10.imu_tilt[2])), i_v_br1)
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

print('true position_XYZ: ', true_position)
print('estimate position_XYZ: ', estimated_position)


coords_1 = (true_position[0], true_position[1])
coords_2 = (estimated_position[0], estimated_position[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()