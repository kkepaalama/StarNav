#usr/bin/python

'''
File returns position estimate for image belonging to s2 class
'''

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
from ost_output_papakolea import s2

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
phi = s2.imu_tilt[0] #0.06068190843512727 #roll
theta = s2.imu_tilt[1] #0.03443821713316293 #pitch
psi = -0.45076307922650694
rx = main.Rx(phi)
ry = main.Ry(theta)
ryx = np.dot(ry, rx)
rz = main.Rz(-psi)
rzyx = np.dot(rx, np.dot(ry, rz))

'''e = main.cel2ecef(s2.time, s2.cel[0], s2.radec, 'car')
e1 = main.cel2ecef(s2.time, s2.cel[1], s2.radec, 'car')
e2 = main.cel2ecef(s2.time, s2.cel[2], s2.radec, 'car')
e3 = main.cel2ecef(s2.time, s2.cel[3], s2.radec, 'car')
e4 = main.cel2ecef(s2.time, s2.cel[4], s2.radec, 'car')
e5 = main.cel2ecef(s2.time, s2.cel[5], s2.radec, 'car')
e6 = main.cel2ecef(s2.time, s2.cel[6], s2.radec, 'car')
e7 = main.cel2ecef(s2.time, s2.cel[7], s2.radec, 'car')
e8 = main.cel2ecef(s2.time, s2.cel[8], s2.radec, 'car')
e9 = main.cel2ecef(s2.time, s2.cel[9], s2.radec, 'car')
e10 = main.cel2ecef(s2.time, s2.cel[10], s2.radec, 'car')
e11 = main.cel2ecef(s2.time, s2.cel[11], s2.radec, 'car')
e12 = main.cel2ecef(s2.time, s2.cel[12], s2.radec, 'car')
e13 = main.cel2ecef(s2.time, s2.cel[13], s2.radec, 'car')'''



i = np.array([-0.90322083, -0.37367258, 0.21109461]) #main.cel2ecef(s2.time, s2.cel[0], s2.radec, 'car')
i1 = np.array([-0.89263542, -0.3883499 , 0.22888065]) #main.cel2ecef(s2.time, s2.cel[1], s2.radec, 'car')
i2 = np.array([-0.85757245, -0.46800765, 0.21339712]) #main.cel2ecef(s2.time, s2.cel[2], s2.radec, 'car')
i3 = np.array([-0.90000306, -0.31623280, 0.29998550]) #main.cel2ecef(s2.time, s2.cel[3], s2.radec, 'car')
i4 = np.array([-0.91333506, -0.22342375,  0.3404422]) #main.cel2ecef(s2.time, s2.cel[4], s2.radec, 'car')
i5 = np.array([-0.89225900, -0.33838557, 0.29894663]) #main.cel2ecef(s2.time, s2.cel[5], s2.radec, 'car')
i6 = np.array([-0.89860768, -0.28621414, 0.33254428]) #main.cel2ecef(s2.time, s2.cel[6], s2.radec, 'car')
i7 = np.array([-0.87691411, -0.33876983, 0.34096430]) #main.cel2ecef(s2.time, s2.cel[7], s2.radec, 'car')
i8 = np.array([-0.87471075, -0.34696232, 0.33837591]) #main.cel2ecef(s2.time, s2.cel[8], s2.radec, 'car')
i9 = np.array([-0.85314156, -0.39968128, 0.33526760]) #main.cel2ecef(s2.time, s2.cel[9], s2.radec, 'car')
i10 = np.array([-0.8811917, -0.24780785, 0.40260706]) #main.cel2ecef(s2.time, s2.cel[10], s2.radec, 'car')
i11 = np.array([-0.86603428, -0.31081278, 0.39163765]) #main.cel2ecef(s2.time, s2.cel[11], s2.radec, 'car')
i12 = np.array([-0.84008633, -0.39514527, 0.37163849]) #main.cel2ecef(s2.time, s2.cel[12], s2.radec, 'car')
i13 = np.array([-0.8496394 , -0.29889619, 0.43448125]) #main.cel2ecef(s2.time, s2.cel[13], s2.radec, 'car')


idx = [1, 2, 0]
b = s2.body[0][idx]
b1 = s2.body[1][idx]
b2 = s2.body[2][idx]
b3 = s2.body[3][idx]
b4 = s2.body[4][idx]
b5 = s2.body[5][idx]
b6 = s2.body[6][idx]
b7 = s2.body[7][idx]
b8 = s2.body[8][idx]
b9 = s2.body[9][idx]
b10 = s2.body[10][idx]
b11 = s2.body[11][idx]

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
iRb = main.q2R(q) #iRb rotation from body to inertial


br = np.dot(iRb, b)
br1 = np.dot(iRb, b1)
br2 = np.dot(iRb, b2)
#ax.quiver([0], o[1], o[2], b[0], b[1], b[2], color="purple")
#ax.quiver([0], o[1], o[2], b1[0], b1[1], b1[2], color="purple")
#ax.quiver([0], o[1], o[2], b2[0], b2[1], b2[2], color="purple")
#ax.quiver([0], o[1], o[2], br1[0], br1[1], br1[2], color="purple")
#ax.quiver([0], o[1], o[2], br2[0], br2[1], br2[2], color="purple")
#ax.quiver([0], o[1], o[2], br[0], br[1], br[2], color="purple")


#ax.quiver([0], o[1], o[2], i[0], i[1], i[2], color="orange")
#ax.quiver([0], o[1], o[2], i1[0], i1[1], i1[2], color="orange")
#ax.quiver([0], o[1], o[2], i2[0], i2[1], i2[2], color="orange")

z_axis_new = np.dot(iRb, z_axis)
ax.quiver([0], o[1], o[2], z_axis_new[0], z_axis_new[1], z_axis_new[2], color="b")

t = np.dot(np.transpose(ryx), z_axis)
ax.quiver(o[0], o[1], o[2], t[0], t[1], t[2], color="green")

tp = np.dot(iRb, t)
ax.quiver([0], o[1], o[2], tp[0], tp[1], tp[2], color="green")

bn = np.dot(ryx, b)
bn1 = np.dot(ryx, b1)
bn2 = np.dot(ryx, b2)
#ax.quiver([0], o[1], o[2], bn[0], bn[1], bn[2], color="y")
#ax.quiver([0], o[1], o[2], bn1[0], bn1[1], bn1[2], color="y")
#ax.quiver([0], o[1], o[2], bn2[0], bn2[1], bn2[2], color="y")

#gtcr = np.dot(np.transpose(iRb), gtc)
#ax.quiver([0], o[1], o[2], gtcr[0], gtcr[1], gtcr[2], color="dodgerblue")

tpp = np.dot(main.Rz(np.pi), t)
#ax.quiver([0], o[1], o[2], tpp[0], tpp[1], tpp[2], color="seagreen")

tpp1 = np.dot(main.Rz(psi), tpp)
ax.quiver([0], o[1], o[2], tpp1[0], tpp1[1], tpp1[2], color="limegreen")

tpp1_r = np.dot(iRb, tpp1)
ax.quiver([0], o[1], o[2], tpp1_r[0], tpp1_r[1], tpp1_r[2], color="limegreen")


true_position = main.car2sph(gtc)
estimated_position = main.car2sph(tpp1_r)
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