#usr/bin/python

###Visual simulation for euler rotaion sequences. The origin frame is fixed to show the progression of the rotating frame.

import matplotlib.pyplot as plt
import numpy as np
import main
import geopy.distance
import ost_output_papakolea as ost

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
theta = np.radians(10)
phi = np.radians(20)
theta = 0.06161487984311333
phi = 0.03387764611236473
psi = -0.4118977034706618
rx = main.Rx(theta)
ry = main.Ry(phi)
rz = main.Rz(psi)
ryx = np.dot(ry, rx)
gtc_p = np.dot(ryx, gtc)

#i = ost.s1.cel[0]
i = np.array([-0.89138523 , -0.39121153, 0.22887966])
idx = [1, 2, 0]
b = ost.s1.body[0][idx]

'''B = main.B(z_axis, i, 1)
K = main.K(B)
q = main.q(K)
rotation_matrix = main.q2R(q)'''

ax.quiver(o[0], o[1], o[2], i[0], i[1], i[2], color="darkviolet")#,normalize=True)
ax.quiver(o[0], o[1], o[2], b[0], b[1], b[2], color="forestgreen")

bp = b + z
ax.quiver(o[0], o[1], o[2], bp[0], bp[1], bp[2], color="deepskyblue")
ax.quiver(z[0], z[1], z[2], b[0], b[1], b[2], color="y")

#bt = np.dot(rotation_matrix, b)
#ax.quiver(i[0], i[1], i[2], bt[0], bt[1], bt[2], color="y")

#btp = np.dot(bt + i)
#btp_norm = btp/np.linalg.norm(btp)
#ax.quiver(i[0], i[1], i[2], bt[0], bt[1], bt[2], color="pink")

bp_norm = bp/np.linalg.norm(bp)
br = z_axis_double - bp_norm
ax.quiver(bp_norm[0], bp_norm[1], bp_norm[2], br[0], br[1], br[2], color="orange")

B = main.B(z_axis, i, 1)
K = main.K(B)
q = main.q(K)
rotation_matrix = main.q2R(q)

brp = np.dot(rotation_matrix, br)
brpp = i + brp
brpp_norm = brpp/np.linalg.norm(brpp)
ax.quiver(i[0], i[1], i[2], brp[0], brp[1], brp[2], color="pink")
ax.quiver(o[0], o[1], o[2], brpp_norm[0], brpp_norm[1], brpp_norm[2], color="olive")

print('brpp_norm: ', brpp_norm)


gtc__sph = main.car2sph(gtc)
gtc_1_sph = main.car2sph(brpp_norm)

coords_1 = (gtc__sph[0], gtc__sph[1])
coords_2 = (gtc_1_sph[0], gtc_1_sph[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

plt.show()
