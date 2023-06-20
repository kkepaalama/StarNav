 #/usr/bin/python

### Purpose of this simulation to to show a visual of a rover located on Earths surface, with the z-axis pointed towards the celestial shpere
### The visuals are not to scale

import numpy as np
import random as ran
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
import main as m
import frame_trans as f

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

#earth frame
oE = np.array([0, 0, 0]) 
xE = np.array([1, 0, 0])
yE = np.array([0, 1, 0])
zE = np.array([0, 0, 1])

ax.quiver(oE[0], oE[1], oE[2], xE[0], xE[1], xE[2], color = 'red')
ax.quiver(oE[0], oE[1], oE[2], yE[0], yE[1], yE[2], color = 'blue')
ax.quiver(oE[0], oE[1], oE[2], zE[0], zE[1], zE[2], color = 'green')

sE = f.s_e
#ax.quiver(oE[0], oE[1], oE[2], sE[0], sE[1], sE[2], color = 'orange')

#local frame
''' 
special orthoganal unit vectors to create a local frame with x, y, and z-axis.
local frame origin is constrained to the surface of the sphere. The normal vector
of the local frame is normal to the surface. The gravity vector of the local frame 
is anti-parallel to the normal vector. The x-axis and y-axis are orthogonal to the 
z-axis which is also the normal vector.
'''

#local frame at random point
#s = ran.uniform(0, np.pi)
#t = ran.uniform(0, np.pi)

#local frame origin at known point
#x_ecef2enu = f.ecef_to_enu(xE, f.est[0], f.est[1])
#y_ecef2enu = f.ecef_to_enu(yE, f.est[0], f.est[1])
#z_ecef2enu = f.ecef_to_enu(zE, f.est[0], f.est[1])

#s_ecef2enu = f.ecef_to_enu(sE, f.est[0], f.est[1])

x_ecef2enu = f.ecef2enu(f.est[0], f.est[1], f.s_e, xE)
y_ecef2enu = f.ecef2enu(f.est[0], f.est[1], f.s_e, yE)
z_ecef2enu = f.ecef2enu(f.est[0], f.est[1], f.s_e, zE)

s_ecef2enu = f.ecef_to_enu(sE, f.est[0], f.est[1])
o_local = f.s_e


#local frame axes
#phi = ran.uniform(-(np.pi/4), (np.pi/4)) #random roll within limits
#theta = ran.uniform(-(np.pi/4), (np.pi/4)) #random pitch within limits
#psi = ran.uniform(0, 2*np.pi) #random yaw within limits
#angle_x = (phi, 0, 0)
#angle_y = (0, 90, 0)
#angle_z = (0, 0, psi)
#Rx = Rot(angle_x)
#RyL = m.R(angle_y)
#Rz = Rot(angle_z)
#yL = np.dot(RyL*0.1, o_local)
#xL = np.cross(yL, o_local)
#zL = np.array([o_local[0], o_local[1], o_local[2]])*0.1
#zb = np.dot(Rz*0.1, o_local)

#rad = np.pi/180
#tilt = [30*rad, 30*rad, 10*rad]
#tilt = [0, 0, 0]
#R = m.R(tilt)
#x_ecef2enu, y_ecef2enu, z_ecef2enu = np.dot(R, x_ecef2enu), np.dot(R, y_ecef2enu), np.dot(R, z_ecef2enu)

ax.quiver(o_local[0], o_local[1], o_local[2], x_ecef2enu[0], x_ecef2enu[1], x_ecef2enu[2], color = 'red') #heading vector
ax.quiver(o_local[0], o_local[1], o_local[2], y_ecef2enu[0], y_ecef2enu[1], y_ecef2enu[2], color = 'blue')
ax.quiver(o_local[0], o_local[1], o_local[2], z_ecef2enu[0], z_ecef2enu[1], z_ecef2enu[2], color = 'green') #normal vecor
#ax.quiver(o_local[0], o_local[1], o_local[2], -o_local[0]*0.1, -o_local[1]*0.1, -o_local[2]*0.1, color = 'orange') #gravity vector

#ax.quiver(o_local[0], o_local[1], o_local[2], s_ecef2enu[0], s_ecef2enu[1], s_ecef2enu[2], color = 'orange') #heading vector




ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.view_init(elev=30, azim=50)


#body frame
'''
special orthoganal unit vectors to create a body frame with x, y, and z-axis.
Body frame origin shares the same origin as local frame.
'''
'''rad = np.pi/180
tilt = [0*rad, 0*rad, 0*rad]
R = R(tilt)
x_B, y_B, z_B = [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]
RxB, RyB, RzB = np.dot(R, x_B), np.dot(R, y_B), np.dot(R, z_B)

#ax.quiver(originb[0],originb[1],originb[2], RxB[0], RxB[1], RxB[2], color="c")#,normalize=True)
#ax.quiver(originb[0],originb[1],originb[2], RyB[0], RyB[1], RyB[2], color="y")#,normalize=True)
#ax.quiver(originb[0],originb[1],originb[2], RzB[0], RzB[1], RzB[2], color="m")#,normalize=True)


xb = Rot(RxB, xL)
yb = Rot(RyB, yL)
zb = Rot(RzB, zL)

xB = np.dot(xb, RxB)*50
yB = np.dot(yb, RyB)*50
zB = np.dot(zb, RzB)*50

ax.quiver(originb[0],originb[1],originb[2], xB[0], xB[1], xB[2], color="c", linestyle = 'dashed')#,normalize=True)
ax.quiver(originb[0],originb[1],originb[2], yB[0], yB[1], yB[2], color="y", linestyle = 'dashed')#,normalize=True)
ax.quiver(originb[0],originb[1],originb[2], zB[0], zB[1], zB[2], color="m", linestyle = 'dashed')#,normalize=True)

#ax.quiver(originb[0],originb[1],originb[2], xB[0], xB[1], xB[2], color="c")#,normalize=True)
#ax.quiver(originb[0],originb[1],originb[2], yB[0], yB[1], yB[2], color="y")#,normalize=True)
#ax.quiver(originb[0],originb[1],originb[2], zB[0], zB[1], zB[2], color="m")#,normalize=True)'''



#earth
''' 
a sphere with radius r = 1 centered at [0, 0, 0] 
'''
'''origin = [0, 0, 0]
xe = np.array([0.1, 0, 0])
ye = np.array([0, 0.1, 0])
ze = np.array([0, 0, 0.1])
ax.quiver(origin[0], origin[1], origin[2], xe[0], xe[1], xe[2], color = 'red')
ax.quiver(origin[0], origin[1], origin[2], ye[0], ye[1], ye[2], color = 'blue')
ax.quiver(origin[0], origin[1], origin[2], ze[0], ze[1], ze[2], color = 'green')
u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)
xe = np.outer(np.sin(u), np.sin(v))
ye = np.outer(np.sin(u), np.cos(v))
ze = np.outer(np.cos(u), np.ones_like(v))
ax.plot_wireframe(xe, ye, ze, color = 'gray', alpha = 0.5)'''

    
#stars
'''
random distribution of stars at distance r = 10
'''
'''lat = np.random.uniform(0, 2*np.pi, 500)
lon = np.random.uniform(-np.pi/2, np.pi/2, 500)
xs = np.cos(lat) * np.cos(lon) * 5
ys = np.cos(lat) * np.sin(lon) * 5
zs = np.sin(lat) * 5

rand_stars = np.array([xs, ys, zs])#rand_stars = np.transpose(rand_stars)'''

#ax.scatter(xs, ys, zs, c='r', marker='o') #rand_stars[0], rand_stars[1], rand_stars[2]


#x = np.arange(0, 10, 0.5)
#y = np.arange(0, 10, 0.5)
#r = 5
#x, y = np.meshgrid(x, y)
#z = (r**2 - (x**2) - (y**2))/2

#ax.plot_surface(x, y, z)


#A0 = np.array([0, 0, 0])
#A1 = np.array([1, 0, 0])
#A0 = originb
#A1 = np.dot(originb, 4.8)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
#truncated_cone(A0, A1, 0.1, 1, 'blue')
plt.show()