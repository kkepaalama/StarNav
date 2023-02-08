#/usr/bin/python

### Purpose of this simulation to to show a visual of a rover located on Earths surface, with the z-axis pointed towards the celestial shpere
### The visuals are not to scale

import numpy as np
import random as ran
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
 
def Rot(angles):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(angles[0])), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
    y_rot = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    z_rot = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
    n = np.dot(x_rot, np.dot(y_rot, z_rot))
    return n


'''fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# Set up the grid in polar
theta = np.linspace(0,2*np.pi,90)
r = np.linspace(0,3,50)
T, R = np.meshgrid(theta, r)

# Then calculate X, Y, and Z
X = R * np.cos(T)
Y = R * np.sin(T)
Z = np.sqrt(X**2 + Y**2)

# Set the Z values outside your range to NaNs so they aren't plotted
#Z[Z < 0] = np.nan
#Z[Z > 2.1] = np.nan
ax.plot_wireframe(X, Y, Z)

ax.set_zlim(0,2)'''


'''from mpl_toolkits.mplot3d import Axes3D    # @UnusedImport

from math import pi, cos, sin

z = np.arange(0, 2, 0.02)
theta = np.arange(0, 2*pi + pi/50, pi/50)

fig = plt.figure()
axes1 = fig.add_subplot(111, projection='3d')
for zval in z:
    x = zval * np.array([cos(q) for q in theta])
    y = zval * np.array([sin(q) for q in theta])
    axes1.plot(x, y, zval, 'b-')


plt.show()'''


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')


def truncated_cone(p0, p1, R0, R1, color):
    """
    Based on https://stackoverflow.com/a/39823124/190597 (astrokeat)
    """
    # vector in direction of axis
    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # print n1,'\t',norm(n1)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 80
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R *
               np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=color, linewidth=0, antialiased=False, alpha = 0.5)

#body frame
''' special orthoganal unit vectors to create a body frame with x, y, and z-axis.
Body frame is given a random orientation and place in a random location on the surface of the sphere. '''
xr = np.array([0.3, 0, 0])
yr = np.array([0, 0.3, 0])
zr = np.array([0, 0, 0.3])

#body frame at random point
#s = ran.uniform(0, np.pi)
#t = ran.uniform(0, np.pi)

#body fram at known point
lat = 19.8968
lon = 360 - 155.5828
x = np.cos(lat) * np.cos(lon)
y = np.cos(lat) * np.sin(lon)
z = np.sin(lat)
originb = [x, y, z]


#phi = ran.uniform(-(np.pi/4), (np.pi/4))
#theta = ran.uniform(-(np.pi/4), (np.pi/4))
#psi = ran.uniform(0, 2*np.pi)
angle_x = (90, 0, 0) #(phi, theta, psi)
Rx = Rot(angle_x)
xb = np.dot(Rx, originb)
ax.quiver(originb[0], originb[1], originb[2], xb[0], xb[1], xb[2], color = 'red')
#ax.quiver(originb[0], originb[1], originb[2], yb[0], yb[1], yb[2], color = 'blue')
#ax.quiver(originb[0], originb[1], originb[2], zb[0], zb[1], zb[2], color = 'red')
ax.quiver(originb[0], originb[1], originb[2], originb[0], originb[1], originb[2], color = 'green')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.view_init(elev=30, azim=50)

#earth
''' a sphere with radius r = 1 centered at [0, 0, 0] '''
origin = [0, 0, 0]
xe = np.array([1, 0, 0])
ye = np.array([0, 1, 0])
ze = np.array([0, 0, 1])
ax.quiver(origin[0], origin[1], origin[2], xe[0], xe[1], xe[2], color = 'red')
ax.quiver(origin[0], origin[1], origin[2], ye[0], ye[1], ye[2], color = 'blue')
ax.quiver(origin[0], origin[1], origin[2], ze[0], ze[1], ze[2], color = 'green')
u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))
ax.plot_wireframe(x, y, z, color = 'gray')

    
#stars
''' random distribution of stars at distance r = 10 '''
lat = np.random.uniform(0, 2*np.pi, 500)
lon = np.random.uniform(-np.pi/2, np.pi/2, 500)
xs = np.cos(lat) * np.cos(lon) * 5
ys = np.cos(lat) * np.sin(lon) * 5
zs = np.sin(lat) * 5

rand_stars = np.array([xs, ys, zs])#rand_stars = np.transpose(rand_stars)

ax.scatter(xs, ys, zs, c='r', marker='o') #rand_stars[0], rand_stars[1], rand_stars[2]


#x = np.arange(0, 10, 0.5)
#y = np.arange(0, 10, 0.5)
#r = 5
#x, y = np.meshgrid(x, y)
#z = (r**2 - (x**2) - (y**2))/2

#ax.plot_surface(x, y, z)


#A0 = np.array([0, 0, 0])
#A1 = np.array([1, 0, 0])
A0 = originb
A1 = np.dot(originb, 4.8)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
truncated_cone(A0, A1, 0.1, 1, 'blue')
plt.show()