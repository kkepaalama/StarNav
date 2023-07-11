#/usr/bin/python

import numpy as np
import math
import geopy.distance
import pymap3d

#astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

#data
from ost_output_papakolea import s1
import main as m

def Rx(theta):
    rx = np.array([[1, 0, 0],
                   [0, (np.cos(theta)), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    return rx
    
def Ry(theta):
    ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    return ry

def Rz(theta):
    rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    return rz

ground_truth = 21.295084, -157.811978 #latitude, longitude of test site in degrees
lat_test = np.radians(18.28441551)
lon_test = np.radians(200.99184445)
tilt = [0.06161487984311333, 0.03387764611236473, -0.4118977034706618]
roll = 0.06161487984311333
pitch = 0.03387764611236473
heading = -0.4118977034706618 #east of north

E_EoS0 = m.cel2ecef(s1.time, s1.cel[0], s1.radec, 'car')
E_EoS1 = m.cel2ecef(s1.time, s1.cel[1], s1.radec, 'car')
E_EoS2 = m.cel2ecef(s1.time, s1.cel[2], s1.radec, 'car')
E_EoS3 = m.cel2ecef(s1.time, s1.cel[3], s1.radec, 'car')
E_EoS4 = m.cel2ecef(s1.time, s1.cel[4], s1.radec, 'car')
E_EoS5 = m.cel2ecef(s1.time, s1.cel[5], s1.radec, 'car')
E_EoS6 = m.cel2ecef(s1.time, s1.cel[6], s1.radec, 'car')
E_EoS7 = m.cel2ecef(s1.time, s1.cel[7], s1.radec, 'car')
E_EoS8 = m.cel2ecef(s1.time, s1.cel[8], s1.radec, 'car')
E_EoS9 = m.cel2ecef(s1.time, s1.cel[9], s1.radec, 'car')

bs = m.cel2ecef(s1.time, s1.cel[0], s1.radec, 'radec2car')

tilt = [0.06161487984311333, 0.03387764611236473, -0.4118977034706618]
roll = 0.06161487984311333
pitch = 0.03387764611236473
heading = -0.4118977034706618 #east of north

idx = [1, 2, 0] #changes vectors from camera to body

B0 = m.B(s1.body[0][idx], E_EoS0, s1.wt[0])
B1 = m.B(s1.body[1][idx], E_EoS1, s1.wt[1])
B2 = m.B(s1.body[2][idx], E_EoS2, s1.wt[2])
B3 = m.B(s1.body[3][idx], E_EoS3, s1.wt[3])
B4 = m.B(s1.body[4][idx], E_EoS4, s1.wt[4])
B5 = m.B(s1.body[5][idx], E_EoS5, s1.wt[5])
B6 = m.B(s1.body[6][idx], E_EoS6, s1.wt[6])
B7 = m.B(s1.body[7][idx], E_EoS7, s1.wt[7])
B8 = m.B(s1.body[8][idx], E_EoS8, s1.wt[8])
B9 = m.B(s1.body[9][idx], E_EoS9, s1.wt[9])


'''B0 = m.B(s1.body[0][idx], s1.cel[0], s1.wt[0])
B1 = m.B(s1.body[1][idx], s1.cel[1], s1.wt[1])
B2 = m.B(s1.body[2][idx], s1.cel[2], s1.wt[2])
B3 = m.B(s1.body[3][idx], s1.cel[3], s1.wt[3])
B4 = m.B(s1.body[4][idx], s1.cel[4], s1.wt[4])
B5 = m.B(s1.body[5][idx], s1.cel[5], s1.wt[5])
B6 = m.B(s1.body[6][idx], s1.cel[6], s1.wt[6])
B7 = m.B(s1.body[7][idx], s1.cel[7], s1.wt[7])
B8 = m.B(s1.body[8][idx], s1.cel[8], s1.wt[8])
B9 = m.B(s1.body[9][idx], s1.cel[9], s1.wt[9])'''

B = B0 + B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9
K = m.K(B)
q = m.q(K)
rotation_matrix = m.q2R(q) #body to inertial or s to i
transpose = np.transpose(rotation_matrix)

phi = np.pi/2 - np.arccos(rotation_matrix[2, 2])
lam = np.arctan2(rotation_matrix[1, 2]/np.cos(phi), rotation_matrix[0, 2]/np.cos(phi))
alpha = np.pi - np.arctan2(rotation_matrix[2, 1]/np.cos(phi), -rotation_matrix[2, 0]/np.cos(phi))

CSI = np.dot(Rz(s1.radec[0]), np.dot(Ry(np.pi/2 - s1.radec[1]), Rz(heading)))


Cvs = np.dot(Ry(pitch), Rx(roll)) #np.transpose(np.dot(Ry(-pitch), Rx(-roll))) #from v to s
Cis = rotation_matrix #np.transpose(rotation_matrix)
Cif = Rz(lam)

Cmf = Rz(lam)
Cmt = Ry(np.pi/2 - phi)
Ctv = Rz(np.pi - alpha)


#Cfv = np.dot(np.transpose(Cif), np.dot(Cis, Csv))
Cfv = np.dot(Cmf, np.dot(Cmt, Ctv)) #Cfv_RHS = np.dot(Rz(lam), np.dot(Ry(np.pi/2 - phi), Rz(np.pi - alpha)))

a = np.dot(Cfv, s1.body[0][idx].reshape(3,1))
b = np.dot(Cfv, s1.body[1][idx].reshape(3,1))
c = np.dot(Cfv, s1.body[2][idx].reshape(3,1))
d = np.dot(Cfv, s1.body[3][idx].reshape(3,1))
e = np.dot(Cfv, s1.body[4][idx].reshape(3,1))
bs = np.dot(Cfv, bs)

a = m.car2sph(a)
b = m.car2sph(b)
c = m.car2sph(c)
d = m.car2sph(d)
e = m.car2sph(e)
bs = m.car2sph(bs)

ground_truth = (ground_truth[0], ground_truth[1])
a = (a[0], a[1])
b = (b[0], b[1])
c = (c[0], c[1])
d = (d[0], d[1])
d = (d[0], d[1])
bs = (bs[0], bs[1])

'''print(a)
print(geopy.distance.geodesic(ground_truth, a).miles, 'miles    ', geopy.distance.geodesic(ground_truth, a).km, 'km')
print(b)
print(geopy.distance.geodesic(ground_truth, b).miles, 'miles    ', geopy.distance.geodesic(ground_truth, b).km, 'km')
print(c)
print(geopy.distance.geodesic(ground_truth, c).miles, 'miles    ', geopy.distance.geodesic(ground_truth, c).km, 'km')
print(d)
print(geopy.distance.geodesic(ground_truth, d).miles, 'miles    ', geopy.distance.geodesic(ground_truth, d).km, 'km')
print(e)
print(geopy.distance.geodesic(ground_truth, e).miles, 'miles    ', geopy.distance.geodesic(ground_truth, e).km, 'km')
print(bs)
print(geopy.distance.geodesic(ground_truth, bs).miles, 'miles    ', geopy.distance.geodesic(ground_truth, bs).km, 'km')'''
