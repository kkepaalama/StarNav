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
from ost_output_papakolea import s1, s15
import main as m


#constannts
gts = 21.295084, -157.811978 # <lat, lon> of ground truth in decimal degrees
gtc = np.array([-0.86272793, -0.35186237,  0.36317129]).reshape(3,1)

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

E_EoSb = m.cel2ecef(s1.time, s1.cel[0], s1.radec, 'radec2car')
#E_EoSb1 = m.cel2ecef(s15.time, s15.cel[0], s15.radec, 'radec2car')

tilt = [0.06161487984311333, 0.03387764611236473, -0.4118977034706618]
#tilt1 = [0.37533126407679634, 0.05805405820772749, -0.46972714620026923]
#tilt = np.radians([0, 0, 0])

#B_BoS0 = m.body2enu(tilt, s1.body[0])
B_BoS0 = np.array([-0.07990099,  0.06023875,  0.99498099]).reshape(3,1) #corrected x, y, z in rover frame
#B_BoS0 = np.array([0.00496625, 0.10761597, 0.99418002])
#B_BoS0 = m.body2enu(tilt, B_BoS0)
'''B_BoS0 = np.array([0.00496625, 0.10761597, 0.99418002])
B_BoS1 = np.array([-0.1644782 ,  0.03523238, 0.98575127]).reshape(3,1)
B_BoS1 = m.body2enu(tilt, B_BoS1)
B_BoS2 = np.array([0.01562646, 0.02335317, 0.99960512])
B_BoS2 = m.body2enu(tilt, B_BoS2)
B_BoS3 = np.array([0.1186599 , 0.02154414, 0.99270111])
B_BoS3 = m.body2enu(tilt, B_BoS3)'''
'''B_BoS4 = m.body2enu(tilt, s1.body[4])
B_BoS5 = m.body2enu(tilt, s1.body[5])
B_BoS6 = m.body2enu(tilt, s1.body[6])
B_BoS7 = m.body2enu(tilt, s1.body[7])
B_BoS8 = m.body2enu(tilt, s1.body[8])
B_BoS9 = m.body2enu(tilt, s1.body[9])'''

B0 = m.B(B_BoS0, E_EoS0, 1)
#B1 = m.B(z, B_BoS0, 1)
'''B1 = m.B(B_BoS1, E_EoS1, s1.wt[1])
B2 = m.B(B_BoS2, E_EoS2, s1.wt[2])
B3 = m.B(B_BoS3, E_EoS3, s1.wt[3])
B4 = m.B(B_BoS4, E_EoS4, s1.wt[4])
B5 = m.B(B_BoS5, E_EoS5, s1.wt[5])
B6 = m.B(B_BoS6, E_EoS6, s1.wt[6])
B7 = m.B(B_BoS7, E_EoS7, s1.wt[7])
B8 = m.B(B_BoS8, E_EoS8, s1.wt[8])
B9 = m.B(B_BoS9, E_EoS9, s1.wt[9])'''

B = B0 #+ B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9
K = m.K(B)
q = m.q(K)
ERB = m.q2R(q)

#K1 = m.K(B1)
#q1 = m.q(K1)
#ZRB = m.q2R(q1)

#B_BoS0 = m.body2enu(tilt, B_BoS0)
#a = np.dot(ERB, B_BoS0)
#a = np.dot(ERG, B_BoS0)

#E_BoS0 = np.dot(ERB, B_BoS0).reshape(3, 1)
 
#EpB = E_EoS0 - E_BoS0
#a = EpB/np.linalg.norm(EpB)
#print(m.car2sph(a))
#tilt = np.radians([0, 0, 0])
#R = m.Rotation_XYZ(tilt)
#E_EoS0 = np.dot(R, E_EoS0)'''


'''global_position = m.car2sph(E_EoSb)
print(global_position)


coords_1 = (gts[0], gts[1])
coords_2 = (global_position[0], global_position[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')


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

E_EoSb = m.cel2ecef(s1.time, s1.cel[0], s1.radec, 'radec2car')

tilt = [0.06161487984311333, 0.03387764611236473, -0.4118977034706618]
roll = 0.06161487984311333
pitch = 0.03387764611236473
heading = -0.4118977034706618 #east of north


B0 = m.B(s1.body[0], E_EoS0, s1.wt[0])
B1 = m.B(s1.body[1], E_EoS1, s1.wt[1])
B2 = m.B(s1.body[2], E_EoS2, s1.wt[2])
B3 = m.B(s1.body[3], E_EoS3, s1.wt[3])
B4 = m.B(s1.body[4], E_EoS4, s1.wt[4])
B5 = m.B(s1.body[5], E_EoS5, s1.wt[5])
B6 = m.B(s1.body[6], E_EoS6, s1.wt[6])
B7 = m.B(s1.body[7], E_EoS7, s1.wt[7])
B8 = m.B(s1.body[8], E_EoS8, s1.wt[8])
B9 = m.B(s1.body[9], E_EoS9, s1.wt[9])

B = B0 + B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9
K = m.K(B)
q = m.q(K)
rotation_matrix = m.q2R(q)'''
