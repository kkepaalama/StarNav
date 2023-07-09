#/usr/bin/python

import numpy as np
import math as m
import geopy.distance
import pymap3d

#astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

#data
from ost_output_papakolea import s1
import main as m


#constannts
rad = np.pi/180
deg = 180/np.pi
gts = 21.295084, -157.811978 # <lat, lon> of ground truth in decimal degrees
gtc = np.array([-0.86272793, -0.35186237,  0.36317129])

#origins-points
#radec = [323.113683*rad, 0.810916*rad]
#latlone = cel2ecef(s1.time, s1.cel, s1.radec, 'radec2sph')
#latlon_e = 18.28441551*rad, 200.99184445*rad #latlon_e shortcut (transfromed vector from cel to ecef)
test = [18.28441551*rad, 200.99184445*rad] #21.295084*rad, -157.811978*rad
#car_e = cel2ecef(s1.time, s1.cel, s1.radec, 'radec2car') #transformed vector from cel to ecef (cartesian)
#s_e = sph2car(latlon_e[0], latlon_e[1]) 
s_e = m.sph2car(test[0], test[1])
init_e = s_e
org_l = [s_e[0], s_e[1], s_e[2]]


#vectors
#init_c = np.array([0, 0, 0]) #initial boresight coordinate in celestial frame in <RA, DEC>
#init_e = cel2ecef(s1.time, s1.cel, s1.radec, 'sph') #vector from org_e to star (since the vector is normalize it is also the point that intersects shpere with r = 1 in <x, y, z>)'''
#init_l = ecef2enu(latlon_e[0], latlon_e[1], org_l, org_l)
init_l = m.ecef2enu(test[0], test[1], org_l, org_l)
#init_l = ecef_to_enu(s_e, 18.28441551, 200.99184445)
#tilt = [1*rad, 0*rad, 0*rad]
#tilt = [0.06161487984311333, 0.03387764611236473, 0*rad]
tilt = [-0.4118977034706618, 0.06161487984311333, 0.03387764611236473]
#init_b = R(tilt, init_l)
#init_l = init_l*10**3
init_b = m.body2enu(tilt, init_l)


R_BE = m.Rot(init_b, init_e) #rotation matrix from B to E
Rdotp = np.dot(R_BE, init_e)
p_E = (Rdotp + init_l)
p_e = p_E/np.linalg.norm(p_E)
pe = m.car2sph(p_e)
print(pe)

B = m.B(p_e, gtc, 1)
K = m.K(B)
q = m.q(K)
r = m.q2R(q)
a = m.car2sph(np.dot(r, p_e))



coords_1 = (gts[0], gts[1])
coords_2 = (pe[0], pe[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

#ideal conditions
'''lat, lon = 21.295084*rad, -157.811978*rad    #initial position
#lat, lon = 21*rad, -157*rad        #initial position
s_e = sph2car(lat, lon)         #conversion from spherical to cartesian
org_e = [0, 0, 0]       #origin of E frame in terms of E
org_l = [s_e[0], s_e[1], s_e[2]]        #origin of L frame
org_b = org_l       #origin of L and B frame coincide
init_e = s_e        #initial position in cartesian
init_l = ecef2enu(lat, lon, s_e, s_e)       #initial position for L frame


tilt = [0.1*rad, 0.0*rad, 0*rad]      #tilt
init_b = enu2body(tilt, init_l)         #initial position of B frame origin
R_BE = Rot(init_b, init_e)      #homogeneous transformation
Rdotp = np.dot(R_BE, init_e)
p_E = Rdotp + init_l
p_e = p_E/np.linalg.norm(p_E)
pe = car2sph(p_e)       #final position estimate
print(pe)'''


'''for i in np.linspace(0, np.pi/2, num=11):
    tilt = [0, i, 0]    
    init_b = enu2body(tilt, init_l)
    R_BE = Rot(init_b, init_e)
    Rdotp = np.dot(R_BE, init_e)
    p_E = Rdotp + init_l
    p_e = p_E/np.linalg.norm(p_E)
    pe = car2sph(p_e)
    print(pe)'''


#origins-points
#radec = [323.113683*rad, 0.810916*rad]
#latlone = cel2ecef(s1.time, s1.cel, s1.radec, 'radec2sph')
#latlon_e = 18.28441551*rad, 200.99184445*rad #latlon_e shortcut (transfromed vector from cel to ecef)
#test = np.array([np.radians(18.28441551), np.radians(200.99184445)]) #21.295084*rad, -157.811978*rad
#est = [18.28441551, 200.99184445]
#car_e = cel2ecef(s1.time, s1.cel, s1.radec, 'radec2car') #transformed vector from cel to ecef (cartesian)
#s_e = sph2car(latlon_e[0], latlon_e[1]) 
#s_e = sph2car(test[0], test[1]) 
#org_l = np.array([s_e[0], s_e[1], s_e[2]])
#init_e = org_l

#vectors
#init_c = np.array([0, 0, 0]) #initial boresight coordinate in celestial frame in <RA, DEC>
#init_e = cel2ecef(s1.time, s1.cel, s1.radec, 'sph') #vector from org_e to star (since the vector is normalize it is also the point that intersects shpere with r = 1 in <x, y, z>)'''
#init_l = ecef2enu(latlon_e[0], latlon_e[1], org_l, org_l)
#s1.ecef = cel2ecef(s1.time, s1.cel[0], s1.radec, 'car')
#s1.ecef = np.array([-0.89138523, -0.39121153,  0.22887966])
#s1.ecef = [s1.ecef[0], s1.ecef[1], s1.ecef[2]]
#init_l = ecef_to_enu(s1.body[0], test[0], test[1])
#init_l = ecef_to_enu(s_e, 18.28441551, 200.99184445)
#tilt = np.array([0, 0, 0])
#tilt = [0.06161487984311333, 0.03387764611236473, -0.4118977034706618]
#init_b = R(tilt, init_l)
#init_b = enu2body(tilt, init_l)




'''#R_BE = Rot(init_b, init_e) #rotation matrix from B to E
#Rdotp = np.dot(R_BE, init_e)
#p_E = (Rdotp + init_l)
#p_e = p_E/np.linalg.norm(p_E)
pe = main.car2sph(EL1)
print(pe)


coords_1 = (gts[0], gts[1])
coords_2 = (pe[0], pe[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')'''
