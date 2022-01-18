import numpy as np
import math as m

def v(x, y, z):
    return np.array([[x], [y], [z]])


def B(w, vb, vi):
    return (w*(vb*np.transpose(vi)))


def K(B):
    S = B + np.transpose(B)
    Z = np.array([[B[1, 2] - B[2, 1]], 
                  [B[2, 0] - B[0, 2]],
                  [B[0, 1] - B[1, 0]]])
    sigma = np.array([[B.trace()]])
    top_right = S - sigma*np.identity(3)
    K1 = np.concatenate((top_right, np.transpose(Z)), axis = 0)
    K2 = np.concatenate((Z, sigma), axis = 0)
    K = np.concatenate((K1, K2), axis = 1)
    return K

def q(K):
    eigval, eigvec = np.linalg.eig(K)
    if np.max(eigval) == eigval[0]:
        return eigvec[:, 0]
    elif np.max(eigval) == eigval[1]:
        return eigvec[:, 1]
    elif np.max(eigval) == eigval[2]:
        return eigvec[:, 2] 
    elif np.max(eigval) == eigval[3]:
        return eigvec[:, 3]

    
def R_bi(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q0 = q[3]
    
    a11 = 2*(q0**2 + q1**2) - 1
    a12 = 2*(q1*q2 - q0*q3)
    a13 = 2*(q1*q3 + q0*q2)
    
    a21 = 2*(q1*q2 + q0*q3)
    a22 = 2*(q0**2 + q2**2) - 1
    a23 = 2*(q2*q3 - q0*q1)
    
    a31 = 2*(q1*q3 - q0*q2)
    a32 = 2*(q2*q3 + q0*q1)
    a33 = 2*(q0**2 + q3**2) -1
    return np.array([[a11, a12, a13], 
                     [a21, a22, a23],
                     [a31, a32, a33]])

def n_sensed(q):
    n1 = q[0]/(np.sin(np.arccos(q[3])))
    n2 = q[1]/(np.sin(np.arccos(q[3])))
    n3 = q[2]/(np.sin(np.arccos(q[3])))
    return np.array([[n1], [n2], [n3]])


def pos(phi, theta, psi, n_sensed):
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-phi*(np.pi/180)), -np.sin(-phi*(np.pi/180))],
                      [0, np.sin(-phi*(np.pi/180)), np.cos(-phi*(np.pi/180))]])
    y_rot = np.array([[np.cos(-theta*(np.pi/180)), 0, np.sin(-theta*(np.pi/180))],
                      [0, 1, 0],
                      [-np.sin(-theta*(np.pi/180)), 0, np.cos(-theta*(np.pi/180))]])
    z_rot = np.array([[np.cos(psi*(np.pi/180)), -np.sin(psi*(np.pi/180)), 0],
                      [np.sin(psi*(np.pi/180)), np.cos(psi*(np.pi/180)), 0],
                      [0, 0, 1]])
    n_true = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), n_sensed)
    lat = np.pi/2 - np.arccos(n_true[2])
    lon = np.arctan(n_true[1]/n_true[0])
    return print(f'true position: {n_true}'), print(f'latitude: {lat}'), print(f'longitude: {lon}')


def gain(q, K):
    eigval, eigvec = np.linalg.eig(K)
    g = np.dot(np.transpose(q), np.dot(K,q)) - np.max(eigval)*np.dot(np.transpose(q), q)
    return g


v1b = v(0.7814, 0.3751, 0.4987)
v2b = v(0.6163, 0.7075, -0.3459)
v1i = v(0.2673, 0.5345, 0.8018)
v2i = v(-0.3124, 0.9370, 0.1562)
w = 1

B1 = B(w, v1b, v1i)
B2 = B(w, v2b, v2i)
B = B1 + B2


K = K(B)


q = q(K)


#print(np.linalg.norm(q)) #check


Rot = np.transpose(R_bi(q))


p = gain(q, K)
print(f'error: {p}')

n = n_sensed(q)

P = pos(0, 0, 0, n)
