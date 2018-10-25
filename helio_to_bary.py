from __future__ import division
import numpy as np
from skyfield.api import Topos, Loader
from scipy.optimize import newton

load = Loader('./Skyfield-Data', expire=False)
planets = load('de423.bsp')

class transformation:
    """
    transfer heliocentric Keplerian elements to barycentric Keplerian elements
    Units: au, radian, JD
      
    """
    def __init__(self, a, e, i, arg, node, M0, epoch):
        self.u_bary = 2.9630927492415936E-04 # standard gravitational parameter, GM, M is the mass of sun + all planets 
        self.u_helio = 2.9591220828559093E-04 # GM, M is the mass of sun
        self.epsilon =  23.43929111 * np.pi/180. # obliquity
        self.epoch = epoch
        self.a, self.e, self.i, self.arg, self.node, self.M =  self.helio_to_bary(a,  e, i, arg, node, M0, epoch)

    def cal_E(self, e, M):
        # compute eccentric anomaly E
        f = lambda E, M, e: E - e * np.sin(E) - M
        E0 = M
        E = newton(f, E0, args=(M, e))
        return E
        
    def kep_to_xyz(self, a, e, i, arg, node, M, u):
        # compute eccentric anomaly E
        E = np.array(list(map(self.cal_E, e, M)))            
        # compute true anomaly v
        v = 2 * np.arctan2((1 + e)**0.5*np.sin(E/2.), (1 - e)**0.5*np.cos(E/2.))
        # compute the distance to the central body r
        r = a * (1 - e*np.cos(E))
        # obtain the position o and velocity ov vector
        ox = r * np.cos(v)
        oy = r * np.sin(v)
        oz = 0
        ovx = (u * a)**0.5 / r * (-np.sin(E))
        ovy = (u * a)**0.5 / r * ((1-e**2)**0.5 * np.cos(E))
        ovz = 0
        # Transform o and ov to the inertial frame
        X = ox * (np.cos(arg)*np.cos(node) - np.sin(arg)*np.sin(node)*np.cos(i)) - oy * (np.sin(arg)*np.cos(node) + np.cos(arg)*np.sin(node)*np.cos(i))
        Y = ox * (np.cos(arg)*np.sin(node) + np.sin(arg)*np.cos(node)*np.cos(i)) + oy * (np.cos(arg)*np.cos(node)*np.cos(i) - np.sin(arg)*np.sin(node))
        Z = ox * (np.sin(arg)*np.sin(i)) + oy * (np.cos(arg)*np.sin(i))
        VX = ovx * (np.cos(arg)*np.cos(node) - np.sin(arg)*np.sin(node)*np.cos(i)) - ovy * (np.sin(arg)*np.cos(node) + np.cos(arg)*np.sin(node)*np.cos(i))
        VY = ovx * (np.cos(arg)*np.sin(node) + np.sin(arg)*np.cos(node)*np.cos(i)) + ovy * (np.cos(arg)*np.cos(node)*np.cos(i) - np.sin(arg)*np.sin(node))
        VZ = ovx * (np.sin(arg)*np.sin(i)) + ovy * (np.cos(arg)*np.sin(i))
        return X, Y, Z, VX, VY, VZ
           
    def xyz_to_kep(self, X, Y, Z, VX, VY, VZ, u):
        # compute the barycentric distance r
        r = (X**2 + Y**2 + Z**2)**0.5
        rrdot = (X*VX + Y*VY + Z*VZ)
        # compute the specific angular momentum h
        hx = Y * VZ - Z * VY
        hy = Z * VX - X * VZ
        hz = X * VY - Y * VX
        h = (hx**2 + hy**2 + hz**2)**0.5
        # compute eccentricity vector
        ex = (VY * hz - VZ * hy)/u - X/r
        ey = (VZ * hx - VX * hz)/u - Y/r
        ez = (VX * hy - VY * hx)/u - Z/r
        e = (ex**2+ey**2+ez**2)**0.5
        # compute vector n
        nx = -hy
        ny = hx
        nz = 0
        n = (nx**2 + ny**2)**0.5
        # compute true anomaly v, the angle between e and r
        v = np.arccos((ex * X + ey * Y + ez * Z) / (e*r))
        v[rrdot<0] = 2*np.pi - v[rrdot<0]
        # compute inclination
        i = np.arccos(hz/h)
        # compute eccentric anomaly E
        E = 2*np.arctan2((1-e)**0.5*np.sin(v/2.), (1+e)**0.5*np.cos(v/2.))
        # compute ascending node
        node = np.arccos(nx/n)
        node[ny<0] = 2*np.pi - node[ny<0]
        # compute argument of periapsis, the angle between e and n
        arg = np.arccos((nx * ex + ny * ey + nz *ez) / (n*e))
        arg[ez<0] = 2*np.pi - arg[ez<0]
        # compute mean anomaly
        M = E - e * np.sin(E)
        M[M<0] += 2*np.pi
        # compute a
        a = 1/(2/r - (VX**2+VY**2+VZ**2)/u)
        return a, e, i, arg, node, M   
        
    def helio_to_bary(self, a0, e0, i0, arg0, node0, M0, epoch0):
        # This is heliocentric xyz postion       
        X0, Y0, Z0, VX0, VY0, VZ0 =  self.kep_to_xyz(a0, e0, i0, arg0, node0, M0, self.u_helio)
        # extract barycentric postion of Sun
        sun = planets['Sun']
        ts = load.timescale()
        t = ts.tai(jd=epoch0+0.000428) #37 leap seconds
        x_sun, y_sun, z_sun = sun.at(t).position.au
        vx_sun, vy_sun, vz_sun = sun.at(t).velocity.au_per_d
        # now we have barycentric xyz postion 
        X = X0 + x_sun
        Y = Y0 + y_sun * np.cos(-self.epsilon) - z_sun * np.sin(-self.epsilon)
        Z = Z0 + y_sun * np.sin(-self.epsilon) + z_sun * np.cos(-self.epsilon)
        VX = VX0 + vx_sun
        VY = VY0 + vy_sun * np.cos(-self.epsilon) - vz_sun * np.sin(-self.epsilon)
        VZ = VZ0 + vy_sun * np.sin(-self.epsilon) + vz_sun * np.cos(-self.epsilon)
        # transfer back to keplerian elements
        a, e, i, arg, node, M = self.xyz_to_kep(X, Y, Z, VX, VY, VZ, self.u_bary)
        return a, e, i, arg, node, M

def helio_to_bary(list_of_object):
        object = np.array(list_of_object)
        h2b = transformation(object.T[0], object.T[1], object.T[2], object.T[3], object.T[4], object.T[5], object.T[6])
        elements = np.array([h2b.a, h2b.e, h2b.i, h2b.arg, h2b.node, h2b.M, h2b.epoch]).T
        return elements

if __name__ == '__main__':
        eris_helio = np.array([6.768797414697519E+01, 4.420790192749195E-01, 4.413609832595201E+01*np.pi/180., 1.511834587301832E+02*np.pi/180., 3.590950696773813E+01*np.pi/180., 2.044966970545773E+02*np.pi/180., 2457375.5])
        GH137_helio = np.array([3.979291654743955E+01, 2.336699438884731E-01, 1.344241830596374E+01*np.pi/180., 1.536827090353943E+02*np.pi/180., 3.572856612008427E+01*np.pi/180., 2.338003676460184E+01*np.pi/180., 2458285.5])
        bp_helio = np.array([420.48310521, 0.91645907, 0.9445533, 6.08211338, 2.35920844, 6.25364812, 2458285.5])
        bary = helio_to_bary([eris_helio, GH137_helio, bp_helio])
        print(bary)