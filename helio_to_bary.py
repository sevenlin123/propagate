#############################################################################
#
# helio_to_bary.py, version 0.2
#
# transfer heliocentric Keplerian elements to barycentric Keplerian elements    
# 
# v0.2: add barycentric to heliocentric transformation
# 
# Author: 
# Edward Lin hsingwel@umich.edu
#############################################################################

from __future__ import division
import numpy as np
#Need skyfield (https://rhodesmill.org/skyfield/) to access JPL ephemeris
from skyfield.api import Topos, Loader
from scipy.optimize import newton

load = Loader('./Skyfield-Data', expire=False)
planets = load('de423.bsp')

class transformation:
    """
    convert between heliocentric Keplerian elements and barycentric Keplerian elements
    Units: au, radian, JD
      
    """
    def __init__(self, a, e, i, arg, node, M0, epoch):
        self.u_bary = 2.9630927492415936E-04 # standard gravitational parameter, GM, M is the mass of sun + all planets 
        self.u_helio = 2.9591220828559093E-04 # GM, M is the mass of sun
        self.epsilon =  23.43929111 * np.pi/180. # obliquity
        self.epoch = epoch
        self.a0 = a
        self.e0 = e
        self.i0 = i
        self.arg0 = arg
        self.node0 = node
        self.M0 = M0

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
        
    def helio_to_bary(self):
        # This is heliocentric xyz postion       
        X0, Y0, Z0, VX0, VY0, VZ0 =  self.kep_to_xyz(self.a0, self.e0, self.i0, self.arg0, self.node0, self.M0, self.u_helio)
        # extract barycentric postion of Sun
        sun = planets['Sun']
        ts = load.timescale()
        t = ts.tai(jd=self.epoch+0.000428) #37 leap seconds
        x_sun, y_sun, z_sun = sun.at(t).position.au
        vx_sun, vy_sun, vz_sun = sun.at(t).velocity.au_per_d
        # now calculate the heliocentric xyz postions of objects
        X = X0 + x_sun
        Y = Y0 + y_sun * np.cos(-self.epsilon) - z_sun * np.sin(-self.epsilon)
        Z = Z0 + y_sun * np.sin(-self.epsilon) + z_sun * np.cos(-self.epsilon)
        VX = VX0 + vx_sun
        VY = VY0 + vy_sun * np.cos(-self.epsilon) - vz_sun * np.sin(-self.epsilon)
        VZ = VZ0 + vy_sun * np.sin(-self.epsilon) + vz_sun * np.cos(-self.epsilon)
        # transfer back to keplerian elements
        self.a, self.e, self.i, self.arg, self.node, self.M = self.xyz_to_kep(X, Y, Z, VX, VY, VZ, self.u_bary)

    def bary_to_helio(self):
        # This is barycentric xyz postion       
        X0, Y0, Z0, VX0, VY0, VZ0 =  self.kep_to_xyz(self.a0, self.e0, self.i0, self.arg0, self.node0, self.M0, self.u_bary)
        # extract barycentric postion of Sun
        sun = planets['Sun']
        ts = load.timescale()
        t = ts.tai(jd=self.epoch+0.000428) #37 leap seconds
        x_sun, y_sun, z_sun = sun.at(t).position.au
        vx_sun, vy_sun, vz_sun = sun.at(t).velocity.au_per_d
        # now calculate the heliocentric xyz postions of objects
        X = X0 - x_sun
        Y = Y0 - y_sun * np.cos(-self.epsilon) + z_sun * np.sin(-self.epsilon)
        Z = Z0 - y_sun * np.sin(-self.epsilon) - z_sun * np.cos(-self.epsilon)
        VX = VX0 - vx_sun
        VY = VY0 - vy_sun * np.cos(-self.epsilon) + vz_sun * np.sin(-self.epsilon)
        VZ = VZ0 - vy_sun * np.sin(-self.epsilon) - vz_sun * np.cos(-self.epsilon)
        # transfer back to keplerian elements
        self.a, self.e, self.i, self.arg, self.node, self.M = self.xyz_to_kep(X, Y, Z, VX, VY, VZ, self.u_helio)


def helio_to_bary(list_of_object):
        object = np.array(list_of_object)
        # the transformation class takes radians, turn elements to radians
        h2b = transformation(object.T[0], object.T[1], object.T[2]*np.pi/180., object.T[3]*np.pi/180., object.T[4]*np.pi/180., object.T[5]*np.pi/180., object.T[6])
        h2b.helio_to_bary()
        # now turn the radians back to degrees
        elements = np.array([h2b.a, h2b.e, h2b.i*180/np.pi, h2b.arg*180/np.pi, h2b.node*180/np.pi, h2b.M*180/np.pi, h2b.epoch]).T
        return elements

def bary_to_helio(list_of_object):
        object = np.array(list_of_object)
        # the transformation class takes radians, turn elements to radians
        b2h = transformation(object.T[0], object.T[1], object.T[2]*np.pi/180., object.T[3]*np.pi/180., object.T[4]*np.pi/180., object.T[5]*np.pi/180., object.T[6])
        b2h.bary_to_helio()
        # now turn the radians back to degrees
        elements = np.array([b2h.a, b2h.e, b2h.i*180/np.pi, b2h.arg*180/np.pi, b2h.node*180/np.pi, b2h.M*180/np.pi, b2h.epoch]).T
        return elements


if __name__ == '__main__':
        eris_helio = np.array([6.768806773065693E+01, 4.420836167285916E-01, 4.413632128654130E+01, 1.511803890073615E+02, 3.590938986466752E+01, 2.045040412784239E+02, 2457375.5])
        ruby_helio = np.array([3.004664043215652E+01, 8.282679611502294E-02, 3.130856167451365E+01, 2.165133203133375E+02, 1.925864845971902E+02, 3.517576218761481E+02, 2457375.5])
        bp_helio = np.array([4.532976896627495E+02, 9.224097494635715E-01, 5.410743429007145E+01, 3.481369148319624E+02, 1.353379395756713E+02, 3.584014239710087E+02, 2457375.5])
        eris_bary = np.array([6.78341374e+01, 4.38452144e-01, 4.39930733e+01, 1.51216518e+02, 3.59764240e+01, 2.04147112e+02, 2.45737550e+06])
        ruby_bary = np.array([3.00866683e+01, 8.37151911e-02, 3.12584148e+01, 2.15447926e+02, 1.92538236e+02, 3.52704806e+02, 2.45737550e+06])
        bp_bary = np.array([4.49085364e+02, 9.21513173e-01, 5.41106748e+01, 3.48060483e+02, 1.35213249e+02, 3.58379172e+02, 2.45737550e+06])
        TV = np.array([1.112092419450883E+02,6.720550112402568E-01,3.114255960758642E+01,2.320933962766367E+02 ,1.810741849959539E+02,3.538989508776386E+02,2452522.85424])
        bary = helio_to_bary([eris_helio, ruby_helio, bp_helio])
        helio = bary_to_helio([eris_bary, ruby_bary, bp_bary, TV])
        print(bary)
        print(helio)
