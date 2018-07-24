#############################################################################
#
# propagate.py, version 0.5.1
#
# calculate equatorial sky positions for given Keplerian orbits and epochs     
#  
# Author: 
# Edward Lin hsingwel@umich.edu
# v0.2: improve efficiency
# v0.3: work for Python3
# v0.4: optional heliocentric elements input 
# v0.4.2: heliocentric elements work correctly
# v0.5: add bary_to_helio function (not use during propagation)
# v0.5.1: add attributes 'delta','r', 'phase_angle', 'elong' 
#         'delta': geocentric distance
#         'r': barycentric distance
#         'phase_angle': phase angle in radian
#         'elong': solar elongation in radian
# v0.5.2: correct elong calculations
##############################################################################

from __future__ import division
import numpy as np
from skyfield.api import Topos, Loader
from scipy.optimize import newton

load = Loader('./Skyfield-Data', expire=False)
planets = load('de423.bsp')
    
class propagate:
    """
    input Keplerian elements in the format of numpy arrays
    Units: au, radian, JD
      
    """
    def __init__(self, a, e, i, arg, node, M0, epoch, obs_date, helio=False):
        self.u_bary = 2.9630927492415936E-04 # standard gravitational parameter, GM, M is the mass of sun + all planets 
        self.u_helio = 2.9591220828559093E-04 # GM, M is the mass of sun
        self.epsilon =  23.43929111 * np.pi/180. # obliquity
        if helio:
            self.a, self.e, self.i, self.arg, self.node, self.M0 =  self.helio_to_bary(a,  e, i, arg, node, M0, epoch)
        else:
            self.a = a
            self.e = e
            self.i = i
            self.arg = arg
            self.node = node
            self.M0 = M0
        
        self.epoch = epoch
        self.obs_date = obs_date
        # compute M for given epoch
        self.M = (self.u_bary/self.a**3)**0.5 * (self.obs_date - self.epoch) + self.M0
        self.X, self.Y, self.Z, self.VX, self.VY, self.VZ = self.kep_to_xyz(self.a, self.e, self.i, self.arg, self.node, self.M, self.u_bary)
        self.ra, self.dec, self.delta, self.r, self.ltt, self.phase_angle, self.elong = self.xyz_to_equa(self.X, self.Y, self.Z, self.obs_date)
 
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

    def bary_to_helio(self, a0, e0, i0, arg0, node0, M0, epoch0):
        # This is barycentric xyz postion       
        X0, Y0, Z0, VX0, VY0, VZ0 =  self.kep_to_xyz(a0, e0, i0, arg0, node0, M0, self.u_bary)
        # extract barycentric postion of Sun
        sun = planets['Sun']
        ts = load.timescale()
        t = ts.tai(jd=epoch0+0.000428) #37 leap seconds
        x_sun, y_sun, z_sun = sun.at(t).position.au
        vx_sun, vy_sun, vz_sun = sun.at(t).velocity.au_per_d
        # now we have barycentric xyz postion 
        X = X0 - x_sun
        Y = Y0 - y_sun * np.cos(-self.epsilon) + z_sun * np.sin(-self.epsilon)
        Z = Z0 - y_sun * np.sin(-self.epsilon) - z_sun * np.cos(-self.epsilon)
        VX = VX0 - vx_sun
        VY = VY0 - vy_sun * np.cos(-self.epsilon) + vz_sun * np.sin(-self.epsilon)
        VZ = VZ0 - vy_sun * np.sin(-self.epsilon) - vz_sun * np.cos(-self.epsilon)
        # transfer back to keplerian elements
        a, e, i, arg, node, M = self.xyz_to_kep(X, Y, Z, VX, VY, VZ, self.u_helio)
        return a, e, i, arg, node, M

    def xyz_to_equa(self, X0, Y0, Z0, epoch):
        c = 173.1446323547978
        r = (X0**2 + Y0**2 + Z0**2)**0.5
        earth = planets['earth']
        earth = earth + Topos('30.169 S', '70.804 W', elevation_m=2200) #turn off the topocentric calculation should run much faster
        ts = load.timescale()
        t = ts.tai(jd=epoch+0.000428) #37 leap seconds
        x_earth, y_earth, z_earth = earth.at(t).position.au # earth ICRS position
        earth_dis = (x_earth**2 + y_earth**2 + z_earth**2)**0.5
        for i in range(3): 
            # transfer ecliptic to ICRS and shift to Geocentric (topocentric)
            X = X0 - x_earth 
            Y = Y0 * np.cos(self.epsilon) - Z0 * np.sin(self.epsilon) - y_earth
            Z = Y0 * np.sin(self.epsilon) + Z0 * np.cos(self.epsilon) - z_earth
            delta = (X**2 + Y**2+ Z**2)**0.5
            ltt = delta / c
            M = (self.u_bary/self.a**3)**0.5 * (-ltt) + self.M
            X0, Y0, Z0, VX0, VY0, VZ0 = self.kep_to_xyz(self.a, self.e, self.i, self.arg, self.node, M, self.u_bary)
        # Cartesian to spherical coordinate
        dec = np.arcsin(Z/(X**2+Y**2+Z**2)**0.5)
        ra = np.arctan2(Y, X) % (2*np.pi)
        phase_angle = np.arccos(-(earth_dis**2-r**2-delta**2)/(2*r*delta))
        elong = np.arccos(-(r**2-delta**2-earth_dis**2)/(2*delta*earth_dis))
        return ra, dec, delta, r, ltt, phase_angle, elong

        
if __name__ == '__main__':
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    #sedna_ephem = propagate_pyephem(540.6238899789372, .859111497291054, 11.92859044711287, 311.100256264511, 144.5022446594254, 358.222163293174, 2456580.5, 2457375.5)
    sedna0 = np.array([5.074246374075075E+02, 8.498746087028334E-01, 1.192855886739805E+01*np.pi/180., 3.113066085007133E+02*np.pi/180., 1.444030959000094E+02*np.pi/180., 3.581035054464383E+02*np.pi/180., 2457375.5, 2457741.5])
    sedna1 = np.array([5.074246374075075E+02, 8.498746087028334E-01, 1.192855886739805E+01*np.pi/180., 3.113066085007133E+02*np.pi/180., 1.444030959000094E+02*np.pi/180., 3.581035054464383E+02*np.pi/180., 2457375.5, 2458106.5])
    sedna2 = np.array([5.074246374075075E+02, 8.498746087028334E-01, 1.192855886739805E+01*np.pi/180., 3.113066085007133E+02*np.pi/180., 1.444030959000094E+02*np.pi/180., 3.581035054464383E+02*np.pi/180., 2457375.5, 2458471.5])
    eris0 = np.array([6.783405036515818E+01, 4.384475538521589E-01, 4.399285116152476E+01*np.pi/180., 1.512196225116616E+02*np.pi/180., 3.597654134974561E+01*np.pi/180., 2.041397376676136E+02*np.pi/180., 2457375.5, 2457740.5])
    eris1 = np.array([6.783405036515818E+01, 4.384475538521589E-01, 4.399285116152476E+01*np.pi/180., 1.512196225116616E+02*np.pi/180., 3.597654134974561E+01*np.pi/180., 2.041397376676136E+02*np.pi/180., 2457375.5, 2458105.5])
    eris2 = np.array([6.783405036515818E+01, 4.384475538521589E-01, 4.399285116152476E+01*np.pi/180., 1.512196225116616E+02*np.pi/180., 3.597654134974561E+01*np.pi/180., 2.041397376676136E+02*np.pi/180., 2457375.5, 2457375.5])
    eris_helio = np.array([6.768797414697519E+01, 4.420790192749195E-01, 4.413609832595201E+01*np.pi/180., 1.511834587301832E+02*np.pi/180., 3.590950696773813E+01*np.pi/180., 2.044966970545773E+02*np.pi/180., 2457375.5, 2457375.5])
    GH137 = np.array([39.439983, 0.228133, 13.468*np.pi/180., 152.542*np.pi/180., 35.741*np.pi/180., 0, 2452153.02, 2458285.54166667])
    GH137_2 = np.array([3.943811437738326E+01, 2.280719112904044E-01, 1.346811006640707E+01*np.pi/180., 1.525597064620324E+02*np.pi/180., 3.574051307337605E+01*np.pi/180., 2.441056774137685E+01*np.pi/180., 2458285.5, 2458285.5])
    GH137_helio = np.array([3.979291654743955E+01, 2.336699438884731E-01, 1.344241830596374E+01*np.pi/180., 1.536827090353943E+02*np.pi/180., 3.572856612008427E+01*np.pi/180., 2.338003676460184E+01*np.pi/180., 2458285.5, 2459285.5])
    BP519 = np.array([4.490784528803734E+02, 9.215119038599956E-01, 5.411067896592579E+01*np.pi/180., 3.480604843629308E+02*np.pi/180., 1.352131890380128E+02*np.pi/180., 3.584734429420708E+02*np.pi/180., 2458285.5, 2459285.5])
    BP519_helio = np.array([4.290994031655086E+02, 9.180967097705615E-01, 5.411891760737154E+01*np.pi/180., 3.483996318327017E+02*np.pi/180., 1.351726869671246E+02*np.pi/180., 3.583603303959885E+02*np.pi/180., 2458285.5, 2459285.5])
    bp_helio = np.array([420.48310521, 0.91645907, 0.9445533, 6.08211338, 2.35920844, 6.25364812, 2458285.5, 2458285.5])
    ruby = np.array([3.008719617754020E+01, 8.373084272981737E-02, 3.125845724207132E+01*np.pi/180., 2.154449425088119E+02*np.pi/180., 1.925382788635224E+02*np.pi/180., 3.502232533923760E+02*np.pi/180., 2456959.8, 2456959.8])
    ruby_helio = np.array([3.011617833264382E+01, 8.498447348993524E-02, 3.128624598019004E+01*np.pi/180., 2.165696064568567E+02*np.pi/180., 1.925663003544522E+02*np.pi/180., 3.492840525539137E+02*np.pi/180., 2456959.8, 2456959.8])
    #object = np.array([sedna0, sedna1, sedna2, eris0, eris1, eris2, GH137, GH137_2, BP519, BP519_helio])
    object = np.array([BP519])
    p = propagate(object.T[0], object.T[1], object.T[2], object.T[3], object.T[4], object.T[5], object.T[6], object.T[7])
    #print(p.a, p.e, p.i, p.arg, p.node, p.M)
    date = np.arange(2458362.5, 2458515.5, 1)
    n_obs = len(date)
    p = propagate(np.zeros(n_obs)+BP519[0], np.zeros(n_obs)+BP519[1], np.zeros(n_obs)+BP519[2], np.zeros(n_obs)+BP519[3], np.zeros(n_obs)+BP519[4], np.zeros(n_obs)+BP519[5], np.zeros(n_obs)+BP519[6], date)
    #print(p.ra, p.dec)
    c = SkyCoord(ra = list(p.ra*180/np.pi), dec = list(p.dec*180/np.pi), frame='icrs', unit='deg')
    #for n, i in enumerate(date):
        #print(i, c[n].ra.hms, c[n].dec.dms)
     #   print('{0}:'.format(i), '{0:02.0f} {1:02.0f} {2:05.2f}'.format(c[n].ra.hms.h, c[n].ra.hms.m, c[n].ra.hms.s), '{0:02.0f} {1:02.0f} {2:05.2f}'.format(c[n].dec.dms .d, c[n].dec.signed_dms .m, c[n].dec.signed_dms .s))
    
    #object = np.array([BP519_helio])
    #p1 = propagate(object.T[0], object.T[1], object.T[2], object.T[3], object.T[4], object.T[5], object.T[6], object.T[7], helio=True)
    #print(p1.a, p1.e, p1.i, p1.arg, p1.node, p1.M)
    #print(p1.ra*180/np.pi, p1.dec*180/np.pi, p.phase_angle*180/np.pi, p.elong*180/np.pi)
    #print(((p1.ra-p.ra)**2+(p1.dec-p.dec)**2)**0.5*180/np.pi*3600)
    #object = np.array([bp_helio])
    #p2 = propagate(object.T[0], object.T[1], object.T[2], object.T[3], object.T[4], object.T[5], object.T[6], object.T[7], helio=True)
    #print(p2.a, p2.e, p2.i, p2.arg, p2.node, p2.M)