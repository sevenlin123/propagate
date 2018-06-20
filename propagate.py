##########################################################################
#
# propagate.py, version 0.3
#
# calculate equatorial sky positions for given Keplerian orbits and epochs     
#  
# Author: 
# Edward Lin hsingwel@umich.edu
# v0.2: improve efficiency
# v0.3: work for Python3
# v0.4: optional heliocentric elements input 
##########################################################################

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
        self.a = a
        self.e = e
        self.i = i
        self.arg = arg
        self.node = node
        self.M0 = M0
        self.epoch = epoch
        self.helio = helio
        self.obs_date = obs_date
        # compute M for given epoch
        self.M = (2.9630927492415936E-04/self.a**3)**0.5 * (self.obs_date - self.epoch) + self.M0
        self.X, self.Y, self.Z = self.kep_to_xyz(self.a, self.e, self.i, self.arg, self.node, self.M)
        self.ra, self.dec = self.xyz_to_equa(self.X, self.Y, self.Z, self.obs_date)
 
    def cal_E(self, e, M):
        # compute eccentric anomaly E
        f = lambda E, M, e: E - e * np.sin(E) - M
        E0 = M
        E = newton(f, E0, args=(M, e))
        return E
        
    def kep_to_xyz(self, a, e, i, arg, node, M):
        E = np.array(list(map(self.cal_E, e, M)))    
        # compute true anomaly v
        v = 2 * np.arctan2((1 + e)**0.5*np.sin(E/2.), (1 - e)**0.5*np.cos(E/2.))
        # compute the barycentric distance r
        r = a * (1 - e*np.cos(E))
        # compute X,Y,Z
        X = r * (np.cos(node) * np.cos(arg + v) - np.sin(node) * np.sin(arg + v) * np.cos(i))
        Y = r * (np.sin(node) * np.cos(arg + v) + np.cos(node) * np.sin(arg + v) * np.cos(i))
        Z = r * (np.sin(i) * np.sin(arg + v))
        return X, Y, Z
           
    def xyz_to_equa(self, X0, Y0, Z0, epoch):
        c = 173.1446323547978
        earth = planets['earth']
        earth = earth + Topos('30.169 S', '70.804 W', elevation_m=2200)
        ts = load.timescale()
        t = ts.tai(jd=epoch+0.000428) #37 leap seconds
        epsilon =  23.43929 * np.pi/180. # obliquity
        x_earth, y_earth, z_earth = earth.at(t).position.au # earth IRCS position
        earth_dis = (x_earth**2 + y_earth**2 + z_earth**2)**0.5
        if self.helio:
            sun = planets['Sun']
            x_sun, y_sun, z_sun = sun.at(t).position.au
            print(x_sun,y_sun,z_sun)
        
        for i in range(3): 
            # transfer ecliptic to IRCS and shift to Geocentric
            if self.helio:
                X = X0 - x_earth + x_sun
                Y = Y0 * np.cos(epsilon) - Z0 * np.sin(epsilon) - y_earth + y_sun
                Z = Y0 * np.sin(epsilon) + Z0 * np.cos(epsilon) - z_earth + z_sun
            else:
                X = X0 - x_earth 
                Y = Y0 * np.cos(epsilon) - Z0 * np.sin(epsilon) - y_earth
                Z = Y0 * np.sin(epsilon) + Z0 * np.cos(epsilon) - z_earth
            
            delta = (X**2 + Y**2+ Z**2)**0.5
            ltt = delta / c
            M = (2.9630927492415936E-04/self.a**3)**0.5 * (-ltt) + self.M
            X0, Y0, Z0 = self.kep_to_xyz(self.a, self.e, self.i, self.arg, self.node, M)
        print(X,Y,Z)    
        # Cartesian to spherical coordinate
        dec = np.arcsin(Z/(X**2+Y**2+Z**2)**0.5)
        ra = np.arctan2(Y, X) % (2*np.pi)
        return ra, dec

 #90377 Sedna,e,11.9292,144.4177,311.5959,482.2392,0.0000931,0.842226,358.0125,03/23.0/2018,2000,H 1.5,0.15
    
#def propagate_pyephem(a, e, i, arg, node, M, epoch, obs_date):
#    epoch = ephem.date(epoch-2415020)
#    yy = str(epoch).split('/')[0]
#    mm = str(epoch).split('/')[1]
#    dd = float(str(epoch).split('/')[2].split()[0]) + float(epoch-0.5) - int(epoch-0.5)
#    date = '{0}/{1}/{2}'.format(mm,dd,yy)
#    object = ephem.readdb("object,e,{0},{1},{2},{3},{4},{5},{6},{7},2000,H 1.5,0.15".format(i, node, arg, a, 0.0000931, e, M, date))
#    object.compute(ephem.date(obs_date-2415020))
#    return object.a_ra, object.a_dec
        

if __name__ == '__main__':
    #sedna_ephem = propagate_pyephem(540.6238899789372, .859111497291054, 11.92859044711287, 311.100256264511, 144.5022446594254, 358.222163293174, 2456580.5, 2457375.5)
    sedna0 = np.array([5.074246374075075E+02, 8.498746087028334E-01, 1.192855886739805E+01*np.pi/180., 3.113066085007133E+02*np.pi/180., 1.444030959000094E+02*np.pi/180., 3.581035054464383E+02*np.pi/180., 2457375.5, 2457741.5])
    sedna1 = np.array([5.074246374075075E+02, 8.498746087028334E-01, 1.192855886739805E+01*np.pi/180., 3.113066085007133E+02*np.pi/180., 1.444030959000094E+02*np.pi/180., 3.581035054464383E+02*np.pi/180., 2457375.5, 2458106.5])
    sedna2 = np.array([5.074246374075075E+02, 8.498746087028334E-01, 1.192855886739805E+01*np.pi/180., 3.113066085007133E+02*np.pi/180., 1.444030959000094E+02*np.pi/180., 3.581035054464383E+02*np.pi/180., 2457375.5, 2458471.5])
    eris0 = np.array([6.783405036515818E+01, 4.384475538521589E-01, 4.399285116152476E+01*np.pi/180., 1.512196225116616E+02*np.pi/180., 3.597654134974561E+01*np.pi/180., 2.041397376676136E+02*np.pi/180., 2457375.5, 2457740.5])
    eris1 = np.array([6.783405036515818E+01, 4.384475538521589E-01, 4.399285116152476E+01*np.pi/180., 1.512196225116616E+02*np.pi/180., 3.597654134974561E+01*np.pi/180., 2.041397376676136E+02*np.pi/180., 2457375.5, 2458105.5])
    eris2 = np.array([6.783405036515818E+01, 4.384475538521589E-01, 4.399285116152476E+01*np.pi/180., 1.512196225116616E+02*np.pi/180., 3.597654134974561E+01*np.pi/180., 2.041397376676136E+02*np.pi/180., 2457375.5, 2458470.5])
    GH137 = np.array([39.439983, 0.228133, 13.468*np.pi/180., 152.542*np.pi/180., 35.741*np.pi/180., 0, 2452153.02, 2458285.54166667])
    GH137_2 = np.array([3.943905045995138E+01, 2.280954040641315E-01, 1.346812359084134E+01*np.pi/180., 1.525555136613203E+02*np.pi/180., 3.574051820754930E+01*np.pi/180., 2.441205526338959E+01*np.pi/180., 2458285.5, 2458285.54166667])
    BP519 = np.array([4.490784528803734E+02, 9.215119038599956E-01, 5.411067896592579E+01*np.pi/180., 3.480604843629308E+02*np.pi/180., 1.352131890380128E+02*np.pi/180., 3.584734429420708E+02*np.pi/180., 2458285.5, 2458285.5])
    BP519_helio = np.array([4.290994031655086E+02, 9.180967097705615E-01, 5.411891760737154E+01*np.pi/180., 3.483996318327017E+02*np.pi/180., 1.351726869671246E+02*np.pi/180., 3.583603303959885E+02*np.pi/180., 2458285.5, 2458285.5])
    #object = np.array([sedna0, sedna1, sedna2, eris0, eris1, eris2, GH137, GH137_2, BP519, BP519_helio])
    object = np.array([BP519])
    p = propagate(object.T[0], object.T[1], object.T[2], object.T[3], object.T[4], object.T[5], object.T[6], object.T[7])
    print(p.ra*180/np.pi, p.dec*180/np.pi, p.X, p.Y, p.Z)
    object = np.array([BP519_helio])
    p1 = propagate(object.T[0], object.T[1], object.T[2], object.T[3], object.T[4], object.T[5], object.T[6], object.T[7], helio=True)
    print(p1.ra*180/np.pi, p1.dec*180/np.pi, p1.X, p1.Y, p1.Z)
    print(((p1.ra-p.ra)**2+(p1.dec-p.dec)**2)**0.5*180/np.pi*3600)