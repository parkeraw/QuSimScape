import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class LandScape:
    """contains a sampled spacial landscape of static electric potential information"""

    def __init__(self,x_dim,y_dim,z_dim):
        self.landscape_ = np.zeros((x_dim,y_dim,z_dim))
        self.charges_ = np.zeros((0,4))
        
    def add_charges(self,charge_list):
        """charge list should be [[x,y,z,q],...]"""
        self.charges_ = np.concatenate((self.charges_,charge_list),axis=0)

    def plot_charge_locations(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.charges_[:,0],
                     self.charges_[:,1],
                     self.charges_[:,2],
                     c = self.charges_[:,3])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def potential_at(self,x,y,z):
        return self.landscape_[x,y,z]
        
    def plot_potential(self,z):
        X = np.array(np.arange(self.landscape_.shape[0]))
        Y = np.array(np.arange(self.landscape_.shape[1]))
        
        Z = self.landscape_[:,:,z]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X,Y,Z, 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def slow_compute(self):
        for x in np.arange(self.landscape_.shape[0]):
            for y in np.arange(self.landscape_.shape[1]):
                for z in np.arange(self.landscape_.shape[2]):
                    for c in self.charges_:
                        dist = np.sqrt((x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2)
                        dist = max(dist,0.5)
                        self.landscape_[x,y,z] += c[3]/dist
        

    
if __name__ == "__main__":
    foo = LandScape(10,10,10)
    foo.add_charges([[5,5,0,-1]])
    foo.plot_charge_locations()
    foo.slow_compute()
    foo.plot_potential(-2)
    plt.show()
