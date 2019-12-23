import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class LandScape:
    """contains a sampled spacial landscape of static electric potential information"""

    def __init__(self,x_dim,y_dim,z_dim):
        #self.landscape_ = np.zeros((x_dim,y_dim,z_dim))
        self.charges_ = np.zeros((0,4))
        self.ax_ = plt.axes(projection='3d')
        self.ax_.set_xlabel('x')
        self.ax_.set_ylabel('y')
        self.ax_.set_zlabel('z')
        
    def add_charges(self,charge_list):
        """charge list should be [[x,y,z,q],...]"""
        self.charges_ = np.concatenate((self.charges_,charge_list),axis=0)

    def plot_charge_locations(self):
        
        
        self.ax_.scatter3D(self.charges_[:,0],
                     self.charges_[:,1],
                     self.charges_[:,2],
                     c = self.charges_[:,3])
        
    def potential(self,X,Y,z):
        #there is still potential for improvments by vectorizing over charge, how to do so isn't yet obvious
        return sum([c[3] / np.sqrt((X-c[0])**2 + (Y-c[1])**2 + (z-c[2])**2) for c in self.charges_])
    
    def plot_potential(self,x,y,z):
        
        X, Y = np.meshgrid(x,y)
        Z = self.potential(X,Y,z)
        
        self.ax_.contour3D(X,Y,Z, 50, cmap='binary')
        

      

    
if __name__ == "__main__":
    foo = LandScape(10,10,10)
    foo.add_charges([[5,5,0,-1]])
    foo.plot_charge_locations()
    X = np.arange(0,10,.1)
    Y = np.arange(0,10,.1)
    foo.plot_potential(X,Y,-1)
    plt.show()
