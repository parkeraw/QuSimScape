import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class LandScape:
    """contains a sampled spacial landscape of static electric potential information"""

    def __init__(self):
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
        return np.sum([c[3] / np.sqrt((X-c[0])**2 + (Y-c[1])**2 + (z-c[2])**2) for c in self.charges_], axis = 0)

    
    def plot_potential(self,x,y,z,contour_density = 50):
        
        X, Y = np.meshgrid(x,y)
        Z = self.potential(X,Y,np.ones(X.shape)*z)
        
        self.ax_.contour3D(X,Y,Z, contour_density, cmap='binary')
        

class ChargeCube:
    def __init__(self, x1,y1,z1, x2,y2,z2,q = -1,step = 1):
        #there might be clever numpy magic to vectorize this but its speed is pretty irrelevant compatered to potential
        
        self.charges_ = np.array([[x,y,z,q] for x in np.arange(x1,x2+step,step) for y in np.arange(y1,y2+step,step) for z in np.arange(z1,z2+step,step)])

    def charges(self):
        return self.charges_
                    
        

      

    
if __name__ == "__main__":
    foo = LandScape()
    foo.add_charges(ChargeCube(0,0,0,5,5,0).charges())
    foo.add_charges(ChargeCube(8,0,0,12,5,0,q=-0.5).charges())
    foo.add_charges(ChargeCube(15,0,0,20,5,0).charges())
    foo.plot_charge_locations()
    X = np.arange(-5,25,1)
    Y = np.arange(-12.5,17.5,1)
    foo.plot_potential(X,Y,-1,contour_density = 50)
    plt.show()
