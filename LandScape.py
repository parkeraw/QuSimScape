import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp




class LandScape:
    """
    contains a sampled spatial landscape of static electric potential information
    """

    def __init__(self):
        #self.landscape_ = np.zeros((x_dim,y_dim,z_dim))
        self.charges_ = np.zeros((0,4))
        self.ax_ = plt.axes(projection='3d')
        self.ax_.set_xlabel('x')
        self.ax_.set_ylabel('y')
        self.ax_.set_zlabel('z')
        
    def add_charges(self,charge_list):
        """
        charge list should be [[x,y,z,q],...]
        """
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
        #print(Z)

        self.ax_.contour3D(X,Y,Z, contour_density, cmap='binary')
        

class ChargeCube:
    '''
    one dilema here is that charge is uniform distributed..

    Im thinking that one way to fix this would be to feed in some 'center of body' coord that charge could grow as it is further away,
    so the density of point charges would still be unform but the charge would increase, which should have roughly the same effect as increasing
    charge density at the parts of the shape that stick out more
    '''

    def __init__(self, x1,y1,z1, x2,y2,z2,density = lambda x,y,z,q : -1,q = -1,step = 1):
        #there might be clever numpy magic to vectorize this but its speed is pretty irrelevant compatered to potential
        self.charges_ = np.array([[x,y,z,density(x-(x1+x2)/2.,y-(y1+y2)/2.,z-(z1+z2)/2.,q)] for x in np.arange(x1,x2+step,step) for y in np.arange(y1,y2+step,step) for z in np.arange(z1,z2+step,step)])
 

    def charges(self):
        return self.charges_
                    
        
class Particle:
    '''
    A Particle that lives in the potential landscape
    
    has its own mass and charge
    
    Obeys an approximation of the TISE for the wave function in this landscape
    '''
    def __init__(self,mass,charge,x0,y0,k0):
        '''
        store impotant values for calculating PDE solutions
        '''
        self.mass = mass
        self.charge = charge
        self.x0 = x0
        self.y0 = y0
        self.k0 = k0
        
    def psuedo_wkb(self,potential,energy,X,Y):
        '''
        cheaty way to approximate the wave function
        '''
        raise NotImplemented("")
        kx = np.sqrt((potential-energy)/self.mass,dtype=complex)
        ky = np.sqrt((potential-energy)/self.mass,dtype=complex)
        psi = np.exp(kx*X+ky*Y)
        return psi
    
    
    def wave_function(self,X,potential,x0,y0,k0):
        '''
        eventually will solve ODEs/PDEs for the TISE
        '''
        raise NotImplemented("")
        
        z = potential[np.shape(potential)[0]//2]
        x = X[np.shape(potential)[0]//2]
        sol = solve_ivp(schrodinger,[x[0],x[-1]],[np.exp(-1*np.sqrt(z[0],dtype=complex)*x[0])],dtype=complex)
        print(sol.y)
        print(sol.t)
        
        fig = plt.axes()
        fig.plot(sol.t,sol.y[0])
       
        
        #coefs = np.polyfit(x,z,5)#what degree to use?
        #V =  make_lambda_from_fit(coefs)
        
        
        

# Attempt at ODE solving
def schrodinger(psi,V,E=1.):
    raise NotImplemented("need to figure out solving system of ODE")
    return E*psi - V*psi

def phi_prime_is_psi(psi,x):
     raise NotImplemented("need to figure out solving system of ODE")


   
# charge density functions 
def const(x,y,z,q): return q
def linear(x,y,z,q): return q*np.sqrt(x*x + y*y + z*z)
def quadratic(x,y,z,q): return q*(x*x + y*y + z*z)
    



    
if __name__ == "__main__":
    
    # Testing the wave function approximations
    '''
    foo = LandScape()
    foo.add_charges(ChargeCube(0,0,0,5,5,0).charges())
    foo.add_charges(ChargeCube(8,0,0,12,5,0,q=-0.5).charges())
    foo.add_charges(ChargeCube(15,0,0,20,5,0).charges())
    X = np.arange(-5,25,1)
    Y = np.arange(-12.5,17.5,1)
    e = Particle(1,1,-5,0,1)
    X, Y = np.meshgrid(X,Y)
    e.wave_function(X,foo.potential(X,Y,-1),0,0,1)
    
    '''
    
    # Make a pretty landscape graph
    foo = LandScape()
    foofoo = Particle(1,1,-5,0,1)
    foo.add_charges(ChargeCube(0,0,0,5,5,0,density=quadratic).charges())
    foo.add_charges(ChargeCube(8,0,0,12,5,0,q=-0.5,density=quadratic).charges())
    foo.add_charges(ChargeCube(15,0,0,20,5,0,density=quadratic).charges())
    foo.plot_charge_locations()
    X = np.arange(-5,25,1)
    Y = np.arange(-12.5,17.5,1)
    foo.plot_potential(X,Y,-1,contour_density = 50)
    plt.show()
    
    
    
    
    
    '''
def make_lambda_from_fit(coefs): return lambda x : eval_poly(coefs,x)
    
def eval_poly(p,x):
    """
    efficiently evaluates a nth order polynomial with coefficients in p at a value x
    
    Parameters:
    -------------
        p: list or array of coefficients 
        
        x: float value
    """
    
    y = p[0]
    for coef in p[1:]:
        y = coef + y * x  
        
    return y
    ''' 
    