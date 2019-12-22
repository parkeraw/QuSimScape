import numpy as np
import matplotlib.pyplot as plt

class LandScape:
    """contains a sampled spacial landscape of static electric potential information"""

    def __init__(self,x_dim,y_dim,z_dim):
        self.landscape_ = np.zeros((x,y,z))
        self.charges_ = np.array([])
        
    def add_charges(charge_list):
        """charge list should be [[x,y,z,q],...]"""
        self.charges_ = np.concatenate((self.charges_,charge_list))
