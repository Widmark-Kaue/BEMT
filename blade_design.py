#%% Packages
import numpy as np
import matplotlib.pyplot as plt

#%% Plot airfoil
x, y = np.loadtxt('S834.dat', skiprows=1, unpack=True)

plt.plot(x, y)
plt.axis('equal')
plt.show()

# %%
