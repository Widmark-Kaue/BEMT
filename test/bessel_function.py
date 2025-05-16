#%%
import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
#%%%
x = np.linspace(0, 10, 100)
order = np.arange(1, 4)
for o in order:
    plt.plot(x, jv(o, x), label = f'Order {o}')
# plt.plot(x, jv(0, x),'k',  label = f'Order {o}')
plt.plot([2, 4, 6],jv(order, [2, 4, 6]), 'ko')



plt.legend()
plt.grid()
plt.show()
# %%
# %%
