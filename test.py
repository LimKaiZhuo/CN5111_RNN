import numpy as np
import matplotlib.pyplot as plt
from own_package.features_labels_setup import load_data_to_fl
from own_package.others import create_results_directory
from own_package.simulation import test
from own_package.pre_processing.sg_data_prep import sg_data

"""
u1 = np.array([100]*48)
u2 = np.copy(u1)
u2[15:17] = 2000
u = np.concatenate((u1.reshape(1,-1),u2.reshape(1,-1)), axis=0)
print(u)
test(u)
"""

fl = load_data_to_fl('./excel/results.xlsx')
price = fl.features_c[:,22,0]
demand = fl.labels[:,22,0]
plt.scatter(price, demand)
plt.show()


"""
x = np.arange(0,60).reshape(-1, 2)
print(x)
x_t = x.reshape(-1, 10, 2)
print(x_t)
x_f = x_t.reshape(-1, 2)
print(x_f)

print('x shape = {}, x_t shape = {}, x_f shape = {}'.format(x.shape, x_t.shape, x_f.shape))
"""

