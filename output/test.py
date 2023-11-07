import numpy as np

data = np.genfromtxt('output.csv', delimiter=',')
out2 = np.genfromtxt('exposures.csv', delimiter=',')

errors = np.genfromtxt('errors_out.csv', delimiter=',')
out_errors = np.genfromtxt('errors.csv', delimiter=',')

print((data-out2).min())
print((data-out2).max())

print((errors-out_errors).min())
print((errors-out_errors).max())