import numpy as np

# Wczytaj plik CSV do zmiennej numpy.ndarray
data = np.genfromtxt('data/output.csv', delimiter=',')
out2 = np.genfromtxt('exposures.csv', delimiter=',')

print((data-out2).min())
print((data-out2).max())