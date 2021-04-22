import argparse
import numpy as np

parser = argparse.ArgumentParser(description='LIME evaluation tool')
parser.add_argument('--n')
args = parser.parse_args()


coeffs = np.array([0.3, 0.1, 0.05, 0.5, 0, 0.3, 0.05, 0.001, 0.7])

coeffs = -np.sort(-coeffs)

lag = np.abs(np.diff(coeffs))

print(coeffs)
print(lag / coeffs[:-1])

#A = {k: v for k, v in vars(args).items() if v is not None}


#print(A)
