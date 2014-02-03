#!/share/cs-instructional/cs5220/local/anaconda/bin/python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def size_name(s):
    "Give a string name to a size in bytes"
    ksize = s/1024
    if ksize >= 1024:
        return "{0}M".format(ksize/1024)
    else:
        return "{0}K".format(ksize)

# Load data from timings.csv
df = pd.read_csv(sys.argv[1] + ".csv")

# Get colormap name (http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps)
cmap = 'jet'
if len(sys.argv) == 3:
    cmap = sys.argv[2]

# Plot time vs stride and working set sizes
plt.figure()
df["working"] = df.size / df.stride
A = np.empty([27,27])
A[:] = np.NAN
for idx,row in df.iterrows():
    i = int(row['size']).bit_length()-1
    j = int(row['stride']).bit_length()-1
    A[i,j] = row['ns']
plt.matshow(A, cmap=plt.get_cmap(cmap))
plt.xlabel('log2(stride)')
plt.ylabel('log2(size)')

plt.xlim([1,27])
plt.ylim([11,27])

plt.colorbar(shrink=0.6)

plt.savefig(sys.argv[1]+'2.pdf', bbox_inches='tight')
