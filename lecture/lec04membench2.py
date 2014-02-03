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
plt.savefig(sys.argv[1]+'2a.pdf', bbox_inches='tight')

# 32K to 64K transition (exceeds L1)
plt.plot([1,27],[15.5,15.5], color='gray', linewidth=3, linestyle='--')

# 256K to 512K transition (exceeds L2)
#plt.plot([1,27],[18.5,18.5], color='gray', linewidth=3, linestyle='--')

# 2 MB to 4 MB transition (exceeds L3)
plt.plot([1,27],[21.5,21.5], color='gray', linewidth=3, linestyle='--')

# 64B to 128B stride (exceeds cache line)
plt.plot([5.5,5.5],[11,27], color='gray', linewidth=3, linestyle='--')

# 4K stride (one hit per page page size)
plt.plot([11.5,11.5],[11,27], color='gray', linewidth=3, linestyle='--')

# Edge of region where working set = 8 (cache associativity)
plt.plot([12,27],[12+3.5,27+3.5], color='gray', linewidth=3, linestyle='--')

# Edge of region where working set = 512 (L2 TLB associativity)
plt.plot([12,27],[12+9.5,27+9.5], color='gray', linewidth=3, linestyle='--')

# Edge of region where working set = 64 (L1 TLB associativity)
plt.plot([12,27],[12+6.5,27+6.5], color='gray', linewidth=3, linestyle='--')

plt.savefig(sys.argv[1]+'2.pdf', bbox_inches='tight')
