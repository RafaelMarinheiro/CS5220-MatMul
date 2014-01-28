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

# Load data from timings.csv and plot it, grouping by size
df = pd.read_csv(sys.argv[1])
for key, grp in df.groupby("size"):
    plt.semilogx(grp['stride'], grp['ns'], label=size_name(key), basex=2)
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Stride (bytes)')
plt.ylabel('Time (ns)')
plt.savefig(sys.argv[1]+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

