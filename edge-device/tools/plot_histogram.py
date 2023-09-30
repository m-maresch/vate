import itertools

import matplotlib.pyplot as plt
import numpy as np

frq, edges = (np.array([1, 1, 1, 4, 5, 33, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 2, 3, 37, 64, 17, 4, 45]),
              np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                        13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., np.inf]))

fig, ax = plt.subplots()

frq = np.append(frq, [0])
edges = edges[:-1]
edges = np.append(edges, [edges[-1] + 1])

bars = ax.bar(edges, frq, width=1, edgecolor="white", align="edge")
bars = bars[:-1]

bars[-1].set_facecolor("purple")

labels = itertools.chain(map(int, edges[:-1]), ["inf"])
ax.set_xticks(edges, labels=labels)

fig.tight_layout()

plt.show()
