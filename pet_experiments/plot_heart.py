import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fname = 'mpet3721a_em1_v1_s4_20-cluster_2018-10-29 13:19:08.csv'

df = pd.read_csv(os.path.join('state_data',fname))

heart_df = df[df.columns[pd.Series(df.columns).str.startswith('organs')]]

ns = len(heart_df.columns)

ixs = [(u,v,w) for u in range(ns) for v in range(ns) for w in range(ns)]
ixs = [r for r in ixs if len(set(r))==3]

for j,k,l in ixs:
	h1 = heart_df.iloc[:,j]
	h2 = heart_df.iloc[:,k]
	h3 = heart_df.iloc[:,l]

	fig = plt.figure()
	ax = Axes3D(fig)	
	ax.plot(h1,h2,h3,'.-')
	plt.title('{} vs {} vs {}'.format(j,k,l))
	plt.show()