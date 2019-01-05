import os
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from functools import reduce
from datetime import datetime

from segmentation import *
from baseimage import PETImage, CTImage, normalize




# if __name__ == "__main__":

try:

	out_folder = 'state_data'

	# pet files
	files = [os.path.join('single_mouse_pet',f) for f in os.listdir('single_mouse_pet') if f.endswith('.pet.img')]

	# load image
	filepath = files[0]
	img = PETImage(filepath=filepath)
	img.load_image()
	Ns = img.img_data.shape
	Ts = Ns[-1]

	# select the middle z=Ws[0],y=Ws[1],x=Ws[2] prism of each frame; y axis might need to be wider
	Ws = (128,60,50)
	roi_lims = [(int((N-W)/2),int((N+W)/2)) for N,W in zip(Ns,Ws)]

	# select options for segmentation and run segmentation
	print('Clustering image voxels...')
	nclusters = 20
	options = {
		'roi_lims' : roi_lims,
		'plot' : True,
		'nfeatures' : 6,
		'nclusters' : nclusters
		}



	# do segmentation
	masks, roi = pca_kmeans_segmentation(img,**options)
	masked_rois = apply_masks(masks,roi)

	# remove tempfile
	img.unload_image()

	# ask for names for each segment
	seg_names = []
	for mroi in masked_rois:
		frame = mroi.sum(axis=1).sum(axis=-1)
		plt.imshow(frame)
		plt.show()
		name = input('Enter name for segment: ')
		seg_names.append(name)

	# make clusters into state data
	flt_rois = [mr.reshape(reduce(lambda x,y: x*y,Ws),Ts) for mr in masked_rois]
	icurves = [fr.sum(axis=0) for fr in flt_rois]
	dmat = np.stack(icurves)
	df = pd.DataFrame(data=dmat).transpose()
	df.columns = seg_names #['x'+str(i) for i in range(len(icurves))]

	# We want to remove time points before tracer is completely injected
	# find this timepoint by viewing below plot
	# ie remove first the t_start data points
	t_start = 2
	df = df.iloc[t_start:]


	df.plot(marker='.')
	plt.show()

	fname = "{0}_{1}-cluster_{2}.csv".format(img.filename.split('.')[0],nclusters,str(datetime.now()).split('.')[0])
	df.to_csv(os.path.join(out_folder,fname))



except Exception as e:
	print(e)
	img.unload_image()
