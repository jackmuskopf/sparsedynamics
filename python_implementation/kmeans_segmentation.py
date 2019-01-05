# Jack Muskopf
# Wash U ESE
# 10/30/18


import os
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.cluster import KMeans

from load_img.baseimage import PETImage, CTImage, normalize

# should probably implement as method for PETImage ultimately
def pca_kmeans_segmentation(pet_img, **kwargs):  # nclusters=10, nfeatures=None, options=None):

	default_options = {
		'nclusters' : 10,
		'n_init' : 20
	}

	# load and unload image outside of this routine
	pet_img.check_data()

	# initialize ROI as whole matrix
	roi = pet_img.img_data
	zd,yd,xd,td = roi.shape

	# get kwargs
	keys = kwargs.keys()

	# specify limits of region to segment
	if 'roi_lims' in keys and kwargs['roi_lims'] is not None:
		(zmin,zmax),(ymin,ymax),(xmin,xmax) = kwargs['roi_lims']
		roi = roi[zmin:zmax,ymin:ymax,xmin:xmax,:]
		zd,yd,xd,td = roi.shape

	# other kwargs		
	nclusters = kwargs['nclusters'] if 'nclusters' in keys else default_options['nclusters']
	n_init = kwargs['n_init'] if 'n_init' in keys else default_options['n_init']
	nfeatures = kwargs['nfeatures'] if 'nfeatures' in keys else td

	# default to use all principal components if not specified
	nfeatures = td if nfeatures is None else nfeatures

	# number of voxels to analyze
	N = zd*yd*xd

	# raw data matrix
	X = roi.reshape(N,td)

	# normalize each voxel over time
	# X = np.apply_along_axis(lambda x: x/np.linalg.norm(x,ord=2), 0, X)

	# SVD of data matrix
	U, S, Vh = np.linalg.svd(X.T.dot(X))

	# transform project each voxel's time series onto eigenvectors
	# U^T: X -> Xh
	X_h = U[:,:nfeatures].T.dot(X.T)

	# use kmeans clustering on transformed data
	kmeans = KMeans(init='k-means++', n_clusters=nclusters, n_init=n_init)
	kmeans.fit(X_h.T)
	Z = kmeans.predict(X_h.T)

	# create masks for clustered data
	root_mask = np.linspace(0,N-1,N)	# a skeleton for actual masks
	masks = []
	for clust in set(Z):
		mask_func = np.vectorize(lambda x:int(Z[int(x)]==clust))
		masks.append(mask_func(root_mask).reshape(zd,yd,xd))

	print('Created {} masks of image'.format(len(masks)))


	return masks, roi


def fourier_kmeans_segmentation(pet_img, **kwargs):  # nclusters=10, nfeatures=None, options=None):

	default_options = {
		'nclusters' : 10,
		'n_init' : 20
	}

	# load and unload image outside of this routine
	pet_img.check_data()

	# initialize ROI as whole matrix
	roi = pet_img.img_data
	zd,yd,xd,td = roi.shape

	# get kwargs
	keys = kwargs.keys()

	# specify limits of region to segment
	if 'roi_lims' in keys and kwargs['roi_lims'] is not None:
		(zmin,zmax),(ymin,ymax),(xmin,xmax) = kwargs['roi_lims']
		roi = roi[zmin:zmax,ymin:ymax,xmin:xmax,:]
		zd,yd,xd,td = roi.shape

	# other kwargs		
	nclusters = kwargs['nclusters'] if 'nclusters' in keys else default_options['nclusters']
	n_init = kwargs['n_init'] if 'n_init' in keys else default_options['n_init']
	nfeatures = kwargs['nfeatures'] if 'nfeatures' in keys else td

	# default to use all principal components if not specified
	nfeatures = td if nfeatures is None else nfeatures

	# number of voxels to analyze
	N = zd*yd*xd

	# raw data matrix
	X = roi.reshape(N,td)

	# fourier transform along time axis
	Xf = abs(np.fft.fft(X,axis=-1))

	# # SVD of data matrix
	# U, S, Vh = np.linalg.svd(Xf.T.dot(Xf))
	# # transform project each voxel's time series onto eigenvectors
	# # U^T: X -> Xh
	# X_h = U[:,:nfeatures].T.dot(Xf.T)

	X_h = Xf[:,0:nfeatures].T

	# use kmeans clustering on transformed data
	kmeans = KMeans(init='k-means++', n_clusters=nclusters, n_init=n_init)
	kmeans.fit(X_h.T)
	Z = kmeans.predict(X_h.T)

	# create masks for clustered data
	root_mask = np.linspace(0,N-1,N)	# a skeleton for actual masks
	masks = []
	for clust in set(Z):
		mask_func = np.vectorize(lambda x:int(Z[int(x)]==clust))
		masks.append(mask_func(root_mask).reshape(zd,yd,xd))

	print('Created {} masks of image'.format(len(masks)))


	return masks, roi




def plot_segments(segments, **kwargs):

	ax_map = {'z':0,'y':1,'x':2,'t':3}

	default_options = {
		'view_ax' : 'y',
		'mask_collapse' : np.sum,
		'time_collapse' : np.max,
		'spatial_collapse' : np.sum,
		'figsize' : (40,50),
		'fontsize' : 36
	}

	keys = kwargs.keys()
	view_ax = kwargs['view_ax'] if 'view_ax' in keys else default_options['view_ax']
	mask_collapse = kwargs['mask_collapse'] if 'mask_collapse' in keys else default_options['mask_collapse']
	time_collapse = kwargs['time_collapse'] if 'time_collapse' in keys else default_options['time_collapse']
	spatial_collapse = kwargs['spatial_collapse'] if 'spatial_collapse' in keys else default_options['spatial_collapse']
	figsize = kwargs['figsize'] if 'figsize' in keys else default_options['figsize']
	fontsize = kwargs['fontsize'] if 'fontsize' in keys else default_options['fontsize']

	# check and set np.ndarray axes
	view_ax = ax_map[view_ax] if view_ax in ax_map.keys() else view_ax
	if view_ax not in list(ax_map.keys()) + list(ax_map.values()):
		raise ValueError('Bad view_ax: {}'.format(view_ax))
	time_ax = 3

	# setup axes for plots
	nc = 6
	fig = plt.figure(figsize=figsize)
	axes = []
	nplots = len(segments)
	for k in range(nplots):
	    ax = fig.add_subplot(1+nplots//nc,nc,k+1)
	    ax.set_aspect('auto')
	    axes.append(ax)


	# plot each mask
	for k,(ax,premask)in enumerate(zip(axes,segments)):	
	    fmask = normalize(spatial_collapse(time_collapse(premask,axis=time_ax),axis=view_ax))
	    ax.imshow(fmask, cmap='gray')
	    ax.set_title('Segment ix: {}'.format(k),fontsize=36)
	plt.tight_layout()
	plt.show()


def apply_masks(masks,roi):
	new_rois = []
	for m in masks:
		nroi = np.stack([np.multiply(roi[:,:,:,k],m) for k in range(roi.shape[-1])],axis=-1)
		new_rois.append(nroi)
	return new_rois


def animate_frames(frames,**kwargs):

	fig = plt.figure()
	ims = [[plt.imshow(frame, cmap="gray", animated=True)] for frame in frames]
	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
	plt.show()

def logger(s):
	if DEBUG:
		print(s)

if __name__ == '__main__':
	matplotlib.use('TkAgg')
	DEBUG = False

	# iterate over image files in directory
	data_dir = 'data'
	for fname in os.listdir(data_dir):

		try:

			if fname.endswith('.pet.img') and not fname.startswith('.'):

				filepath = os.path.join(data_dir,fname)

				# load image
				img = PETImage(filepath=filepath)
				img.load_image()
				Ns = img.img_data.shape

				# select the middle z=Ws[0],y=Ws[1],x=Ws[2] prism of each frame; y axis might need to be wider
				Ws = (128,60,40)
				roi_lims = [(int((N-W)/2),int((N+W)/2)) for N,W in zip(Ns,Ws)]

				# select options for segmentation and run segmentation
				print('Clustering image voxels...')
				options = {
					'roi_lims' : roi_lims,
					'plot' : True,
					'nfeatures' : 6,
					'nclusters' : 20
					}
				masks, roi = pca_kmeans_segmentation(img,**options)

				# done with original image data
				img.unload_image()

				# apply mask to roi
				new_rois = apply_masks(masks,roi)

				# animate new ROIs
				collapse_method = 'sum'	# use sum or max
				view_ax = 'y'
				with_roi = True		# whether to show original roi next to animated mask
				wroi_map = {
					2 : 0,	# x->y
					0 : 1,	# z->x
					1 : 1	# y->x
				}

				# # animate each segment
				# view_ax = img.get_axis(view_ax)
				# for nroi in new_rois:
				# 	nroi = normalize(getattr(nroi,collapse_method)(axis=view_ax))
				# 	if with_roi:
				# 		wroi_ax = wroi_map[view_ax]
				# 		roi_frames  = normalize(getattr(roi,collapse_method)(axis=view_ax))
				# 		nroi = np.concatenate([nroi,roi_frames],axis=wroi_ax)
				# 	frames = np.split(nroi, nroi.shape[-1], axis=-1)
				# 	frames = [np.squeeze(f) for f in frames]
				# 	animate_frames(frames)

		except KeyboardInterrupt:
			plt.close()
			cont = input('\nContinue to next image?\nEnter empty string to continue; enter any other string to stop.')
			if cont != '':
				break





