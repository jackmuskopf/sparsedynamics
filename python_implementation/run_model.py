# Jack Muskopf
# 12/3/18
# jam.muskopf@gmail.com

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
fontP = FontProperties()
fontP.set_size('small')
import itertools
from decimal import Decimal
import scipy
from scipy.signal import gausspulse
from scipy.integrate import odeint
from mouse_model import OrganModel, real_mouse, simple_mouse
from utility_functions import sparsifyDynamics, poolData



def gauss_env(x,mu=.8,sig=.1):
	return np.exp(-.5*(x-mu)**2/(sig**2))

def optimal_linear_trajectory(X,A,times):
    # X should be np.ndarray, each row a state at time t
    # A should be df output by sindy, should be linear
    # times should be times at which we evaluate x(t)
    
    def print_nan(mat):
        print(np.count_nonzero(np.isnan(mat)))
    shape = X.shape
    true_states = X.flatten()
    
    # get transition matrix
    A = A.iloc[:,1:].values
    transition_matrices = [scipy.linalg.expm(A*t) for t in times]
    phi = np.concatenate(transition_matrices)
    opt_x0 = np.linalg.inv(phi.T.dot(phi)).dot(phi.T).dot(true_states)
    
    
    # resulting soln
    return phi.dot(opt_x0).reshape(shape)

def predicted_linear_trajectory(x0,A,times):
    # x0 should be np.ndarray
    # A should be df output by sindy, should be linear
    # times should be times at which we evaluate x(t)
    A = A.iloc[:,1:].values
    transition_matrices = [scipy.linalg.expm(A*t) for t in times]
    states = [phi.dot(x0) for phi in transition_matrices]
    trajectory = np.array(states)
    return trajectory


    
    # resulting soln
    return phi.dot(opt_x0).reshape(shape)


def SINDy(tX, dX, columns, lam=10**-3, polyorder=1, usesine=0, max_iter=1000):
	# SINDy params
	nVars = len(cols)


	# generate data
	Theta = poolData(tX,nVars, polyorder, usesine)
	dot_labels = pd.Index([s + ' [dot]' for s in cols])

	# run SINDy
	dXh = sparsifyDynamics(Theta,dX,lam, max_iter=max_iter)
	dXh = dXh.set_index(dot_labels)

	return dXh

def SNR(x,noisy,dB=True):
	rms = lambda v: np.sqrt((np.linalg.norm(v)**2)/len(v))
	x_rms = rms(x)
	noise = noisy - x
	noise_rms = rms(noise)

	snr = (x_rms/noise_rms)**2

	return 10*np.log10(snr) if dB else snr



def plot_traj(name):
	ix = mouse.get_var(name).ix
	traj = trajectory[:,ix]
	plt.plot(t_span,traj)
	plt.title(name)
	plt.show()

def plot_ss(name1,name2):
	ixs = mouse.get_var(name1).ix, mouse.get_var(name2).ix
	trajs = trajectory[:,ixs]
	plt.plot(trajs[0,:],trajs[1,:])
	plt.title('{} vs {}'.format(name1,name2))
	plt.show()


mouse=real_mouse()


n_states = len(mouse.state_vars)

# impulse input to arterial blood
arterial_blood_ix = mouse.get_var('Arterial Blood').ix

impulse_mag = 100
input_fn = lambda t: np.array([gauss_env(t)*int(ix==arterial_blood_ix) for ix in range(n_states)])
mouse.input_fn = lambda x: 0 #input_fn


# # test eval_derivative fn and input fn
# rstate = lambda : np.random.normal(size=n_states)
# zstate = np.zeros(n_states)
# d_dt = mouse.eval_derivative(at_state=zstate, t=0)
# print(d_dt)

cols = ["({})".format(sv.name) for sv in mouse.state_vars]
TX = pd.DataFrame(columns=cols)
DX = pd.DataFrame(columns=cols)


# SINDy params
polyorder = 1
lam = 10**-5
usesine = 0
max_iter=10**6
plot=True
# noise_power = 1*10**-6

lam_noise = [
	(2*10**-5, 0),
	(6*10**-4, 10**-3),     # (5*10**-4, 10**-3)
	(8*10**-4, 10**-2),
	(5*10**-3, .05)
]
fontsize=20

for i,(lam,noise_power) in enumerate(lam_noise):
	# intial state
	x0 = np.array([int(i==arterial_blood_ix)*100 for i in range(n_states)]) #np.zeros(n_states) # 10*np.random.normal(np.zeros(n_states)) + np.array([int(i==arterial_blood_ix)*100 for i in range(n_states)])
	t_start = 0
	t_end = 70
	t_step = .05
	t_span = np.arange(t_start,t_end,t_step)

	# odeint(dynamic func, initial point, time points to evaluate
	trajectory = odeint(mouse.eval_derivative,x0,t_span)

	scale = trajectory.max()


	# # drop t before impulse is gone
	# t_impulse_gone = 1
	# ix_impulse_gone = int((t_impulse_gone-t_start)/t_step)

	# add noise
	noise = np.random.normal(size=trajectory.shape)*np.sqrt(noise_power)
	noisy_traj = trajectory + noise


	X = pd.DataFrame(data=noisy_traj, columns=cols)
	# X = X[ix_impulse_gone:]

	# noise = 1*np.random.normal(size=X.shape)
	# X = X+noise


	dX = X.diff().dropna().reset_index(drop=True)/t_step
	tX = X[:-1].reset_index(drop=True)
	trajectory = trajectory[:-1,:] # trajectory[ix_impulse_gone:-1,:]



	TX = pd.concat([TX,tX], axis=0)
	DX = pd.concat([DX,dX], axis=0)


	print('evaluating dynamics...')
	dXh = SINDy(tX,dX,cols,lam=lam,polyorder=polyorder,usesine=usesine, max_iter=max_iter)

	print(dXh)

	tXmat= tX.values
	t_span=t_span[1:] #t_span[ix_impulse_gone+1:]


	# reconstruct x(t_span)
	if polyorder == 1:
	    # X_s = optimal_linear_trajectory(tXmat,dXh,t_span)
	    X_s = predicted_linear_trajectory(x0,dXh,t_span)
	else:

		# find dynamic_fn: x -> x_dot
		dynamic_fn = lambda x,t: np.squeeze(np.matmul(
		        Xhat_df.values,
		        poolData(pd.DataFrame(x).T,nVars, polyorder, usesine).values.T
		))

		# how to choose x0?
		x0 = tX.iloc[0].values
		X_s = odeint(dynamic_fn,x0,t_span)

	if plot:
		# plot results

		# setup axes for plots
		nc = 3
		axes = []
		nplots = len(cols)

		fig = plt.figure(figsize=(28,20))
		# gs1 = gridspec.GridSpec(1+nplots//nc, nc)
		# gs1.update(wspace=0.025, hspace=0.025)

		for k in range(nplots):
			ax1 = fig.add_subplot(1+nplots//nc,nc,k+1)
			# ax1 = plt.subplot(gs1[k])
			ax1.set_aspect('auto')

			sim_arr = X_s[:,k]
			orig_arr = tXmat[:,k]
			ideal_arr = trajectory[:,k]

			in_snr = SNR(ideal_arr, orig_arr)
			out_snr = SNR(ideal_arr, sim_arr)
			# sim_arr = sim_arr/scale #np.linalg.norm(sim_arr)
			# orig_arr = orig_arr/scale #np.linalg.norm(orig_arr)

			mse = (np.linalg.norm(sim_arr-orig_arr)**2)/len(sim_arr)


			ax1.plot(t_span,sim_arr)
			ax1.plot(t_span,orig_arr)
			ax1.plot(t_span,ideal_arr)
			ax1.set_title('{}; input SNR {:.2f} dB; output SNR {:.2f}'.format(cols[k], in_snr, out_snr),fontsize=fontsize)
			ax1.set_xlabel('Time', fontsize=fontsize)
			ax1.set_ylabel('Intensity',fontsize=fontsize)
			for tick in ax1.xaxis.get_major_ticks() + ax1.yaxis.get_major_ticks():
				tick.label.set_fontsize(fontsize) 
		plt.subplots_adjust(wspace=.5, hspace=.5)
		plt.legend(['reconstructed','noisy', 'original'],loc=9, bbox_to_anchor=(2, .5),fontsize=fontsize+4)
		plt.suptitle('Lambda = ' + '%.1E' % Decimal(lam),fontsize=fontsize)
		# plt.tight_layout()
		
		# plt.show()
		plt.savefig(os.path.join('plots','{}.png'.format(i)))

		# for x in range(len(cols)):
		# 	begin_off = 0

		# 	fulldf = pd.concat([pd.DataFrame(X_s),tX],axis=1)

		# 	f, (ax1) = plt.subplots(1, 1, figsize=(9, 8))
		# 	sim_arr = X_s[begin_off:,x]
		# 	orig_arr = tXmat[begin_off:,x]

		# 	sim_arr = sim_arr/np.linalg.norm(sim_arr)
		# 	orig_arr = orig_arr/np.linalg.norm(orig_arr)

		# 	print(sim_arr.shape,orig_arr.shape,(sim_arr-orig_arr).shape)

		# 	mse = (np.linalg.norm(sim_arr-orig_arr)**2)/len(sim_arr)


		# 	ax1.plot(t_span[begin_off:],sim_arr)
		# 	ax1.plot(t_span[begin_off:],orig_arr)
		# 	ax1.set_title('{}; MSE: {} '.format(cols[x],mse))
		# 	ax1.legend(['reconstructed','original'])
		# 	plt.show()


DX = DX.reset_index(drop=True)
TX = TX.reset_index(drop=True)

dXh = SINDy(TX,DX,cols,lam=lam,polyorder=polyorder,usesine=usesine, max_iter=max_iter)




# plot_traj('Arterial Blood')


# # plot some trajectories

# var_names = [sv.name for sv in mouse.state_vars]

# pairs = itertools.combinations(var_names,2)
# pairs = list(pairs)

# while True:
# 	pair = pairs[np.random.choice(len(pairs))]
# 	v1,v2 = pair
# 	plot_traj(v1)


