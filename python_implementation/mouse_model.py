import numpy as np

class StateVar:

	def __init__(self, name, initial_val=0):
		self.name = name
		self.value = initial_val
		self.ix = None
		self.d_dt = 0
		self.metabolic_fn = lambda x: 0


class OrganModel:

	def __init__(self):
		self.state_vars = []
		self.flows = []
		self.input_fn = None

	def add_var(self, state_var):
		# convert to StateVar obj if string
		if type(state_var) is str:
			state_var = StateVar(state_var)

		# unique naming
		names = [sv.name for sv in self.state_vars]
		if state_var.name in names:
			raise ValueError('Variable named {} already exitsts.')

		self.state_vars.append(state_var)
		state_var.ix = len(self.state_vars)-1

	def get_var(self, name):
		# return StateVar obj by name
		for sv in self.state_vars:
			if sv.name == name:
				return sv
		raise ValueError('No state_var named {}'.format(name))

	def get_val(self,name):
		return get_var(name).value

	def add_flow(self, from_var, to_var, fn):
		flows = [set([f[0],f[1]]) for f in self.flows]
		set_flows = set([from_var, to_var])

		# make sure flow is not defined yet and is proper
		if set_flows in flows:
			raise ValueError('Flow between {} already defined'.format(set_flows))
		elif len(set_flows) == 1:
			raise ValueError('Only one flow var defined')

		# check var names in flow
		for f in set_flows:
			self.get_var(f)

		self.flows.append((from_var, to_var, fn))

	def eval_derivative(self, at_state=None, t=None):

		orig_state = self.get_state()

		self.set_state(at_state)

		# evaluate each flow term and sum them together
		for from_name, to_name, fn in self.flows:
			from_var = self.get_var(from_name)
			to_var = self.get_var(to_name)

			from_var.d_dt = from_var.d_dt - fn(from_var.value)
			to_var.d_dt = to_var.d_dt + fn(from_var.value)

		# evaluate metabolic rate and add to d_dt
		for sv in self.state_vars:
			sv.d_dt += sv.metabolic_fn(sv.value)

		d_dt_vec = np.array([sv.d_dt for sv in self.state_vars])

		if (self.input_fn is not None) or (t is not None):


			if self.input_fn is None:
				raise ValueError('t specified but no input_fn has been defined.')

			d_dt_vec += self.input_fn(t)


		# reset d_dt to 0
		for sv in self.state_vars:
			sv.d_dt = 0

		# reset state
		self.set_state(orig_state)

		return d_dt_vec



	def set_input_fn(self, input_fn):
		# should be vector fn corresponding to states
		self.input_fn = input_fn

	def get_state(self):
		return np.array([[sv.value] for sv in self.state_vars])

	def set_state(self,state):
		# specify state to eval d_dt_vec or just use current state
		if state is not None:

			if len(state) != len(self.state_vars):
				raise ValueError('Size of input vector does not match N states.')

			for i, var in enumerate(self.state_vars):
				var.value = state[i]

	def get_derivative(self,t):
		return np.array([[sv.eval_derivative() for sv in self.state_vars]]) + input_fn(t)



def real_mouse():
	mouse = OrganModel()

	# make state vars
	mouse.add_var('Venous Blood')
	mouse.add_var('Lung')
	mouse.add_var('Other Tissue')
	# mouse.add_var('Fat')
	# mouse.add_var('Bone')
	# mouse.add_var('Brain')
	# mouse.add_var('Heart')
	# mouse.add_var('Muscle')
	# mouse.add_var('Skin')
	mouse.add_var('Liver')
	mouse.add_var('Kidney')
	mouse.add_var('Gut')
	mouse.add_var('Spleen')
	mouse.add_var('Arterial Blood')

	# define all flows from https://www.sciencedirect.com/science/article/pii/S221138351630082X


	d_art = .1
	d_ven = .1
	d_lung = .3
	d_liv = .1
	metab = -.05

	# venous flows
	mouse.add_flow('Venous Blood', 'Lung', lambda x: d_lung*x)
	mouse.add_flow('Other Tissue', 'Venous Blood', lambda x:d_ven*x)
	# mouse.add_flow('Fat', 'Venous Blood', lambda x: 1.0*x)
	# mouse.add_flow('Bone', 'Venous Blood', lambda x: 1.0*x)
	# mouse.add_flow('Brain', 'Venous Blood', lambda x: 1.0*x)
	# mouse.add_flow('Heart', 'Venous Blood', lambda x: 1.0*x)
	# mouse.add_flow('Muscle', 'Venous Blood', lambda x: 1.0*x)
	# mouse.add_flow('Skin', 'Venous Blood', lambda x: 1.0*x)
	mouse.add_flow('Liver', 'Venous Blood', lambda x: d_ven*x)
	mouse.add_flow('Kidney', 'Venous Blood', lambda x: d_ven*x)

	# arterial flows
	mouse.add_flow('Lung', 'Arterial Blood', lambda x: .01*d_lung*x)
	mouse.add_flow('Arterial Blood', 'Other Tissue', lambda x:d_art*x)
	# mouse.add_flow('Arterial Blood', 'Fat', lambda x: 1.0*x)
	# mouse.add_flow('Arterial Blood', 'Bone', lambda x: 1.0*x)
	# mouse.add_flow('Arterial Blood', 'Brain', lambda x: 1.0*x)
	# mouse.add_flow('Arterial Blood', 'Heart', lambda x: 1.0*x)
	# mouse.add_flow('Arterial Blood', 'Muscle', lambda x: 1.0*x)
	# mouse.add_flow('Arterial Blood', 'Skin', lambda x: 1.0*x)
	mouse.add_flow('Arterial Blood', 'Gut', lambda x: d_art*x)
	mouse.add_flow('Arterial Blood', 'Liver', lambda x: d_art*x)
	mouse.add_flow('Arterial Blood', 'Spleen', lambda x: d_art*x)
	mouse.add_flow('Arterial Blood', 'Kidney', lambda x: d_art*x)

	# other flows
	mouse.add_flow('Gut', 'Liver', lambda x: d_liv*x)
	mouse.add_flow('Spleen', 'Liver', lambda x: d_liv*x)

	# define metabolic rates
	mouse.get_var('Liver').metabolic_fn = lambda x: metab*x
	mouse.get_var('Kidney').metabolic_fn = lambda x: metab*x


	return mouse

def simple_mouse():

	mouse = OrganModel()

	# make state vars
	mouse.add_var('Venous Blood')
	mouse.add_var('Lung')
	mouse.add_var('Other Tissue')
	mouse.add_var('Kidney')
	mouse.add_var('Spleen')
	mouse.add_var('Arterial Blood')

	# define all flows from https://www.sciencedirect.com/science/article/pii/S221138351630082X


	d_art = .1
	d_ven = .1

	# venous flows
	mouse.add_flow('Venous Blood', 'Lung', lambda x: 1.0*x)
	mouse.add_flow('Other Tissue', 'Venous Blood', lambda x:d_ven*x)
	mouse.add_flow('Kidney', 'Venous Blood', lambda x: d_ven*x)

	# arterial flows
	mouse.add_flow('Lung', 'Arterial Blood', lambda x: 1.0*x)
	mouse.add_flow('Arterial Blood', 'Other Tissue', lambda x:d_art*x)
	mouse.add_flow('Arterial Blood', 'Spleen', lambda x: d_art*x)
	mouse.add_flow('Arterial Blood', 'Kidney', lambda x: d_art*x)

	mouse.get_var('Kidney').metabolic_fn = lambda x: -0.5*x


	return mouse



