import numpy as np
import scipy

class SingleStraightDislocation:
    def __init__(self, nett_b, line_dir, core_mean_loc,
                 core_smearing_parameters={"dist_type": "delta",
                                           "dist_params": None,
                                           "core_spread_dir": None,
                                           "discretization_points": None}):
        self._nett_b = nett_b
        self._line_dir = line_dir
        self._core_mean_loc = core_mean_loc
        self._core_smearing_parameters = core_smearing_parameters
        if not np.isclose(np.dot(self._core_smearing_parameters["core_spread_dir"],
                                 self._line_dir), 0.0):
            raise ValueError("Core smearing direction must be perpendicular to the "+
                             "dislocation line direction")
        self._generate_discretized_smeared_out_dislocations()

    @property
    def nett_b(self):
        return self._nett_b

    @property
    def line_dir(self):
        return self._line_dir

    @property
    def core_mean_loc(self):
        return self._core_mean_loc

    @property
    def core_smearing_parameters(self):
        return self._core_smearing_parameters

    def _generate_discretized_smeared_out_dislocations(self):
        # Discretized slip density for parameterized smeared out dislocation core
        if self.core_smearing_parameters['dist_type'] == 'gaussian':
            sigma = self.core_smearing_parameters["dist_params"]['sigma']
            dist_func = lambda x: 1.0 / np.sqrt(2.0*np.pi) / sigma * np.exp(-x**2.0/2.0/sigma**2.0)
        elif self.core_smearing_parameters['dist_type'] == 'lorentzian':
            zeta = self.core_smearing_parameters["dist_params"]['zeta']
            dist_func = lambda x: zeta/np.pi/(x**2+zeta**2)
        elif self.core_smearing_parameters['dist_type'] == 'lorentzian-gaussian_mixture':
            p_lorentz = self.core_smearing_parameters["dist_params"]['p_lorentz'] # proportion of Lorentzian
            if not (p_lorentz >= 0.0 and p_lorentz <= 1.0):
                raise ValueError('Proportion of Lorentzian density must be a fraction in [0,1]')
            zeta = self.core_smearing_parameters["dist_params"]['zeta'] # Lorentzian scale parameter (inverse length)
            sigma = self.core_smearing_parameters["dist_params"]['sigma'] # Gaussian scale parameter (in length units)
            dist_func = lambda x: ( ( p_lorentz * ( zeta/np.pi/(x**2+zeta**2) ) ) + 
                                    ( (1-p_lorentz) * ( 1.0 / np.sqrt(2.0*np.pi) / sigma * np.exp(-x**2.0/2.0/sigma**2.0) ) ) )
        elif self.core_smearing_parameters['dist_type'] == 'logistic':
            t = self.core_smearing_parameters["dist_params"]['t']
            dist_func = lambda x: 1.0/t/(np.exp(0.5*x/t)+np.exp(-0.5*x/t))**2
        elif self.core_smearing_parameters['dist_type'] == 'user-defined':
            dist_func = self.core_smearing_parameters["dist_params"]['dist_func']
            if not callable(dist_func):
                raise ValueError("If dist_type is 'user-defined' then ['dist_params']['dist_func'] " +
                                "must be a callable object")         

        if self.core_smearing_parameters['dist_type'] == 'delta':
            dist_func_values = np.array([1.0])
        elif self.core_smearing_parameters['dist_type'] == 'discrete':
            dist_func_values = self.core_smearing_parameters["dist_params"]['func_values']
            discrete_pts = self.core_smearing_parameters['discretization_points']
            if np.shape(discrete_pts) != np.shape(dist_func_values):
                raise ValueError("If dist_type is discrete, then params['func_values'] must be " + 
                                "of same shape as key 'discretization_points'")
            if not np.isclose(scipy.integrate.simps(dist_func_values, discrete_pts), 1.0):
                raise RuntimeError('The dist_func does not integrate to 1.0')
            dist_mean = scipy.integrate.simps(discrete_pts*dist_func_values, discrete_pts)
        else:
            if not np.isclose(scipy.integrate.quad(dist_func, -np.inf, np.inf)[0], 1.0):
                raise RuntimeError('The dist_func does not integrate to 1.0')
            discrete_pts = self.core_smearing_parameters['discretization_points']
            dist_mean = scipy.integrate.quad(lambda x: x*dist_func(x), -np.inf, np.inf)[0]    
            dist_func_array = np.frompyfunc(dist_func, 1, 1)  
            dist_func_values = dist_func_array(discrete_pts).astype(float)

        discretized_b = self.nett_b*dist_func_values[:,None]
        if self.core_smearing_parameters['dist_type'] != 'delta':
            sel_discretization = np.invert(np.all(np.isclose(discretized_b, 0.0), axis = 1))
            discrete_pts = discrete_pts[sel_discretization]
            discretized_b = discretized_b[sel_discretization]
            chosen_b_inds = np.nonzero(sel_discretization)[0]

        if self.core_smearing_parameters['dist_type'] == 'delta':
            discrete_b_pos = np.array([self.core_mean_loc])
        else:
            core_spread_dir = self.core_smearing_parameters['core_spread_dir']
            core_spread_dir /= np.linalg.norm(core_spread_dir)
            discrete_b_pos = self.core_mean_loc + ((discrete_pts-dist_mean)[:, None]*core_spread_dir)
        dtype = np.dtype([
            ("discrete_b_pos", float, (3,)),
            ("discrete_b_vec", float, (3,))
        ])
        self._discretized_dislocation = np.array(list(zip(discrete_b_pos, discretized_b)), 
                                                dtype=dtype)
        #detection_pts_translated = detection_points - discrete_b_pos[:, np.newaxis, :]
        #if np.any(np.all(np.isclose(detection_pts_translated, 0.0), axis = 2)):
        #    raise RuntimeError('Detection point cannot be at the origin')

    @property
    def discretized_dislocation(self):
        return self._discretized_dislocation

    def compute_displacements(self, media, detection_points, 
                              cut_plane = None):
        
        detection_pts_translated = detection_points - discrete_b_pos[:, np.newaxis, :]
        if np.any(np.all(np.isclose(detection_pts_translated, 0.0), axis = 2)):
            raise RuntimeError('Detection point cannot be at the origin')
        

    def compute_stress_n_strain(self, media, detection_points):
        pass

class PeriodicArrayStraightDislocations:
    def compute_displacements(self, media, detection_points, 
                              cut_planes=None):
        pass

    def compute_stress_n_strain(self, media, detection_points):
        pass
