import numpy as np
import scipy
from .dislocation_fields import isotropic_soln

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

    def compute_dislocation_fields(self, mu, nu, detection_points, 
                                   m0=None, n0=None):
        
        detection_pts_translated = (detection_points - 
                                    self.discretized_dislocation["discrete_b_pos"][:,None,:])
        if np.any(np.all(np.isclose(detection_pts_translated, 0.0), axis = 2)):
            raise RuntimeError('Detection point(s) coincides with dislocation core')
        
        u = np.zeros((len(detection_points), 3))
        stress = np.zeros((len(detection_points), 6))

        for i, b in enumerate(self.discretized_dislocation["discrete_b_vec"]):
            fields = isotropic_soln(mu, nu, self.line_dir, b, 
                                    detection_pts_translated[i], m0, n0)
            u += fields[0]; stress += fields[1]

        return u, stress



class PeriodicArrayStraightDislocations:

    def __init__(self, straight_disl, lattice_primitive_vectors, 
                 lattice_origin_loc):
        self._straight_disl = straight_disl
        self._lattice_primitive_vectors = lattice_primitive_vectors
        #### USING SETATTR MAKE self._lattice_primitive_vectors a 2D ARRAY EVEN WHEN THERE IS ONLY ONE VECTOR
        self._lattice_origin_loc = lattice_origin_loc
        line_dir = self._straight_disl.line_dir
        if not np.allclose(np.dot(self._lattice_primitive_vectors, 
                                    line_dir), 0):
            raise ValueError("The periodic dislocation array lattice must be perpendicular " +
                             "to the dislocation line direction")


    @property
    def straight_disl(self):
        return self._straight_disl
    
    @property
    def lattice_primitive_vectors(self):
        return self._lattice_primitive_vectors
    
    @property
    def lattice_origin_loc(self):
        return self._lattice_origin_loc

    def compute_dislocation_fields(self, mu, nu, detection_points, 
                                   lattice_extents,
                                   m0=None, n0=None):
        
        if len(lattice_extents) != len(self.lattice_primitive_vectors):
            raise ValueError("Lattice extents must be provided for each "+
                             "lattice primitive vectors")
        if len(lattice_extents) == 1:
            X = np.arange(*lattice_extents)
            dislocation_positions = ((X[:,None]*self.lattice_primitive_vectors) + 
                                     self.lattice_origin_loc)
        elif len(lattice_extents) == 2:
            X,Y = np.meshgrid(*[np.arange(*x) for x in lattice_extents])
            dislocation_positions = (np.dot(np.c_[X.ravel(), Y.ravel()],
                                            self.lattice_primitive_vectors) + 
                                     self.lattice_origin_loc)
            
        nett_b = self.straight_disl.nett_b
        line_dir = self.straight_disl.line_dir
        core_smearing_parameters = self.straight_disl.core_smearing_parameters
        #SPECIFY m0, n0 restrictions more clearly
        if m0 is None:
            m0 = [m0]*len(dislocation_positions)
        if n0 is None:
            n0 = [n0]*len(dislocation_positions)

        u = np.zeros((len(detection_points), 3))
        stress = np.zeros((len(detection_points), 6))
        for i, disl_pos in enumerate(dislocation_positions):
            str_disl = SingleStraightDislocation(nett_b, line_dir, disl_pos,
                                                 core_smearing_parameters)
            fields = str_disl.compute_dislocation_fields(
                                                mu, nu, detection_points, 
                                                m0[i], n0[i])
            u += fields[0]; stress += fields[1]

        return u, stress
            
        
        

