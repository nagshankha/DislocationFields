import numpy as np
import voigt_fourth_rank_conversion_stiffness_matrix as stiff_conv
import scipy
from scipy.integrate import quad
import sympy
from sympy.utilities.autowrap import ufuncify

##########################  STROH FORMALISM  #################################
##########################      (BEGIN)      #################################

def stroh_params1(C, m, n):

   # If C is in voigt indices then converting it
   # full 3*3*3*3 stiffness tensor
   if np.shape(C) == (6,6):
      C = stiff_conv.C_voigt_to_fourth_rank(C)
   elif np.shape(C) == (3,3,3,3):
      stiff_conv.check_fourth(C)
   else:
      raise RuntimeError('Stiffness matrix C must either be 6*6 in' + 
                         ' voigt notation or full 3*3*3*3 stiffness tensor')


   # Creating operator matrix N for the eigen operation
   upper_left = -np.dot(np.linalg.inv(op(n,C,n)), op(n,C,m))
   upper_right = -np.linalg.inv(op(n,C,n))
   lower_left = -(np.dot(np.dot(op(m,C,n), np.linalg.inv(op(n,C,n))), op(n,C,m)) - op(m,C,m))
   lower_right = -np.dot(op(m,C,n), np.linalg.inv(op(n,C,n))) # The minus sign is missing in Hirth and Lothe

   # upper_right and lower_left matrices must be symmetric
   if not np.all(np.isclose(upper_right, upper_right.T)):
      raise RuntimeError('upper_right matrix must be symmetric')
   if not np.all(np.isclose(lower_left, lower_left.T)):
      raise RuntimeError('lower_left matrix must be symmetric')

   # upper_left and lower_right matrices must be transpose of each other
   if not np.all(np.isclose(lower_right, upper_left.T)):
      raise RuntimeError('upper_left and lower_right matrices must be transpose of each other')

   """
   # upper_right matrix must be positive definite
   if not all(np.linalg.eigvals(upper_right) > np.finfo(float).eps):
      raise RuntimeError('upper_right matrix must be positive definite. Eigen values = {0}'.format(np.linalg.eigvals(upper_right)))
   # -lower_left matrix must be positive semidefinite
   if not all(np.linalg.eigvals(-lower_left) > -np.finfo(float).eps):
      raise RuntimeError('-lower_left matrix must be positive semi-definite. Eigen values = {0}'.format(np.linalg.eigvals(-lower_left)))
   """

   N = np.vstack(( np.hstack(( upper_left, upper_right )), np.hstack(( lower_left, lower_right )) ))  
   #print 'N = ', N
   #N2 = np.vstack(( np.hstack(( upper_left, -upper_right )), np.hstack(( -lower_left, lower_right )) ))   
   
   # Solving the eigen value problem for eigen values p
   # and corresponding eigen vectors zeta
   
   p, zeta = np.linalg.eig(N)
   """
   p2, zeta2 = np.linalg.eig(N2)
   print p, p2
   real_zeta = np.real(zeta)
   imag_zeta = np.imag(zeta)
   real_zeta[np.isclose(real_zeta, 0)] = 0
   imag_zeta[np.isclose(imag_zeta, 0)] = 0
   real_zeta2 = np.real(zeta2)
   imag_zeta2 = np.imag(zeta2)
   real_zeta2[np.isclose(real_zeta2, 0)] = 0
   imag_zeta2[np.isclose(imag_zeta2, 0)] = 0
   np.savetxt('zeta1', real_zeta + (imag_zeta*1j), fmt='%.5e')
   np.savetxt('zeta2', real_zeta2 + (imag_zeta2*1j), fmt='%.5e')

   """

   ##### Checking the correctness of the solution ##########

   real_p = np.real(p)
   imag_p = np.imag(p)   
   
   #Checking whether all p(s) are complex with nonzero imaginary parts
   if any(np.isclose(imag_p, 0.0)):
      raise RuntimeError('All the p(s) must have nonzero imaginary parts')


   #Checking whether the p(s) are in conjugate pairs
   unique_real, unique_real_inverse, unique_real_counts = np.unique(np.round(real_p, decimals = 10 ), 
                                                                    return_inverse = True, return_counts = True)
   
   if all(unique_real_counts == np.array([3, 3, 3])):
      print('p = {0}'.format(p))
      raise RuntimeError('Roots of p are not in 3 pairs of complex conjugate')
   else:
      imag_p2 = np.array([imag_p[unique_real_inverse == x] for x in range(3)])
      #print imag_p2
      if not all(np.isclose(imag_p2[:,0], -imag_p2[:,1])):
         print('p = {0}'.format(p))
         raise RuntimeError('Roots of p are not in 3 pairs of complex conjugate')
   
   for i in range(6):
      
      #Checking Eq. 13.167 of Hirth and Lothe
      if not all(np.isclose(zeta[3:,i], -np.dot(op(n,C,m) + (p[i] * op(n,C,n)), zeta[:3,i]), rtol = 0)):
         raise RuntimeError('Eq. 13.167 of Hirth and Lothe not satisfied. ' + 
                            'LHS = {0} and RHS = {1}'.format(zeta[3:,i], -np.dot(op(n,C,m) + (p[i] * op(n,C,n)), zeta[:3,i])))
      #Checking Eq. 13.170 of Hirth and Lothe
      elif not np.isclose(np.linalg.det(N - (p[i] * np.identity(6))), 0):
         raise RuntimeError('Eq. 13.170 of Hirth and Lothe not satisfied. The det is {0}'.format(np.linalg.det(N - (p[i] * np.identity(6)))))


   # Normalize A and L, such that 2*A*L=1 (equation 13-178)
   norm = 2.0 * np.sum(zeta[:3] * zeta[3:], axis = 0)
   zeta /= np.sqrt(norm)
   

   return p, zeta

#########################################################################

def stroh_params2(p, zeta, b):

   D = np.sign(np.imag(p)) * np.sum(zeta[3:, :].T * b, axis = 1)
   A = zeta[:3, :]

   return D, A

#########################################################################

def disp_strain_stress_stroh(D, A, C, m, n, p, x):

   # If C is in the form of full 3*3*3*3 stiffness tensor
   # then converting it to 6*6 voigt contracted notation
   if np.shape(C) == (3,3,3,3):
      C = stiff_conv.C_fourth_rank_to_voigt(C)
   elif np.shape(C) == (6,6):
      stiff_conv.check_voigt(C)
   else:
      raise RuntimeError('Stiffness matrix C must either be 6*6 in' + 
                         ' voigt notation or full 3*3*3*3 stiffness tensor')

   if any(np.all(np.isclose(np.column_stack(( np.sum(m*x, axis = 1), np.sum(n*x, axis = 1) )), 0), axis = 1)):
      raise RuntimeError('Singularity at dislocation core !!!') 

   if np.shape(p) != (6,):
      raise RuntimeError('p should have 6 element')

   eta = np.tile(np.sum(m*x, axis = 1), (6,1)).T + p * np.tile(np.sum(n*x, axis = 1), (6,1)).T
   u = 1 / (2*np.pi*1j) * np.sum(A * D * np.log(eta)[:, np.newaxis, :], axis = 2)
   inter_mat1 = np.outer(n,p) + np.tile(m, (6,1)).T
   inter_mat2 = inter_mat1.T[:,:,None] * A.T[:, None]
   deriv_u = 1 / (2*np.pi*1j) * np.sum(inter_mat2 * D[:, np.newaxis, np.newaxis] / eta[:, :, np.newaxis, np.newaxis], axis = 1)
   strain = 0.5 * (deriv_u + np.einsum('ikj', deriv_u))
   strain = strain[:, np.array([0,1,2,1,0,0]), np.array([0,1,2,2,2,1])]
   strain[:,3:] *= 2
   
   if not np.all(np.isclose(np.imag(u), 0)):
      raise RuntimeError('Displacements have nonzero imaginary parts')
   if not np.all(np.isclose(np.imag(strain), 0)):
      raise RuntimeError('Strains have nonzero imaginary parts')

   u = np.real(u)
   strain = np.real(strain)
   stress = np.dot(C, strain.T).T

   return u , strain, stress

##########################  STROH FORMALISM  #################################
##########################      (END)        #################################


##########################  INTEGRAL METHOD  #################################
##########################      (BEGIN)      #################################

def compute_integrals(C, line_dir, m0 = None, n0 = None):

   # If C is in voigt indices then converting it
   # full 3*3*3*3 stiffness tensor
   if np.shape(C) == (6,6):
      C = stiff_conv.C_voigt_to_fourth_rank(C)
   elif np.shape(C) == (3,3,3,3):
      stiff_conv.check_fourth(C)
   else:
      raise RuntimeError('Stiffness matrix C must either be 6*6 in' +
                         ' voigt notation or full 3*3*3*3 stiffness tensor')

   if np.all(m0 == None) or np.all(n0 == None):
      m0, n0 = find_m_n(line_dir)
   else:
      m0 = m0 / np.linalg.norm(m0)
      n0 = n0 / np.linalg.norm(n0)
      if not all(np.isclose(np.cross(m0,n0), line_dir / np.linalg.norm(line_dir))):
         raise RuntimeError('m cross n is not directed toward the dislocation line direction')


   theta = sympy.Symbol('theta')
   m = (sympy.cos(theta)*m0) + (sympy.sin(theta)*n0)
   n = (-sympy.sin(theta)*m0) + (sympy.cos(theta)*n0)

   S_integrand = np.dot(np.array(sympy.Matrix(op(n, C, n)).inv()), op(n, C, m) )
   Q_integrand = np.array(sympy.Matrix(op(n, C, n)).inv())
   B_integrand = np.dot(np.dot(op(m, C, n), np.array(sympy.Matrix(op(n, C, n)).inv()) ),
                        op(n, C, m)) - op(m, C, m)

   # Create ufunc of sympy...unfuncify
   ufunc_ufuncify = np.frompyfunc(sympy.utilities.autowrap.ufuncify, 2, 1)

   S_integrand_func_array = ufunc_ufuncify([theta], S_integrand)
   Q_integrand_func_array = ufunc_ufuncify([theta], Q_integrand)
   B_integrand_func_array = ufunc_ufuncify([theta], B_integrand)

   # Create ufunc of quad to integrate from 0 to 2pi
   ufunc_contour_int = np.frompyfunc(lambda x: scipy.integrate.quad(x, 0.0, 2*np.pi)[0], 1, 1)
   ufunc_contour_int_err = np.frompyfunc(lambda x: scipy.integrate.quad(x, 0.0, 2*np.pi)[1], 1, 1) 
   
   S_integrand_eval = ufunc_contour_int(S_integrand_func_array).astype(float)
   Q_integrand_eval = ufunc_contour_int(Q_integrand_func_array).astype(float)
   B_integrand_eval = ufunc_contour_int(B_integrand_func_array).astype(float)

   S_integrand_eval_err = ufunc_contour_int_err(S_integrand_func_array).astype(float)
   Q_integrand_eval_err = ufunc_contour_int_err(Q_integrand_func_array).astype(float)
   B_integrand_eval_err = ufunc_contour_int_err(B_integrand_func_array).astype(float)

   S = - 1 / (2 * np.pi) * S_integrand_eval
   Q = - 1 / (2 * np.pi) * Q_integrand_eval
   B = - 1 / (2 * np.pi) * B_integrand_eval

   print(S)
   print(Q)
   print(B)
   exit()
   
   return S, Q, B

##############################################################################

def disp_strain_stress_integral(C, S, Q, B, b, x, mx, nx, m0, n0, line_dir):

   # If C is in the form of full 3*3*3*3 stiffness tensor
   # then converting it to 6*6 voigt contracted notation (C_v)
   if np.shape(C) == (3,3,3,3):
      C_v = stiff_conv.C_fourth_rank_to_voigt(C)
   elif np.shape(C) == (6,6):
      C_v = C
      C = stiff_conv.C_voigt_to_fourth_rank(C_v)
   else:
      raise RuntimeError('Stiffness matrix C must either be 6*6 in' +
                         ' voigt notation or full 3*3*3*3 stiffness tensor')

   if np.shape(mx) != np.shape(x):
      raise RuntimeError('mx must be of the same shape as x')
   elif np.shape(nx) != np.shape(x):
      raise RuntimeError('nx must be of the same shape as x')   

   if not all(np.isclose(np.linalg.norm(mx, axis = 1), 1.0)):
      raise RuntimeError('mx are not unit vectors')
   elif not all(np.isclose(np.linalg.norm(nx, axis = 1), 1.0)):
      raise RuntimeError('nx are not unit vectors')
   elif not np.isclose(np.linalg.norm(m0), 1.0):
      raise RuntimeError('m0 is not a unit vector')
   elif not np.isclose(np.linalg.norm(n0), 1.0):
      raise RuntimeError('n0 is not a unit vector')

   r = np.sum(mx * x, axis = 1)
   
   if any(r < 0.0):
      raise RuntimeError('r must not be negative !!!')

   angle = angular_distance_of_rotated_m(mx, m0, line_dir) 
   theta = sympy.Symbol('theta')
   m = (sympy.cos(theta)*m0) + (sympy.sin(theta)*n0)
   n = (-sympy.sin(theta)*m0) + (sympy.cos(theta)*n0)

   u_integrand1 = np.array(sympy.Matrix(op(n, C, n)).inv())
   u_integrand2 = np.dot(np.array(sympy.Matrix(op(n, C, n)).inv()), op(n, C, m))

   # Create ufunc of sympy...unfuncify
   ufunc_ufuncify = np.frompyfunc(sympy.utilities.autowrap.ufuncify, 2, 1)

   u_integrand1_func_array = ufunc_ufuncify([theta], u_integrand1)
   u_integrand2_func_array = ufunc_ufuncify([theta], u_integrand2)

   # Create ufunc of quad to integrate from 0 to 2pi
   ufunc_angular_int = np.frompyfunc(lambda x, y: scipy.integrate.quad(x, 0.0, y)[0], 2, 1)
   ufunc_angular_int_err = np.frompyfunc(lambda x, y: scipy.integrate.quad(x, 0.0, y)[1], 2, 1)

   u_integrand1_eval = ufunc_angular_int(u_integrand1_func_array, angle[:, np.newaxis, np.newaxis]).astype(float)
   u_integrand2_eval = ufunc_angular_int(u_integrand2_func_array, angle[:, np.newaxis, np.newaxis]).astype(float)

   u = np.einsum('j,aij->ai', b/(2*np.pi), -S*np.log(r[:,np.newaxis,np.newaxis]) +
                         np.einsum('aij,jk->aik', u_integrand1_eval, B) +
                         np.einsum('aij,jk->aik', u_integrand2_eval, S) )

   deriv_u = np.einsum('ak,aijk->aij', 
             b / (2*np.pi*r[:,np.newaxis]), 
             -np.einsum('ai,jk', mx, S) +
             np.einsum('ai,ajk,kl->aijl', nx, np.linalg.inv(np.einsum('ai, ijkl, al->ajk', nx, C, nx)), B) +
             np.einsum('ai,ajk,akl,lr->aijr', nx, np.linalg.inv(np.einsum('ai, ijkl, al->ajk', nx, C, nx)), 
                       np.einsum('ai, ijkl, al->ajk', nx, C, mx), S) ) 

   strain = 0.5 * (deriv_u + np.einsum('ikj', deriv_u))
   strain = strain[:, np.array([0,1,2,1,0,0]), np.array([0,1,2,2,2,1])]
   strain[:,3:] *= 2
   stress = np.dot(C_v, strain.T).T

   return u, strain, stress

##############################################################################

def choose_m_n_dir_x(line_dir, x):

   m = np.cross(line_dir, np.cross(line_dir, x, axisb = 1), axisb = 1)
   m = (m.T / np.linalg.norm(m, axis = 1)).T
   m[np.sum(m*x, axis = 1) < 0] *= -1 
   n = np.cross(line_dir/np.linalg.norm(line_dir), m, axisb = 1)

   return m, n

##############################################################################

def angular_distance_of_rotated_m(m, m0, line_dir):

   if any(np.logical_or(np.all(np.isclose(m,m0), axis=1), np.all(np.isclose(m,m0), axis=1))):
      raise RuntimeError('There must be no points of detection on the cut plane. '+
                         'The entire plane is not the cut plane; yet the points on the '+
                         'entire plane is forbidden for convenience sake. This mean while '+
                         'applying the field to atomistic care must be taken that the cut '+
                         'is always between two planes of atoms.' )
   line_dir = line_dir / np.linalg.norm(line_dir)
   angle = np.arccos(np.sum(m*m0, axis = 1))
   cross_prod = np.cross(m0, m, axisb = 1)
   cross_prod_dir = (cross_prod.T/np.linalg.norm(cross_prod, axis = 1)).T
   if np.any(np.invert(np.logical_or(np.all(np.isclose(cross_prod_dir, line_dir), axis = 1), 
                                     np.all(np.isclose(cross_prod_dir, -line_dir), axis = 1)))):
      raise RuntimeError('Either m or m0 is not perpendicular to line_dir which is weird.')
   sel = np.invert(np.all(np.isclose(cross_prod_dir, line_dir), axis = 1))
   angle[sel] *= -1
   angle[sel] += 2*np.pi

   return angle

##############################################################################

##########################  INTEGRAL METHOD  #################################
##########################       (END)       #################################


def op(a,C,b):
   
   #return np.einsum('i,ijkl,l', a, C, b)
   # Alternative compatible with symbolic math
   return np.dot(b, np.einsum('ikj', np.dot(a, np.einsum('jkil', C)))).T

##########################################################################

def find_m_n(line_dir):

   if np.shape(line_dir) != (3,):
      raise RuntimeError('Line direction must be a three element vector')

   if all(np.isclose(line_dir, 0)):
      raise RuntimeError('Line direction cannot be a null vector')

   line_dir = line_dir.astype(float)

   m = np.zeros(3, dtype = float)

   if any(np.isclose(line_dir, 0)):
      if np.count_nonzero(np.isclose(line_dir, 0)) == 2:
         m[np.isclose(line_dir, 0)] = np.random.randint(low = 1, high = 10, size = (2,))
      else:
         m[np.isclose(line_dir, 0)] = np.random.randint(low = 1, high = 10)
         m[np.nonzero(np.invert(np.isclose(line_dir, 0)))[0][0]] = np.random.randint(low = 1, high = 10)
         m[m == 0] = -np.dot(m, line_dir) / line_dir[m == 0]
   else:
      m[:2] = np.random.randint(low = 1, high = 10, size = (2,))
      m[2] = -np.dot(m, line_dir) / line_dir[2]

   # Small check
   if not np.isclose(np.dot(line_dir, m), 0):
      raise RuntimeError('The line_direction and m are not in right angle')

   n = np.cross(line_dir, m)

   # Small check
   if not all(np.isclose(np.cross(m,n) / np.linalg.norm(np.cross(m,n)), 
                         line_dir / np.linalg.norm(line_dir))):
      raise RuntimeError('m cross n is not directed toward the dislocation line direction')

   m = m / np.linalg.norm(m)
   n = n / np.linalg.norm(n)

   return m, n

###################################################################################

def anisotropic_soln(C, line_dir, b, x, m0 = None, n0 = None, m0_int = None, n0_int = None, method = 'integral'):

   if not ((np.shape(C) == (3,3,3,3)) or (np.shape(C) == (6,6))):
      raise RuntimeError('Stiffness matrix must either be full 3*3*3*3 tensor or 6*6 voigt contracted matrix')

   if np.shape(C) == (6,6):
      stiff_conv.check_voigt(C)
   else:
      stiff_conv.check_fourth(C)

   if np.shape(b) != (3,):
      raise RuntimeError('Burger vector b must be a three element vector')

   if len(np.shape(x)) != 2:
      raise RuntimeError('shape of x must be 2 even if it just has one point')
   elif np.shape(x)[0] == 0:
      raise RuntimeError('x must have at least one point')
   elif np.shape(x)[1] != 3:
      raise RuntimeError('points in x must have three elements which are positions components in 3D')

   if np.all(m0 == None) or np.all(n0 == None):
      m0, n0 = find_m_n(line_dir)
   else:
      m0 = m0 / np.linalg.norm(m0)
      n0 = n0 / np.linalg.norm(n0)
      if not all(np.isclose(np.cross(m0,n0), line_dir / np.linalg.norm(line_dir))):
         raise RuntimeError('m cross n is not directed toward the dislocation line direction')

   if method == 'integral':
      S, Q, B = compute_integrals(C, line_dir, m0_int, n0_int)
      mx, nx = choose_m_n_dir_x(line_dir, x)
      u, strain, stress = disp_strain_stress_integral(C, S, Q, B, b, x, mx, nx, m0, n0, line_dir)
   elif method == 'stroh':
      p, zeta = stroh_params1(C, m0, n0)
      D, A = stroh_params2(p, zeta, b)
      u, strain, stress = disp_strain_stress_stroh(D, A, C, m0, n0, p, x)
   else:
      raise RuntimeError('"method" must be either "integral" or "stroh"')

   return u, strain, stress


##################################################################################

###################################################################################
####################### Isotropic solution ########################################
###################################################################################

def isotropic_soln(mu, nu, line_dir, b, x, m0=None, n0=None):

   if np.shape(b) != (3,):
      raise RuntimeError('Burger vector b must be a three element vector')

   if len(np.shape(x)) != 2:
      raise RuntimeError('shape of x must be 2 even if it just has one point')
   elif np.shape(x)[0] == 0:
      raise RuntimeError('x must have at least one point')
   elif np.shape(x)[1] != 3:
      raise RuntimeError('points in x must have three elements which are positions components in 3D')

   
   line_dir = line_dir / np.linalg.norm(line_dir)

   bz = np.dot(b, line_dir)
   
   if np.isclose(np.linalg.norm(b), abs(bz)):
      if m0 is None and n0 is None:
         m0, n0 = find_m_n(line_dir)
      elif any([m0 is None, n0 is None]):
         raise ValueError('Either both m0 and n0 are None or neither of them are None')
      else:
         if np.shape(m0) != (3,) or np.shape(m0) != (3,):
            raise RuntimeError('Both m0 and n0 must be three element vectors')
         n0 = np.array(n0) / np.linalg.norm(n0)
         m0 = np.array(m0) / np.linalg.norm(m0)
         if not np.allclose(np.cross(m0, n0), line_dir):
            raise RuntimeError('m cross n is not directed toward the dislocation line direction')         
      bx = 0.0
   else:
      n0 = np.cross(line_dir, b)
      n0 = n0 / np.linalg.norm(n0)
      m0 = np.cross(n0, line_dir)
      m0 = m0 / np.linalg.norm(m0)
      bx = np.dot(b, m0)
      
   if (bx < 0 and (not np.isclose(bx, 0.0))):
      print(f'bz = {bz}, bx = {bx}, b = {b}, \n m0 = {m0}, n0 = {n0}, line_dir = {line_dir}')
      raise RuntimeError('Something is weird since m0 must direct towards the edge component of b')
   if not np.allclose(np.cross(m0,n0), line_dir):
      print(f'm0 = {m0}, n0 = {n0}, line_dir = {line_dir}')
      raise RuntimeError('m0 cross n0 is not directed toward the dislocation line direction')
   if not np.isclose(np.sqrt(bx**2 + bz**2), np.linalg.norm(b)):
      print(f'bz = {bz}, bx = {bx}, b = {b}')
      raise RuntimeError('norm of b is not the same after rotation ---> Weird!!!')
      
   #print(f'bz = {bz}, bx = {bx}, b = {b}, \n m0 = {m0}, n0 = {n0}, line_dir = {line_dir}')

   rot_mat = np.dot(np.vstack((m0,n0,line_dir)), np.array([[1,0,0],[0,1,0],[0,0,1]]))
   x_rot = np.dot(rot_mat, x.T).T


   #if not all(np.isclose(np.array([bx, 0.0, bz]), b)):
   #   raise RuntimeError('b rotation problem')
   #if not np.all(np.isclose(x, x_rot)):
   #   raise RuntimeError('x rotation error')

   u = np.zeros(np.shape(x_rot))
   stress = np.zeros((np.shape(x_rot)[0],6))
   # For screw component
   u[:,2] = bz*np.arctan2(x_rot[:,1],x_rot[:,0])/2.0/np.pi
   stress[:,3] = mu*bz/2/np.pi * x_rot[:,0]/(x_rot[:,0]**2 + x_rot[:,1]**2)
   stress[:,4] = -mu*bz/2/np.pi * x_rot[:,1]/(x_rot[:,0]**2 + x_rot[:,1]**2)

   # For edge component
   u[:,0] = bx/2.0/np.pi * ( np.arctan2(x_rot[:,1],x_rot[:,0]) + (x_rot[:,0]*x_rot[:,1]/2.0/(1-nu)/(x_rot[:,0]**2 + x_rot[:,1]**2)) )
   u[:,1] = -bx/2.0/np.pi * ( ((1-(2.0*nu))/(4.0*(1-nu))*np.log(x_rot[:,0]**2 + x_rot[:,1]**2)) +
                            ((x_rot[:,0]**2 - x_rot[:,1]**2)/4.0/(1-nu)/(x_rot[:,0]**2 + x_rot[:,1]**2)) )

   stress[:,0] = -mu*bx/2/np.pi/(1-nu) * x_rot[:,1]*((3*x_rot[:,0]**2)+x_rot[:,1]**2) / (x_rot[:,0]**2 + x_rot[:,1]**2)**2
   stress[:,1] = mu*bx/2/np.pi/(1-nu) * x_rot[:,1]*(x_rot[:,0]**2-x_rot[:,1]**2) / (x_rot[:,0]**2 + x_rot[:,1]**2)**2
   stress[:,2] = nu*(stress[:,0]+stress[:,1])
   stress[:,5] = mu*bx/2/np.pi/(1-nu) * x_rot[:,0]*(x_rot[:,0]**2-x_rot[:,1]**2) / (x_rot[:,0]**2 + x_rot[:,1]**2)**2

   # Rotate back the u and stress
   rot_mat = np.dot(np.array([[1,0,0],[0,1,0],[0,0,1]]), np.vstack((m0,n0,line_dir)).T)
   u = np.dot(rot_mat, u.T).T
   stress_full = np.zeros((np.shape(stress)[0],3,3))
   stress_full[:,np.repeat([0,1,2],3),np.tile([0,1,2],3)] = stress[:, np.array([0,5,4,5,1,3,4,3,2])]
   stress_full = np.einsum('pj,qk,ijk->ipq', rot_mat, rot_mat, stress_full)
   if not np.all(np.isclose(stress_full, np.einsum('ikj', stress_full))):
      raise RuntimeError('stress matrix is not symmetric after rotation') 
   stress_contracted = np.zeros(np.shape(stress))
   stress_contracted[:,np.arange(6)] = stress_full[:,np.array([0,1,2,1,0,0]),np.array([0,1,2,2,2,1])]

   
   #if not np.all(np.isclose(stress, stress_contracted)):
   #   raise RuntimeError('stress rotation error')

   return u, stress_contracted
   

############################## END (Isotropic soln) ######################################################
##########################################################################################################


def compute_anisotropic_elasticity_tensor(C, t, n, method = 'integral'):

   """
   Inputs: C: Stiffness tensor, t: dislocation line direction
           n: slip plane normal, method: integral or stroh (anisotropic formalism)
   Output: Second-order anisotropic elasticity tensors B and S

   Note: C t and n must be expressed w.r.t. to same coordinate system.
   """

   if np.shape(C) == (6,6):
      C = stiff_conv.C_voigt_to_fourth_rank(C)
   elif np.shape(C) == (3,3,3,3):
      stiff_conv.check_fourth(C)
   else:
      raise RuntimeError('Stiffness matrix C must either be 6*6 in' +
                         ' voigt notation or full 3*3*3*3 stiffness tensor')

   if np.isclose(np.linalg.norm(t), 0.0):
      raise RuntimeError('Magnitude of t is zero')
   elif np.isclose(np.linalg.norm(n), 0.0):
      raise RuntimeError('Magnitude of n is zero')
   else:
      t = t/np.linalg.norm(t)
      n = n/np.linalg.norm(n)

   m = np.cross(n, t)
   
   if method == 'integral':

      theta = sympy.Symbol('theta')
      M = m*sympy.cos(theta) + n*sympy.sin(theta)
      N = -m*sympy.sin(theta) + n*sympy.cos(theta)

   
      integrand_B = np.array( sympy.Matrix(np.dot(M, np.dot(C, M)))
                            - ( ( sympy.Matrix(np.dot(M, np.dot(C, N))) * 
                                  sympy.Matrix(np.dot(N, np.dot(C, N))).inv() ) *
                                  sympy.Matrix(np.dot(N, np.dot(C, M))) ) )

      integrand_S = ( sympy.Matrix(np.dot(N, np.dot(C, N))).inv() *
                      sympy.Matrix(np.dot(N, np.dot(C, M))) )
   
      # Create ufunc of sympy...unfuncify
      ufunc_ufuncify = np.frompyfunc(sympy.utilities.autowrap.ufuncify, 2, 1)

      integrand_B_func_array = ufunc_ufuncify([theta], integrand_B)
      integrand_S_func_array = ufunc_ufuncify([theta], integrand_S)

      # Create ufunc of quad to integrate from 0 to pi
      ufunc_angular_int = np.frompyfunc(lambda x: scipy.integrate.quad(x, 0.0, np.pi)[0], 1, 1)
   
      integrand_B_eval = ufunc_angular_int(integrand_B_func_array).astype(float)
      integrand_S_eval = ufunc_angular_int(integrand_S_func_array).astype(float)

      B = integrand_B_eval / (4*np.pi**2)
      S = - integrand_S_eval / np.pi
      

   elif method == 'stroh':

      p, zeta = stroh_params1(C, m, n)
      A = zeta[:3]
      L = zeta[3:]
      B = 0; S = 0
      for i in range(6):
         B += np.sign(np.imag(p[i])) * np.outer(L[:,i], L[:,i])
         S += np.sign(np.imag(p[i])) * np.outer(A[:,i], L[:,i])

      B /= -4*np.pi*1j; S *= 1j
      B = np.real(B); S = np.real(S)

      
   return B, S



##########################################################################################################








   

   


