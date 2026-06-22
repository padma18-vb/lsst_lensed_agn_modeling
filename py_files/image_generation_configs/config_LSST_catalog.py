# Includes a PEMD deflector with external shear, and Sersic sources. 
# Designed to be similar to LSST-like images (though background noise is not yet implemented.)

import numpy as np
from scipy.stats import norm, truncnorm, uniform
import sys
import paltas.Sampling.distributions as dist

from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource
from lenstronomy.Util import kernel_util
from lenstronomy.Util.param_util import phi_q2_ellipticity
import os
import paltas
root_path = os.path.dirname(os.getcwd())
print(root_path)


# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# This is always the number of pixels for the CCD. If drizzle is used, the
# final image will be larger.
numpix = 33

# Define arguments that will be used multiple times
output_ab_zeropoint = 28.17
n_years = 5
subtract_lens=False
subtract_source=False
doubles_quads_only=False
no_singles=True
catalog = True
compute_caustic_area = False
no_noise=False
apply_psf=True

# load in data
psf_kernels = np.load(os.path.join(root_path, 'data/norm_resize_psf.npy'), mmap_mode='r+')

def draw_psf_kernel():
	random_psf_index = np.random.randint(psf_kernels.shape[0])
	chosen_psf = psf_kernels[random_psf_index, :, :]
	chosen_psf[chosen_psf<0]=0
	return chosen_psf

# this can be None, single int, or a list/np.array/pd.Series of elements
index=None

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'file': os.path.join(root_path, 'data/updated_deflectors.csv'),
		'parameters':{
			'z_lens': 'ZLENS', # fixed in OM10
			#'gamma':2,
			'gamma': 'gamma_lens',
			'theta_E': 'EINSTEIN', # computed
			'e1': 'e1_mass_dinos', # added to catalog
			'e2': 'e2_mass_dinos', # added to catalog
			'center_x': 'XLENS', # fixed in OM10
			'center_y': 'YLENS', # fixed in OM10
			'gamma1': 'gamma1', # added to catalog
			'gamma2': 'gamma2', # added to catalog
			'ra_0':0.0, 'dec_0':0.0,
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'file': os.path.join(root_path, 'data/updated_deflectors.csv'),
		'parameters':{
			'z_source':'ZLENS',
			'mag_app':'APMAG_I', # LENS APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic': 'actual_size',
			'n_sersic': 'n_sersic',
			'e1': 'e1_light_dinos', # added to catalog
			'e2': 'e2_light_dinos', # added to catalog
			'center_x':'XLENS',
			'center_y':'YLENS'
			}
	},
	'source':{
		'class': SingleSersicSource,
		'file': os.path.join(root_path, 'data/sources3.csv'),
		'parameters':{
			'z_source': 'redshift',
			# 'mag_app':norm(loc=24, scale = 2).rvs,
			'mag_app': 'mag_true_i', # SOURCE APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			# 'R_sersic': truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs,
            'R_sersic': 'actual_size',
			'n_sersic': 'n_sersic',
			'e1':'ellipticity_1_true', # added to catalog
			'e2':'ellipticity_2_true', # added to catalog
			'center_x': 'XSRC',
			'center_y': 'YSRC'
		}
	},
    'point_source':{
		'class': SinglePointSource,
		'file': os.path.join(root_path, 'data/sources3.csv'),
		'parameters':{
			'z_source': 'redshift',
            'z_point_source':'ZSRC',
			'x_point_source':'XSRC',
			'y_point_source':'YSRC',
			'mag_app': 'MAGI_IN', # POINT SOURCE APPARENT MAG
			'mag_pert': dist.MultipleValues(dist=truncnorm(-1/0.3,np.inf,1,0.3).rvs,num=10),
			'output_ab_zeropoint':output_ab_zeropoint,
			'compute_time_delays': False
		}
	},
	'cosmology':{
		'file': None,
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
    'psf':{
		'file': None,

		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source':draw_psf_kernel,
			'point_source_supersampling_factor':1
		}
	},

	'lens_equation_solver_parameters':{
		'file': None,
		'solver': 'lenstronomy',
	},

# Currently using the lenstronomy values:
	'detector':{
		'file': None,
		'parameters':{
			'pixel_scale':0.2,'ccd_gain':2.3,'read_noise':9,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':15,'sky_brightness':20.46,
			'num_exposures':30*n_years,'background_noise':None
		}
	}
}

# read noise = total instrumental noise per visit = 12.7; instrumental noise per exposure = 9
# total readnoise from the camera = 8.8 photo-electrons/pixel/exposure
# magnitude_zero_point = output AB zeropoint = 
# sky_brightness = 20.46