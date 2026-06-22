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

root_path = f'{paltas.__path__[0][:-7]}/...'

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':...}

# This is always the number of pixels for the CCD; size of your cutout 
numpix = ...

# Define arguments that will be used multiple times
catalog = True

# other flags (all are booleans) you can use: subtract_lens, subtract_source ,
# doubles_quads_only, no_singles, compute_caustic_area

# this can be None, single int, or a list/np.array/pd.Series of elements
index = None

def draw_psf_kernel():
    ...
    return

# you're appending root path to each file path you provide!
# so they should be in the same directory
config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'file': os.path.join(root_path, '...'),
		'parameters':{
			'z_lens': '...',
			#'gamma':2,
			'gamma': '...',
			'theta_E': '...',
			'e1': '...', 
			'e2': '...',
			'center_x': '...', 
			'center_y':'...',
			'gamma1': '...', 
			'gamma2': '...', 
			'ra_0':0.0, 'dec_0':0.0,
		}
	},
	'lens_light':{
		'class': SingleSersicSource,
		'file': os.path.join(root_path, '...'),
		'parameters':{
			'z_source': '...',
			'mag_app': '...', # LENS APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic': '...',
			'n_sersic': '...',
			'e1': '...', # added to catalog
			'e2': '...', # added to catalog
			'center_x': '...',
			'center_y': '...'
			}
	},
	'source':{
		'class': SingleSersicSource,
		'file': os.path.join(root_path, '...'),
		'parameters':{
			'z_source': ...,
			'mag_app': ..., # SOURCE APPARENT MAG
			'output_ab_zeropoint':output_ab_zeropoint,
			# 'R_sersic': truncnorm(-0.5, np.inf, loc=0.7,scale=1).rvs,
            'R_sersic': ...,
			'n_sersic': ...,
			'e1':..., # added to catalog
			'e2':..., # added to catalog
			'center_x':...,
			'center_y': ...
		}
	},
    'point_source':{
		'class': SinglePointSource,
		'file': os.path.join(root_path, 'data/sources3.csv'),
		'parameters':{
			'z_source': '...',
            'z_point_source':'...',
			'x_point_source':'...',
			'y_point_source':'...',
			'mag_app': '...', # POINT SOURCE APPARENT MAG
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
			'pixel_scale':...,'ccd_gain':...,'read_noise':...,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':...,'sky_brightness':...,
			'num_exposures':...,'background_noise':...
		}
	}
}

# read noise = total instrumental noise per visit = 12.7; instrumental noise per exposure = 9
# total readnoise from the camera = 8.8 photo-electrons/pixel/exposure
# magnitude_zero_point = output AB zeropoint = 
# sky_brightness = 20.46