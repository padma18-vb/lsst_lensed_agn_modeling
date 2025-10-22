import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm, uniform, beta, multivariate_normal
import emcee
import corner
import numba
import pandas as pd
import os, re, sys

# FILE MANAGEMENT
import h5py
from IPython.utils import io

# DATA MANIPULATION
import numpy as np
import pandas as pd

# # PALTAS FUNCTIONS
# import network_predictions
# import paltas
# from paltas import generate
# from paltas.Analysis import hierarchical_inference
# from paltas.Analysis import posterior_functions as pf

# VISUALIZATION
import matplotlib
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import matplotlib.colors as mpc
import corner
from matplotlib.patches import Patch
import astropy.visualization as asviz
from matplotlib.lines import Line2D

SMALL_SIZE = 17
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# MACHINE LEARNING
# import tensorflow as tf

# # HI
# sys.path.insert(0, '/global/homes/v/vpadma/training/deep-lens-modeling')
# from Inference import network_hierarchical_inference as bhi

### GLOBAL VARIABLES
scratch_dir = '/pscratch/sd/v/vpadma'

learning_params = ['main_deflector_parameters_theta_E', 'main_deflector_parameters_gamma1',
                   'main_deflector_parameters_gamma2', 'main_deflector_parameters_gamma', 
                   'main_deflector_parameters_e1', 'main_deflector_parameters_e2', 'main_deflector_parameters_center_x',
                  'main_deflector_parameters_center_y', 'source_parameters_R_sersic', 'source_parameters_mag_app']
labels =["$\\theta_E$", "$\gamma_1$", "$\gamma_2$", "$\gamma_{lens}$", "$e_1$", "$e_2$",'$x_D$', '$y_D$', '$R_{src}$', "$m_{i}$"]
# learning_params = ['main_deflector_parameters_theta_E', 'main_deflector_parameters_gamma1',
#                    'main_deflector_parameters_gamma2', 'main_deflector_parameters_gamma', 
#                    'main_deflector_parameters_e1', 'main_deflector_parameters_e2', 'main_deflector_parameters_center_x',
#                   'main_deflector_parameters_center_y', 'source_parameters_center_x', 'source_parameters_center_y',
#                   'lens_light_parameters_e1','lens_light_parameters_e2']
#                 #   ]
# # labels =["$\\theta_E$", "$\gamma_1$", "$\gamma_2$", "$\gamma_{lens}$", "$e_1$", "$e_2$",'$x_D$', '$y_D$', '$R_{src}$', "$m_{i}$"]
# labels = ["$\\theta_E$", "$\gamma_1$", "$\gamma_2$", "$\gamma_{lens}$", "$e_1$", "$e_2$",'$x_D$', '$y_D$','$x_S$', '$y_S$',
#         "$le_1$", "$le_2$",]

labels_dict = dict(zip(learning_params, labels))

def early_stopping_epoch(log_file,num_before_stopping=10):
    """
    Compute the early stopping epoch given csv containing loss values.
    """

	df = pd.read_csv(log_file)
	val_loss = df['val_loss'].to_numpy()

	min_val_loss = np.inf
	chosen_epoch = np.nan
	num_waited = 0
	for i,v in enumerate(val_loss):
		if v < min_val_loss:
			min_val_loss = v
			chosen_epoch = i+1 
			num_waited = 0
		else: 
			num_waited += 1
						
			if num_waited == num_before_stopping:
				break			
	return chosen_epoch

def get_epoch_from_weights_file(file):
    """Gets epoch number from weights file name.

    Args:
        file (str): h5 file contains weights at a particular epoch

    Returns:
        int: epoch number corresponding to input file
    """
    return int(re.split('/_(.*)--',file)[1])

def get_values_from_dict(dic, keys):
    """Gets list of values for key list from dict.

    Args:
        dic (dict): dictionary containing keys and values
        keys (list): list of keys

    Returns:
        list: values corresponding to list of keys
    """
    return [dic[i] for i in keys]

def get_files_in_order(directory, extension=None):
    """Gets files in order in which they were created.

    Args:
        directory (str): directory containing files
        extension (str, optional): .extension to select. Defaults to None.

    Returns:
        list[str]: list of files
    """
    files = os.listdir(directory)
    # files = filter(os.path.isfile, data_dir)
    files = [os.path.join(directory, f) for f in files]
    if extension is not None:
        files= [f for f in files if f.endswith(extension)]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files

def get_weights_files(directory, mode = 'last_best'):
    """Picks best weights from directory containing all models

    Args:
        directory (str): directory containing all model weights
        mode (str, optional): 'last_best' or 'early_stopping'. Defaults to 'last_best'.

    Returns:
        tuple (int, str): best epoch number and string specifying best weights file
    """
    files = get_files_in_order(directory)
    
    if mode == 'last_best':
    
        files_best = [file for file in files if 'best' in file]
        chosen_file = files_best[-1]
        epoch_number = get_epoch_from_weights_file(chosen_file)
        
        #print(files_best)
    elif mode == 'early_stopping':
        epoch_number = early_stopping_epoch(os.path.join(directory, 'losses.csv'))
        chosen_file = get_file_for_epoch(directory,epoch_number)
    return epoch_number, chosen_file


def get_file_for_epoch(directory, epoch, extension=".h5"):
    """Gets the file corresponding to a particular epoch.

    Args:
        directory (str): Directory containing the files.
        epoch (int): The epoch number to find the file for.
        extension (str, optional): File extension to filter. Defaults to ".h5".

    Returns:
        str: File path corresponding to the given epoch, or None if not found.
    """
    files = get_files_in_order(directory, extension=extension)
    for file in files:
        if f"_{epoch}--" in file:
            return file
    return None
# TRACK TRAINING LOSS
def training_loss(files, scatter_point=None, set_ylim=None, ax=None):
    """Plots training loss for each loss file in files.

    Args:
        files (list): list of loss files
        scatter_point (list, optional): list of points to draw on loss track. Defaults to None.
        set_ylim (tuple, optional): limit y-axis. Defaults to None.
        ax (ax, optional): ax to plot on. Defaults to None.
    Returns:
        matplotlib.Figure: training/validation loss figure
    """
    losses = pd.DataFrame()
    for file in files:
        losses = pd.concat([losses, pd.read_csv(file, index_col=0)])

    # print(f'Number of epochs: {len(losses)}')
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(len(losses)), losses['val_loss'], color='purple',label='Validation Loss')
    ax.plot(np.arange(len(losses)), losses['loss'], color='cyan',label='Training Loss')
    if set_ylim is None:
        ax.set_ylim(min(losses['loss']), max(losses['loss']))
    else:
        ax.set_ylim(set_ylim)
    ax.set_title('Training Progress - 500k images')
    ax.set_xlabel("Num of Epochs")
    ax.set_ylabel("$-\log(p(\\xi| D) $")
    if scatter_point is not None:
        ax.scatter(scatter_point, marker='*', color='r')
    ax.grid(ls='--')
    ax.set_xscale('log')
    ax.legend()
    return fig
    
# HELPER FUNCTIONS TO MAKE CORNER PLOTS
def get_mean(dist):
    """Get n-mean from n-dist.

    Args:
        dist (n-d array): array containing predictions/truths etc

    Returns:
        array: array of means along axis 0.
    """
    return np.mean(dist, axis=0)

def get_precision_errprec_scatter(y_test, y_pred, std_pred):
    """Gets precision, error percent and scatter from truth, prediction,
    uncertainty of prediction

    Args:
        y_test (array): truth values; should be same shape as other input
        y_pred (array): predicted values; should be same shape as other input
        std_pred (array): predicted value uncertainties; should be same shape as other input

    Returns:
        tuple: tuple containing three arrays.
    """
    assert y_test.shape==y_pred.shape
    assert y_test.shape==std_pred.shape

    precisions = np.median(std_pred[:, :]*100/np.abs(y_test[:, :]) , axis=0)
    error = y_pred[:, :]-y_test[:, :]
    errperc = np.mean((y_pred[:, :]-y_test[:, :])*100/y_test[:, :] , axis=0)
    scatters = np.median(np.abs(error[:,:])*100/y_test[:, :], axis=0)
    return precisions, errperc, scatters

def get_stats(y_test, y_pred, std_pred, params=np.arange(8)):
    """Gets correlation, mean error (or bias), median absolute error (MAE), 
    median precision (uncertainty) 

    Args:
        y_test (array): truth values; should be same shape as other input
        y_pred (array): predicted values; should be same shape as other input
        std_pred (array): predicted value uncertainties; should be same shape as other input
        params (array, Optional):

    Returns:
        tuple: tuple containing four arrays.
    """
    y_test = y_test[:, params]
    y_pred = y_pred[:, params]
    std_pred = std_pred[:, params]
    correlation = np.array([
        np.round(np.corrcoef(y_test[:, i], y_pred[:, i])[0][1], 2) for i in range(y_pred.shape[1])])
    mean_error = np.round(np.mean(y_pred[:, :] - y_test[:, :],axis=0), 2)
    median_absolute_error = np.round(np.median(np.abs(y_pred[:, :] - y_test[:, :]),axis=0), 2)
    median_precision = np.round(np.median(std_pred[:, :],axis=0), 2)
    return correlation, mean_error, median_absolute_error, median_precision

def get_range(dist):
    """Gets n-range for n-dist

    Args:
        dist (np.ndarray): distribution of n parameters

    Returns:
        list[tuple]: list of n tuples containing distribution range of n parameters
    """
    range_arr = np.array([np.min(dist, axis=0),
                          np.max(dist, axis=0)]).T
    final = [tuple(i) for i in range_arr]
    return final

def get_train_data(results_df, prep, file_num = range(1, 6)):
    """Get training data metadata.

    Args:
        results_df (pd.DataFrame): dataframe containing training results for all preps
        prep (str): image prepartion (index in results_df)
        file_num (list, optional): list of number of batches to use. Defaults to range(1, 6).

    Returns:
        pd.DataFrame: dataframe produced from training metadata csv
    """
    return pd.concat([pd.read_csv(f'{results_df.loc[prep, "path_to_train_images"]}{i}/metadata.csv') for i in file_num])

def get_valid_data(results_df, prep, file_num = range(1, 6)):
    """Get validation data metadata.

    Args:
        results_df (pd.DataFrame): dataframe containing validation set used for training 
        for all preps
        prep (str): image prepartion (index in results_df)
        file_num (list, optional): list of number of batches to use. Defaults to range(1, 6).

    Returns:
        pd.DataFrame: dataframe produced from training metadata csv
    """
    return pd.concat([pd.read_csv(f'{results_df.loc[prep, "path_to_valid_images"]}{i}/metadata.csv') for i in file_num])

def make_analysis_table(results_df, prep_name, obj=None):
    """Makes 

    Args:
        results_df (pd.DataFrame): dataframe containing training results for all preps
        prep (str): image prepartion (index in results_df)
        obj (object, Optional): prepRes object

    Returns:
        pd.DataFrame: _description_
    """
    all_df_orig = pd.read_csv(os.path.join(results_df.loc[prep_name, 'path_to_test_images'], 'metadata.csv'))
    all_df_orig['distance_from_lens'] = np.sqrt((all_df_orig['source_parameters_center_x'] - all_df_orig['main_deflector_parameters_center_x'])**2 + (all_df_orig['source_parameters_center_y'] - all_df_orig['main_deflector_parameters_center_y'])**2)
    interesting_columns = ['lens_light_parameters_R_sersic', 'lens_light_parameters_e1', 'lens_light_parameters_e2', 
                           'lens_light_parameters_mag_app', 'main_deflector_parameters_z_lens',
                           'point_source_parameters_mag_app', 'point_source_parameters_num_images',
                           'source_parameters_R_sersic', 'source_parameters_e1', 'source_parameters_e1', 'source_parameters_mag_app','source_parameters_z_source',
                           'source_parameters_center_x','source_parameters_center_y',
                           'distance_from_lens', 'psf_fwhm', 'lens_light_parameters_z_source']
    all_df = pd.DataFrame()
    i = 0
    for param in results_df.loc[prep_name, 'learning_params'][:]:
        try:
            all_df[f'{param}'] = results_df.loc[prep_name, 'y_test'][:, i]
            all_df[f'{param}_pred'] = results_df.loc[prep_name, 'y_pred'][:, i]
            all_df[f'{param}_stdpred'] = results_df.loc[prep_name, 'std_pred'][:, i]
        except:
            if obj is not None:
                all_df[f'{param}'] = obj.y_test[:, i]
                all_df[f'{param}_pred'] = obj.y_pred[:, i]
                all_df[f'{param}_stdpred'] = obj.std_pred[:, i]
        all_df[f'{param}_error'] = all_df[f'{param}_pred'] - all_df[f'{param}']
        error = all_df[f'{param}_pred'] - all_df[f'{param}']
        all_df[f'{param}_stderr'] = error/all_df[f'{param}_stdpred']
        i+=1

    for i in interesting_columns:
        all_df[i] = all_df_orig[i]
    return all_df
# def make_metrics_table_from_df(analysis_df, params=None):
#     """_summary_

#     Args:
#         analysis_df (pd.DataFrame): output df of make_analysis_table
#         params (list, optional): list of params we need metrics for. Defaults to None.

#     Returns
#         pd.DataFrame: metrics dataframe
#     """
#     if params is None:
#         params = learning_params
#     columns = pd.unique([col for col in analysis_df.columns for i in params if (i+'_' in col)])
#     df = analysis_df[columns]
#     error_df = df[[col for col in df.columns if ('error' in col)]]
#     mean_error = error_df.describe()[1:].T['mean']
#     median_absolute_error = error_df.abs().describe()[1:].T['50%']
#     precision = df[[col for col in df.columns if ('stdpred' in col)]].describe()[1:].T['50%']
#     return pd.DataFrame(data = np.array([mean_error, median_absolute_error, precision]).T, 
#                  index = get_values_from_dict(labels_dict, params), columns = ['Mean Error', "Median Absolute Error", "Median Precision"])

def make_metrics_table(y_test,y_pred,std_pred,params):
    """
    Args:
        y_test (np.array): 2D array containing all truth values
        y_pred (np.array): 2D array containing all predicted posterior mean values
        std_pred (np.array): 2D array containing all predicted posterior uncertainty values

    Returns:
        tuple: mean_error, mse_error, precision
    """
    params = np.array(params)
    mean_error = ((y_pred[:, params]- y_test[:, params])/std_pred[:, params]).mean(axis=0)
    # mse_error = np.sqrt(((y_pred[:, params] - y_test[:, params])**2).mean(axis=0))
    mae_error = np.median(np.abs(y_pred[:, params]- y_test[:, params])/std_pred[:, params],axis=0)
    med_prec = np.median(std_pred[:, params], axis=0)
    return mean_error, mae_error, med_prec

CORNER_KWARGS = dict(
    smooth = 0.9,
    label_kwargs=dict(fontsize=35),
    title_kwargs=dict(fontsize=30,loc='left' ),
    plot_density=True,
    plot_datapoints=False,
    fill_contours=True,
    show_titles=False,
    max_n_ticks=4,
    bins=20
)
def make_contour(list_of_dists, 
                 labels, categories, colors, range_for_bin=False, 
                 show_correlation=False, 
                 truths_list=None,show_every_title=False, save_fig=False,show_truths=True):
    """
    Generate a combined corner/contour plot for multiple multivariate sample sets.
                This function wraps the corner.corner plotting routine to overlay contours/histograms
                for multiple distributions (e.g. different categories or models) on a single set of
                axes, add legends, optional truth/mean lines, optional per-pair correlation annotations,
                and optional custom title styling. It returns the matplotlib Figure containing the
                assembled corner plot.
                Parameters
                ----------
                list_of_dists : sequence of array-like
                    A list (or other iterable) of 2D arrays of samples. Each entry should have shape
                    (n_samples, n_dimensions). All entries are plotted on the same corner-grid axes.
                labels : sequence of str
                    Names for each parameter/dimension; passed to corner.corner as axis labels.
                categories : sequence
                    Category identifiers (used to build the legend). Length must match `colors`.
                colors : sequence
                    Colors (strings or matplotlib color specifications) for each distribution. The
                    length must equal `categories`. Colors are used for histogram fills, contours,
                    overplotted means, and text annotations.
                range_for_bin : bool or sequence, optional
                    If truthy, compute and pass a common `range` to the per-distribution histograms
                    using the first distribution via get_range(exemplar_dist). If False, no range is
                    provided (corner handles defaults). Default: False.
                show_correlation : bool, optional
                    If True, compute Pearson correlation coefficients for each 2D pair in each
                    distribution and overlay the (rounded) coefficient text on the corresponding
                    axes in the main figure. Default: False.
                truths_list : sequence of array-like, optional
                    If provided, this should be a list of arrays of "true" parameter values (one per
                    distribution) to be drawn on the plots. If None, the per-distribution means are
                    used as default "truths" (unless show_truths is False). Default: None.
                show_every_title : bool, optional
                    If True, attempt to draw per-distribution titles/values above each subplot column
                    in color-coded rows. When False, a single title style is used and title color is
                    taken from the last color in `colors`. Default: False.
                save_fig : bool or str, optional
                    If truthy, the figure will be saved. If a string, it is used as the filename.
                    If True (but not a string), a default filename 'newfig.pdf' is used. Default: False.
                show_truths : bool, optional
                    If False, do not draw any truth/true-value lines. Default: True.
                Other behavior and notes
                ------------------------
                - The function asserts that len(categories) == len(colors); if categories contains
                  duplicate entries a warning about the legend may be printed.
                - A legend is constructed using the mapping from category -> color and placed on the
                  assembled figure.
                - Per-distribution histograms use density=True and a relatively thick line width by
                  default. Contour levels are fixed at [0.68, 0.95].
                - The function attempts to use and update a global CORNER_KWARGS dict to set common
                  corner.corner keyword arguments (including title styling). If CORNER_KWARGS is not
                  defined or lacks expected keys, reasonable defaults are set.
                - For each distribution, the function overplots the per-dimension sample means (via
                  corner.overplot_lines). If truths_list is provided, truths are overplotted in green.
                - If show_correlation is True the function builds a temporary mini-corner figure to
                  compute correlations from the scatter plot lines for each pair and annotates the main
                  figure's axes with the coefficients.
                - Axis tick styling and minor ticks are adjusted for all axes of the assembled figure.
                - Saving the figure uses bbox_inches='tight' and facecolor='white'.
                Returns
                -------
                matplotlib.figure.Figure
                    The matplotlib Figure object containing the combined corner/contour plots.
                Raises
                ------
                AssertionError
                    If len(categories) != len(colors).
                See also
                --------
                corner.corner : for the underlying plotting functionality used by this wrapper.
                get_range, get_mean : helper functions expected to be available in the module,
                                     used for range computation and mean calculation respectively.
                Example
                -------
                # Basic usage
                fig = make_contour([samples_a, samples_b],
                                   labels=['x', 'y', 'z'],
                                   categories=['A', 'B'],
                                   colors=['C0', 'C1'],
                                   range_for_bin=True,
                                   truths_list=[truths_a, truths_b],
                                   save_fig='comparison.pdf')
    """
    assert(len(categories)==len(colors)), print('make sure you have a color for every category!')
    cat_to_col = dict(zip(categories, colors))
    # if categories has a two of the same elements, they'll have the same color!
    if len(np.unique(categories))< len(categories):
        print("Your legend is going to be weird with the same color!")
    legend_elements = []

    for cat in categories:
        legend_elements.append(Patch(facecolor=cat_to_col[cat], edgecolor=cat_to_col[cat], label=cat))
    
    exemplar_dist = list_of_dists[0]
    if range_for_bin:
        bin_range = get_range(exemplar_dist)
    else:
        bin_range=None
        
    fig,ax = plt.subplots(exemplar_dist.shape[1],exemplar_dist.shape[1],figsize=(20,22))
    if show_every_title:
        title_color='white'
        title_fontsize=1
    else:
        title_color=colors[-1]
        try:
            title_fontsize=CORNER_KWARGS['title_kwargs']['fontsize']
        except:
            title_fontsize = 30
    try:
        CORNER_KWARGS['title_kwargs'].update(color=title_color, fontsize=title_fontsize)
    except:
        CORNER_KWARGS['title_kwargs'] = dict(fontsize=title_fontsize, color=title_color)
    i = 0
    alpha = 0.3
    for ax in fig.get_axes():
        ax.tick_params(axis='both', length=10, labelsize=17)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', length=5, labelsize=10)
        
    for dist in list_of_dists:
        means = get_mean(dist)
        if truths_list is None:
            truths = means
            truth_color=colors[i]
        else:
            truths = truths_list[i]
            truth_color='green'
        if not show_truths:
            truths = None
        
        corner.corner(
            data=dist,
            labels=labels,
            color=colors[i],
            truths= truths,
            hist_kwargs=dict(density=True,lw=5, color=colors[i], range=bin_range),
            levels=[0.68,0.95],
            truth_color=truth_color,
            **CORNER_KWARGS,
            title_fmt = '.2f',
            fig=fig,
            alpha=alpha
            );
        corner.overplot_lines(fig, means, color=colors[i])
        if truths_list is not None:
            corner.overplot_lines(fig, truths, color='green', linewidth=4)
        alpha = alpha + len(list_of_dists)/10
        alpha = max(1, alpha)
        props1 = dict(boxstyle='round', facecolor='white')
        if show_correlation:
            mini_fig = corner.corner(
                data=dist);
            plt.figure(visible=False)
            to_put = {}
            ax_i = 0
            for ax in mini_fig.get_axes():
                line = ax.lines
                try:
                    #print(line[0].get_xdata(), line[0].get_ydata())
                    r_coef = np.corrcoef(line[0].get_xdata(), line[0].get_ydata())[0][1]
                    #print(r_coef)
                    to_put[ax_i] = np.round(r_coef,2)
                except:
                    pass
                ax_i += 1
            fig_ax_list = fig.get_axes()
            for ax_stored in to_put.keys():
                ax = fig_ax_list[ax_stored]
                ax.text(0.6, 0.9 - i/10, to_put[ax_stored], size=18, color=colors[i],transform=ax.transAxes, bbox=props1)
        i+=1
        if show_every_title:
            titles_old = []
            for ax in fig.axes:
                #print(ax.get_title('left'))
                titles_old.append(ax.get_title('left'))
            titles_old=np.array(titles_old)
            titles_old=titles_old.reshape(i,len(titles_old)//i, ).T
            new_tit_i=0
    
            for ax in fig.axes:
                curr_ax_title= ax.get_title('left')
                if curr_ax_title != '':
                    titles_curr = titles_old[new_tit_i]
                    start=''
                    color_i=0
                    inch=0
                    for tit in titles_curr:
                        ax.text(0, 1.25-inch, tit,color=colors[color_i],weight=5,fontsize=35,transform=ax.transAxes)
                        inch+=0.1
                        color_i+=1
                    new_tit_i+=i
            
        # titles_old.append(ax.get_title('left'))
    #print(np.array(fig.get_axes()).shape)

    fig.legend(handles=legend_elements, frameon=False, ncol=1 ,loc=(0.56, 0.6), fontsize=40)
    # fig.tight_layout()
    if save_fig:
        if type(save_fig) == str:
            plt.savefig(save_fig,facecolor='white', bbox_inches = 'tight')
        else:
            plt.savefig('newfig.pdf', facecolor='white', bbox_inches = 'tight')
    return fig


def retrieve_chains_h5(file_path):
    """Retrieves MCMC h5 outputs.

    Args:
        file_path (str): path to chains

    Returns:
        np.ndarray: MCMC results from hierarchical inference chains
    """
    h5f = h5py.File(file_path, 'r')
    chain_names = list(h5f.keys())

    for name in chain_names:
        print(name,h5f.get(name))
        chains = h5f.get(name)[()]
    h5f.close()

    return chains
    
    
### ALGORITHM 1
def plot_TARP_HDP_all_params(results_df, prep, num_samples = 10000, fig=None, ax=None):
    """Plots the TARP-HDP calibration for a given experiment.

    Args:
        results_df (pd.DataFrame): dataframe containing training results for all preps
        prep (str): image prepartion (index in results_df)
        num_samples (int, optional): number of samples. Defaults to 10000.
        fig (optional): figure to plot into. Defaults to None.
        ax (optional): ax in figure. Defaults to None.

    Returns:
        _type_: _description_
    """
    y_test = results_df.loc[prep, 'y_test']
    y_pred = results_df.loc[prep, 'y_pred']
    cov_pred = results_df.loc[prep, 'cov_pred']
    N_sims = y_test.shape[0]
    store_f_i = np.array([])
    alphas = np.linspace(0, 1, num_samples//10)
    ECPS = np.array([])
    
    for i in range(N_sims):
        # generate samples
        p_hat_mean = y_pred[i, :]
        #p_hat_mean = p_hat_mean_emulate[i, :]
        true_val = y_test[i, :]
        p_hat_cov = cov_pred[i, :]    
        p_hat = multivariate_normal.rvs(p_hat_mean, p_hat_cov, size=num_samples)

        probs_of_p_hat = multivariate_normal.pdf(p_hat, mean = p_hat_mean, cov = p_hat_cov,)

        true_prob =  multivariate_normal.pdf(true_val, mean = p_hat_mean, cov = p_hat_cov,)

        f_i = (1 / num_samples) * np.sum(probs_of_p_hat < true_prob)
        store_f_i = np.append(store_f_i, f_i)

    for alpha in alphas:
        ECP = (1/N_sims) * np.sum(store_f_i < (1-alpha))
        ECPS = np.append(ECPS, ECP)
    #print(ECPS)
    if fig is None:
        fig, ax = plt.subplots()
    ax.plot(1-alphas, ECPS, color=results_df.loc[prep, 'color'])
    ax.plot(1-alphas, 1-alphas,ls = '--', color='k')
    ax.set_title("TARP-HDP Calibration")
    ax.set_xlabel("Credibility Level")
    ax.set_ylabel("Coverage")
    return fig, ax


def plot_calibration(results_df, prep_list, obj=None,obj_index=None,num_params=8, pf_fig=None,ax=None):
    """Plots calibration for every prep in prep_list.

    Args:
        results_df (pd.DataFrame): dataframe containing training results for all preps
        prep_list (list): list of image prepartions (index in results_df)

    Returns:
        matplotlib.Figure: calibration plot figure
    """
    if pf_fig is None and ax is None:
        pf_fig, ax  = plt.subplots(figsize=(10,10))
    results_df = results_df.loc[prep_list]
    custom_lines = []
    for prep in results_df.index:
        prep_row = results_df.loc[prep]
        if obj:
            y_test = obj.y_test[obj_index,:num_params]
            y_pred = obj.y_pred[obj_index, :num_params]
            # std_pred = obj.std_pred[:, :]
            cov_pred =  obj.cov_pred[obj_index, :num_params,:num_params]
        else:
            y_test = prep_row.loc['y_test'][:,:]
            y_pred = prep_row.loc['y_pred'][:, :]
            # std_pred = prep_row.loc['std_pred'][:, :]
            cov_pred =  prep_row.loc['cov_pred'][:, :,:]
        n_params = prep_row.loc['n_params']
        predict_samps = np.array([multivariate_normal.rvs(y_pred[i, :], cov_pred[i, :, :], size=y_pred.shape[0]) for i in range(y_pred.shape[0])])
        print(predict_samps.shape)
        pf_fig = pf.plot_calibration(predict_samps=predict_samps, y_test=y_test, figure = pf_fig,ax=ax, block=False,
                                    show_plot=False, legend = [],color_map=['k', prep_row.loc['color']])

        custom_lines.append(Line2D([0], [0], color=prep_row.loc['color'], lw=4))
        ax.set_xlabel('Percentage of Probability Volume', size=15)
        ax.set_ylabel('Percent of Lenses With True Value in the Volume', size=15)

    pf_fig.legend(custom_lines, list(results_df['name']), loc=(0.5, 0.2))
    return pf_fig, ax
    
    
def plot_calibration_per_param(results_df, prep_list, params=None):
    """Plots parameter-wise calibration for every preparation of data in prep_list.

    Args:
        results_df (pd.DataFrame): dataframe containing training results for all preps
        prep_list (list): list of image prepartions (index in results_df)
        params (list, optional): list of parameters. Defaults to None.
    """
    results_df = results_df.loc[prep_list]
    if params is None:
        params=np.arange(len(learning_params))
    fig, ax =plt.subplots(2, len(params)//2, figsize=(33, 15))
    ax = ax.flatten()
    for prep in results_df.index:
        prep_row = results_df.loc[prep]
        for i in range(len(params)):
            y_test = prep_row.loc['y_test']
            y_pred = prep_row.loc['y_pred']
            std_pred = prep_row.loc['std_pred']
            predict_samps = norm(y_pred[:, params[i]], std_pred[:, params[i]]).rvs((5000, len(y_pred[:, params[i]]),))
            pf.plot_calibration(predict_samps=predict_samps, y_test=y_test[:, params[i]], figure=fig, ax=ax[i], 
                                title=labels[i],block=True,legend = [],color_map=['k', prep_row.loc['color']])


def plot_calibration_one_prep_all_param(results_df,prep,obj_index=None,num_params=8, obj=None,params=None):
    """Plots parameter-wise calibration for every preparation of data in prep_list.

    Args:
        results_df (pd.DataFrame): dataframe containing training results for all preps
        prep_list (list): list of image prepartions (index in results_df)
        params (list, optional): list of parameters. Defaults to None.
    """
    if params is None:
        params=np.arange(len(learning_params))
    fig, ax =plt.subplots(figsize=(10,10))
    custom_lines = []

    # for prep in results_df.index:
    prep_row = results_df.loc[prep]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

    for i in range(len(params)):
        if obj is None:
            y_test = prep_row.loc['y_test']
            y_pred = prep_row.loc['y_pred']
            std_pred = prep_row.loc['std_pred']
        else:
            if obj_index is None:
                obj_index = len(obj.y_test[:,:])
            y_test = obj.y_test[obj_index,:]
            y_pred = obj.y_pred[obj_index, :]
            std_pred = obj.std_pred[obj_index, :]
            cov_pred =  obj.cov_pred[obj_index, :,:]
        predict_samps = norm(y_pred[:, params[i]], std_pred[:, params[i]]).rvs((5000, len(y_pred[:, params[i]]),))
        pf_fig = pf.plot_calibration(predict_samps=predict_samps, y_test=y_test[:, params[i]], figure=fig, ax=ax, 
                            title='Calibration per parameter',block=True,legend = [],color_map=['k', colors[i]])
        custom_lines.append(Line2D([0], [0], color=colors[i], lw=4))
        ax.set_xlabel('Percentage of Probability Volume', size=15)
        ax.set_ylabel('Percent of Lenses With True Value in the Volume', size=15)
    pf_fig.legend(custom_lines, list(labels[params]), loc=(0.5, 0.2),ncols=2)
    return pf_fig, ax

from lenstronomy.Util import param_util
def get_shear(param_array):
    # returns phi, gamma
    return param_util.shear_cartesian2polar(param_array[:, 1], param_array[:, 2])

def get_ellip(param_array):
    # return phi, q
    return param_util.ellipticity2phi_q(param_array[:, 4], param_array[:, 5])

def plot_performance_across_sample(results_df, prep, param_list = None, save=True, obj=None,polar=False):
    """Plots recovery plot for each param in param_list (or all params if none specified)
    for a given preparation of data.

    Args:
        results_df (pd.DataFrame): dataframe containing training results for all preps
        prep (str): image prepartion (index in results_df)
        param_list (list, optional): list[int] of parameters indices. Defaults to None (all parameters).
    """
    prep_row = results_df.loc[prep]
    try:
        y_pred = prep_row['y_pred']
        y_test = prep_row['y_test']
        std_pred = prep_row['std_pred']
    except:
        y_pred = obj.y_pred
        y_test = obj.y_test
        std_pred = obj.std_pred

    face_color = prep_row['color']
    props1 = dict(boxstyle='round', facecolor='turquoise', alpha=0.5)
    props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    props3 = dict(boxstyle='round', facecolor='thistle', alpha=0.5)
    # mean_errors = []
    # maes = []
    # precisions = []
    train_data = get_train_data(results_df, prep)
    precisions = np.median(std_pred[:, :]*100/np.abs(y_test[:, :]) , axis=0)
    error = y_pred[:, :]-y_test[:, :]
    errperc = np.mean((y_pred[:, :]-y_test[:, :])*100/y_test[:, :] , axis=0)
    scatters = np.median(np.abs(error[:,:])*100/y_test[:, :], axis=0)
    if param_list is None:
        param_list = range(y_pred.shape[1])
    
    
    for i in param_list:

        fig,ax =plt.subplots(figsize=(80,25))
        param_i = prep_row['learning_params'][i]
        ax.axhline(train_data[param_i].mean(), ls='--', color='gray', lw=20)
        ax.errorbar(y_test[:,i], y_pred[:,i],yerr=std_pred[:,i], fmt='o', mec=face_color,ecolor=face_color, mfc=face_color,marker='s', ms=15, alpha=0.7)
        # # plt.errorbar(wy_test[:,i], wy_pred[:,i],yerr=wstd_pred[:,i], fmt='o', mfc='r',marker='s', ms=3, alpha=0.3)
        ax.plot(y_test[:,i], y_test[:,i], color='red', alpha=0.6,lw=20)
        ax.set_title(labels[i],size=100, color='black')
        ax.set_xlabel('Truth',size=90, color='black')
        ax.set_ylabel('Prediction',size=90, color='black')

        correlation = np.round(np.corrcoef(y_test[:, i], y_pred[:, i])[0][1], 2)
        mean_error = np.round(np.mean(y_pred[:, i] - y_test[:, i]), 2)
        median_absolute_error = np.round(np.median(np.abs(y_pred[:, i] - y_test[:, i])), 2)
        median_precision = np.round(np.median(std_pred[:, i]), 2)


        ax.text(0.8,0.1,
             'Correlation: %.3f'%(correlation),{'fontsize':80}, transform=ax.transAxes)

        if i == 0 or i == 3 or i == 9:
            ax.text(0.05,0.7,
            'Mean Error/Lens: %.2f'%(errperc[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props3)
            ax.text(0.05,0.8,
            'Scatter/Lens: %.2f'%(scatters[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props1)
            ax.text(0.05,0.9,
                    'Precision/Lens: %.2f'%(precisions[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props2)
        else:
            ax.text(0.05,0.7,
            'Mean Error: %.2f'%(mean_error),{'fontsize':80},transform=ax.transAxes, bbox=props3)
            ax.text(0.05,0.8,
            'MAE: %.2f'%(median_absolute_error),{'fontsize':80},transform=ax.transAxes, bbox=props1)
            ax.text(0.05,0.9,
                    'Median Precision: %.2f'%(median_precision),{'fontsize':80},transform=ax.transAxes, bbox=props2)

        ax.tick_params(axis='both', length=25, which='major', labelsize=60, colors='black')
        ax.tick_params(axis='both', length=20, which='minor', labelsize=60, colors='black')
        ax.text(0.8, 1.02, prep_row['name'], {'fontsize':80, 'color': 'black'},transform=ax.transAxes)
        ax.set_ylim(ax.get_xlim())
        for spine in ax.spines.values():
            spine.set_linewidth(2)

            
        plt.grid()

        fig.set_facecolor('white')
        if save:
            plt.savefig(f"{param_i}_{prep}.png", bbox_inches = 'tight')

    plt.show()



def plot_performance_across_sample_with_input(results_df, prep, y_pred_input, y_test_input, std_pred_input, param_list = None,polar=False):
    prep_row = results_df.loc[prep]
    y_pred = prep_row['y_pred']
    y_test = prep_row['y_test']
    std_pred = prep_row['std_pred']
    if polar:
        y_pred[:,1], y_pred[:,2] = get_shear(y_pred)
        y_test[:,1], y_test[:,2] = get_shear(y_test)
        std_pred[:,1], std_pred[:,2] = np.sqrt(std_pred[:, 1]**2 + std_pred[:,2]**2)

        y_pred[:,4], y_pred[:,5] = get_ellipticity(y_pred)
        y_test[:,4], y_test[:,5] = get_ellipticity(y_test)
        std_pred[:,4], std_pred[:,5] = np.sqrt(std_pred[:, 4]**2 + std_pred[:,5]**2)
    face_color = prep_row['color']
    props1 = dict(boxstyle='round', facecolor='turquoise', alpha=0.5)
    props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    props3 = dict(boxstyle='round', facecolor='thistle', alpha=0.5)
    # mean_errors = []
    # maes = []
    # precisions = []
    train_data = get_train_data(results_df, prep)

    if param_list is None:
        param_list = range(max(y_pred.shape[1], y_pred_input.shape[1]))


    for i in param_list:
        fig,ax =plt.subplots(figsize=(80,25))
        
        ax.axhline(train_data[prep_row['learning_params'][i]].mean(), ls='--', color='gray', lw=20)
        ax.errorbar(y_test[:,i], y_pred[:,i],yerr=std_pred[:,i], fmt='o', mec='salmon', ecolor='salmon', mfc='salmon',marker='s', ms=15, alpha=0.9)

        ax.errorbar(y_test_input[:,i], y_pred_input[:,i],yerr=std_pred_input[:,i], fmt='o', mec='seagreen', ecolor = 'seagreen',mfc='seagreen',marker='s', ms=15, alpha=0.5)


        # # plt.errorbar(wy_test[:,i], wy_pred[:,i],yerr=wstd_pred[:,i], fmt='o', mfc='r',marker='s', ms=3, alpha=0.3)
        ax.plot(y_test[:,i], y_test[:,i], color='darkred', lw=25)
        ax.plot(y_test_input[:,i], y_test_input[:,i], color='darkgreen', lw=25)

        ax.set_title(labels[i],size=100, color='black')
        ax.set_xlabel('Truth',size=90, color='black')
        ax.set_ylabel('Prediction',size=90, color='black')
        
        precisions, errperc,scatters = get_precision_errprec_scatter(y_test, 
                                                                    y_pred,
                                                                    std_pred)
        correlation, mean_error, median_absolute_error, median_precision =get_stats(y_test, y_pred, std_pred)
        precisions2, errperc2,scatters2 = get_precision_errprec_scatter(y_test_input, 
                                                                    y_pred_input,
                                                                    std_pred_input)
        correlation2, mean_error2, median_absolute_error2, median_precision2 =get_stats(y_test_input, y_pred_input, std_pred_input)
        ax.text(0.8,0.1,
             'LAGN Correlation: %.2f'%(correlation[i]),{'fontsize':80}, transform=ax.transAxes)

#         ax.text(0.05,0.8,
#              'Mean Error: %.2f'%(mean_error),{'fontsize':80}, transform=ax.transAxes, bbox=props3)
        if i == 0 or i == 3 or i == 9:
            ax.text(0.05,0.7,
            'Mean Error/Lens: LAGN %.0f'%(errperc[i])+'%'+';LG: %.0f'%(errperc2[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props3)
            ax.text(0.05,0.8,
            'Scatter/Lens: LAGN %.0f'%(scatters[i])+'%'+';LG: %.0f'%(scatters2[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props1)
            ax.text(0.05,0.9,
                    'Precision/Lens: LAGN %.0f'%(precisions[i])+'%'+';LG: %.0f'%(precisions2[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props2)
        else:
            ax.text(0.05,0.7,
            'Mean Error: LAGN %.2f'%(mean_error[i])+';LG %.2f'%(mean_error2[i]),{'fontsize':80},transform=ax.transAxes, bbox=props3)
            ax.text(0.05,0.8,
            'MAE: LAGN %.2f'%(median_absolute_error[i])+';LG %.2f'%(median_absolute_error2[i]),{'fontsize':80},transform=ax.transAxes, bbox=props1)
            ax.text(0.05,0.9,
                    'Median Precision: LAGN %.2f'%(median_precision[i])+';LG %.2f'%(median_precision2[i]),{'fontsize':80},transform=ax.transAxes, bbox=props2)

        # ax.text(0.05,0.7,
        #     'Scatter/Lens: %.0f'%(scatters[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props1)
        # ax.text(0.05,0.8,'Precision/Lens: %.0f'%(precisions[i])+'%',{'fontsize':80},transform=ax.transAxes, bbox=props2)
        ax.tick_params(axis='both', length=25, which='major', labelsize=60, colors='black')
        ax.tick_params(axis='both', length=20, which='minor', labelsize=60, colors='black')
        ax.text(0.8, 1.02, prep_row['name'], {'fontsize':80, 'color': 'black'},transform=ax.transAxes)
        
        
        ax.text(0.8,0.2,
             'LG Correlation: %.2f'%(correlation2[i]),{'fontsize':80}, transform=ax.transAxes)

        plt.grid()

        # plt.savefig(f"{i}_plsss.png")

        fig.set_facecolor(face_color)
        # plt.savefig(f'{prep}_{re.split("parameters_", learning_params[i])[1]}_recovery.pdf')

    plt.show()
# ax[0].axhline(0.8, color='r')

def get_obj_of_wide_posteriors(results_df, prep, params=None, round_degree = 2):
    if params is None:
        params=np.arange(results_df.loc[prep, 'n_params'])
    train_data = get_train_data(results_df, prep)
    learning_params = np.array(results_df.loc[prep, 'learning_params'])[params]
    std_pred = results_df.loc[prep, 'std_pred'][:, params]
    boolean_arrays = std_pred[:, :]>np.array(np.round(train_data[learning_params].std(), round_degree))
    baddies = []
    baddies_dict = {}
    # for every object
    for i in range(len(boolean_arrays)):
        # if the posterior width is wider than the prior width for one or more parameters
        if sum(boolean_arrays[i])>=1:
            # append the object to the "baddies" list
            baddies.append(i)
            baddies_dict[i] = boolean_arrays[i]

    baddies = np.array(baddies)
    return baddies, baddies_dict

def get_obj_of_wide_posteriors_obj(obj, params=None, round_degree = 2, use_cov_pred=True):
    """
    Identify objects whose posterior parameter uncertainties are wider than the training (prior) widths.

    This function compares the per-object posterior standard deviations (std_pred) for a selected
    set of parameters with the corresponding standard deviations computed from the training data
    (prior widths). If any selected parameter's posterior std for an object is larger than the
    rounded prior std, that object is considered "wide" and its index is returned.

    Parameters
    ----------
    obj : object
        An object containing at least the following attributes:
          - df : pandas.DataFrame
              A dataframe containing metadata and configuration for the dataset.
          - prep : indexable
              A label/index used to select a row from df (df.loc[prep, ...]).
          - std_pred : numpy.ndarray, shape (n_objects, n_parameters)
              Per-object posterior standard deviations for each parameter.
        The function also expects an external helper `get_train_data(df, prep)` to be available
        in the calling scope; it is used to obtain the training data for computing prior widths.
    params : array-like of int or None, optional
        Indices of the model parameters (relative to the `learning_params` entry in df) to check.
        If None (default), all parameters are checked (0..n_params-1 where n_params is read from
        df.loc[prep, 'n_params']).
    round_degree : int, optional
        Number of decimal places to which the prior (training) standard deviations are rounded
        before comparison. Default is 2.
    use_cov_pred : bool, optional
        Placeholder flag (default True). The current implementation does not use this flag;
        std_pred is always taken from obj.std_pred. Provided for API compatibility.

    Returns
    -------
    numpy.ndarray
        1D array of integer indices (relative to the ordering in obj.std_pred) identifying objects
        that have at least one selected parameter whose posterior std exceeds the rounded prior std.
        If no objects meet the criterion, an empty numpy array is returned.

    Notes
    -----
    - The function computes training/prior widths by calling get_train_data(df, prep) and then
      taking the std() of the selected learning parameters. It expects that df.loc[prep, 'learning_params']
      yields an iterable of parameter names that index into the training data columns.
    - The comparison performed is strictly greater (posterior_std > rounded_prior_std).
    - A dictionary of boolean arrays per object (`baddies_dict`) is constructed in the original code
      but not returned; only the indices of offending objects are returned.

    Raises
    ------
    KeyError, IndexError, TypeError
        May be raised if expected dataframe keys/columns or array shapes are missing or incompatible,
        or if `params` contains invalid indices.

    Example
    -------
    >>> # assuming `obj` has the required attributes and get_train_data is available:
    >>> bad_indices = get_obj_of_wide_posteriors_obj(obj, params=[0,2,4], round_degree=3)
    >>> # bad_indices is an array like: array([0, 5, 13])
    """
    """"""
    df = obj.df
    prep = obj.prep
    if params is None:
        params=np.arange(df.loc[prep, 'n_params'])
    train_data = get_train_data(df, prep)
    learning_params = np.array(df.loc[prep, 'learning_params'])[params]
    # print(learning_params)
    # if not use_cov_pred:
    #     print('using std_pred')
    std_pred = obj.std_pred[:, params]
    # if use_cov_pred:
        # print('using cov_pred!')
    # std_pred = np.array([np.sqrt(np.diag(obj.cov_pred[i, :, :])) for i in range(obj.num_obj)])
    boolean_arrays = std_pred[:, :]>np.array(np.round(train_data[learning_params].std(), round_degree))
    baddies = []
    baddies_dict = {}
    # for every object
    for i in range(len(boolean_arrays)):
        # if the posterior width is wider than the prior width for one or more parameters
        if sum(boolean_arrays[i])>=1:
            # append the object to the "baddies" list
            baddies.append(i)
            baddies_dict[i] = boolean_arrays[i]

    baddies = np.array(baddies)
    return baddies

from paltas.Analysis import loss_functions
#dear god we are going to try to make a results df

def make_results_df_without_training(index_list, names, weights_files, mode_of_stopping,
                    loss_type_list, train_folders, test_folders, colors,learned_params=None, nparams_learned=None,absolute_test_folders=None):
    
    assert len(index_list)==len(weights_files)
    assert len(index_list)==len(loss_type_list)
    assert len(index_list)==len(colors)
    num_indices = len(index_list)
    # import network_predictions
    results_table = pd.DataFrame(index=index_list, columns=['path_to_train_images',
                                                                               'path_to_valid_images', 
                                                                               'path_to_test_images',
                                                                               'path_to_norms',
                                                                               'path_to_results'])
    epochs, paths_to_weights = zip(*[get_weights_files(os.path.join(scratch_dir,i), mode=mode_of_stopping) for i in weights_files])
    results_table['epochs'] = epochs
    results_table['chosen_weights'] = paths_to_weights
    paths_to_norms = [os.path.join(scratch_dir,  i, 'norms.csv') for i in weights_files]
    results_table['path_to_norms'] = paths_to_norms
    results_folders = [os.path.join(scratch_dir, i) for i in weights_files]
    results_table['path_to_results'] = results_folders
    if nparams_learned is None:
        nparams_learned = [10] * num_indices
    results_table['n_params'] = nparams_learned
    if learned_params is None:
        results_table['learning_params'] = [learning_params[:i] for i in nparams_learned]
    else:
        results_table['learning_params'] = [learned_params[:i] for i in nparams_learned]
    assert len(loss_type_list) == num_indices, "Provide a loss type for each preparation"
    results_table['loss_type'] = loss_type_list
    train_folders = [f'{folder}/batch' for folder in train_folders]
    results_table['path_to_train_images'] = [os.path.join(scratch_dir, 'generated_images','train', i) for i in train_folders]
    results_table['path_to_valid_images'] = [os.path.join(scratch_dir, 'generated_images','valid', i) for i in train_folders]
    if absolute_test_folders is None:
        results_table['path_to_test_images'] = [os.path.join(scratch_dir, 'generated_images','test',i) for i in test_folders]
    else:
        results_table['path_to_test_images'] = absolute_test_folders
    results_table['name'] = names
    results_table['color'] = colors
    return results_table
    
def make_results_df (index_list, names, weights_files, mode_of_stopping,
                    loss_type_list, train_folders, test_folders, colors, nparams_learned=None,
                     image_size=None,batch_size=None,save_df=False ,absolute_test_folders=None):
    """
    Create a results DataFrame by loading trained network weights, instantiating prediction
                    wrappers, and running predictions on test image folders.
                    This function centralizes the common boilerplate required to evaluate multiple trained
                    models (preparations). For each item in index_list it:
                    - resolves chosen model weights and epoch using get_weights_files
                    - builds filesystem paths for norms, results, and generated image folders (train/valid/test)
                    - constructs a NetworkPredictions object
                    - runs predictions on the specified test folder gen_network_predictions
                    - stores metadata, the instantiated network, and prediction arrays in a pandas.DataFrame
                    Parameters
                    ----------
                    index_list : Sequence
                            Identifiers for each preparation. These values are used as the resulting DataFrame index
                            and in saved filenames when save_df is provided.
                    names : Sequence[str]
                            Display names (one per preparation) that will be stored in the DataFrame column 'name'.
                    weights_files : Sequence[str]
                            Per-preparation folder names (relative to the global scratch_dir) containing model
                            weights and a 'norms.csv' file. Each entry is passed to get_weights_files(scratch_dir/entry, mode=...).
                    mode_of_stopping : str
                            Mode argument forwarded to get_weights_files to determine which checkpoint to choose.
                    loss_type_list : Sequence[str]
                            Loss type label for each preparation; stored in the 'loss_type' column.
                    train_folders : Sequence[str]
                            Per-preparation names of training data folders. Each entry will be appended with '/batch'
                            and then prefixed with scratch_dir/generated_images/{train,valid}/ to form train/valid paths.
                    test_folders : Sequence[str]
                            Per-preparation test folder names (relative). By default these are resolved to
                            scratch_dir/generated_images/test/{test_folders[i]} unless absolute_test_folders is provided.
                    colors : Sequence
                            Display color (one per preparation) stored in the 'color' column.
                    nparams_learned : Sequence[int], optional
                            Number of learned parameters for each preparation. If None, defaults to 10 for all
                            preparations. Used to compute the 'learning_params' slice from a global learning_params
                            sequence (learning_params must be available in the calling environment).
                    image_size : int, optional
                            Image size forwarded to NetworkPredictions when instantiating the model.
                    batch_size : int, optional
                            Batch size forwarded to NetworkPredictions when instantiating the model.
                    save_df : bool or str, optional
                            If falsy (default False) predictions are not saved to disk. If a string/path is given,
                            the function will attempt to save, for each preparation, a numpy file named
                            "{prep}_{loss_type}_network.npy" containing an array of
                            [y_test, y_pred, std_pred, prec_pred, cov_pred]. Parent directories will be created
                            if necessary.
                    absolute_test_folders : Sequence[str], optional
                            If provided, these absolute paths are used directly for the 'path_to_test_images' column
                            instead of constructing paths under scratch_dir/generated_images/test/.
                    Returns
                    -------
                    pandas.DataFrame
                            A DataFrame indexed by index_list with the following columns (not exhaustive):
                            - path_to_train_images, path_to_valid_images, path_to_test_images : str
                                Filesystem paths to image folders used for train/valid/test.
                            - path_to_norms : str
                                Path to the norms.csv file for each preparation (scratch_dir/<weights_file>/norms.csv).
                            - path_to_results : str
                                The results folder (scratch_dir/<weights_file>).
                            - epochs : int
                                Epoch number selected by get_weights_files.
                            - chosen_weights : str
                                Path to the selected model weights file.
                            - n_params : int
                                Number of parameters learned for the preparation.
                            - learning_params : sequence
                                Slice of the global learning_params up to n_params for that preparation.
                            - loss_type : str
                                Loss label used for bookkeeping.
                            - trained_network : NetworkPredictions
                                Instantiated network wrapper for running predictions.
                            - y_test, y_pred, std_pred, prec_pred, cov_pred : numpy arrays or None
                                Prediction outputs produced by the network.
                            - name, color : metadata fields copied from the corresponding arguments.
                    Side effects and requirements
                    -----------------------------
                    - The function relies on the following objects/functions being available in the calling
                        environment or importable at runtime:
                            - scratch_dir (global or imported variable) used as a base path for weights and generated images
                            - learning_params (sequence) used to slice learning parameters by nparams_learned
                            - get_weights_files(path, mode=...) function to return (epoch, path_to_weights)
                            - NetworkPredictions class providing gen_network_predictions
                    - The function will load model weights and may perform I/O/compute heavy operations when
                        instantiating networks and running predictions. Expect this to be slow and to allocate
                        GPU/CPU resources as required by NetworkPredictions.
                    - It captures stdout/stderr emitted during network prediction calls (via io.capture_output)
                        so that notebook output is suppressed by default.
                    Errors
                    ------
                    - AssertionError is raised if the lengths of index_list, weights_files, loss_type_list, or
                        colors do not match (function expects exactly one entry per index).
                    - Other errors may propagate from get_weights_files, NetworkPredictions initialization, or
                        the prediction functions when files or weights are missing or incompatible.
                    Example
                    -------
                    >>> # minimal usage (assumes appropriate globals and modules are available)
                    >>> df = make_results_df(
                    ...     index_list=['prepA', 'prepB'],
                    ...     names=['A', 'B'],
                    ...     weights_files=['weightsA', 'weightsB'],
                    ...     mode_of_stopping='best',
                    ...     loss_type_list=['mse', 'mse'],
                    ...     train_folders=['trainA', 'trainB'],
                    ...     test_folders=['testA', 'testB'],
                    ...     colors=['red', 'blue'],
                    ...     nparams_learned=[6, 10],
                    ...     image_size=64,
                    ...     batch_size=32,
                    ...     save_df='/path/to/save'
    """
    assert len(index_list)==len(weights_files)
    assert len(index_list)==len(loss_type_list)
    assert len(index_list)==len(colors)
    num_indices = len(index_list)
    import network_predictions
    results_table = pd.DataFrame(index=index_list, columns=['path_to_train_images',
                                                                               'path_to_valid_images', 
                                                                               'path_to_test_images',
                                                                               'path_to_norms',
                                                                               'path_to_results'])
    
    epochs, paths_to_weights = zip(*[get_weights_files(os.path.join(scratch_dir,i), mode=mode_of_stopping) for i in weights_files])
    results_table['epochs'] = epochs
    results_table['chosen_weights'] = paths_to_weights
    paths_to_norms = [os.path.join(scratch_dir,  i, 'norms.csv') for i in weights_files]
    results_table['path_to_norms'] = paths_to_norms
    results_folders = [os.path.join(scratch_dir, i) for i in weights_files]
    results_table['path_to_results'] = results_folders
    if nparams_learned is None:
        nparams_learned = [10] * num_indices
    results_table['n_params'] = nparams_learned
    results_table['learning_params'] = [learning_params[:i] for i in nparams_learned]
    assert len(loss_type_list) == num_indices, "Provide a loss type for each preparation"
    results_table['loss_type'] = loss_type_list
    train_folders = [f'{folder}/batch' for folder in train_folders]
    results_table['path_to_train_images'] = [os.path.join(scratch_dir, 'generated_images','train', i) for i in train_folders]
    results_table['path_to_valid_images'] = [os.path.join(scratch_dir, 'generated_images','valid', i) for i in train_folders]
    if absolute_test_folders is None:
        results_table['path_to_test_images'] = [os.path.join(scratch_dir, 'generated_images','test',i) for i in test_folders]
    else:
        results_table['path_to_test_images'] = absolute_test_folders

    results_table['trained_network'] = None
    for prep in results_table.index:
        results_table.loc[prep, 'trained_network'] = network_predictions.NetworkPredictions(
            path_to_model_weights=results_table.loc[prep, 'chosen_weights'],
            path_to_model_norms=results_table.loc[prep, 'path_to_norms'],
            learning_params=results_table.loc[prep, 'learning_params'],
            loss_type=results_table.loc[prep, 'loss_type'],
            model_type='xresnet34',
            norm_type='norm', image_size=image_size,batch_size=batch_size)
    
    # trained.model is gonna take all the inputs, do all the calculations and gives you output on your test object
    # it is the trained nn
    # precision is inverse of covariance matrix
    results_table['y_test'] = None
    results_table['y_pred'] = None
    results_table['std_pred'] = None
    results_table['prec_pred'] = None
    results_table['cov_pred'] = None

    for prep in results_table.index:
        prep_row = results_table.loc[prep]
        network = prep_row.loc['trained_network']
        test_file = prep_row.loc['path_to_test_images']
        with io.capture_output() as captured:
            # if not phil_func:

            prep_row.loc['y_test'], prep_row.loc['y_pred'], prep_row.loc['std_pred'], prep_row.loc['prec_pred'], prep_row.loc['cov_pred'] = network.gen_network_predictions(test_file,samples=False,shuffle=False)
            results_table.loc[prep] = prep_row
                # write_preds_to_h5(self,write_path,y_pred,std_pred,prec_pred
            # if phil_func:
                
            #     prep_row.loc['y_test'], prep_row.loc['y_pred'], prep_row.loc['std_pred'], prep_row.loc['prec_pred'] = network.get_network_predictions_phil(test_folder = test_file,
            #                 is_h5=False,N_max=None,
            #                 diag_cov = False,
            #                 specific_file_with_no_ground_truth=False,return_prec=True)
            #     results_table.loc[prep] = prep_row
            

    results_table['name'] = names
    results_table['color']=colors
    if save_df:
        for prep in results_table.index:
            res_arr = results_table.loc[prep, ['y_test', 'y_pred', 'std_pred', 'prec_pred', 'cov_pred']].to_numpy()
            
            file_name = f"{prep}_{results_table.loc[prep, 'loss_type']}_network.npy"
            save_path = os.path.join(save_df, file_name)
            try:
                np.save(save_path, res_arr)
            except:
                os.makedirs(os.path.dirname(save_path))
                np.save(save_path, res_arr)
    return results_table


### A SINGLE PREP: load in train data
def plot_train_test(results_df, prep, file_num=[1,2,3,4,5], params=learning_params, labels=None,obj=None):
    train_data = get_train_data(results_df, prep, file_num = file_num)
    test_data = make_analysis_table(results_df, prep,obj=obj)
    params_is_learning_params = True
    for param in params:
        if param not in learning_params:
            params_is_learning_params = False
    if labels is None:
        if params_is_learning_params:
            labels = [labels_dict[i] for i in params]
        else:
            labels =[re.split('parameters_', i)[1] for i in params]
    make_contour([train_data[params], test_data[params]],
                 labels=labels,
                 categories=["Training Data", "Test Data"],
                 colors=['gainsboro','red'],
                 range_for_bin=True,
                 show_correlation=False)
    
    
def HI_results(results_df, prep_list, n_emcee_samps,params_list=None, 
               delete_bad=False,truncate_width=False,prior='uniform',
              chains_filepath=None,):
    num_preps = len(prep_list)
    results_df = results_df.loc[prep_list]
    categories = np.array(results_df['name'])
    cat_to_col = dict(zip(categories, np.array(results_df['color'])))
    legend_elements = []
    for cat in categories:
        legend_elements.append(Patch(facecolor="w", edgecolor=cat_to_col[cat], label=cat))
    if params_list is None:
        params_list = [range(10)]*num_preps
    prep_to_params_list=dict(zip(prep_list, params_list))
    for prep in prep_list:
        params_list=np.array(prep_to_params_list[prep])
        n_params=len(params_list)
        print(prep,params_list)
        learning_params = np.array(results_df.loc[prep, 'learning_params'])[params_list]
        train_data = get_train_data(results_df, prep)[learning_params]
        

        mu_train = np.array(train_data.mean(), dtype=np.float32)
        std_train = np.array(train_data.std(), dtype=np.float32)
        num_obj = len(np.array(results_df.loc[prep, 'y_pred'], dtype=np.float32)[:, :n_params])
        if delete_bad:
            bad_obj_index=get_obj_of_wide_posteriors(results_df, prep)
            obj_index=np.array([i for i in range(num_obj) if i not in bad_obj_index])
        else:
            obj_index=np.arange(num_obj)
        y_pred = np.array(results_df.loc[prep, 'y_pred'], dtype=np.float32)[obj_index, :n_params]
        prec_pred = np.array(results_df.loc[prep, 'prec_pred'], dtype=np.float32)[obj_index, :n_params, :n_params]
        print(prec_pred.shape)
        test_data = np.array(results_df.loc[prep, 'y_test'], dtype=np.float32)[obj_index, :n_params]
        mu_test = np.mean(test_data, axis=0)
        std_test = np.std(test_data, axis=0)
        lower = np.concatenate((mu_test - 1*std_test, np.ones(n_params)*0.01))
        upper = np.concatenate((mu_test + 1*std_test, std_train))
        print('Train mean: ', mu_train)
        print('Train std: ', std_train)
        print('Test mean: ', mu_test)
        print('Test std: ', std_test)
        HI_obj = bhi.NetworkHierarchicalInference(y_train = mu_train,
                                       std_train = std_train,
                                       y_pred_list = [y_pred],
                                       prec_pred_list = [prec_pred],
                                       hypermodel_type = 'regular',
                                       method = 'analytical',
                                       sigmas_log_uniform = False,
                                       n_emcee_samps = n_emcee_samps,
                                       n_params = n_params,
                                       initial_state_bounds=(lower, upper),
                                        n_walkers=40,
                                        type_of_prior=prior
                                       )
        if chains_filepath is not None:
            file_path = prep+'_'+chains_filepath
        else:
            file_path= None
        chain_list = HI_obj.run_HI(file_path)   
        
    return HI_obj, chain_list




class prepRes:
        """
    Lightweight container that loads saved network prediction results and prepares
    a small summary dataframe for downstream analysis or plotting.
    Parameters
    ----------
    loss : str
        Identifier for the loss used during training (used to locate the saved
        results file and to annotate the summary dataframe).
    prep : str
        Preprocessing or experiment identifier (used to locate the saved results
        file and as an index in the resulting dataframe).
    n_params : int
        Number of physical/model parameters predicted by the network.
    name : str
        Human-readable display name for the experiment (used in the dataframe).
    color : str
        Color string for plotting/legend purposes (stored in the dataframe).
    train_image : str
        Path or identifier for the training image/folder used to build the
        dataframe entry.
    test_image : str
        Path or identifier for the test image/folder used to build the
        dataframe entry.
    weights_path : str
        Filepath to the trained network weights (used when constructing the
        dataframe describing the run).
    save_results : str
        Directory where the network prediction results are saved. The class will
        attempt to load a numpy file named "{prep}_{loss}_network.npy" from this
        directory.
    Attributes
    ----------
    prep : str
        See parameter.
    loss : str
        See parameter.
    results : numpy.ndarray
        Loaded numpy array from the saved results file. Expected layout:
        results[0] -> y_test
        results[1] -> y_pred
        results[2] -> std_pred
        results[3] -> prec_pred
        results[4] -> cov_pred
        (The exact contents and shapes are determined by the code that wrote
        the file; the class exposes convenient properties to access these.)
    name : str
        See parameter.
    color : str
        See parameter.
    df : pandas.DataFrame
        Summary dataframe produced by make_results_df_without_training using the
        provided metadata (prep, name, loss, file paths, color, n_params, ...).
    Properties
    ----------
    y_test : ndarray
        Ground-truth/target parameter array for the test set. Expected shape
        (N_objects, N_params).
    y_pred : ndarray
        Predicted parameter means for the test set. Expected shape
        (N_objects, N_params).
    std_pred : ndarray
        Predicted standard deviations for each parameter. Expected shape
        commonly (N_objects, N_params) when per-parameter uncertainties are stored.
    prec_pred : ndarray or None
        Predicted precision information (inverse variances or precision matrices).
        Shape/format is model-dependent: could be (N_objects, N_params) for a
        diagonal precision per parameter or (N_objects, N_params, N_params) for a
        full precision matrix. May be None if not available.
    cov_pred : ndarray or None
        Predicted covariance information. Shape/format mirrors prec_pred: either
        diagonal form (N_objects, N_params) or full covariance matrices
        (N_objects, N_params, N_params). May be None if not available.
    num_obj : int
        Number of test objects (N_objects). Derived from y_test.shape[0].
    num_param : int
        Number of parameters predicted per object (N_params). Derived from
        y_test.shape[1].
    Raises
    ------
    FileNotFoundError, OSError, ValueError
        If the expected results file cannot be found or the loaded array does not
        have the expected layout/length, numpy.load or subsequent indexing may
        raise errors. Consumers should handle or validate the loaded data before
        downstream use.
    Notes
    -----
    - This class assumes numpy has been imported as np and that the helper
      function make_results_df_without_training is available in the execution
      environment.
    - The exact internal layout and shapes of results[...] are governed by the
      code that produced the "{prep}_{loss}_network.npy" file; the properties here
      provide a consistent, documented interface to that data.
    """
    def __init__(self, loss, prep, n_params, l_params, name, color, train_image, test_image, weights_path, save_results,mode_of_stopping='early_stopping'):
        self.prep = prep
        self.loss = loss
        self.results = np.load(f'{save_results}/{self.prep}_{self.loss}_network.npy', allow_pickle=True)
        self.name = name
        self.color = color
        self.df = make_results_df_without_training(
            index_list=[prep],
            names=[name],
            weights_files=[weights_path],
            mode_of_stopping=mode_of_stopping,
            loss_type_list=[loss],
            train_folders=[train_image],
            test_folders=[test_image],
            colors=[color],
            nparams_learned=[n_params],
            learned_params=l_params,
        ) 
    
    @property
    def y_test(self):
        return self.results[0]

    @property
    def y_pred(self):
        return self.results[1]    
    
    @property
    def std_pred(self):
        return self.results[2]
    
    @property
    def prec_pred(self):
        return self.results[3]
    
    @property
    def cov_pred(self):
        return self.results[4]
    
    @property
    def num_obj(self):
        return self.y_test.shape[0]

    @property
    def num_param(self):
        return self.y_test.shape[1]