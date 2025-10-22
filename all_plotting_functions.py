import numpy as np
import matplotlib.pyplot as pl
import glob
import h5py
from tqdm import tqdm
import matplotlib.image as mpimg    
from astropy.visualization import ImageNormalize,SqrtStretch,LinearStretch,LogStretch,MinMaxInterval,ZScaleInterval
'''
All functions in this file were written by Phil Holloway @P-1884.
'''
def plot_collage(image_folder: str,N_rows=2,N_cols=2,vmin=None,vmax=None,stretch=None,idx_list = None,interval=None,\
                 figsize=None,tight_layout=False,title=None,fontsize=18,h5_filename = None,cmap=None,return_fig = False,ax=None,seed=1):
    assert interval in [None,'zscale','minmax']
    assert stretch in [None,'sqrt','linear','log']
    assert (interval is None and stretch is None) or (interval in ['zscale','minmax'] and stretch in ['sqrt','linear','log'])
    if stretch is None or interval is None:
        print(f'Will use vmin/vmax of {(vmin,vmax)} and ignore interval/stretch values')
    else:
        print(f'Will use interval/stretch of {(interval,stretch)} and vmin/vmax of {(vmin,vmax)}')
    np.random.seed(seed)
    N_images = int(N_rows*N_cols)
    image_npy_list = glob.glob(f'{image_folder}/*.npy')
    image_h5_list = glob.glob(f'{image_folder}/*.h5')
    image_jpeg_list = glob.glob(f'{image_folder}/*.jpeg')
    if len(image_npy_list)>0:
        print('Plotting from npy files')
        if idx_list is None:
            random_indx = np.random.choice(np.arange(len(image_npy_list)),replace=False,size=N_images)
        else:
            random_indx = idx_list
        image_list = [np.load(image_npy_list[random_indx[n_im]]) for n_im in range(N_images)]
    elif len(image_h5_list)>0:
        print('Plotting from h5 files')
        if h5_filename is None: assert len(image_h5_list)==1 #There should only be one h5 file in the folder
        else: image_h5_list = [h5_filename]
        with h5py.File(image_h5_list[0],'r') as f0:
            number_of_files = f0['data'].shape[0]
            h5_file_array = f0['data'][()]
            if idx_list is None:
                random_indx = np.random.choice(np.arange((number_of_files)),replace=False,size=N_images)
            else:
                random_indx = idx_list
            image_list = [h5_file_array[random_indx[n_im]] for n_im in tqdm(range(N_images))]
    elif len(image_jpeg_list)>0:
        print('Plotting from jpeg files')
        if idx_list is None:
            random_indx = np.random.choice(np.arange(len(image_jpeg_list)),replace=False,size=N_images)
        else:
            random_indx = idx_list
        print(image_jpeg_list[random_indx[0]])
        image_list = [mpimg.imread(image_jpeg_list[random_indx[n_im]]) for n_im in tqdm(range(N_images))]
    if figsize is None: figsize = (N_cols,N_rows)
    if not return_fig: fig,ax = pl.subplots(N_rows,N_cols,figsize=figsize)
    for n_im in range(N_images):
        x = n_im%N_rows
        y = np.floor(n_im/N_rows).astype('int')
        if stretch is None or interval is None:
            ax[x,y].imshow(image_list[n_im],vmin=vmin,vmax=vmax)
        else: 
            norm_dict = {'minmax':MinMaxInterval,'zscale':ZScaleInterval,'sqrt':SqrtStretch,'linear':LinearStretch,'log':LogStretch}
            norm = ImageNormalize(image_list[n_im], interval=norm_dict[interval](),
                                stretch=norm_dict[stretch](),vmin=vmin,vmax=vmax)
            print(image_list)
            ax[x,y].imshow(image_list[n_im],norm=norm,cmap=cmap)
        #Remove axis ticks:
        ax[x,y].xaxis.set_tick_params(labelbottom=False)
        ax[x,y].yaxis.set_tick_params(labelleft=False)
        # Hide X and Y axes tick marks
        ax[x,y].set_xticks([])
        ax[x,y].set_yticks([])
    if title is not None: pl.suptitle(title,fontsize=fontsize,fontweight='bold')
    if tight_layout: pl.tight_layout()
    if return_fig: return ax, image_list
    pl.show()


def obtain_collage(image_folder: str,N_images = 10,idx_list = None,h5_filename = None,seed=1):
    image_npy_list = glob.glob(f'{image_folder}/*.npy')
    image_h5_list = glob.glob(f'{image_folder}/*.h5')
    image_jpeg_list = glob.glob(f'{image_folder}/*.jpeg')
    if len(image_npy_list)>0:
        print('Plotting from npy files')
        if idx_list is None:
            random_indx = np.random.choice(np.arange(len(image_npy_list)),replace=False,size=N_images)
        else:
            random_indx = idx_list
        image_list = [np.load(image_npy_list[random_indx[n_im]]) for n_im in range(N_images)]
    elif len(image_h5_list)>0:
        print('Plotting from h5 files')
        if h5_filename is None: assert len(image_h5_list)==1 #There should only be one h5 file in the folder
        else: image_h5_list = [h5_filename]
        with h5py.File(image_h5_list[0],'r') as f0:
            number_of_files = f0['data'].shape[0]
            h5_file_array = f0['data'][()]
            if idx_list is None:
                random_indx = np.random.choice(np.arange((number_of_files)),replace=False,size=N_images)
            else:
                random_indx = idx_list
            image_list = [h5_file_array[random_indx[n_im]] for n_im in tqdm(range(N_images))]
    elif len(image_jpeg_list)>0:
        print('Plotting from jpeg files')
        if idx_list is None:
            random_indx = np.random.choice(np.arange(len(image_jpeg_list)),replace=False,size=N_images)
        else:
            random_indx = idx_list
        print(image_jpeg_list[random_indx[0]])
        image_list = [mpimg.imread(image_jpeg_list[random_indx[n_im]]) for n_im in tqdm(range(N_images))]


    return image_list