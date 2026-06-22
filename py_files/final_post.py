### Code used to compute final posteriors in Venkatraman et al 2025

import numpy as np
from scipy.stats import multivariate_normal


np.random.seed(4)
# REQUIRED IMPORTS
# FILE MANAGEMENT
import h5py
import multiprocessing

# DATA MANIPULATION
import numpy as np

scratch_dir = '/pscratch/sd/v/vpadma'
from latils import retrieve_chains_h5, get_obj_of_wide_posteriors_obj, make_results_df_without_training, learning_params

class prepRes:
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
    

from scipy.stats import multivariate_normal
from tqdm import tqdm

# i, y_pred, cov_pred, n_lenses, n_params, sigmas, mus,
#             train_mean, train_scatter, chain, n_samps

def reweighted_lens_posteriors(i, y_pred,cov_pred,n_lenses, n_params,sigmas, mus, 
        train_mean,train_scatter,n_samps=None):
        """
        Loops through all lenses (length of y_pred) and computes weights for
            samples from the NPE that will re-weight to account for bias from
            the interim training prior
        
        Args:
            y_pred (array[float]): Shape:(n_lenses,n_params)
            prec_pred (array[float]) Shape:(n_lenses,n_params,n_params)
            train_mean (array[float]): Shape:(n_params)
            train_scatter (array[float]): Shape:(n_params)
            samps_weights_path (string): path to .h5 file to store samples & 
                weights for each lens. If None, does not save the info.
            debug (bool): If True, stops after trying one lens
            reweight_indices (list[int]): Which lenses to re-weight (i.e. could
                use whole list to inform HI, but only reweight a subset of lenses)
            check_chains (bool): If true, saves plots to make sure chains are
                moving around.
        """
        

        NPE_multivariate_sampler = multivariate_normal(mean=y_pred[i,:n_params],cov=cov_pred[i,:n_params,:n_params])
        NPE_samples = NPE_multivariate_sampler.rvs(size=int(n_samps))
        # calculate weights using chain from sampler
        weights = np.empty(np.shape(NPE_samples)[0])

        for k in tqdm(range(0,np.shape(NPE_samples)[0])):
            xi_k = NPE_samples[k,:] # 1-d array - (8,)
            exponent = -0.5*(np.sum(((xi_k - mus)/sigmas)**2,axis=1)) # 40 * 3000
            to_sum = (1/(np.product(sigmas,axis=1)))*np.exp(exponent) # 120000
            final_sum = np.sum(to_sum) # 1 value
            interim_exponent = 0.5*np.sum(((xi_k - train_mean)/train_scatter)**2)
            to_multiply = np.prod(train_scatter)*np.exp(interim_exponent)
            weights[k] = NPE_multivariate_sampler.pdf(xi_k) * final_sum * to_multiply / np.shape(mus)[0]
        return i, NPE_samples, weights

from latils import get_train_data, learning_params
def main():
    # samples_list = []
    # weights_list = []
    ALobj = prepRes('full', 'all',8,learning_params[:8],'All Light Included','rebeccapurple','0118/all','0325/all_no_single','full/0118/all', '0325_results')
    full_df = ALobj.df
    prep='all'
    wide_post = get_obj_of_wide_posteriors_obj(ALobj)
    obj_index=np.array([i for i in range(len(ALobj.y_test)) if i not in wide_post])

    y_pred= np.delete(ALobj.y_pred, wide_post, axis=0)


    cov_pred = np.delete(ALobj.cov_pred, wide_post, axis=0)
    train_data = get_train_data(full_df, prep)
    train_mean = np.array(train_data[learning_params[:8]].mean(axis=0))
    train_scatter = np.array(train_data[learning_params[:8]].std(axis=0))
    chain = retrieve_chains_h5('all_lsst_0325_uniform.h5')

    # obj_idx = np.random.choice(obj_index, 100)
    obj_idx = np.arange(len(obj_index))
    n_lenses = np.shape(y_pred)[0]
    n_params = np.shape(y_pred)[1]

    samps_weights_path = 'final_posteriors_0910.h5'
    # if samps_weights_path is not None:
    #     h5f = h5py.File(samps_weights_path, 'w')
    
    

    burnin = 3000
    chain_HI = chain[:,burnin:,:].reshape((-1,n_params*2))
    mus = chain_HI[:, :n_params]
        #print(mus.shape)
    sigmas = chain_HI[:, n_params:]
    n_samps = 5e3
    # Prepare arguments for parallel processing

    args_list = [
        (
            i, y_pred, cov_pred, n_lenses, n_params, sigmas, mus,
            train_mean, train_scatter, n_samps
        )
        for i in obj_idx
    ]

    with multiprocessing.Pool(processes=64) as pool:
        result = tqdm(pool.starmap(reweighted_lens_posteriors, args_list))
    
    if samps_weights_path is not None:
        h5f = h5py.File(samps_weights_path, 'w')
        for r in result:
            h5f.create_dataset('samples_%d'%(r[0]), data=r[1])
            h5f.create_dataset('weights_%d'%(r[0]), data=r[2])
        h5f.close()

        
    # for i in range(obj_idx):
    #     samps,weights = reweighted_lens_posteriors(i, h5f, y_pred,cov_pred,n_lenses, n_params,sigmas, mus,
    #             train_mean,train_scatter,chains_list=chain,
    #             reweight_indices=obj_idx, n_samps=5000, burnin=3000)


    
    pool.close()

if __name__ == '__main__':
    main()
    # parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
    #                            description="Create the data set and organise the file system for a new measurement",
    #                            formatter_class=ap.RawTextHelpFormatter)
    # help_lensname = "name of the lens to process"
    # help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    # help_work_dir = "name of the working directory"
    # parser.add_argument(dest='lensname', type=str,
    #                     metavar='lens_name', action='store',
    #                     help=help_lensname)
    # parser.add_argument(dest='dataname', type=str,
    #                     metavar='dataname', action='store',
    #                     help=help_dataname)
    # parser.add_argument(dest='dataname', type=str,
    #                     metavar='dataname', action='store',
    #                     help=help_dataname)
    # parser.add_argument('--dir', dest='work_dir', type=str,
    #                     metavar='', action='store', default='./',
    #                     help=help_work_dir)
    # args = parser.parse_args()
    # main(args.lensname, args.dataname, work_dir=args.work_dir)