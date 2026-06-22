import os

batch_size = 1024
# The number of epochs to train for
n_epochs = 150
# The size of the images in the training set
img_size = (33, 33, 1)
# A random seed to use
random_seed = 2
# The list of learning parameters to use
learning_params = ['main_deflector_parameters_theta_E',
	'main_deflector_parameters_gamma1','main_deflector_parameters_gamma2',
	'main_deflector_parameters_gamma','main_deflector_parameters_e1',
	'main_deflector_parameters_e2','main_deflector_parameters_center_x',
	'main_deflector_parameters_center_y']
# Which parameters to consider flipping
flip_pairs = None
# Which terms to reweight
weight_terms = None
# The path to the folder containing the npy images
# for training
npy_folders_train = ['/pscratch/sd/v/vpadma/generated_images/train/nolens-deconv-lownoise200_batch1/',
                     '/pscratch/sd/v/vpadma/generated_images/train/nolens-deconv-lownoise200_batch2/',
                     '/pscratch/sd/v/vpadma/generated_images/train/nolens-deconv-lownoise200_batch3/',
                     '/pscratch/sd/v/vpadma/generated_images/train/nolens-deconv-lownoise200_batch4/',
                    '/pscratch/sd/v/vpadma/generated_images/train/nolens-deconv-lownoise200_batch5/']
# The path to the tf_record for the training images
tfr_train_paths = [
	os.path.join(path,'data.tfrecord') for path in npy_folders_train]
metadata_paths_train = [
	os.path.join(path,'metadata.csv') for path in npy_folders_train]
# The path to the folder containing the npy images for validation
npy_folder_val = ('/pscratch/sd/v/vpadma/generated_images/valid/nolens-deconv-lownoise200_batch1/')
# The path to the tf_record for the validation images
tfr_val_path = os.path.join(npy_folder_val,'data.tfrecord')
# The path to the training metadata
# The path to the validation metadata
metadata_path_val = os.path.join(npy_folder_val,'metadata.csv')
# The path to the csv file to read from / write to for normalization
# of learning parameters.
input_norm_path = os.path.join('/pscratch/sd/v/vpadma/full/nolens-deconv-lownoise6_results', 'norms.csv')
# The detector kwargs to use for on-the-fly noise generation
kwargs_detector = None
# Whether or not to normalize the images by the standard deviation
norm_images = True
# A string with which loss function to use.
loss_function = 'full'
# A string specifying which model to use
model_type = 'xresnet34'
# A string specifying which optimizer to use
optimizer = 'Adam'
# Where to save the model weights
model_weights = ('/pscratch/sd/v/vpadma/full/nolens-deconv-lownoise6_results/_{epoch:02d}-{val_loss:.2f}.h5')
model_weights_init = None
# The learning rate for the model
# learning_rate = starting_rate * (decay_rate ** (num_epochs * num_images_total) / (batch_size * steps_per_decay)
# learning_rate = 1e-4 * (0.98**(50 * 1e5) / (1024 * 97))
learning_rate = 1e-4
# Whether or not to use random rotation of the input images
random_rotation = True
# Only train the head
train_only_head = False
# number of steps after which you change the decay rate
# csv path
csv_path = '/pscratch/sd/v/vpadma/full/nolens-deconv-lownoise6_results/losses.csv'