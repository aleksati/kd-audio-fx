import pickle
from UtilsForTrainings import checkpoints
from ModelsForCreateDatasets import create_model_LSTM_DK
import matplotlib.pyplot as plt
import random
import numpy as np
from DatasetsClassDK1 import DataGeneratorPickles
import tensorflow as tf
import os


def trainDK1(**kwargs):
    """
      :param data_dir: the directory in which dataset are stored [string]
      :param save_folder: the directory in which the models are saved [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [int]
      :param input_dim: the input size [int]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param dataset: name of the datset to use [string]
      :param epochs: the number of epochs [int]
      :param teacher: if True it is inferring the training set and store in save_folder [bool]
      :param fs: the sampling rate [int]
      :param conditioning_size: the numeber of parameters to be included [int]
    """

    batch_size = kwargs.get('batch_size', 1)
    input_dim = kwargs.get('input_dim', 1)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    dataset_test = kwargs.get('dataset_test', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    conditioning_size = kwargs.get('conditioning_size', 0)
    trial = kwargs.get("trial", [])

    # set all the seed in case reproducibility is desired
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])

    # create the model
    model = create_model_LSTM_DK(units=trial,
        input_dim=1, conditioning_size=conditioning_size, b_size=batch_size)

    # define callbacks: where to store the weights
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(
        model_save_dir, save_folder)

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        raise ValueError("Something is wrong.")

    last_layer_weights = model.layers[-1].get_weights()

    print('Saving the new dataset...')
    # create the DataGenerator object to retrieve the data in the training set
    train_gen = DataGeneratorPickles(data_dir, dataset_test + '_train.pickle',
                                     input_size=input_dim, conditioning_size=conditioning_size, batch_size=batch_size)

    predictions = model.predict(train_gen, verbose=0)
    z = {'x': train_gen.x.reshape(
        1, -1), 'y': predictions[0].reshape(1, -1), 'z': train_gen.z,
        #'y_l0': predictions[1],
        #'y_l1': predictions[2],
        #'y_l2': predictions[3],
        #'y_l3': predictions[4],
        'y_l4': predictions[5],
        'y_l5': predictions[6],
        'y_l6': predictions[7], 'w': last_layer_weights}

    file_data = open(os.path.normpath(
        '/'.join([data_dir, 'DK_Teacher__conditioned_' + dataset_test + '_train.pickle'])), 'wb')
    pickle.dump(z, file_data)
    file_data.close()

    return 42
