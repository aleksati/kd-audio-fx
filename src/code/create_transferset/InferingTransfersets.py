import pickle
from UtilsForTrainings import checkpoints
from Models import create_LSTM_DK_model, create_cond_LSTM_DK_model
import matplotlib.pyplot as plt
import random
import numpy as np
from DatasetsClass import DataGeneratorPickles, DataGeneratorCondPickles
import tensorflow as tf
import os


def LSTM_KD_infer_transferset(**kwargs):
    """
      Run inference on an LSTM teacher network to generete transfer dataset for KD tasks.

      :param input_dim: the input size [int]
      :param data_dir: the directory in which dataset are stored [string]
      :param save_dir: the directory in which the models are saved/located [string]
      :param model_save_dir: the directory in which models are stored [string]
      :param dataset: name of the teacher datset to use [string]
    """

    input_dim = kwargs.get('input_dim', 1)
    model_save_dir = kwargs.get('model_save_dir', '../../../models/teachers')
    conditioning = kwargs.get('conditioning', False)
    save_dir = kwargs.get('save_dir', 'LSTM_DEVICE_teacher')
    dataset = kwargs.get('dataset', None)
    data_dir = kwargs.get('data_dir', '../../../datasets')
    mini_batch_size = kwargs.get('mini_batch_size', 2048)
    
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
    if (conditioning):
        model = create_cond_LSTM_DK_model(input_dim=1, mini_batch_size=mini_batch_size, b_size=1, stateful=True)
        
        train_gen = DataGeneratorCondPickles(data_dir, dataset + '_train.pickle', mini_batch_size=mini_batch_size, input_size=input_dim, batch_size=1)
    else:
        model = create_LSTM_DK_model(input_dim=1, mini_batch_size=mini_batch_size, b_size=1, stateful=True)

        train_gen = DataGeneratorPickles(data_dir, dataset + '_train.pickle', mini_batch_size=mini_batch_size, input_size=input_dim, batch_size=1)

    # define callbacks: where to store the weights
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(
        model_save_dir, save_dir)

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
    
    predictions = model.predict(train_gen, verbose=0)
    z = {'x': train_gen.x.reshape(1, -1),
        'y': predictions[0].reshape(1, -1),
        'z': train_gen.z,
        'y_l6': predictions[1],
        'y_true': train_gen.y.reshape(1, -1),
        'w': last_layer_weights}
  
    file_data = open(os.path.normpath(
        '/'.join([data_dir, dataset +'_transferset_train.pickle'])), 'wb')
    pickle.dump(z, file_data)
    file_data.close()
    print('end')

    # train_gen = DataGeneratorPickles(data_dir, dataset + '_test.pickle', mini_batch_size=mini_batch_size,
    #                                  input_size=input_dim, batch_size=1)
    #
    # model.reset_states()
    # predictions = model.predict(train_gen, verbose=0)
    # z = {'x': train_gen.x.reshape(1, -1),
    #      'y': predictions[0].reshape(1, -1),
    #      'z': train_gen.z,
    #      'y_l6': predictions[1],
    #      'y_true': train_gen.y.reshape(1, -1),
    #      'w': last_layer_weights}
    #
    # file_data = open(os.path.normpath(
    #     '/'.join([data_dir, 'DK_Teacher_' + dataset + '_test.pickle'])), 'wb')
    # pickle.dump(z, file_data)
    # file_data.close()

    return 42
