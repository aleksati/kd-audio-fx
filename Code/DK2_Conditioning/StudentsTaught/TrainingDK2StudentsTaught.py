from Metrics import ESR, RMSE, STFT_loss
from ModelsForStudentsTaught import create_model_LSTM_DK2
from UtilsGridSearch import filterAudio
from UtilsForTrainingsGridSearch import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from DatasetsClassDK2 import DataGeneratorPicklesTest, DataGeneratorPicklesTrain
import tensorflow as tf
import os
import sys


def trainDK2(**kwargs):
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
    learning_rate = kwargs.get('learning_rate', 1e-1)
    input_dim = kwargs.get('input_dim', 1)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    dataset_train = kwargs.get('dataset_train', None)
    dataset_test = kwargs.get('dataset_test', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', 60)
    fs = kwargs.get('fs', 48000)
    conditioning_size = kwargs.get('conditioning_size', 0)
    units = kwargs.get("units", 2)

    # set all the seed in case reproducibility is desired
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # start the timer for all the trianing script
    global_start = time.time()

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])

    # create the model
    model = create_model_LSTM_DK2(conditioning_size=conditioning_size,
                                  input_dim=1, units=units, b_size=batch_size, training=True)

    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(
        model_save_dir, save_folder)

    # create the DataGenerator object to retrieve the data in the test set
    test_gen = DataGeneratorPicklesTest(data_dir, dataset_test + '_test.pickle',
                                        input_size=input_dim, conditioning_size=conditioning_size, batch_size=batch_size)

    # if inference is True, it jump directly to the inference section without train the model
    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        # load the weights of the last epoch, if any
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")

        # create the DataGenerator object to retrive the data in the training set
        train_gen = DataGeneratorPicklesTrain(data_dir, dataset_train + '_train.pickle',
                                              input_size=input_dim, conditioning_size=conditioning_size, batch_size=batch_size)

        # the number of total training steps
        training_steps = train_gen.training_steps*30
        # define the Adam optimizer with initial learning rate, training steps
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(
            learning_rate, training_steps), clipnorm=1)

        # compile the model with the optimizer and selected loss function
        if dataset_test == 'DrDriveCond_DK':
            model.compile(loss='mae', optimizer=opt)
        elif dataset_test == 'CL1B_DK':
            model.compile(loss='mse', optimizer=opt)

        # defining the array taking the training and validation losses
        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        # counting for early stopping
        count = 0

        # training loop
        for i in range(0, epochs, 1):
            # start the timer for each epoch
            start = time.time()
            print('epochs:', i)

            # reset the model's states
            model.reset_states()
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=train_gen,
                                callbacks=callbacks)

            # store the training and validation loss
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            # if validation loss is smaller then the best loss, the early stopping counting is reset
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            # if not count is increased by one and if equal to 20 the training is stopped
            else:
                count = count + 1
                if count == 20:
                    break

            avg_time_epoch = (time.time() - start)
            sys.stdout.write(
                f" Average time/epoch {'{:.3f}'.format(avg_time_epoch / 60)} min")
            sys.stdout.write("\n")

        # write and save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs)

        # plot the training and validation loss for all the training
        loss_training = np.array(loss_training)[:i]
        loss_val = np.array(loss_val)[:i]
        plotTraining(loss_training, loss_val, model_save_dir,
                     save_folder, str(epochs))

        print("Training done")
        print("\n")

    avg_time_epoch = (time.time() - global_start)
    sys.stdout.write(
        f" Average time training{'{:.3f}'.format(avg_time_epoch / 60)} min")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # re-create the model to include last layer
    model = create_model_LSTM_DK2(
        input_dim=1, units=units, b_size=batch_size, training=False)

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")

    model.layers[-1].set_weights(train_gen.weights)
    model.layers[-2].set_weights(train_gen.weights_film)

    # reset the states before predicting
    model.reset_states()
    # predict the test set
    predictions = model.predict(test_gen, verbose=0).reshape(-1)

    # plot and render the output audio file, together with the input and target
    predictWaves(predictions, test_gen.x,  test_gen.y,
                 model_save_dir, save_folder, fs, '0')
    predictions = np.array(filterAudio(predictions), dtype=np.float32)
    y = np.array(filterAudio(test_gen.y), dtype=np.float32)

    # compute the metrics: mse, mae, esr and rmse
    mse = tf.get_static_value(
        tf.keras.metrics.mean_squared_error(y, predictions))
    mae = tf.get_static_value(
        tf.keras.metrics.mean_absolute_error(y, predictions))
    esr = tf.get_static_value(ESR(y, predictions))
    rmse = tf.get_static_value(RMSE(y, predictions))
    stft = tf.get_static_value(STFT_loss(y, predictions))

    # write and store the metrics values
    results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse, 'stft': stft}
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, save_folder + '_results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)

    return 42
