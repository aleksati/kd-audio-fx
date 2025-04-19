from Metrics import ESR, RMSE, STFT_loss
from Models import create_LSTM_DK_model, create_cond_LSTM_DK_model
from UtilsForTrainingsStudent import plotTraining, writeResults, checkpoints, predictWaves
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from DatasetsClass import DataGeneratorPickles, DataGeneratorCondPickles
import tensorflow as tf
import os
import sys


def LSTM_KD_nondistilled_student(**kwargs):
    """
      Trains an LSTM network to act as a non-distilled student for KD tasks. Can also be used to run pure inference.

      :param data_dir: the directory in which dataset are stored [string]
      :param save_dir: the directory in which the models are saved [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [int]
      :param model_save_dir: the directory in which models are stored [string]
      :param inference: When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool]
      :param conditioning: Flag True for conditioned training, False for unconditioned [bool]
      :param dataset: name of the datset to use [string]
      :param epochs: the number of epochs [int]
      :param input_dim: the input size [int]
      :param fs: the sampling rate [int]
    """

    batch_size = kwargs.get('batch_size', 8)
    mini_batch_size = kwargs.get('mini_batch_size', 2048)
    learning_rate = kwargs.get('learning_rate', 3e-4)
    input_dim = kwargs.get('input_dim', 1)
    model_save_dir = kwargs.get(
        'model_save_dir', '../../../models/students_non_distilled')
    save_dir = kwargs.get('save_dir', 'LSTM_DEVICE_UNITS')
    conditioning = kwargs.get('conditioning', False)
    inference = kwargs.get('only_inference', False)
    dataset_train = kwargs.get('dataset_train', None)
    dataset_test = kwargs.get('dataset_test', None)
    data_dir = kwargs.get('data_dir', '../../../datasets')
    epochs = kwargs.get('epochs', 60)
    fs = kwargs.get('fs', 48000)
    units = kwargs.get("units", 8)

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
    model = create_LSTM_DK_model(
        input_dim=1, mini_batch_size=mini_batch_size, units=units, b_size=batch_size)

    # create the model and proces the test and training data correctly
    if (conditioning):
        model = create_cond_LSTM_DK_model(
            input_dim=1, mini_batch_size=mini_batch_size, b_size=batch_size, units=units)

        test_gen = DataGeneratorCondPickles(data_dir, dataset_test + '_test.pickle',
                                            mini_batch_size=mini_batch_size, input_size=input_dim, batch_size=batch_size)

        train_gen = DataGeneratorCondPickles(data_dir, dataset_train + '_train.pickle',
                                             mini_batch_size=mini_batch_size, input_size=input_dim, batch_size=batch_size)
    else:
        model = create_LSTM_DK_model(
            input_dim=1, mini_batch_size=mini_batch_size, b_size=batch_size, units=units)

        test_gen = DataGeneratorPickles(data_dir, dataset_test + '_test.pickle',
                                        mini_batch_size=mini_batch_size, input_size=input_dim, batch_size=batch_size)

        train_gen = DataGeneratorPickles(data_dir, dataset_train + '_train.pickle',
                                         mini_batch_size=mini_batch_size, input_size=input_dim, batch_size=batch_size)

    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(
        model_save_dir, save_dir)

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

        # the number of total training steps
        # training_steps = train_gen.training_steps*30
        # define the Adam optimizer with initial learning rate, training steps
        # opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # compile the model with the optimizer and selected loss function
        model.compile(loss='mae', optimizer=opt)

        # defining the array taking the training and validation losses
        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        # counting for early stopping
        count = 0
        count2 = 0
        # training loop
        for i in range(0, epochs, 1):
            # start the timer for each epoch
            start = time.time()
            print('epochs:', i)

            # reset the model's states
            model.reset_states()
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=test_gen,
                                callbacks=callbacks)

            # store the training and validation loss
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            # if validation loss is smaller then the best loss, the early stopping counting is reset
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
                count2 = 0
            # if not count is increased by one and if equal to 20 the training is stopped
            else:
                count = count + 1
                count2 = count2 + 1

                if count2 == 5:
                    model.optimizer.learning_rate = model.optimizer.learning_rate/2
                    count2 = 0
                if count == 50:
                    break

            avg_time_epoch = (time.time() - start)
            sys.stdout.write(
                f" Average time/epoch {'{:.3f}'.format(avg_time_epoch / 60)} min")
            sys.stdout.write("\n")

        # write and save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_dir, epochs)

        # plot the training and validation loss for all the training
        loss_training = np.array(loss_training)[:i]
        loss_val = np.array(loss_val)[:i]
        plotTraining(loss_training, loss_val, model_save_dir,
                     save_dir, str(epochs))

        print("Training done")
        print("\n")

    avg_time_epoch = (time.time() - global_start)
    sys.stdout.write(
        f" Average time training{'{:.3f}'.format(avg_time_epoch / 60)} min")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")

    # reset the states before predicting
    model.reset_states()
    # predict the test set
    predictions = model.predict(test_gen, verbose=0).flatten()
    y = np.array((test_gen.y[0, :len(predictions)]), dtype=np.float32)
    x = np.array((test_gen.x[0, :len(predictions)]), dtype=np.float32)

    # plot and render the output audio file, together with the input and target
    predictWaves(predictions, x,  y, model_save_dir, save_dir, fs, '1')
    # predictions = np.array(filterAudio(predictions), dtype=np.float32)
    # y = np.array(filterAudio(test_gen.y[0, :len(predictions)]), dtype=np.float32)

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
    with open(os.path.normpath('/'.join([model_save_dir, save_dir, save_dir + '_results2.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)
    print('end')
    return 42
