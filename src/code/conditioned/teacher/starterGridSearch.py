from TrainingGridDK1 import trainDK1

"""
main script

"""
USER = "PC"

DK = 'DK_'

# number of epochs
EPOCHS = 1000
# number of parameters
PARAMETER_NUMBER = 0
# batch size
BATCH_SIZE = 8
MINI_BATCH_SIZE = 2048
# initial learning rate
LR = 3e-4
INFERENCE = False

print('Welcome back ', USER)


# the directory in which datasets are stored
data_dir = '../../../datasets'
# where to store the results ++
model_save_dir = '../../../models/teachers'

# name of the model to be used
model = 'LSTM_'
# name of dataset to be used
datasets = ["drdrive_cond_dk"]

units = [[8, 16, 32, 64, 32, 16, 8]]

for dataset in datasets:
    # we are a teacher
    dataset_train = dataset
    print("######### Preparing for Teacher training #########")
    print("\n")

    trainDK1(data_dir=data_dir,
             save_folder=f'LSTM_{dataset}_teacher',
             model_save_dir=model_save_dir,
             dataset_train=dataset_train,
             dataset_test=dataset,
             batch_size=BATCH_SIZE,
             mini_batch_size=MINI_BATCH_SIZE,
             learning_rate=LR,
             epochs=EPOCHS,
             model=model,
             inference=INFERENCE)
