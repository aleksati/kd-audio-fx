from TrainingGridDK1 import trainDK1


"""
main script

"""
USER = "RIC"
#USER = "ALE"


DK = 'DK1_'
print('DK1 conditioning phase')

# number of conditioning parameters
CONDITIONING = 1

# number of epochs
EPOCHS = 1

# batch size
BATCH_SIZE = 2400
# initial learning rate
LR = 3e-4

INFERENCE = False

print('Welcome back ', USER)

if USER == 'ALE':
    # the directory in which datasets are stored
    data_dir = 'C:\\Users\\aleks\\Documents\\GitHub\\KnowledgeDistillationVA\\Datasets'
    # where to store the results ++
    model_save_dir = 'C:\\Users\\aleks\\Documents\\GitHub\\KnowledgeDistillationVA\\TrainedModels\\DK2_NoConditioning'
elif USER == 'RIC':
    # the directory in which datasets are stored
    # Riccardo's folder
    data_dir = 'C:\\Users\\riccarsi\\OneDrive - Universitetet i Oslo\\Datasets\\DK'
    # where to store the results
    model_save_dir = '../../../'  # Riccardo's folder
elif USER == 'PC':
    # the directory in which datasets are stored
    data_dir = '../../Files'
    # where to store the results ++
    model_save_dir = '../../TrainedModels'  # Riccardo's folder

# name of the model to be used
model = 'LSTM_'

# name of dataset to be used
dataset = "DrDriveCond_DK"  # 'CL1B_DK'  #
dataset_train = dataset

# trials = [[2, 4, 8, 16, 8, 4, 2],
# [2, 4, 8, 16, 32, 16, 8, 4, 2],
# [2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2],
# [8, 16, 32, 64, 32, 16, 8],
# [32, 32, 32, 32, 32, 32],
# [16, 32, 64, 32, 16]]
trials = [[8, 16, 32, 64, 32, 16, 8]]

for trial in trials:
    # we are a teacher
    # UNITS = 64
    dataset_train = dataset
    name = '_teacher_conditioned'
    print("######### Preparing for Teacher training #########")
    print("\n")

    trainDK1(data_dir=data_dir,
             save_folder=DK + model+dataset+str(trial) + name,
             model_save_dir=model_save_dir,
             dataset_train=dataset_train,
             dataset_test=dataset,
             batch_size=BATCH_SIZE,
             learning_rate=LR,
             trial=trial,
             epochs=EPOCHS,
             model=model,
             conditioning_size=CONDITIONING,
             inference=INFERENCE)
