from TrainingDKStudent import trainDK1


"""
main script

"""
USER = "RIC"
# USER = "ALE"


DK = 'DK1_'
print('DK1 phase')

# number of conditioning parameters
CONDITIONING = 1

# number of epochs
EPOCHS = 300
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
    # where to store the results ++
    model_save_dir = '../../../Models'  # Riccardo's folder
elif USER == 'PC':
    # the directory in which datasets are stored
    data_dir = '../../Files'
    # where to store the results ++
    model_save_dir = '../../TrainedModels'  # Riccardo's folder

# name of the model to be used
model = 'LSTM_'

# name of dataset to be used
dataset = "DrDriveCond_DK"
# dataset = "CL1B_DK"
dataset_train = dataset

units = [8, 16, 32, 64]
name = '_student_self_taught_conditioned'

for unit in units:
    # we are a student being taught
    print("######### Preparing for Student taught training #########")
    print("\n")

    trainDK1(data_dir=data_dir,
             save_folder=DK + model+dataset+str(unit) + name,
             model_save_dir=model_save_dir,
             dataset_train=dataset_train,
             dataset_test=dataset,
             batch_size=BATCH_SIZE,
             learning_rate=LR,
             units=unit,
             epochs=EPOCHS,
             model=model,
             conditioning_size=CONDITIONING,
             inference=INFERENCE)
