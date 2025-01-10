from TrainingDKStudent import trainDK1


"""
main script

"""
USER = "RIC"
#USER = "ALE"
#USER = "PC"

DK = 'DK1_'
print('DK1 phase')

# number of epochs
EPOCHS = 1000

# batch size
BATCH_SIZE = 8
MINI_BATCH_SIZE = 2048
# initial learning rate
LR = 3e-4
INFERENCE = False

print('Welcome back ', USER)

if USER == 'ALE':
    # the directory in which datasets are stored
    data_dir = 'C:\\Users\\aleks\\Documents\\GitHub\\KnowledgeDistillationVA\\Datasets'
    # where to store the results ++
    model_save_dir = 'C:\\Users\\aleks\\Documents\\GitHub\\KnowledgeDistillationVA\\TrainedModels\\DK2'
elif USER == 'RIC':
    # the directory in which datasets are stored
    data_dir = 'C:\\Users\\riccarsi\\OneDrive - Universitetet i Oslo\\Datasets\\DK' # Riccardo's folder
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
#dataset = "DrDrive_DK"
#dataset = "CL1B_DK"
datasets = ["ht1-", "muff-"] # 'CL1B_DK'  #
datasets = ["DrDrive_DK"] # 'CL1B_DK'  #


units = [8, 16, 32, 64]
#units = [8]
name = '_student_self_taught_esr'

for dataset in datasets:
    dataset_train = dataset

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
                 mini_batch_size=MINI_BATCH_SIZE,
                 learning_rate=LR,
                 units=unit,
                 epochs=EPOCHS,
                 model=model,
                 inference=INFERENCE)
