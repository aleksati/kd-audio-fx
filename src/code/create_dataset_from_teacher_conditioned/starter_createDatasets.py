from InferingDatasets_DK import trainDK1


"""
main script

"""
#USER = "RIC"
#USER = "ALE"
USER = "PC"


DK = 'DK_'
print('DK phase')

# number of epochs
EPOCHS = 300
# number of parameters
PARAMETER_NUMBER = 0
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
datasets = ["DrDriveCond_DK"]  # 'CL1B_DK'  #


for dataset in datasets:
    
    # we are a teacher
    name = '_teacher_conditioned_64'
    print("######### Preparing for creating the Datasets #########")
    print("\n")

    trainDK1(data_dir=data_dir,
             save_folder=DK + model+dataset + name,
             model_save_dir=model_save_dir,
             dataset=dataset,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             model=model,
             parameter_numbers=PARAMETER_NUMBER)
