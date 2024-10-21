from InferingDatasets_DK import trainDK1


"""
main script

"""
USER = "RIC"
# USER = "ALE"
# USER = "PC"


DK = 'DK1_'
print('DK1 phase')

# number of conditioning parameters
CONDITIONING = 1

# number of epochs
EPOCHS = 300
# batch size
BATCH_SIZE = 2400

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
dataset = "DrDrive_DK"  # 'CL1B_DK'  #

# trials = [[8, 16, 32, 64, 32, 16, 8]]

# we are a teacher
name = '_teacher_conditioned'
print("######### Preparing for creating the Datasets #########")
print("\n")

trainDK1(data_dir=data_dir,
         save_folder=DK + model+dataset + name,
         model_save_dir=model_save_dir,
         dataset_test=dataset,
         batch_size=BATCH_SIZE,
         epochs=EPOCHS,
         model=model,
         trial=[8, 16, 32, 64, 32, 16, 8],
         conditioning_size=CONDITIONING)
