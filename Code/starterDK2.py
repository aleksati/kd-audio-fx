from TrainingDK2 import trainDK2
from TrainingDK2_fineTuning import trainDK2_fineTuning


"""
main script

"""
USER = "RIC"
#USER = "ALE"


####### Define what type of training
Teacher = False
Student_taught = True
Student_self_taught = False
#######


train = trainDK2
DK = 'DK2_'
print('DK2 phase')

# number of epochs
EPOCHS = 1#200
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
    model_save_dir = '../' # Riccardo's folder

# name of the mdoel to be used
model = 'LSTM'

# name of dataset to be used
dataset = "DrDrive_DK"  # 'CL1B_DK'  #
dataset_train = dataset

if Teacher:
    # we are a teacher
    UNITS = 64
    name = '_teacher'
    dataset_train = dataset
    enable_second_output = False
    print("######### Preparing for training the Teacher #########")
    print("\n")

    train(data_dir=data_dir,
          #   save_folder=model+dataset+str(UNITS) + name,
          save_folder=DK + model+dataset+"8-16-32-64-32-16-8" + name,
          model_save_dir=model_save_dir,
          dataset_train=dataset_train,
          dataset_test=dataset,
          batch_size=BATCH_SIZE,
          learning_rate=LR,
          units=UNITS,
          epochs=EPOCHS,
          model=model,
          parameter_numbers=PARAMETER_NUMBER,
          teacher=True,
          enable_second_output=enable_second_output,
          inference=INFERENCE)



if Student_taught:

    # we are student but we are been taught
    enable_second_output = True
    UNITS = 8
    dataset_train = DK + 'Teacher_' + dataset_train
    name = '_student_taught'

    print("######### Preparing for training the Student using the teaching #########")
    print("\n")

    train(data_dir=data_dir,
              #   save_folder=model+dataset+str(UNITS) + name,
              save_folder=DK + model + dataset + "8-16-32-64-32-16-8" + name,
              model_save_dir=model_save_dir,
              dataset_train=dataset_train,
              dataset_test=dataset,
              batch_size=BATCH_SIZE,
              learning_rate=LR,
              units=UNITS,
              epochs=EPOCHS,
              model=model,
              parameter_numbers=PARAMETER_NUMBER,
              teacher=False,
              enable_second_output=enable_second_output,
              inference=INFERENCE)


    print("#########  Preparing for fine tuning the Student #########")
    print("\n")
    dataset_train = dataset
    enable_second_output = False

    if not INFERENCE:

        trainDK2_fineTuning(data_dir=data_dir,
                #   save_folder=model+dataset+str(UNITS) + name,
                save_folder=DK + model + dataset + "8-16-32-64-32-16-8" + name,
                model_save_dir=model_save_dir,
                dataset_train=dataset_train,
                dataset_test=dataset,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                units=UNITS,
                epochs=EPOCHS,
                model=model,
                parameter_numbers=PARAMETER_NUMBER,
                enable_second_output=enable_second_output)

if Student_self_taught:
    # we are student but self-taught

    enable_second_output = False
    UNITS = 8
    dataset_train = dataset
    name = '_student_self_taught'
    print("######### Preparing for training the Student by self-learning #########")
    print("\n")

    train(data_dir=data_dir,
          #   save_folder=model+dataset+str(UNITS) + name,
          save_folder=DK + model+dataset+"8-16-32-64-32-16-8" + name,
          model_save_dir=model_save_dir,
          dataset_train=dataset_train,
          dataset_test=dataset,
          batch_size=BATCH_SIZE,
          learning_rate=LR,
          units=UNITS,
          epochs=EPOCHS,
          model=model,
          parameter_numbers=PARAMETER_NUMBER,
          teacher=False,
          enable_second_output=enable_second_output,
          inference=INFERENCE)
