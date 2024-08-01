from TrainingDK1 import trainDK1
from TrainingDK2 import trainDK2


"""
main script

"""
DK1 = False
DK2 = not DK1
if DK1:
    train = trainDK1
    DK = 'DK1_'
    print('DK1 phase')
else:
    train = trainDK2
    DK = 'DK2_'
    print('DK2 phase')

# number of epochs
EPOCHS = 60
# number of parameters
PARAMETER_NUMBER = 0
# batch size
BATCH_SIZE = 2400
# initial learning rate
LR = 3e-4

# the directory in which datasets are stored
data_dir = 'C:\\Users\\aleks\\Documents\\GitHub\\KnowledgeDistillationVA\\Datasets'
# where to store the results ++
model_save_dir = 'C:\\Users\\aleks\\Documents\\GitHub\\KnowledgeDistillationVA\\TrainedModels\\DK2'
# name of the mdoel to be used
model = 'LSTM'


# flag to true when we train teacher
teacher = False
teaching = False
# name of dataset to be used
dataset = 'CL1B_DK'  # DrDrive_DK
dataset_train = dataset
# [teacher, teaching]
cases = [[True, False], [False, True], [False, False]]
# cases = [[False, True]]

for case in cases:
    teacher = case[0]
    teaching = case[1]
    enable_second_output = False

    if teacher:
        # we are a teacher
        UNITS = 64
        name = '_teacher'
        enable_second_output = False

    elif not teacher and teaching:
        # we are student but we are been taught
        if DK2:
            enable_second_output = True
        UNITS = 8
        dataset_train = DK + 'Teacher_' + dataset_train
        name = '_student_taught'
    else:
        if DK2:
            break
        enable_second_output = False
        # we are student but self-taught
        UNITS = 8
        dataset_train = dataset
        name = '_student_self_taught'

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
          teacher=teacher,
          enable_second_output=enable_second_output,
          inference=False)
