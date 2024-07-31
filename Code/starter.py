from Training import train

"""
main script

"""

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
model_save_dir = 'C:\\Users\\aleks\\Documents\\GitHub\\KnowledgeDistillationVA\\TrainedModels'
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
# cases = [[False, True],  [False, False]]

for case in cases:
    teacher = case[0]
    teaching = case[1]

    if teacher:
        # we are a teacher
        UNITS = 64
        name = '_teacher'

    elif not teacher and teaching:
        UNITS = 8
        # we are student but we are been taught
        dataset_train = 'Teacher_' + dataset_train
        name = '_student_taught'
    else:
        # we are student but self-taught
        UNITS = 8
        dataset_train = dataset
        name = '_student_self_taught'

    train(data_dir=data_dir,
          #   save_folder=model+dataset+str(UNITS) + name,
          save_folder=model+dataset+"8-16-32-64-32-16-8" + name,
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
          inference=False)
