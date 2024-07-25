from Training import train

"""
main script

"""

# number of epochs
EPOCHS = 1
# number of parameters
PARAMETER_NUMBER = 0
# batch size
BATCH_SIZE = 2400
# initial learning rate
LR = 3e-4
# number of model's units
UNITS = 8

# data_dir: the directory in which datasets are stored
data_dir = '../../Files/OD/'
# name of the mdoel to be used
model = 'LSTM'


# flag to true when we train teacher
teacher = False
teaching = False
# name of dataset to be used
dataset = 'OD'
dataset_train = dataset
#[teacher, teaching]
cases = [[True, False], [False, True], [False, False]]

for case in cases:
      UNITS = 8
      teacher = case[0]
      teaching = case[1]

      if teacher:
            # we are a teacher
            UNITS = 64
            name = '_teacher'

      elif not teacher and teaching:
            # we are student but we are been taught
            dataset_train = 'Teacher_' + dataset_train
            name = '_student_taught'
      else:
            # we are student but self-taught
            name = '_student_self_taught'



      train(data_dir=data_dir,
            save_folder=model+dataset+str(UNITS) + name,
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