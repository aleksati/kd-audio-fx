from utils import validate, convert_keras_to_pytorch, save_best_keras_weights, create_neutone_model
import time

# Config
UNITS = 8
DEVICE = "ht1"
MODEL_TYPE = "distilled"

# Edit below only if necessary
NAME = f'{DEVICE}_student_{MODEL_TYPE}_{UNITS}' # model name
MODELS_DIR = "./models" # Where to store the new models
CKPT_DIR = f'../../models/students_{MODEL_TYPE}/LSTM_{DEVICE}_dk_{UNITS}/checkpoints/best/best.ckpt' # Path to best weights


if __name__ == "__main__":
    print("--------------- Step 1 ---------------")
    save_best_keras_weights(units=UNITS, model_name=NAME, models_dir=MODELS_DIR, ckpt_dir=CKPT_DIR)
    print("\n")

    # just to make sure everything is save
    # correctly before moving on. Cheap async.
    time.sleep(2)

    print("--------------- Step 2 ---------------")
    convert_keras_to_pytorch(model_name=NAME, models_dir=MODELS_DIR, units=UNITS)
    print("\n")

    time.sleep(2)

    print("--------------- Step 3 ---------------")
    validate(model_name=NAME, models_dir=MODELS_DIR, units=UNITS)
    print("\n")

    time.sleep(2)

    print("--------------- Step 4 ---------------")
    create_neutone_model(models_dir=MODELS_DIR, units=UNITS, model_name=NAME)