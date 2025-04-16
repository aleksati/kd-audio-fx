from Training import LSTM_KD_nondistilled_student
import argparse

"""
Main starter script for training an LSTM network to act as a non-distilled student for KD tasks. Can also be used to run pure inference.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Trains an LSTM network to act as a non-distilled student for KD tasks. Can also be used to run pure inference.')

    parser.add_argument('--datasets', default=["drdrive"], nargs='+', help='The names of the datasets to use')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--hidden_layer_sizes', default=[8], nargs='+', help='Hidden layer sizes (amount of units) of the LSTM network. To train multiple networks with different hidden layer sizes, simply select more than one value. for instance, "--hidden_layer_sizes 8 16 32" will train three seperate networks, each with different units.')

    parser.add_argument('--mini_batch_size', default=2048, type=int, nargs='?', help='Mini batch size.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--inference', default=False, type=bool, nargs='?', help='Flag on whether to run inference on the trained model.')

    parser.add_argument('--data_dir', default='../../../datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--model_save_dir', default='../../../models/unconditioned/teachers', type=str, nargs='?', help='Folder directory in which to store the model (and other results).')

    return parser.parse_args()


def train_student(args):    
    datasets = args.datasets
    for dataset in datasets:
        dataset_train = dataset
        units = args.hidden_layer_sizes
        for unit in units:
            print("######### Preparing training for unconditioned non-distilled student #########")
            print("\n")

            LSTM_KD_nondistilled_student(data_dir=args.data_dir,
                    save_dir=f'LSTM_{dataset}_{unit}_TESTING',
                    model_save_dir=args.model_save_dir,
                    dataset_train=dataset_train,
                    dataset_test=dataset,
                    batch_size=args.batch_size,
                    mini_batch_size=args.mini_batch_size,
                    learning_rate=args.learning_rate,
                    units=unit,
                    epochs=args.epochs,
                    inference=args.inference)


def main():
    args = parse_args()
    train_student(args)

if __name__ == '__main__':
    main()