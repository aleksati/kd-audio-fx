from Training import LSTM_KD_teacher
import argparse

"""
Main starter script for training an LSTM network to act as a teacher for KD tasks. Can also be used to run pure inference.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Trains an LSTM network to act as a teacher for KD tasks. Can also be used to run pure inference.')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--mini_batch_size', default=2048, type=int, nargs='?', help='Mini batch size.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    parser.add_argument('--conditioning', default=False, type=bool, nargs='?', help='Flag True for conditioned training, False for unconditioned.')

    parser.add_argument('--input_dim', default=1, type=int, nargs='?', help='Input dimension of the training data.')
    
    parser.add_argument('--datasets', default=["drdrive_dk"], nargs='+', help='The names of the datasets to use. For instance, "drdrive_dk" for unconditional training. For conditional training, it would be "drdrive_cond_dk".')

    parser.add_argument('--data_dir', default='../../../datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--model_save_dir', default='../../../models/teachers', type=str, nargs='?', help='Folder directory in which to store the model (and other results).')

    return parser.parse_args()


def train_teacher(args):
    datasets = args.datasets
    for dataset in datasets:
        dataset_train = dataset
        print("######### Preparing for teacher training #########")
        print("\n")

        LSTM_KD_teacher(data_dir=args.data_dir,
                model_save_dir=args.model_save_dir,
                save_dir=f'LSTM_{dataset}_teacher',
                input_dim=args.input_dim,
                dataset_train=dataset_train,
                dataset_test=dataset,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                conditioning=args.conditioning,
                only_inference=args.only_inference)


def main():
    args = parse_args()
    train_teacher(args)

if __name__ == '__main__':
    main()