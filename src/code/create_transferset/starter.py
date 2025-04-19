from InferingTransfersets import LSTM_KD_infer_transferset
import argparse

"""
Run inference on an LSTM teacher network to generete transfer dataset for KD tasks.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on an LSTM teacher network to generete transfer dataset for KD tasks.')

    parser.add_argument('--mini_batch_size', default=2048, type=int, nargs='?', help='Mini batch size.')

    parser.add_argument('--data_dir', default='../../../datasets', type=str, nargs='?', help='Folder directory in which to store the datasets.')

    parser.add_argument('--input_dim', default=1, type=int, nargs='?', help='Input dimension of the training data.')

    parser.add_argument('--conditioning', default=False, type=bool, nargs='?', help='Flag True for conditioned training, False for unconditioned.')
    
    parser.add_argument('--datasets', default=["drdrive_dk"], nargs='+', help='The names of the datasets to use. For instance, "drdrive_dk" for unconditional training. For conditional training, it would be "drdrive_cond_dk".')

    parser.add_argument('--model_save_dir', default='../../../models/teachers', type=str, nargs='?', help='Folder directory of the model to use.')

    return parser.parse_args()


def gen_transfer_set(args):
    datasets = args.datasets
    for dataset in datasets:
        print("######### Preparing for transferset creation from teacher #########")
        print("\n")

        LSTM_KD_infer_transferset(data_dir=args.data_dir,
                save_dir=f'LSTM_{dataset}_teacher',
                model_save_dir=args.model_save_dir,
                input_dim=args.input_dim,
                dataset=dataset,
                conditioning=args.conditioning,
                mini_batch_size=args.mini_batch_size)


def main():
    args = parse_args()
    gen_transfer_set(args)

if __name__ == '__main__':
    main()