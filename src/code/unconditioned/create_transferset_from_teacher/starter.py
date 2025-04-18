from InferingDatasets import LSTM_KD_infer_transfer_data
import argparse

"""
Run inference on an LSTM teacher network to generete transfer dataset for KD tasks.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on an LSTM teacher network to generete transfer dataset for KD tasks.')

    parser.add_argument('--mini_batch_size', default=2048, type=int, nargs='?', help='Mini batch size.')

    parser.add_argument('--data_dir', default='../../../datasets', type=str, nargs='?', help='Folder directory in which to store the datasets.')

    parser.add_argument('--datasets', default=["drdrive"], nargs='+', help='The names of the teacher datasets to use')

    parser.add_argument('--model_save_dir', default='../../../models/unconditioned/teachers', type=str, nargs='?', help='Folder directory of the model to use.')

    return parser.parse_args()


def gen_transfer_set(args):
    datasets = args.datasets
    for dataset in datasets:
        print("######### Preparing for data/transfer set creation #########")
        print("\n")

        LSTM_KD_infer_transfer_data(data_dir=args.data_dir,
                save_dir=f'LSTM_{dataset}_teacher',
                model_save_dir=args.model_save_dir,
                dataset=dataset,
                mini_batch_size=args.mini_batch_size)


def main():
    args = parse_args()
    gen_transfer_set(args)

if __name__ == '__main__':
    main()