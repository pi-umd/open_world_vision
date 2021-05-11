import argparse
import os
import sys

import torch
import torch.nn.parallel
import torch.nn.functional as nn_func
import torchvision.transforms as transforms
import torch.utils.data.distributed
from tqdm import tqdm

# Replace this with your data loader and edit the calls accordingly
from data_loader import EvalDataset


def get_parser():
    """"Defines the command line arguments"""
    parser = argparse.ArgumentParser(description='Open World Vision')
    parser.add_argument('--input_file', required=True,
                        help='path to a .txt/.csv file containing paths of input images in first column of each row. '
                             '\',\' will be used as a delimiter if a csv is provided. In text format, each row should'
                             ' only contain the path of an image.')
    parser.add_argument('--out_dir', required=True,
                        help='directory to be used to save the results. We will save a \',\' separated csv which will'
                             ' be named by the next argument: <exp_name> ')
    parser.add_argument('--exp_name', required=True,
                        help='unique name for this run of the evaluation')
    parser.add_argument('--model_path', required=True,
                        help='path to model file')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--accimage', action='store_true',
                        help='use if accimage module is available on the system and you want to use it')
    return parser


def run(model_path, data_file, exp_name, out_dir, accimage=False, batch_size=32, workers=4):
    """Runs the model on given data and saves the class probabilities

    Args:
        model_path (str): path to pytorch model file
        data_file (str): path to txt/csv file containing input images. If txt each line should only contain the path of
            an image. If csv,  1st column should have image paths
        exp_name (str): unique name for the experiment. Will be used to save output
        out_dir (str): path to dump output files
        accimage (bool): whether to use accimage loader. If calling this function outside this module, please make sure
            that accimage is importable in your python env
        batch_size (int): batchsize for model
        workers (int): no. of workers to be used in dataloader
    """
    try:
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            # Replace this with your data-loader
            test_set = EvalDataset(
                data_file=data_file,
                accimage=accimage,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]),
                    ]),
                header=True,
            )

            img_path_list = test_set.data_list
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True)

            img_idx_list = list()
            output_list = list()
            for img_idx, images in tqdm(test_loader):
                images = images.cuda()
                output = model(images)

                # Adjust these according to your model
                # output = nn_func.softmax(output[0], 1)
                output = nn_func.softmax(output[:, :413], 1).cpu()
                zero_vec = torch.zeros((output.shape[0], 1))
                output = torch.cat([zero_vec, output], dim=1)

                img_idx_list.append(img_idx)
                output_list.append(output)
            img_idx_list = torch.cat(img_idx_list, 0)
            output_list = torch.cat(output_list, 0)

            lines = list()
            for i, img_idx in enumerate(img_idx_list):
                line = [str(x) for x in output_list[i].tolist()]
                lines.append(','.join([img_path_list[img_idx]] + line))
            with open(os.path.join(out_dir, f'{exp_name}.csv'), 'w') as f:
                f.write('\n'.join(lines))
    except FileNotFoundError:
        print(f'Could not find the model file at {model_path}')
    except KeyError:
        print(f'Saved model does not have expected format. We expect the checkpoint to have \'model\' and '
              f'\'state_dict\' keys')
    except Exception as e:
        print(e)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.accimage:
        try:
            import accimage
        except ModuleNotFoundError:
            print('You opted for using accimage but we are unable to import it. Process will be terminated.')
            sys.exit()

    run(args.model_path, args.input_file, args.exp_name, args.out_dir, args.accimage, args.batch_size, args.workers)


if __name__ == '__main__':
    main()
