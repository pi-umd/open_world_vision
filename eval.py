import argparse
import os
import sys

import torch
import torch.nn.parallel
import torch.nn.functional as nn_func
import torchvision.transforms as transforms
import torch.utils.data.distributed
from tqdm import tqdm

# Replace this with your data loader
from data_loader import EvalDataset


def get_arg_parser():
    """"Defines the command line arguments"""
    parser = argparse.ArgumentParser(description='Open World Vision')
    parser.add_argument('--input_file', required=True,
                        help='path to a .txt/.csv file containing paths of input images in first column of each row. '
                             '\',\' will be used as a delimiter if a csv is provided. In text format, each row should'
                             ' only contain the path of an image.')
    parser.add_argument('--output_dir', required=True,
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


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.accimage:
        try:
            import accimage
        except ModuleNotFoundError:
            print('You opted for using accimage but we are unable to import it. Process will be terminated.')
            sys.exit()

    try:
        checkpoint = torch.load(args.model_path)
        model = checkpoint['model']
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            # Replace this with your data-loader
            test_set = EvalDataset(
                data_file=args.input_file,
                accimage=args.accimage,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]),
                ]))

            img_path_list = test_set.data_list
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True)

            img_idx_list = list()
            output_list = list()
            for img_idx, images in tqdm(test_loader):
                images = images.cuda()
                output = model(images)
                output = nn_func.softmax(output[0], 1)
                img_idx_list.append(img_idx)
                output_list.append(output)
            img_idx_list = torch.cat(img_idx_list, 0)
            output_list = torch.cat(output_list, 0)

            lines = list()
            for i, img_idx in enumerate(img_idx_list):
                line = [str(x) for x in output_list[i].tolist()]
                lines.append(','.join([img_path_list[img_idx]] + line))
            with open(os.path.join(args.output_dir, f'{args.exp_name}.csv'), 'w') as f:
                f.writelines(lines)
    except FileNotFoundError:
        print(f'Could not find the model file at {args.model_path}')
    except KeyError:
        print(f'Saved model does not have expected format. We expect the checkpoint to have \'model\' and '
              f'\'state_dict\' keys')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
