import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
#from dv import AedatFile
#from dv import NetworkEventInput


#filepath = '/Smat/Nemo/dataset/kitti_odometry_dataset/kitti_training_dataset/train/Voxel_grid_between_frames_1/'



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=False, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('-r', '--row_file', type = str,default=None)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=True)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=100.00, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=True)
    parser.add_argument('-s', '--saving_folder', required=True, type=str, help='path to saving voxel grid tensors')
    parser.add_argument('--num_bins', default=5, type=int,
                        help='number of bins between frames')

    set_inference_options(parser)

    args = parser.parse_args()

    # Path to saving voxel grid
    filepath = args.saving_folder

    # Read sensor size from the first first line of the event file
    path_to_events = args.input_file

    header = pd.read_csv(path_to_events, delim_whitespace=True, header=None, names=['width', 'height'],
                         dtype={'width': np.int, 'height': np.int},
                         nrows=1)
    width, height = header.values[0]

    print('Sensor size: {} x {}'.format(width, height))

    # Load model
    #model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    #model = model.to(device)
    #model.eval()

    #reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    """ Read chunks of events using Pandas """

    # Loop through the events and reconstruct images
    N = None
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    if args.fixed_duration:
        event_window_iterator = FixedDurationEventReader(path_to_events,
                                                         duration_ms=args.window_duration,
                                                         start_index=start_index)
    else:
        event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)
    
    i = 0
    with Timer('Processing entire dataset'):
        for event_window in event_window_iterator:
            print(event_window)
            last_timestamp = event_window[-1, 0]
            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=args.num_bins,
                                                        width=width,
                                                        height=height)
                    event_tensor = torch.from_numpy(event_tensor)
                    e_numpy = event_tensor.numpy()
                    np.save(filepath + 'event_tensor_{:06d}.npy'.format(i), e_numpy)
                    i +=1
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=args.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)
                    e_numpy = event_tensor.numpy()
                    np.save(filepath + 'event_tensor_{:06d}.npy'.format(i), e_numpy)
                    i +=1

            num_events_in_window = event_window.shape[0]
            #reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

            start_index += num_events_in_window

