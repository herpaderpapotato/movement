import os
import dotenv
dotenv.load_dotenv()
dll_direnv = os.getenv('DLL_DIR', None)
if dll_direnv:
    os.add_dll_directory(dll_direnv)

from glob import glob
import argparse
import json
import cv2
from ultralytics import YOLO
import torch
from torchaudio.io import StreamReader, StreamWriter
import pickle
import time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract pose from video files')
    parser.add_argument('--input_dir', type=str, help='Input directory containing video files', default='input\\videos', required=False)
    parser.add_argument('--output_dir', type=str, help='Output directory to save extracted pose', default='intermediate\\extractedpose', required=False)
    parser.add_argument('--video', type=str, help='Video file to extract pose', default=None, required=False)
    parser.add_argument('--model', type=str, help='Model to use for pose extraction', default=None, required=False)
    parser.add_argument('--start_time', type=int, help='Start time in seconds to extract pose', default=0, required=False)
    parser.add_argument('--end_time', type=int, help='End time in seconds to extract pose', default=None, required=False)
    parser.add_argument('--batch_size', type=int, help='Batch size for pose extraction.', default=120, required=False)
    parser.add_argument('--device', type=str, help='Device to use for pose extraction', default='auto', required=False, choices=['cuda', 'cpu', 'cuda:0', 'cuda:1', 'auto'])

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    model = args.model
    video_file_filter = args.video
    start_time = args.start_time
    end_time = args.end_time
    batch_size = args.batch_size
    device = args.device

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is not available on this machine')
    
    if device.startswith('cuda'):
        if device == 'cuda':
            device_id = 0
        else:
            device_id = int(device.split(':')[-1])
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3
        gpu_name = torch.cuda.get_device_name(device_id)
        print(f'Using CUDA on GPU {device_id} {gpu_name} with {gpu_memory:.0f}GB memory')
        

    if not model:
        models = glob('input\\models\\*.pt')
        if not models:
            raise ValueError('No pose models found in input\\models directory and no model specified')
        # filter to models with -pose in the name
        models = [model for model in models if '-pose' in model]
        if not models:
            raise ValueError('No pose models found in input\\models directory and no model specified')
        # sort models by date modified
        models = sorted(models, key=os.path.getmtime, reverse=True)
        # choose the most recent model
        model_file_name = models[0]

    print(f'Using model {model_file_name} for pose extraction')
    model = None
    print()

    video_files = glob(os.path.join(input_dir, '*.mp4'))
    if not video_files:
        raise ValueError(f'No video files found in {input_dir}')
    
    if video_file_filter:
        video_files = [video_file for video_file in video_files if video_file_filter in video_file]
        if not video_files:
            raise ValueError(f'No video files found in {input_dir} matching the filter {video_file_filter}')
        print(f'Found {len(video_files)} video files in {input_dir} matching the filter {video_file_filter}')
    else:
        print(f'Found {len(video_files)} video files in {input_dir}')

    
    
    for video_file in video_files:
        extract_start_time = time.time()
        try:
            print(f'Extracting pose from {video_file}')
            video_name = os.path.basename(video_file)

            if start_time or end_time:
                video_name = f'{start_time}_{end_time}_{video_name}'
            
            output_file = os.path.join(output_dir, f'{video_name}.pickle')
            print(f'Output file will be {output_file}')

            if os.path.exists(output_file):
                print(f'{output_file} already exists. Skipping...')
                continue

            if not model:
                model = YOLO(model_file_name).to(device)

            print(f'Extracting pose from {video_file}')
            print()

            cap = cv2.VideoCapture(video_file)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not start_time:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                start_frame = 0
            if end_time:
                cap.set(cv2.CAP_PROP_POS_MSEC, end_time * 1000)
                end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                duration_frames = end_frame - start_frame
            else:
                duration_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame

            cap.release()

            s = StreamReader(video_file)
            srcinfo = s.get_src_stream_info(0)
            num_frames = srcinfo.num_frames
            frame_rate = srcinfo.frame_rate
            width = srcinfo.width
            height = srcinfo.height
            codec = srcinfo.codec
            #print(str(srcinfo))
            if codec == 'hevc':
                config = {
                    "decoder": "hevc_cuvid",  # Use CUDA HW decoder
                    "hw_accel": "cuda",  # Then keep the memory on CUDA
                    "decoder_option": {
                        "crop": f"0x0x{frame_width}x0",
                        "resize": "640x640",
                    }
                }
            elif codec == 'h264':
                config = {
                    "decoder": "h264_cuvid",  # Use CUDA HW decoder
                    "hw_accel": "cuda",  # Then keep the memory on CUDA
                    "decoder_option": {
                        "crop": f"0x0x{frame_width}x0",
                        "resize": "640x640",
                    }
                }
            else:
                raise ValueError(f'Unsupported codec {codec}')
            
            s.add_video_stream(batch_size, **config)
            if not start_time:
                s.seek(0)
            else:
                s.seek(start_time)

            rgb_frames = torch.zeros((120, 3, 640, 640), dtype=torch.float32, device=device)
            y = torch.zeros((120, 640, 640), dtype=torch.float32, device=device)
            u = torch.zeros((120, 640, 640), dtype=torch.float32, device=device)
            v = torch.zeros((120, 640, 640), dtype=torch.float32, device=device)
            r = torch.zeros((120, 640, 640), dtype=torch.float32, device=device)
            g = torch.zeros((120, 640, 640), dtype=torch.float32, device=device)
            b = torch.zeros((120, 640, 640), dtype=torch.float32, device=device)
            results_superset = []
            for i, (chunk, ) in enumerate(s.stream()):
                chunk = chunk.float() / 255.0

                if batch_size == chunk.shape[0]:
                    y[:, :, :] = chunk[:, 0, :, :]
                    u[:, :, :] = chunk[:, 1, :, :] - 0.5
                    v[:, :, :] = chunk[:, 2, :, :] - 0.5
                    # r[:, :, :] = y + 1.14 * v
                    # g[:, :, :] = y -0.396 * u - 0.581 * v
                    # b[:, :, :] = y + 2.029 * u
                    rgb_frames[:, 0, :, :] = y + 2.029 * u # b
                    rgb_frames[:, 1, :, :] = y -0.396 * u - 0.581 * v # g
                    rgb_frames[:, 2, :, :] = y + 1.14 * v # r
                    rgb_frames = rgb_frames.clamp(0.0, 1.0)
                else:
                    # last chunk
                    y = chunk[:, 0, :, :]
                    u = chunk[:, 1, :, :] - 0.5
                    v = chunk[:, 2, :, :] - 0.5
                    rgb_frames = torch.zeros((chunk.shape[0], 3, 640, 640), dtype=torch.float32, device=device)
                    rgb_frames[:, 0, :, :] = y + 2.029 * u # b
                    rgb_frames[:, 1, :, :] = y -0.396 * u - 0.581 * v # g
                    rgb_frames[:, 2, :, :] = y + 1.14 * v # r
                    rgb_frames = rgb_frames.clamp(0.0, 1.0)
                    
                results = model.predict(rgb_frames, verbose=False, save=False, save_frames=False, save_txt=False)
                for j in range(len(results)):
                    results[j].orig_img = None
                results_superset.extend(results)
                
                if (i+1) % 10 == 0:
                    i_plus_1 = i + 1
                    processed_frames = i_plus_1 * batch_size
                    processed_percent = processed_frames / num_frames * 100
                    playback_position = i_plus_1 * batch_size / frame_rate
                    playback_position_str = time.strftime('%H:%M:%S', time.gmtime(playback_position))
                    working_time = time.time() - extract_start_time
                    remaining_time = working_time / processed_frames * (num_frames - processed_frames)
                    working_time_str = time.strftime('%H:%M:%S', time.gmtime(working_time))
                    remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    estimated_end_time = time.time() + remaining_time
                    estimated_end_time_str = time.strftime('%H:%M:%S', time.localtime(estimated_end_time))

                    # table layout
                    print(f'| {processed_frames:8} | {processed_percent:5.2f} | {playback_position_str} | {working_time_str} | {remaining_time_str} | {estimated_end_time_str} |')

                if (i) % 200 == 0:
                    #      |     3600 |  2.16 | 00:00:60 | 00:00:36 | 00:27:38 | 18:43:11 |
                    print("|  Frames  |   %   | Play Pos | Work Time| Rem. Time| End Time |")


                if end_time and (i+1)*batch_size >= duration_frames:
                    break
                if (i+1)*batch_size >= num_frames:
                    break              


        except KeyboardInterrupt:
            print('KeyboardInterrupt. Exiting...')
            pickle.dump(results_superset, open(output_file, 'wb'))
            break
        
        except Exception as e:
            print(f'Error processing {video_file}')
            pickle.dump(results_superset, open(output_file, 'wb'))
            print(e)
            continue
        pickle.dump(results_superset, open(output_file, 'wb'))
    

       


