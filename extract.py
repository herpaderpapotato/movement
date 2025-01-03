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

failback_model_url = "https://huggingface.co/herpaderpapotato/pose-vrlens-finetunes-multiclass/resolve/main/yolo11m-pose/yolo11m-pose.pt"
failback_model_size = 43.0 # MB

# loose code flow:
# 1. parse arguments
# 2. find models
# 3. find videos
# 4. for each video, extract pose with torchaudio (wierd that it's not a torchvision thing)
# 5. save the results to a pickle file at the end of each video
#
#
# nano model processing time for 120 frames batch of 8k x 4k video is about 1.1 seconds on a 3090.
#   ~ 25 minutes for a 50 minute 4096p video because torchaudio loads and processes the frames on the GPU, and then the model processes them on the GPU.
# medium model processing time for 120 frames batch of 8k x 4k video is about 1.6 seconds on a 3090.
#   ~ 37 minutes for a 50 minute 4096p video
#
# Bigger batch doesn't seem to be faster. Smaller batch do get slower.
# VRAM usage TBA. Current benchmark is 120 frames at 4096p input, 640p output, with yolo11m-pose model 22GB VRAM usage. 21.1gb exactly. l is 22.2gb
# anecdotally, nano model is 7mb, medium model is 43mb. i.e 120x7mb = 840mb, 120x43mb = 5160mb. TBA is this is accurate.
#
# The best way to capture the results was to just keep them in memory and save them at the end, but the orig_img field is a memory hog so it's set to None as soon as possible. 
# 46minute video = 211mb of results in a pickle file. YOLO results are kept on the CPU, so it's not a VRAM hog.


def download_model(url, model_dir):
    import requests
    import shutil
    model_file_name = url.split('/')[-1]
    print(f'Downloading model {model_file_name} from {url}, it should be quick, about {failback_model_size:.1f}MB')
    r = requests.get(url, stream=True)
    with open(os.path.join(model_dir, model_file_name), 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return model_file_name


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
    batch_size = args.batch_size  # should do something with auto batch sizing based on VRAM
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
            #raise ValueError('No pose models found in input\\models directory and no model specified')
            # download the model instead of raising an error
            model_file_name = download_model(failback_model_url, 'input\\models')
            # check to see if model was downloaded
            models = glob('input\\models\\*.pt')
            if not models:
                raise ValueError('No pose models found in input\\models directory and no model specified and failed to download model')
        # filter to models with -pose in the name
        models = [model for model in models if '-pose' in model]
        if not models:
            #raise ValueError('No pose models found in input\\models directory and no model specified')
            # download the model instead of raising an error
            model_file_name = download_model(failback_model_url, 'input\\models')
            # check to see if model was downloaded
            models = glob('input\\models\\*.pt')
            if not models:
                raise ValueError('No pose models found in input\\models directory and no model specified and failed to download model')
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
            print()

            cap = cv2.VideoCapture(video_file)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if start_time:
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
            
            if start_time or end_time:
                video_name = f'{start_frame}_{end_frame}_{video_name}'
            
            output_file = os.path.join(output_dir, f'{video_name}.pickle')
            print(f'Output file will be {output_file}')

            if os.path.exists(output_file):
                print(f'{output_file} already exists. Skipping...')
                continue

            if not model:
                model = YOLO(model_file_name).to(device)

            results_superset = []
            for i in range(0, duration_frames):
                results_superset.append([])

            s = StreamReader(video_file) # there will be an error to catch here if there's issues opening the file, including missing ffmpeg
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
                    "hw_accel": device,  # Then keep the memory on CUDA
                    "decoder_option": {
                        "crop": f"0x0x{frame_width}x0",
                        "resize": "640x640",
                    }
                }
            elif codec == 'h264':
                config = {
                    "decoder": "h264_cuvid",  # Use CUDA HW decoder
                    "hw_accel": device,  # Then keep the memory on CUDA
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
    

       


