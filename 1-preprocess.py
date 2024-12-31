from glob import glob
import os
import ffmpeg
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess video files')
    parser.add_argument('--input_dir', type=str, help='Input directory containing video files', default='input', required=False)
    parser.add_argument('--output_dir', type=str, help='Output directory to save preprocessed video files', default='intermediate\\preprocessedvideos', required=False)
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for faster processing', default=True, required=False)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    use_cuda = args.cuda

    os.makedirs(output_dir, exist_ok=True)

    # get all video files in the input directory
    video_files = glob(os.path.join(input_dir, '*.mp4'))
    print(f'Found {len(video_files)} video files in {input_dir}')
    print(f'Saving preprocessed video files to {output_dir}')

    # list all video files in the input directory
    for video_file in video_files:
        print(video_file)

    for video_file in video_files:
        print(f'Preprocessing {video_file}')
        video_name = os.path.basename(video_file)
        output_file = os.path.join(output_dir, video_name)
        print(f'Output file will be {output_file}')

        probe = ffmpeg.probe(video_file)
        print("Video Source file properties")
        print(json.dumps(probe, indent=2))

        if os.path.exists(output_file):
            print(f'{output_file} already exists. Skipping...')
            continue

        input_args = {
            'hwaccel': 'cuda' if use_cuda else 'none',
            'c:v': 'hevc_cuvid' if use_cuda else 'none',
            'resize': '1280x640'
        }
        output_args = {
            'c:v': 'hevc_nvenc' if use_cuda else 'libx264',
            'vf': 'crop=640:640:0:0',
            'crf': 23
        }

        stream = ffmpeg.input(video_file, **input_args)
        stream = ffmpeg.output(stream, output_file, **output_args)
        command = ffmpeg.compile(stream)
        print()
        print("Using command")
        print(command)
        print()
        ffmpeg.run(stream, overwrite_output=True)

        output_probe = ffmpeg.probe(output_file)
        print("Video Output file properties")
        print(json.dumps(output_probe, indent=2))
        