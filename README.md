# movement

conda deactivate

conda create -p %cd%\\.conda python=3.12

conda activate %cd%\\.conda

pip install -r requirements.txt

Have https://github.com/BtbN/FFmpeg-Builds/releases, ffmpeg-master-latest-win64-gpl-shared.zip or similar somewhere that torch can find it.

Any applicable model put in input\models
- Intended for finetunes from https://huggingface.co/herpaderpapotato/pose-vrlens-finetunes-multiclass

Put applicable video in input\videos
- or use --input_dir
- also there's --video which is just for filtering by name, not path.

Only dumps predictions to a pickle at the moment.
