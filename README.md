# wdv3-timm-bulk

use `timm` to run the WD Tagger V3 models.

## How To Use

1. clone the repository and enter the directory:
```sh
git clone https://github.com/nawka12/wdv3-timm-bulk
cd wd3-timm-bulk
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.10 -m venv venv
# Activate it
source venv/bin/activate # or venv\Scripts\activate for windows
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install PyTorch manually (e.g. if you are not using an nVidia GPU)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # or https://download.pytorch.org/whl/cu118 for utilizing GPU
# Install requirements
python -m pip install -r requirements.txt
```

3. Run the script, picking one of the 3 models to use:
```sh
python wdv3_timm.py <swinv2|convnext|vit> path/to/dir
```

Example output from `python wdv3_timm.py swinv2 dataset`:
```sh
Using cuda
Loading model 'swinv2' from 'SmilingWolf/wd-swinv2-tagger-v3'...
Loading tag list...
Creating data transform...
100%|██████████████████████████████████████████████████████████████████████████| 1.05k/1.05k [00:58<00:00, 17.9image/s]
Done!
```
The script will create a new file `imagename.txt` containing the tags.
