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
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install requirements
python -m pip install -r requirements.txt
```

3. Run the script, picking one of the 3 models to use:
```sh
python wdv3_timm.py <swinv2|convnext|vit> path/to/dir
```

Example output from `python wdv3_timm.py swinv2 dataset`:
```sh
Loading model 'swinv2' from 'SmilingWolf/wd-swinv2-tagger-v3'...
Loading tag list...
Creating data transform...
Processing image: 0067f57972698ed0db8721c5f3240cb8.png
Processing image: 009322c6122655d1ac93674865c49591.jpeg
Processing image: 00a10cb3159cc078d347a9e74eb48114.jpeg
Processing image: 00d7957e0e59254b5f7235faab50adf4.jpeg
Processing image: 010d0d6b47aba7c7c8fbbcdd1f948049.jpeg
Processing image: 010e3d98f91e5704548c3d76f5b82ba9.png
Done!
```
The script will create a new file `imagename.txt` containing the tags.
