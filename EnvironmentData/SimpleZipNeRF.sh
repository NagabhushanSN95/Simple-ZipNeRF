conda create --name SimpleZipNeRF python=3.9 -y 
conda activate SimpleZipNeRF
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -y
pip install -r ../src/requirements.txt
pip install ../src/extensions/cuda
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
CUDA=cu117
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
conda install -c conda-forge pandas deepdiff sk-video simplejson -y
conda install -c conda-forge libstdcxx-ng=12 -y
pip install numpy==1.23 lpips
