# Simple-ZipNeRF
Official code release accompanying the paper "Simple-RF: Regularizing Sparse Input Radiance Fields with Simpler Solutions"

* [Project Page](https://nagabhushansn95.github.io/publications/2024/Simple-RF.html)
* [Published Data (OneDrive)](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/nagabhushans_iisc_ac_in/Egl0tSqfBnBIqKYZP0nJPFoB24jhJe0EQQe1KBrav63ohQ?e=0GSEmV)

> [!NOTE]
> This repository contains the code for Simple-ZipNeRF only. The integrated code for Simple-NeRF and Simple-TensoRF can be found at [NagabhushanSN95/Simple-RF](https://github.com/NagabhushanSN95/Simple-RF).

## Setup

### Python Environment
Environment details are available in `EnvironmentData/SimpleZipNeRF.yml`. The environment can be created using conda
```shell
cd EnvironmentData
bash Install_SimpleZipNeRF.sh
cd ..
```

> [!IMPORTANT]
> Document the installation issue here.

### Add the source directory to PYTHONPATH
```shell
export PYTHONPATH=<ABSOLUTE_PATH_TO_SIMPLERF_DIR>/src:$PYTHONPATH
```

### Set-up Databases
Please follow the instructions in [database_utils/README.md](src/database_utils/README.md) file to set up various databases. Instructions for custom databases are also included here.

## Training and Inference
The files `TrainerTester08_MipNeRF360.sh`, `TrainerTester04_NeRF_Synthetic.sh` contain the code for training, testing and quality assessment along with the configs for the respective databases.
```shell
cd src/
bash TrainerTester08_MipNeRF360.sh ../runs/training/train7140/Configs.gin train_set_num=4
bash TrainerTester04_NeRF_Synthetic.sh ../runs/training/train3012/Configs.gin train_set_num=2
cd ../
```

### Inference with Pre-trained Models
The train configs are also provided in `runs/training/train****` folders for each of the scenes. Please download the trained models from `runs/training` directory in the published data (link available at the top) and place them in the appropriate folders. Disable the train call in the [TrainerTester](src/TrainerTester08_MipNeRF360.py#L116) files and run the respective files. This will run inference using the pre-trained models and also evaluate the synthesized images and reports the performance.

### Evaluation
Evaluation of the rendered images will be automatically done after rendering the images. To compute depth based metrics, ground truth depth maps are needed. We obtain (pseudo) ground truth depth maps by training the vanilla ZipNeRF with dense input views. Download these depth maps from `data` directory in the published data (link available at the top) and place them in the appropriate folders.

## License
MIT License

Copyright (c) 2024 Nagabhushan Somraj, Sai Harsha Mupparaju, Adithyan Karanayil, Rajiv Soundararajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Citation
If you use this code for your research, please cite our paper

```bibtex
@article{somraj2024simplerf,
    title = {{Simple-RF}: Regularizing Sparse Input Radiance Fields with Simpler Solutions},
    author = {Somraj, Nagabhushan and Mupparaju, Sai Harsha and Karanayil, Adithyan and Soundararajan, Rajiv},
    journal = {arXiv: 2404.19015},
    month = {May},
    year = {2024},
    doi = {10.48550/arXiv.2404.19015},
}
```
If you use outputs/results of Simple-RF model in your publication, please specify the version as well. The current version is 1.0.

## Acknowledgements
Our code is built on top of [zipnerf-pytorch](https://github.com/SuLvXiangXin/zipnerf-pytorch) and [SimpleNeRF](https://github.com/NagabhushanSN95/SimpleNeRF) codebases.


For any queries or bugs regarding Simple-ZipNeRF, please raise an issue.
