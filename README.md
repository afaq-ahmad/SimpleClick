## [SimpleClick: Interactive Image Segmentation with Simple Vision Transformers](https://arxiv.org/abs/2210.11006)
<p align="center">
    <a href="https://paperswithcode.com/sota/interactive-segmentation-on-sbd?p=simpleclick-interactive-image-segmentation">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simpleclick-interactive-image-segmentation/interactive-segmentation-on-sbd"/>
    </a>
</p>

<p align="center">
  <img src="./assets/img/simpleclick_framework.png" alt="drawing", width="650"/>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
    </a>
    <a href="https://arxiv.org/pdf/2210.11006.pdf">
        <img src="https://img.shields.io/badge/arXiv-2102.06583-b31b1b"/>
    </a>    
</p>

## Environment
Training and evaluation environment: Python3.8.8, PyTorch 1.11.0, Ubuntu 20.4, CUDA 11.0. Run the following command to install required packages.
```
pip3 install -r requirements.txt
```
You can build a container with the configured environment using our [Dockerfiles](https://github.com/uncbiag/SimpleClick/tree/main/docker). 
We only provide the Dockerfiles for CUDA 11.0/11.4/11.6. If you use different drivers, you need to modify the base image in the Dockerfile.
You also need to configue the paths to the datasets in [config.yml](https://github.com/uncbiag/SimpleClick/blob/main/config.yml) before training or testing.

## Demo
<p align="center">
  <img src="./assets/demo_sheep.gif" alt="drawing", width="500"/>
</p>

An example script to run the demo. 
```
python3 demo.py --checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth --gpu 0
```
Some test images can be find [here](https://github.com/uncbiag/SimpleClick/tree/main/assets/test_imgs).

## Download 
Pre-trained SimpleClick models: [Google Drive](https://drive.google.com/drive/folders/1qpK0gtAPkVMF7VC42UA9XF4xMWr5KJmL?usp=sharing)

BraTS dataset (369 cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing) 

OAI-ZIB dataset (150 cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing)

Other datasets: [RITM Github](https://github.com/saic-vul/ritm_interactive_segmentation)

## Notes
[10/25/2022] Add docker files.

[10/02/2022] Release the main models. This repository is still under active development.

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Citation
```bibtex
@article{liu2022simpleclick,
  title={SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
  author={Liu, Qin and Xu, Zhenlin and Bertasius, Gedas and Niethammer, Marc},
  journal={arXiv preprint arXiv:2210.11006},
  year={2022}
}
```
## Acknowledgement
Our project is developed based on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation). Thanks for the nice demo GUI :)
