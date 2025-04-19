## DeNVeR: Deformable Neural Vessel Representations for Unsupervised Video Vessel Segmentation
**CVPR 2025**

[![_](https://img.shields.io/badge/Project%20Page-DeNveR-orange)](https://kirito878.github.io/DeNVeR/) 
[![_](https://img.shields.io/badge/arXiv-2406.01591-b31b1b.svg)](https://arxiv.org/abs/2406.01591) 
[![_](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IYGiJECwAaoLPq7KGHQE_dvtrdHz9fUA?authuser=2&hl=zh-tw#scrollTo=n1ppvOhqbRkV)
## Dataset path 
* Download XACV from [google drive](https://drive.google.com/file/d/11e5SmynT8qitWwSGBj5nn3JVYTNG5VZP/view?usp=sharing)
* Replace `xca_dataset` with our [XACV](https://drive.google.com/file/d/11e5SmynT8qitWwSGBj5nn3JVYTNG5VZP/view?usp=sharing) dataset.


```
XACV
├── CVAI-2828
│   ├── ground_truth
│   │   ├── CVAI-2828RAO2_CRA32
│   │   │   ├── 00056.png
│   │   │   └── 00065.png
│   │   └── CVAI-2828RAO2_CRA32CATH
│   │       ├── 00056.png
│   │       └── 00065.png
│   └── images
│       └── CVAI-2828RAO2_CRA32
│           ├── 00000.png
│           ├── 00001.png
│           ├── 00002.png
│           ├── 00003.png
│           ├── 00004.png
│           ├── ...
├── CVAI-2829
├── ...
```
## Requirements
Python versionn = 3.9
```
conda env create -f environment.yml
```

## Download RAFT checkpoint
```
cd RAFT
./download_models.sh
```
## Test-time training
```
python main.py -d {seq_name}
```
e.g. 
```
python main.py -d CVAI-2828RAO2_CRA32
```
## Batch running
```
python launch.py -f ./job_specs/vessel.txt --gpus {gpu_ids}
```

## Evaluation
```
cd eval
python gt_filename.py -d {exp_name}
python eval.py
```
`exp_name` is the last directory name in the `dir` path specified in `confs/config.yaml`.
e.g. 
```
python gt_filename.py -d init_model
```


## Citation

Please cite us if our work is useful for your research.

```
@inproceedings{wu2025denver,
  title={DeNVeR: Deformable Neural Vessel Representations for Unsupervised Video Vessel Segmentation},
  author={Wu, Chun-Hung and Chen, Shih-Hong and Hu, Chih Yao and Wu, Hsin-Yu and Chen, Kai-Hsin and Chen, Yu-You and Su, Chih-Hai and Lee, Chih-Kuo and Liu, Yu-Lun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```


## Acknowledgement

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.
