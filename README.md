## DeNVeR: Deformable Neural Vessel Representations for Unsupervised Video Vessel Segmentation
**CVPR 2025**

[![_](https://img.shields.io/badge/Project%20Page-DeNveR-orange)](https://kirito878.github.io/DeNVeR/) 
[![_](https://img.shields.io/badge/arXiv-2406.01591-b31b1b.svg)](https://arxiv.org/abs/2406.01591) 
[![_](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IYGiJECwAaoLPq7KGHQE_dvtrdHz9fUA?authuser=2&hl=zh-tw#scrollTo=n1ppvOhqbRkV)
## Dataset path 
* Download from [google drive](https://drive.google.com/file/d/11e5SmynT8qitWwSGBj5nn3JVYTNG5VZP/view?usp=sharing)
* Put the dataset into the `xca_dataset` folder.

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
@article{wu2024denver,
    title={DeNVeR: Deformable Neural Vessel Representations for Unsupervised Video Vessel Segmentation},
    author={Chun-Hung Wu and Shih-Hong Chen and Chih Yao Hu and Hsin-Yu Wu and Kai-Hsin Chen and Yuyou-chen and Chih-Hai Su and Chih-Kuo Lee and Yu-Lun Liu},
    journal={arXiv},
    year={2025}
}
```


## Acknowledgement

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.
