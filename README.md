# DeNVeR: Deformable Neural Vessel Representations for Unsupervised Video Vessel Segmentation

## Requirements
Python versionn = 3.9
```
conda env create -f environment.yml
```

## Computing Optical Flow 

```
cd scripts
python dataset_raft.py  --root ../custom_videos/ --dtype custom
```

## Test-time training
```
python main.py -d CVAI-2829RAO9_CRA37
```
