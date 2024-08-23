# DeNVeR: Deformable Neural Vessel Representations for Unsupervised Video Vessel Segmentation

## Dataset path 
將 dataset 放到xca_dataset 資料夾中

## Requirements
Python versionn = 3.9
```
conda env create -f environment.yml
```
## Change Root

將 confs/data/custom.yaml 之中的root 更換路徑。

## Test-time training
```
python main.py -d CVAI-2829RAO9_CRA37
```

## Batch running
```
python launch.py -f /project/wujh1123/denver/job_specs/vessel.txt --gpus 0 1
```

## Evaluation
```
cd eval
python gt_filename.py -d {data_path}
python eval.py
```
data_path 為 confs/config.yaml 中 dir 的最後一個路徑名
e.g. 
```
python gt_filename.py -d init_model
```