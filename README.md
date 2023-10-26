# LViT


This repo is the official implementation of "**LViT: Language meets Vision Transformer in Medical Image Segmentation**" 
[Arxiv](https://arxiv.org/abs/2206.14718), [ResearchGate](https://www.researchgate.net/publication/371833348_LViT_Language_meets_Vision_Transformer_in_Medical_Image_Segmentation), [IEEEXplore](https://ieeexplore.ieee.org/document/10172039)

![image](https://github.com/HUANGLIZI/LViT/blob/main/IMG/LViT.png)

## Requirements

Python == 3.7 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
Questions about NumPy version conflict. The NumPy version we use is 1.17.5. We can install bert-embedding first, and install NumPy then.

## Usage

### 1. Data Preparation
#### 1.1. QaTa-COV19, MosMedData+ and MoNuSeg Datasets (demo dataset)
The original data can be downloaded in following links:
* QaTa-COV19 Dataset - [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

* MosMedData+ Dataset - [Link (Original)](http://medicalsegmentation.com/covid19/) or [Kaggle](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)

* MoNuSeG Dataset (demo dataset) - [Link (Original)](https://monuseg.grand-challenge.org/Data/)

* ESO-CT Dataset [1] [2]

[1] Jin, Dakai, et al. "DeepTarget: Gross tumor and clinical target volume segmentation in esophageal cancer radiotherapy." Medical Image Analysis 68 (2021): 101909.

[2] Ye, Xianghua, et al. "Multi-institutional validation of two-streamed deep learning method for automated delineation of esophageal gross tumor volume using planning CT and FDG-PET/CT." Frontiers in Oncology 11 (2022): 785788.

The text annotation of QaTa-COV19 has been released!

  *(Note: The text annotation of QaTa-COV19 train and val datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDkm5jsTw5dX_RpuRr?e=uaZq6W).
  The partition of train set and val set of QaTa-COV19 dataset [download link](https://1drv.ms/u/s!AihndoV8PhTDgt82Do5kj33mUee33g?e=kzWl8y).
  The text annotation of QaTa-COV19 test dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDkj1vvvLt2jDCHqiM?e=954uDF).)*

  ***(Note: The contrastive label is available in the repo.)***
  
***(Note: The text annotation of MosMedData+ train dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDguIIKCRfYB9Z0NL8Dw?e=8rj6rY).
The text annotation of MosMedData+ val dataset [download link](https://1drv.ms/u/s!AihndoV8PhTDguIGtAgZiRQFYfsAjw?e=tqowkJ).
The text annotation of MosMedData+ test dataset [download link](https://1drv.ms/u/s!AihndoV8PhTDguIHdHkwXMxGlgU9Tg?e=PbcllF).)***
  
  *If you use the datasets provided by us, please cite the LViT.*

#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── QaTa-Covid19
    │   ├── Test_Folder
    |   |   ├── Test_text.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    |   |   ├── Train_text.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    |	    ├── Val_text.xlsx
    │       ├── img
    │       └── labelcol
    └── MosMedDataPlus
        ├── Test_Folder
        |   ├── Test_text.xlsx
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        |   ├── Train_text.xlsx
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── Val_text.xlsx
            ├── img
            └── labelcol
```



### 2. Training

#### 2.1. Pre-training
You can replace LVIT with U-Net for pre training and run:
```angular2html
python train_model.py
```

#### 2.2. Training

You can train to get your own model. It should be noted that using the pre-trained model in the step 2.1 will get better performance or you can simply change the model_name from LViT to LViT_pretrain in config.

```angular2html
python train_model.py
```




### 3. Evaluation

#### 3.1. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase. Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 



### 4. Results

| Dataset    | 	   Model Name 	   | Dice (%) | IoU (%) |
| ---------- | ------------------- | -------- | ------- |
| QaTa-COV19 | U-Net      	       | 79.02    | 69.46   |
| QaTa-COV19 | LViT-T     	       | 83.66    | 75.11   |
| MosMedData+ | U-Net      	       | 64.60    |  50.73   |
| MosMedData+ | LViT-T     	       | 74.57    |  61.33   |
| MoNuSeg    | U-Net      	       | 76.45    | 62.86   |
| MoNuSeg    | LViT-T     	       | 80.36    | 67.31   |
| MoNuSeg    | LViT-T w/o pretrain | 79.98    | 66.83   |

#### 4.1. More Results on other datasets

| Dataset    | 	   Model Name 	   | Dice (%) | IoU (%) |
| ---------- | ------------------- | -------- | ------- |
| [BKAI-Poly](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data)       | LViT-TW    	       | 92.07  | 80.93    |
| ESO-CT | LViT-TW    	       | 68.27    | 57.02    |


### 5. Reproducibility

In our code, we carefully set the random seed and set cudnn as 'deterministic' mode to eliminate the randomness. However, there still exsist some factors which may cause different training results, e.g., the cuda version, GPU types, the number of GPUs and etc. The GPU used in our experiments is 2-card NVIDIA V100 (32G) and the cuda version is 11.2. And the upsampling operation has big problems with randomness for multi-GPU cases.
See https://pytorch.org/docs/stable/notes/randomness.html for more details.



## Reference


* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)


## Citation

```bash
@article{li2023lvit,
  title={Lvit: language meets vision transformer in medical image segmentation},
  author={Li, Zihan and Li, Yunxiang and Li, Qingde and Wang, Puyang and Guo, Dazhou and Lu, Le and Jin, Dakai and Zhang, You and Hong, Qingqi},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```

[![Stargazers repo roster for @HUANGLIZI/LViT](https://reporoster.com/stars/HUANGLIZI/LViT)](https://github.com/HUANGLIZI/LViT/stargazers)

[![Forkers repo roster for @HUANGLIZI/LViT](https://reporoster.com/forks/HUANGLIZI/LViT)](https://github.com/HUANGLIZI/LViT/network/members)
