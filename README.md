# LViT


This repo is the official implementation of "**LViT: Language meets Vision Transformer in Medical Image Segmentation**" 
[Paper Link](https://arxiv.org/abs/2206.14718)

![image](https://github.com/HUANGLIZI/LViT/blob/main/IMG/LViT.png)

## Requirements

Install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
Questions about NumPy version conflict. The NumPy version I use is 1.17.5. We can install bert-embedding first, and install NumPy then.

## Usage

### 1. Data Preparation
#### 1.1. QaTa-COV19 and MoNuSeg Datasets
The original data can be downloaded in following links:
* QaTa-COV19 Dataset - [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

* MoNuSeG Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)

  *(Note: The text annotation of QaTa-COV19 dataset will be released in the future.)*

#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── Covid19
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
    └── MoNuSeg
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
#### 3.1. Get Pre-trained Models
Here, we provide pre-trained weights on QaTa-COV19 and MoNuSeg, if you do not want to train the models by yourself, you can download them in the following links:

*(Note: the pre-trained model will be released in the future.)*

* QaTa-COV19: 
* MoNuSeg: 
#### 3.2. Test the Model and Visualize the Segmentation Results
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
| MoNuSeg    | U-Net      	       | 76.45    | 62.86   |
| MoNuSeg    | LViT-T     	       | 80.36    | 67.31   |
| MoNuSeg    | LViT-T w/o pretrain | 79.98    | 66.83   |



### 5. Reproducibility

In our code, we carefully set the random seed and set cudnn as 'deterministic' mode to eliminate the randomness. However, there still exsist some factors which may cause different training results, e.g., the cuda version, GPU types, the number of GPUs and etc. The GPU used in our experiments is 2-card NVIDIA V100 (32G) and the cuda version is 11.2. And the upsampling operation has big problems with randomness for multi-GPU cases.
See https://pytorch.org/docs/stable/notes/randomness.html for more details.



## Reference


* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)


## Citation

```bash
@article{Li2022LViTLM,
  title={LViT: Language meets Vision Transformer in Medical Image Segmentation},
  author = {Li, Zihan and Li, Yunxiang and Li, Qingde and Zhang, You and Wang, Puyang and Guo, Dazhou and Lu, Le and Jin, Dakai and Hong, Qingqi},
  journal={arXiv preprint arXiv:2206.14718},
  year={2022}
}
```
