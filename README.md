# 《动手深度学习 2》PyTorch 最小化实现
为了方便教学，各章节代码**解耦**，互不依赖。

移除原作者 `d2l` 库的所有代码，减少混淆。

移除Notebook，带中文**注释**`.py`文件，方便学生实操/移植。

支持 MacOS + Apple Silicon 芯片（即mps）**运算加速**

**大部分章节仅使用CPU即可完成*


# 环境
Python 版本：`3.10`

已最小化依赖，**仅使用常见库**：
- Pytorch 核心
- Matplotlib 绘图
- Pandas 数据处理（仅在部分章节使用）
- Pillow 图像处理（仅在部分章节使用）

**💡建议使用uv进行依赖管理，使用 `uv sync` 命令一键处理。**


> 或手动安装以下依赖：
> - PyTorch：2.7.0
> - Matplotlib：3.7.2
> - Pandas：2.3.1
> ```pip install torch==2.7.0 matplotlib==3.7.2 pandas==2.3.1```

# 数据集下载及结构说明
### 图像分类数据集 FashionMNIST（3.5章）
torchvision自带数据集，运行代码自动下载

使用章节
- 3.6章
- 3.7章
- 4.2章
- 4.3章
- 4.6章
- 6.6章
- 7.1 - 7.7章

### 房价预测数据集 Kaggle（4.10章）
House Prices - Advanced Regression Techniques 
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv
- http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv

并放在`../data/house-prices-advanced-regression-techniques`目录下：
- /train.csv
- /test.csv

使用章节
- 4.10章

### 语料库 Time Machine（8.2章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt

并保存为`../data/timemachine.txt`

使用章节
- 8.2 - 8.6章
- 9.1 - 9.3章

### 机器翻译数据集 英语-法语（9.5章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip

并解压到`../data/fra-eng`目录下：
- /fra.txt

使用章节
- 9.5 - 9.7章
- 10.4章
- 10.7章

### 图像微调数据集 热狗识别（13.2章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/hotdog.zip

并解压到`../data/hotdog`目录下：
- /train
- /test

使用章节
- 13.2章

### 目标检测数据集 香蕉检测（13.6章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip

并解压到`../data/banana-detection`目录下：
- /bananas_train
- /bananas_val

使用章节
- 13.7章

### 语义分割和数据集 Pascal VOC 2012（13.9章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/VOCtrainval_11-May-2012.tar

并解压到`../data/VOCdevkit/VOC2012`目录下：
- /JPEGImages
- /SegmentationClass
...

使用章节
- 13.11章

### 用于预训练词嵌入的数据集 Penn Treebank（14.3章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip

并解压到`../data/ptb`目录下：
- /ptb.train.txt
- /ptb.valid.txt
- /ptb.test.txt

使用章节
- 14.3 - 14.4章

### 预训练词向量 GloVe-50d（14.7章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/glove.6B.50d.zip

并解压到`../data/glove.6B.50d`目录下：
- /vec.txt

使用章节
- 14.7章

### 用于预训练BERT的数据集 WikiText2（14.9章）
!!官方链接已经失效，可前往huggingface下载，读取代码已经修改
- https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-v1

保存到`../data/wikitext-2-v1`目录下：
- /train-00000-of-00001.parquet
- /test-00000-of-00001.parquet
- /validation-00000-of-00001.parquet

使用章节
- 14.9章
- 14.10章

### 预训练词向量 GloVe-100d（15.2章）
下载以下数据集
- http://d2l-data.s3-accelerate.amazonaws.com/glove.6B.100d.zip

并解压到`../data/glove.6B.100d`目录下：
- /vec.txt

使用章节
- 15.2章
- 15.3章
- 15.5章


### 情感分析及数据集-IMDb电影评论（15.1章）
下载以下数据集
- http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

并解压到`../data/acllmdb`目录下：
- /imdb.vocab
- /train
- /test

使用章节
- 15.2章
- 15.3章
- 15.5章

### 斯坦福自然语言推断（SNLI）数据集 (15.4章)
下载以下数据集
- https://nlp.stanford.edu/projects/snli/snli_1.0.zip

并解压到`../data/snli_1.0`目录下：
- /snli_1.0_train.txt
- /snli_1.0_test.txt
...

使用章节
- 15.5章
- 15.7章

## References
- https://zh.d2l.ai/index.html
- https://github.com/d2l-ai/d2l-zh