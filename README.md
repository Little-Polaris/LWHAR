# 基于时空骨架序列的人类行为识别  

## 推荐软件环境
- Python 3.13  
- PyTorch 2.6.0
 
其余依赖项请参考requirements.txt

## 数据预处理  

### 文件目录结构  
请按照以下目录结构放置数据集  
```
-dataset/
    - nturgb+d_skeletons/
        - raw_data/ #在该文件夹下放置ntu60的原始数据
            ...
    - nturgb+d_skeletons120/
        - raw_data/ #在该文件夹下放置ntu120额外的原始数据
            ...
```

### 生成数据集
- 生成NTU RGB+D 60或NTU RGB+D 120数据集：
```
python ./data_preprocess/preprocess.py --config-path ./config/data_preprocess_config/ntu60.json  

python ./data_preprocess/preprocess.py --config-path ./config/data_preprocess_config/ntu120.json  
```
请根据机器配置修改配置文件中的`process_num`参数

## 训练和验证  
### 训练  
模型的训练配置共提供了两个数据集、每个数据集两种评估协议、每种评估协议四种数据模态，共16种训练配置文件，在训练时请修改`--config-path`参数即可进行训练  
```
python main.py --config-path ./config/train_config/ntu60_cs_bone.json  
```

### 验证
模型的验证配置共提供了两个数据集、每个数据集两种评估协议,共4种验证配置文件，在验证时请修改`--config-path`参数即可进行验证。  
  
**请注意！** 在验证之前请修改配置文件种的四种数据模态对应权重文件的位置

```
python ensemble.py --config-path ./config/evaluate_config/ntu60_cs.json  
```

验证后将自动绘制混淆矩阵并给出分类报告。
