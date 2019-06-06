# Siamese-GRU-for-API-query

### 目录结构
 - 原始数据 (./data/corpus)  
 - 训练数据 (./data/*.pickle)  
 - 训练数据生成 (./data/data_maker.py)  
 - 模型代码 (./gruRNN.py)  
 - 训练代码 (./run.py)  

### 环境
 - conda 4.6.14  
 - Python 3.6.8  
 - tensorflow 1.13.1  
 - numpy 1.16.4  
 - scipy 1.2.1  
 - nltk 3.4.1  

### 项目日志
 - 6月1日，模型里写错了loss函数，结果训练有效果  
 - 6月2日，修改模型loss函数为MSE（[参考论文](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)），tensorboard曲线光滑度设为0.995，未见loss有下降趋势。再尝试回滚到昨天的版本查找哪里出了问题，结果仍未见loss有下降趋势（自闭）  
 - 6月3日，修复上述致命bug，准备加入注意力机制和调参  
 - 6月4日，原始模型训练完成，测试集MSE值0.021390，pearson_r值0.811558  
 - 6月6日，制作问句特征向量pickle (./sent_represent.pickle)，pickle内为一字典，key为问句id，value为模型输出的问句特征向量  
 	+ 另，在 (./data/data_maker.py) 中增加了用于输出所有数据的代码，但生成的文件 (./data/all.pickle) 太大，无法上传至Github  
