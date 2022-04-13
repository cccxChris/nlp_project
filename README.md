## 测试Bert/ELMo词向量
- 任务: 文本分类 (text classification)
- 数据集： 
- negative_train.txt 抑郁数据包含噪声
- negative_clean_train.txt 抑郁数据不包含噪声
- positive_train.txt 非抑郁数据不包含噪声
- 模型 word embeddings + encoder:
    - word embeddings:
        - GloVe
    - encoder:
        - CNN+MaxPooling

博客总结：[Bert/ELMo文本分类](http://shomy.top/2020/07/06/bert-elmo-cls/)


## 使用方法 Usage
- 环境:
    - python3.6+
    - pytorch 1.4+
    - transformers
    - AllenNLP
    - sklearn
    - fire
- 克隆代码到本地, 依据`data/readme.md`说明 下载Bert/ELMo/GloVe的词向量文件
- 运行代码：
    ```
    python main.py train --emb_method='glove' --enc_method='cnn'
    ```
- 可配置项:
    - emb_method: [glove]
    - enc_method: [cnn]

其余可参看`config.py`, 比如使用`--gpu_id=1`来指定使用的GPU

## 结果

运行环境:
- GPU: 1080Ti
- CPU: E5 2680 V4


