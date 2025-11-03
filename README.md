# 手工构建transformer架构
本实验实现了decoder-only的transformer架构。分别做了不同注意头数下的对比实验和删除了位置编码的消融实验。
## 数据集
本实验使用了莎士比亚数据集，数据集地址为<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
### 数据集下载
`wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
## 安装指南
`pip install requirements.txt`
## 运行指南
`python decoder-only.py`
## 运行结果
运行结果放在results文件夹下。n_head=3的损失曲线放在result.jpg中，n_head=6的损失曲线放在result2.jpg中，n_head=12的损失曲线放在result3.jpg中。n_head=3的生成文本放在output.txt中，n_head=6的生成文本放在output2.txt中，n_head=12的生成文本放在output3.txt中。消融实验的损失曲线放在result4.jpg中，生成文本放在output4.txt中
