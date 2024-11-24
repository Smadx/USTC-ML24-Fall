# MNIST

> <p><strong>机器学习概论</strong>lab3</p>
> <p><strong>Author:</strong>@Rosykunai</p>
> <p><strong>Date</strong>2024 年 11 月</p>

[toc]

## Deadline

2024.12.15 23:59

## 实验环境

**Conda** (推荐):(Python=3.9)

```bash
# conda create -n ml24 python=3.9
conda activate ml24
pip install -r requirements.txt
```

本次实验需要用到 pytorch，如果你想要安装 GPU 版 (需要 N 卡) 的 pytorch，参考下面的方法：

换源：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels nvidia
conda config --set show_channel_urls yes
```

安装 GPU 版的 pytorch:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 # CUDA版本
```

上述配置完成后，运行`load_resources.py`,最后会输出你的 pytorch 使用的设备。

\*没有 N 卡不影响完成本次实验，我们只会用一些小模型做推理，CPU 运行时间稍长一点 (<3min)

## 提交要求

上传一个`学号 + 姓名+LAB3.zip`文件，包含：

- `results/{datetime}`
  - `gmm`
    - config.json
    - gmm.safetensors
  - `pca`
    - config.json
    - pca.safetensors
  - config.yaml
- submission.py
- report.pdf

## 成绩评定

- Code(40%): 见 MNIST.pdf

- Performance(30%):

  ​ 我们在原始数据空间 (784 维) 中使用`davies_bouldin_score`度量聚类的性能：

  ​ $ grade=30 \* \frac{ DB*{ sklearn } }{ DB*{ yours } } $

  ​ \*如果你的模型性能超越了 sklearn，你会在这部分得到满分。

- Report(30%):

  - 记录实验流程 (2%)
  - 记录你调试超参数的过程 (5%)
  - 报告最好的聚类和生成结果 (输出的图片) (2%)
  - 回答问题 (20%):见 GPU Performance.pdf
  - 反馈 (1%):见 GPU Performance.pdf

- Warning:
  - 请独立完成实验，不得抄袭
  - 如果助教根据你提供的超参数无法复现你所报告的结果，你不会在 Performance 部分得到任何分数
