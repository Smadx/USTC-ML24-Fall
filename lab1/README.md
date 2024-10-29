# GPU Performace

> <p><strong>机器学习概论</strong>lab1</p>
>
> Author:@Rosykunai(如果你认为实验出的有问题，请在微信群私聊)
>
> <p><strong>Date:</strong> 2024 年 9 月</p>

[toc]

## Deadline

2024.10.13 23:59

## 实验环境

**Conda** (推荐):

```bash
conda create -n ml24 python=3.9
pip install -r requirements.txt
conda activate ml24
```

如果你想在现有环境中安装 (Python=3.9),使用下面的命令：

```bash
pip install -r requirements.txt
```

## 提交要求

上传一个`学号 + 姓名+LAB1.zip`文件，包含：

- results

  - \_Regression

    ​ -- config.yaml

    ​ -- model.pkl

  - \_Classification

    ​ -- config.yaml

    ​ -- model.pkl

- submission.py

- report.pdf

## 成绩评定

- Code(40%):见 GPU Performance.pdf

- Performance(30%):

  - Regression(15%):

    $$grade=5\times (1-10\times\underbrace{relative\_error}_\text{On the testset you divided})+10\times (1-10\times \underbrace{relative\_error}_\text{On TA's testset})$$

    $$R^2 = 1-\frac{\sum_{i=1}^N(y_i-\hat{y_i})^2}{\sum_{i=1}^N(y_i-\bar{y})^2}$$

    如果$relative\_error>0.1$或$R^2<-1$,你不会在这部分得到任何分数；

    如果$-1<R^2<=0$,这部分分数会$\times 0.8$；

    \*如果你的模型性能超越了 sklearn，你会在这部分得到满分。

  - Classification(15%):

    $$grade=5\times (\underbrace{accuracy}_\text{On the testset you divided}-0.5)/0.5+10 \times (\underbrace{accuracy}_\text{On TA's testset}-0.5)/0.5$$

    如果$accuracy<0.5$,你不会在这部分得到任何分数；

    如果第一部分的$R^2<0$,这部分分数会$\times 0.8$ ；

    \*如果你的模型性能超越了 sklearn，你会在这部分得到满分。

- Report(30%):

  - 记录实验流程 (2%)
  - 分析 loss 曲线，记录你调试超参数的过程 (5%)
  - 报告你在自己划分的数据集上的最好结果 (2%)
  - 回答问题 (20%):见 GPU Performance.pdf
  - 反馈 (1%):见 GPU Performance.pdf

- Warning:

  - 请独立完成实验，不得抄袭
  - 如果助教根据你提供的超参数无法复现你所报告的结果，你不会在 Performance 部分得到任何分数
