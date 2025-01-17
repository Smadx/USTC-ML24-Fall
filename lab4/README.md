# Agent Playground

> <p><strong>机器学习概论</strong>lab4</p>
>
> <p><strong>Author:</strong>@Rosykunai</p>
>
> <p><strong>Date:</strong>2024 年 12 月</p>

[toc]

## Deadline

2025-1-21 23:59:59

## 实验环境

**Conda** (推荐):(Python=3.9)

```bash
# conda create -n ml24 python=3.9
conda activate ml24
pip install -r requirements.txt
```

## 提交要求

上传一个`学号 + 姓名+LAB4.zip`文件，包含：

- `results`
  - `tabular`
    - config.json
    - config.yaml
    - tabular.safetensors
  - `value-iteration`
    - config.json
    - config.yaml
    - mcvi.safetensors
  - `[Optional] reinforce`
    - `final`
      - config.json
      - model.safetensors
    - config.yaml
- submission.py
- report.pdf

## 成绩评定

- Code(40%): 见 Agent Playground.pdf

- Performance(30%):

  每个 agent 会进行 1000 轮游戏，在规定时间内到达山顶即为获胜

  $$grade=15*(win\_rate_{mcvi}+win\_rate_{tabular})$$

- Report(30%):

  - 记录实验流程 (2%)
  - 记录你调试超参数的过程 (5%)
  - 报告最好的游戏结果 (输出的图片) (2%)
  - 回答问题 (20%):见 Agent Playground.pdf
  - 反馈 (1%):见 Agent Playground.pdf

- Warning:
  - 请独立完成实验，不得抄袭
  - 如果助教根据你提供的超参数无法复现你所报告的结果，你不会在 Performance 部分得到任何分数
