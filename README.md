# 实验作业一：手写数字 MLP

## 快速运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python mlp_digits.py --exp all             # 全部对照组、失败实验
python mlp_digits.py --exp best --repeats 3 # 最优组合三次复测
```