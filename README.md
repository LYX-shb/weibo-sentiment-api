# 微博情感分析 Demo

基于 BERT 的中文微博 happy/angry 二分类 Demo，提供 FastAPI 推理接口和 Streamlit 展示页。Streamlit 的实时预测会请求 FastAPI，不再使用本地关键词规则。

## 环境与模型文件

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

模型权重不包含在仓库中：`.gitignore` 会忽略 `bert-base-chinese/` 与 `*.pt`。运行 API 前需要先准备预训练模型目录和微调权重：

```bash
cd src
python download_model.py
python bert_finetune_binary_focal.py
cd ..
```

训练脚本会在 `results/binary_focal.pt` 生成 API 所需权重；完整训练依赖仓库内 `data/` 数据，并可能耗时较长。若已有预训练模型目录和微调权重，可跳过下载和训练步骤，按 API 的相对路径放好文件。

## 启动

在第一个终端（项目根目录）启动 API：

```bash
python api.py
```

在第二个终端启动 Streamlit：

```bash
streamlit run app.py
```

默认调用 `http://127.0.0.1:8002/predict`。如 API 地址不同，可设置 `SENTIMENT_API_URL` 环境变量。`/health` 只检查进程响应，不代表模型效果或数据质量。

## 指标

页面二分类对比表读取仓库实验记录：`results/binary_baseline.csv` 与 `results/focal_comparison.csv`。F1 以 angry 为正类（标签 1），不是宏平均 F1。不同训练/测试划分的数字不可混用；跨域结果见 `results/binary_results.csv`。

## API

`POST /predict` 请求示例：

```json
{"text": "今天天气真好，心情特别开心！"}
```

响应包含原文本、预测标签和该标签概率。接口仅支持 happy/angry 二分类，不应将概率解释为经过校准的置信度。

## 验证

准备好模型目录和权重后，在项目根目录运行 `pytest -q`，执行已有的 5 个接口用例。没有模型文件时，导入 API 会失败；这些用例需要真实模型，不会自动使用假数据。

启动 API 和 Streamlit 后，在实时预测页输入文本，结果应来自 `/predict` 的响应。停止 API 后再次预测，页面应显示连接错误；空白输入不会发起请求。模型标签和预测类别概率取决于当前训练权重。

## 实验记录的边界

- `src/bert_finetune_binary_focal.py` 的当前损失函数先将类别权重计入交叉熵，再通过 `exp(-ce_loss)` 构造调制项。该实现与标准 Focal Loss 的目标类别概率计算存在偏差，不能当作已验证的标准 Focal Loss；现有 CSV 仅作为该实现的历史实验记录。若修正为标准公式，需要重新训练，并更新权重及指标，不能沿用旧分数证明新实现的效果。
- 训练脚本未固定随机种子，CSV 是单次实验记录，未提供多次运行的均值与方差。下载代码后的重训结果可能不同。
- `src/attention_visualize.py` 使用预训练 BERT，不加载二分类微调权重；其热力图用于观察预训练模型注意力，不能作为情感分类器的解释验证。
