import streamlit as st
import pandas as pd
import os
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def experiment_table(filename):
    table = pd.read_csv(ROOT / "results" / filename)
    table["ACC"] = table["ACC"].map(lambda value: f"{value:.2%}")
    table["F1"] = table["F1"].map(lambda value: f"{value:.4f}")
    return table

st.set_page_config(page_title="微博情感分析系统", layout="wide")

st.title("🎭 微博情感分析系统")
st.markdown("基于BERT的中文微博情感分类与跨领域迁移分析")

page = st.sidebar.radio("选择页面", [
    "📊 项目概览",
    "📈 数据分析",
    "🔬 实验结果",
    "🤖 实时预测",
    "📋 报告总结"
])

if page == "📊 项目概览":
    st.header("项目概览")
    col1, col2, col3 = st.columns(3)
    with col1:
        total = sum(len(pd.read_csv(path, usecols=[0])) for path in (ROOT / "data").glob("*.csv"))
        st.metric("仓库数据总量", f"{total:,}条")
    with col2:
        st.metric("情感类别", "6类 → 2类")
    with col3:
        transfer = pd.read_csv(ROOT / "results/binary_results.csv")
        st.metric("二分类实验最高ACC", f"{transfer['ACC'].max():.2%}")
    
    st.markdown("""
    ### 研究内容
    1. **六分类探索**：happy/sad/angry/surprise/fear/neutral
    2. **二分类优化**：聚焦happy vs angry（ACC~95%）
    3. **跨领域迁移**：usual↔virus跨域评估
    4. **Focal Loss**：与标准交叉熵比较实验结果
    5. **注意力可视化**：辅助观察 token 权重分布
    """)

elif page == "📈 数据分析":
    st.header("数据分析")
    st.subheader("标签分布")
    if os.path.exists('results/label_distribution.png'):
        st.image('results/label_distribution.png', use_column_width=True)
    
    st.subheader("数据统计")
    data = {
        '数据集': ['usual_train', 'usual_eval', 'usual_test', 'virus_train', 'virus_eval', 'virus_test'],
        '样本数': [len(pd.read_csv(ROOT / 'data' / name, usecols=[0])) for name in ['usual_train.csv', 'usual_eval_labeled.csv', 'usual_test_labeled.csv', 'virus_train.csv', 'virus_eval_labeled.csv', 'virus_test_labeled.csv']]
    }
    st.dataframe(pd.DataFrame(data))

elif page == "🔬 实验结果":
    st.header("实验结果对比")
    
    st.subheader("六分类探索实验")
    st.caption("以下分别展示仓库中的基线与 BERT 实验记录。")
    st.table(pd.read_csv(ROOT / "results/baseline_results.csv"))
    st.table(experiment_table("bert_results_v2.csv"))

    st.subheader("二分类实验")
    st.caption("TF-IDF 基线与交叉熵/Focal Loss 对比单独展示；F1 以 angry 为正类。")
    st.table(experiment_table("binary_baseline.csv"))
    st.table(experiment_table("focal_comparison.csv"))

    st.subheader("跨领域迁移分析")
    st.table(experiment_table("binary_results.csv"))

    st.subheader("训练曲线对比")
    if os.path.exists('results/training_curve_comparison.png'):
        st.image('results/training_curve_comparison.png', use_column_width=True)
    
    st.subheader("注意力热力图")
    if os.path.exists('results/attention_0.png'):
        cols = st.columns(2)
        for i in range(4):
            with cols[i % 2]:
                if os.path.exists(f'results/attention_{i}.png'):
                    st.image(f'results/attention_{i}.png', caption=f'样本{i+1}')

elif page == "🤖 实时预测":
    st.header("实时情感预测")
    text_input = st.text_area("输入微博内容：", "今天天气真好，心情特别开心！")
    
    if st.button("预测"):
        if not text_input.strip():
            st.warning("请输入微博文本。")
        else:
            api_url = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8002/predict")
            try:
                response = requests.post(
                    api_url,
                    json={"text": text_input.strip()},
                    timeout=30,
                )
                response.raise_for_status()
                prediction = response.json()
                label = prediction["label"]
                confidence = float(prediction["confidence"])
                if label not in {"happy", "angry"} or not 0 <= confidence <= 1:
                    raise ValueError("模型接口返回值不符合预期。")

                result = "😊 happy" if label == "happy" else "😠 angry"
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("预测结果", result)
                with col2:
                    st.metric("预测类别概率", f"{confidence * 100:.1f}%")
            except requests.RequestException:
                st.error("无法连接情感分析 API。请先启动 FastAPI 服务并检查 SENTIMENT_API_URL。")
            except (KeyError, TypeError, ValueError):
                st.error("情感分析 API 返回了无效结果，请检查后端服务。")

elif page == "📋 报告总结":
    st.header("报告总结")
    focal = pd.read_csv(ROOT / "results/focal_comparison.csv").set_index("方法")
    standard = focal.loc["标准交叉熵"]
    improved = focal.loc["Focal Loss"]
    st.markdown(f"""
    ### 实验记录
    - 标准交叉熵：ACC {standard['ACC']:.2%}，angry 类 F1 {standard['F1']:.4f}。
    - Focal Loss：ACC {improved['ACC']:.2%}，angry 类 F1 {improved['F1']:.4f}。
    - 同一对比实验中，ACC 提升 {(improved['ACC'] - standard['ACC']) * 100:.2f} 个百分点。
    - 跨域结果见实验结果页，领域内与跨域指标分开记录。
    - 注意力热力图用于辅助观察权重分布，不能据此证明模型的因果解释。

    ### 实验范围
    - happy/angry 二分类与六分类探索。
    - 通用微博和疫情微博之间的跨域评估。
    - 本地 FastAPI 推理与 Streamlit 页面展示。
    """)

st.sidebar.markdown("---")
st.sidebar.info("作者：李玥珣")