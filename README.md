# vehicle_security_assistant

基于 RAG 的新能源汽车标准智能问答系统。以 16 个 GB/GBT/QCT 新能源汽车标准文档为知识源，
对 分块 → 查询 → 检索 → 上下文 → 生成 全链路做了系统性优化对比，最终提供一个
**HyPE + 意图路由 + 混合检索(BM25) + RRF 重排 + U型排序** 的最优检索管线，并用 gradio 提供对话界面。

---

## 1. 快速启动（拉起流程）

### 1.1 环境准备

- Python 虚拟环境：`veichle_rag`（`C:\project\envs\veichle_rag`）
- 依赖：见 `requirements.txt`；核心为 `langchain` / `langchain-milvus` / `pymilvus` / `ragas` / `gradio` / `sentence-transformers`
- 模型：Embedding 用 `BAAI/bge-large-zh-v1.5`（本地缓存），LLM 可选 DeepSeek 云 API 或本地 ollama
- 中间件：**Milvus**（向量库，Docker 运行）、**ollama**（本地 LLM）

### 1.2 启动 Milvus

```bash
docker start milvus-standalone milvus-minio milvus-etcd
# 验证：http://localhost:19530 有响应
```

### 1.3 启动本地 LLM（ollama + qwen2:7b）

```bash
# 确保 ollama 服务在运行（Windows 安装后通常自启，否则手动启动）
ollama serve

# 拉取 qwen2:7b 量化模型（4-bit，可运行于 8GB 显存显卡）
ollama pull qwen2:7b   # 约 4.4GB

# 验证模型可调用
ollama list
ollama run qwen2:7b "你好"
```

### 1.4 构建向量库（首次运行）

```bash
# 生成 30 题精选测试集（从 117 题分层抽样）
python scripts/step1b_select_testset.py

# 构建向量集合：pdf_collection / md_collection（db=vehicle）
python scripts/step2_build_vector.py

# 构建优化集合：md_semantic_collection / md_proposition_collection / md_hype_collection
python scripts/step2c_build_optimizations.py
```

### 1.5 启动问答界面（gradio）

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
# 界面内可切换：DeepSeek 云 API / 本地 ollama qwen2:7b
```

### 1.6 命令行测试 pipeline

```bash
# DeepSeek 云 API
python utils/rag_pipeline.py

# 或在 Python 中切换本地模型
# from utils.rag_pipeline import get_pipeline
# pipe = get_pipeline("ollama")
# print(pipe.answer("车窗防夹力应不大于多少牛？"))
```

---

## 2. 项目结构

```
vehicle_security_assistant/
├── data/
│   ├── origin_data/      # 原始 PDF 标准文档（16 个）
│   ├── transfer_data/    # MinerU 转换后的 MD（含结构化表格/图片文字）
│   └── zip_data/         # MinerU 中间产物
├── ref/                  # 参考实现（proposition/semantic/HyDE/HyPE/RSE/重排等）
├── utils/
│   ├── llm_utils.py      # LLM / embedding 统一初始化（支持 deepseek/qwen/openai/ollama）
│   ├── data_process.py   # PDF/MD 加载分块
│   ├── chunking.py       # Semantic / Proposition / HyPE 分块策略
│   ├── query_opt.py      # 意图识别 + 查询重写 + HyDE
│   ├── hybrid.py         # BM25 稀疏检索 + RRF/CrossEncoder/RankLLM/ColBERT 重排
│   ├── context_opt.py    # RSE / 窗口增强 / 上下文压缩 / U型排序
│   ├── eval_rag.py       # RAG 问答
│   └── rag_pipeline.py   # 最终完整检索管线
├── rag_eval/ragas_eval.py    # ragas 评测封装
├── scripts/                  # 各步骤脚本（数据→建库→评测→报告）
├── res/                      # 评测结果与对比报告
├── output/                   # 中间产物（测试集等）
├── app.py                    # gradio 对话界面
└── config.py                 # 集中配置
```

---

## 3. 评测流程（可选复现）

```bash
# 1. 生成测试集（117 题）
python scripts/step1_generate_testset.py
# 2. 精选 30 题
python scripts/step1b_select_testset.py
# 3. 建向量库
python scripts/step2_build_vector.py
python scripts/step2c_build_optimizations.py
# 4. 分块优化对比
python scripts/step3b_evaluate_optimizations.py
python scripts/step4b_generate_opt_report.py
# 5. 查询优化对比
python scripts/step3c_evaluate_query_opt.py
python scripts/step4c_generate_query_report.py
# 6. 混合检索+重排对比
python scripts/step3d_evaluate_hybrid.py
python scripts/step4d_generate_hybrid_report.py
# 7. 上下文优化对比
python scripts/step3e_evaluate_context_opt.py
python scripts/step4e_generate_context_report.py
# 8. 三模型对比（生成模型：DeepSeek 云 / 本地 ollama / 微调后 qwen2:7b）
python scripts/step3f_evaluate_models.py deepseek
python scripts/step3f_evaluate_models.py ollama
python scripts/step3f_evaluate_models.py peft    # 需先完成微调（见下）
python scripts/step4f_generate_model_report.py
# 9. 构造微调数据 + QLoRA 微调（可选）
python scripts/step5_prepare_finetune_data.py     # 生成 ~957 条标准问答指令数据
python scripts/step6_finetune_qwen.py --smoke     # 0.5B 验证训练栈
python scripts/step6_finetune_qwen.py             # 正式微调 qwen2-7b
```

---

## 4. 优化结果摘要（30 题，ragas）

| 阶段 | 方案 | 忠实度 | 相关性 | 精确率 | 召回率 |
|---|---|---|---|---|---|
| 基线 | 原始 MD(512) | 0.889 | 0.693 | 0.497 | 0.650 |
| 分块 | **HyPE** 假设性问题 | 0.890 | 0.836 | 0.786 | 0.883 |
| 查询 | + 意图路由 | 0.935 | 0.828 | 0.781 | 0.883 |
| 检索 | + BM25 + RRF | **0.954** | **0.851** | **0.815** | **0.967** |

**最终推荐管线**：意图路由 → HyPE 稠密 + BM25 稀疏 → RRF 融合 → U型排序 → LLM 生成。

---

## 5. 本地模型部署说明

- 本机为 Windows + RTX 4060（8GB 显存），用 4-bit 量化（GGUF）可本地运行 **qwen2:7b**（4.4GB）做推理。
- **微调**：8GB 显存下 QLoRA 微调 7B 较紧张（评测/生成与 bge embedding 共争 8GB 显存，加载需显式 `max_memory={0:"6GiB"}`）。
- **三模型对比结论（30 题，裁判 DeepSeek，详见 `res/models/对比报告.md`）**：

| 生成模型 | 忠实度 | 答案相关性 |
|---|---|---|
| DeepSeek 云 | **0.9224** | **0.8629** |
| 本地 qwen2:7b（未微调） | 0.6450 | 0.8487 |
| qwen2:7b QLoRA 微调（1 epoch） | 0.1933 | 0.4776 |

  本地 7B 在忠实度上差距明显；**1 epoch QLoRA 微调（裸问答数据、未收敛）反而进一步拉低忠实度**，已接受该负面结论，不再续训。生产 RAG 生成用 DeepSeek 云，本地 qwen2:7b 保留为离线兜底。若再试微调，应让训练数据与推理提示格式一致（带 `<docs>` 上下文）并训练至收敛。
