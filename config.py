"""集中配置：路径、模型、Milvus、分块参数等。

各脚本从本文件读取统一常量，避免散落硬编码。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------- 目录 ----------
DATA_DIR = PROJECT_ROOT / "data"
ORIGIN_DIR = DATA_DIR / "origin_data"      # 原始 PDF 数据
TRANSFER_DIR = DATA_DIR / "transfer_data"  # MinerU 转换后的 MD 数据
OUTPUT_DIR = PROJECT_ROOT / "output"       # 中间产物（测试集等）
RES_DIR = PROJECT_ROOT / "res"             # 对比结果与报告

# ---------- LLM ----------
# 通过环境变量 LLM_TYPE 切换：deepseek | qwen | openai(gpt-4o-mini)
# 默认 deepseek；若 DeepSeek 余额不足，可临时设为 qwen 或 openai。
LLM_TYPE = os.getenv("LLM_TYPE", "deepseek")

# 索引构建（Proposition/HyPE 的 LLM 生成）使用的供应商。
# deepseek-v4-flash 对"提取命题/JSON 数组"这类长列表生成会挂起，
# 故默认用 qwen（中文质量好、稳定），可用环境变量 INDEX_LLM_TYPE 覆盖。
INDEX_LLM_TYPE = os.getenv("INDEX_LLM_TYPE", "qwen")

LLM_CONFIGS = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": "deepseek-v4-flash",
    },
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY"),
        "base_url": os.getenv("QWEN_BASE_URL"),
        "model": "qwen-max",
    },
    "openai": {
        "api_key": os.getenv("LAOZHANG_API_KEY"),
        "base_url": os.getenv("LAOZHANG_BASE_URL"),
        "model": "gpt-4o-mini",
    },
}

LLM_API_KEY = LLM_CONFIGS[LLM_TYPE]["api_key"]
LLM_BASE_URL = LLM_CONFIGS[LLM_TYPE]["base_url"]
LLM_MODEL = LLM_CONFIGS[LLM_TYPE]["model"]
LLM_TEMPERATURE = 0.0

# ---------- Embedding (bge large) ----------
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# ---------- Milvus ----------
MILVUS_URI = "http://localhost:19530"
DB_NAME = "vehicle"
PDF_COLLECTION = "pdf_collection"
MD_COLLECTION = "md_collection"
# 优化方案集合
SEMANTIC_COLLECTION = "md_semantic_collection"
PROPOSITION_COLLECTION = "md_proposition_collection"
HYPE_COLLECTION = "md_hype_collection"

# ---------- 分块 / 检索 ----------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K = 5

# ---------- 测试集 / 评测结果 ----------
TESTSET_JSON = OUTPUT_DIR / "testset.json"
TESTSET_CSV = OUTPUT_DIR / "testset.csv"
TESTSET_30 = OUTPUT_DIR / "testset_30.json"  # 精简后的优化评测集
OPT_RES_DIR = RES_DIR / "optimization"        # 优化方案对比结果
EVAL_PDF_CSV = RES_DIR / "eval_pdf.csv"
EVAL_MD_CSV = RES_DIR / "eval_md.csv"
REPORT_MD = RES_DIR / "对比报告.md"
REPORT_SUMMARY_CSV = RES_DIR / "对比汇总.csv"
