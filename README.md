# 轻量化企业级 RAG 知识库与智能检索系统

基于原生 Python 与轻量化向量检索架构设计的 RAG（检索增强生成）系统，旨在解决传统大模型在垂直领域幻觉率高、本地 Windows 环境部署冲突等痛点。

## 核心特性
- **智能数据治理与切片**：支持自动化 Word (`.docx`) 文档解析，采用动态 Chunking 与重叠度（Overlap）策略，有效防范“Lost in the Middle”效应。
- **高精度余弦检索**：实现基于本地持久化的 Top-K 向量相似度召回。
- **严谨的抗幻觉对齐**：对接硅基流动（SiliconFlow）大模型 API，内置严格的未知兜底拒答机制。

## 项目结构
mini-enterprise-rag/
│
├── core/
│   └── vector_db.py     # 核心向量计算与持久化检索模块
├── data/
│   └── my_book.docx     # 知识库源文档
├── .env.example         # 环境变量配置模板
├── main.py              # 主程序入口（交互闭环）
└── requirements.txt     # 项目依赖包