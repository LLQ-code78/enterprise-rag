import os
from pypdf import PdfReader
from docx import Document


def load_document_text(file_path: str) -> str:
    """根据文件后缀自动选择对应 Parser 提取全文内容"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return ""

    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    else:
        print(f"暂不支持的格式: {ext}")

    return text


def chunk_text(file_path: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """读取并切分文档为规范的文本块"""
    full_text = load_document_text(file_path)
    if not full_text:
        return []

    raw_lines = [line.strip() for line in full_text.split("\n") if line.strip()]

    chunks = []
    for line in raw_lines:
        if len(line) <= chunk_size:
            chunks.append(line)
        else:
            start = 0
            while start < len(line):
                end = start + chunk_size
                chunks.append(line[start:end])
                start += (chunk_size - overlap)

    return chunks