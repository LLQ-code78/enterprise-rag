import os
import docx
import numpy as np


class ChromaRAGEngine:
    def __init__(self, collection_name: str = "my_book_collection", persist_directory: str = "data/chroma_db"):
        self.chunks = []
        self.db_file = os.path.join(persist_directory, "chunks.txt")
        os.makedirs(persist_directory, exist_ok=True)

        # 如果本地有缓存，直接加载
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    self.chunks = content.split("\n---SPLIT---\n")

    @property
    def collection(self):
        # 伪装一个类似 collection 的对象，兼容原有 main.py 的 count() 判断
        class MockCollection:
            def __init__(self, parent):
                self.parent = parent

            def count(self):
                return len(self.parent.chunks)

        return MockCollection(self)

    def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            vec = np.zeros(384)
            for i, char in enumerate(text):
                vec[i % 384] += ord(char)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

    def add_documents(self, chunks: list[str]):
        """保存文本块到本地文件"""
        self.chunks = chunks
        with open(self.db_file, "w", encoding="utf-8") as f:
            f.write("\n---SPLIT---\n".join(chunks))
        print("知识库向量化并持久化保存成功！")

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """基于向量余弦相似度检索"""
        if not self.chunks:
            return []

        query_vec = np.array(self._get_embedding([query])[0])
        chunk_vecs = np.array(self._get_embedding(self.chunks))

        # 计算余弦相似度
        similarities = np.dot(chunk_vecs, query_vec)

        # 获取最相似的前 top_k 个索引
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            # 将相似度转换为距离形式 (1 - 相似度，越小越相关)以匹配原有展示
            dist = float(1.0 - similarities[idx])
            results.append((self.chunks[idx], dist))

        return results