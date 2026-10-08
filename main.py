import os
import traceback
from dotenv import load_dotenv
from openai import OpenAI

# 自动加载当前目录下的 .env 文件
load_dotenv()

print("=== 启动轻量化企业级 RAG 系统（含大模型生成） ===")

try:
    from core.vector_db import ChromaRAGEngine

    # 1. 初始化数据库引擎
    rag = ChromaRAGEngine(collection_name="my_book_collection")
    print(f"知识库已就绪，当前共加载 {rag.collection.count()} 个知识切片。")

    # 2. 从环境变量中初始化硅基流动大模型客户端
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("未检测到 OPENAI_API_KEY，请检查 .env 文件配置！")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else "https://api.siliconflow.cn/v1"
    )

    # 使用硅基流动的稳定模型
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    # 3. 命令行交互检索与生成闭环
    print("\n" + "=" * 50)
    print("🎯 RAG 智能问答系统已上线，请输入您想查询的问题。")

    while True:
        query = input("\n请输入问题 (输入 'q' 退出): ").strip()

        if query.lower() == 'q':
            print("退出系统，再见！")
            break
        if not query:
            continue

        print(f"正在检索相关知识切片...")
        results = rag.search(query, top_k=3)

        if not results:
            print("未找到相关匹配内容。")
            continue

        # 提取召回的文本片段作为上下文
        context_text = "\n\n".join([doc for doc, dist in results])

        # 构造 RAG 提示词
        prompt = f"""请根据以下提供的已知参考信息，简明、准确地回答用户的问题。如果无法从已知信息中得到答案，请直接回答“根据已知信息无法回答”。

【已知参考信息】
{context_text}

【用户问题】
{query}
"""

        print("正在调用大模型生成回答...")
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的企业级 AI 助手，擅长根据检索到的知识库文档为用户提供准确解答。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=False
            )
            answer = response.choices[0].message.content

            print("\n--- 🤖 RAG 生成回答 ---")
            print(answer)
            print("-" * 30)

        except Exception as api_err:
            print(f"调用大模型 API 发生错误: {api_err}")

except Exception as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    traceback.print_exc()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    input("程序发生错误，按回车键退出...")