import os

import numpy as np
import openai
from openai import OpenAI

prompt="""以下是相同语义、不同语义以及完整语义的定义，按照定义将文本映射过去，按照输出格式输出结果，只保留输出格式内容。
相同语义
定义：所有文本内容都是数据，共享同一概念的数据。
示例：如“羊群”中的每只羊或“苹果”的不同实例，尽管个体特征各异，但都归类为同一个概念。
不同语义
定义：涉及信息、智慧和意图的不同层面。
信息：具体事实或不同认知下体现差异。
智慧：从信息中提炼出的高级理解或推断。
意图：二元组形式，输入（问题或现象）与输出（目标或期望结果）之间的关系。
完整语义
定义：构成知识的完整信息。
相对完整：完成特定任务所需的所有信息。
绝对完整：不仅包含完成任务所需的信息，还包括相关扩展知识。
输出格式：

相同语义(属性1): [数据1,数据2， ...]
...
不同语义(信息):[信息1,信息2， ...]
不同语义(智慧): [智慧1, 智慧2, ...]
不同语义(意图): [[意图1, 意图2, ...]
...
完整语义(绝对/相对): [知识1, 知识2, ...]
...
"""

def get_response(question,prompt):
    client=OpenAI(api_key="")
    try:
     completion = client.chat.completions.create(
     model="gpt-3.5-turbo", ##select chatbot's model
     messages=[{"role": "system", "content": prompt},
              {"role": "user", "content": question},],
     temperature = 0.5,
     max_tokens = 800
     )
     response = completion.model_dump()
     answer = response["choices"][0]['message']['content']
     return answer
    except Exception as e:
       print(f"Failed to get response due to error: {e}")
       return None

def get_embedding(text, model="text-embedding-ada-002"):
    """获取文本的嵌入向量"""
    client = OpenAI(
        api_key="")

    try:
      response = client.embeddings.create(
        input=text,
        model=model
       )
      embedding = response.data[0].embedding
      return embedding
    except openai.OpenAIError as e:
     print(f"Error occurred: {e}")
     return None

def calculate_cosine_similarity(vec_a, vec_b):
    """计算两个向量之间的余弦相似度"""
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0  # 避免除以零的情况
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def check_semantic_similarity(original_text, processed_text, threshold=0.8):
    """
    检查处理后的文本与原始文本的语义相似度。

    参数:
    - original_text: 用户输入的原始文本。
    - processed_text: 处理后的文本。
    - threshold: 判断语义相似度的阈值，默认为 0.85。

    返回:
    - 如果相似度大于等于阈值，返回 True 表示语义相似；
      否则返回 False 表示语义不相似。
    """
    original_emb = get_embedding(original_text)
    processed_emb = get_embedding(processed_text)

    if original_emb is None or processed_emb is None:
        print("Failed to obtain embeddings.")
        return False

    similarity = calculate_cosine_similarity(original_emb, processed_emb)
    print(f"Semantic similarity: {similarity:.4f}")

    return similarity >= threshold