API_KEY="sk-ddff266fc036492b8854d5411ff0e5d5"

from openai import OpenAI

from tqdm import tqdm

def listwise_rank(client, query: str, candidates: list[str], model: str = "deepseek-chat") -> list[str]:
    """
    用 DeepSeek API 对候选字符串做 listwise 排序
    
    Args:
        query: 查询字符串
        candidates: 待排序的字符串列表
        model: 使用的模型，deepseek-chat 或 deepseek-reasoner
    
    Returns:
        按相关性从高到低排好序的字符串列表
    """
    # 构建候选列表的编号文本
    numbered = "\n".join(f"[{i+1}] {text}" for i, text in enumerate(candidates))
    n = len(candidates)
    
    prompt = f"""你是一个精通德语/法语/意大利语的文本相关性排序助手。
请根据查询（Query），对以下 {n} 条候选文本按相关性从高到低排序。

Query: {query}

候选文本：
{numbered}

请只输出排序后的编号列表，格式为方括号内的数字，用 > 分隔，例如：
[3] > [1] > [4] > [2]

不要输出任何其他内容。"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的文本相关性排序助手，只输出排序结果，不做任何解释。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,  # 排序任务用 0，保证确定性
    )
    
    raw_output = response.choices[0].message.content.strip()
    # print(f"模型输出: {raw_output}")
    
    # 解析输出，提取编号顺序
    import re
    indices = re.findall(r'\[(\d+)\]', raw_output)
    indices = [int(i) - 1 for i in indices]  # 转为 0-based index
    
    # 处理模型漏掉某些编号的情况
    all_indices = set(range(n))
    missing = all_indices - set(indices)
    indices.extend(list(missing))  # 漏掉的追加到末尾
    
    # 按排序后的顺序返回原字符串
    ranked = [candidates[i] for i in indices if i < n]
    return ranked

def sliding_window_rank(client, query: str, candidates: list[str], window_size: int = 20, step: int = 5) -> list[str]:
    """
    滑动窗口 listwise 排序，适合候选数量较多的场景
    从后往前滑动，每次对窗口内的文本做局部排序
    """
    ranked = candidates.copy()
    n = len(ranked)
    
    # 从列表末尾开始向前滑动
    start = max(0, n - window_size)
    while start >= 0:
        end = min(start + window_size, n)
        window = ranked[start:end]
        
        # 对当前窗口做 listwise 排序
        sorted_window = listwise_rank(client, query, window)
        ranked[start:end] = sorted_window
        
        if start == 0:
            break
        start = max(0, start - step)
    
    return ranked

import text_chunk
def map_reduce(client, query: str, candidates: list[str], window_size: int = 40, top_k_in_map=5):
    chunks = text_chunk.sliding_window_merge_last_unique(candidates, window_size, window_size)

    map_result = []
    
    for chunk in chunks:
        map_result.extend(listwise_rank(client, query, chunk)[:top_k_in_map])

    return listwise_rank(client, query, map_result) # reduce

# ========== 使用示例 ==========
if __name__ == "__main__":
    query = "如何学习机器学习"
    
    candidates = [
        "深度学习入门教程：神经网络基础",
        "今天天气怎么样",
        "机器学习算法：从线性回归到随机森林",
        "Python 数据科学实战课程",
        "如何做一道红烧肉",
        "Scikit-learn 官方文档与最佳实践",
    ]
    
    ranked_results = listwise_rank(query, candidates)
    
    print("\n排序结果（相关性从高到低）：")
    for i, text in enumerate(ranked_results, 1):
        print(f"{i}. {text}")