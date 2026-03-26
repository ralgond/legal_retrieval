import re
from tqdm import tqdm

def chunk_with_sliding_window(
    text: str, 
    chunk_size: int, 
    overlap: int,
) -> list[str]:
    # 1. 按空白符分割成 token 列表
    tokens = re.split(r"\s+", text)
    
    # 2. 参数验证
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) 必须小于 chunk_size ({chunk_size})")
    
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    
    # 3. 处理边界情况
    if len(tokens) <= chunk_size:
        return [text]  # 文本太短，直接返回原文
    
    # 4. 滑动窗口切块
    chunks = []
    step = chunk_size - overlap  # 每次滑动的步长

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        
        # 如果剩余 token 不足 chunk_size，将剩余部分合并到最后一个块
        if len(chunk_tokens) < chunk_size and chunks:
            chunks[-1] = ' '.join(tokens[start:])
            break
        
        chunk_text = ' '.join(chunk_tokens)
        chunks.append(chunk_text)
        
        # 如果已经到达末尾，退出
        if end >= len(tokens):
            break
    
    return chunks
    

def batch_chunk_with_sliding_window(
    documents: list[str], 
    chunk_size: int, 
    overlap: int,
) -> list[str]:
    ret = []
    for doc in tqdm(documents, total=len(documents), desc="chunk docuemnts"):
        chunks = chunk_with_sliding_window(doc['text'], chunk_size, overlap)
        for chunk in chunks:
            new_doc = {'text': chunk, 'citation': doc['citation']}
            ret.append(new_doc)
    return ret

def sliding_window_merge_last_unique(
    data: list[str],
    window_size: int,
    step: int
) -> list[list[str]]:
    n = len(data)
    windows = []

    i = 0
    while i < n:
        window = data[i:i + window_size]

        if len(window) < window_size:
            if windows:
                prev = windows[-1]
                # 只追加不重复的部分
                prev.extend(x for x in window if x not in prev)
            else:
                windows.append(window)
            break
        else:
            windows.append(window)

        i += step

    return [' '.join(w) for w in windows]

if __name__ == "__main__":
    lst = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    print(sliding_window_merge_last_unique(lst, 4, 2))

