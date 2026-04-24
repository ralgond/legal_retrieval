# 数据的质量

metric.py计算recall@200，结果为0.5348431244200559

# 数据结构
data = [
    {
        "query": "...",
        "cc_list": [
            {
                "text": "...",
                "is_positive": 1,  # 是否包含 gold citation
                "citations": [
                    "Art. 228 Abs. 1 StPO",
                    ...
                ],
                "gold_citation": "Art. 228 Abs. 1 StPO"  # 只有正样本有
            },
            ...
        ]
    }
]