# 主要是处理难样本

## 难样本的定义：
用feature来排
- 覆盖的cc数量
- SUM(cc_score*1/math.log(2+first_sentence_idx)[:3])