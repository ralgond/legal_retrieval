# Citation Gold Classifier
========================
目标：对CC中每个citation，预测它是否为gold citation（二分类）
 
输入信息（你实际拥有的）：
  1. citation前后的上下文文本
  2. citation在文档中被引用的次数（文档级频率）
  3. citation在court consideration中的第几句
 
架构选择：
  - 轻量版：TF-IDF上下文特征 + 位置/频率 → LogisticRegression（可解释，无需GPU）
  - 重量版：multilingual BERT (bert-base-multilingual-cased) 编码上下文 → 分类头
  - 推荐：先跑轻量版验证信号，再上BERT

