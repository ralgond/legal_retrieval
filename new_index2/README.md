# Citation-Centric + Multi-scale

1. 对每个 citation：
   - 抽取 ±128 token（主窗口）

2. 同时构造：
   - ±64 token（小窗口）
   - ±256 token（大窗口）

3. 全部入库

4. query：
   - dense + BM25 混合召回