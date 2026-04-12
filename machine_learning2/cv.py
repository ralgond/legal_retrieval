import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import numpy as np
from sklearn.model_selection import GroupKFold

X = np.load("../data/ml2/lgbm_data/train_features.npy")
y = np.load("../data/ml2/lgbm_data/train_labels.npy") 
group_sizes = np.load("../data/ml2/lgbm_data/train_groups.npy")

# ==========================================
# 2. 核心步骤：从 group 列表构建 query_id
# ==========================================
# GroupKFold 需要一个与 X 行数相同的 query_id 数组。
# 我们需要把 [10, 5, ...] 这种 group 信息转换成 [0,0,0...0, 1,1...1, 2...] 这种形式。
query_ids = np.repeat(np.arange(len(group_sizes)), group_sizes)

# 简单验证一下长度是否匹配
assert len(query_ids) == X.shape[0], "Query IDs 长度与特征 X 行数不匹配！"


# ==========================================
# 3. 创建分组折 (GroupKFold)
# ==========================================
gkf = GroupKFold(n_splits=5)
# 生成 (train_index, val_index) 的索引对
folds = list(gkf.split(X, y, groups=query_ids))

# ==========================================
# 4. 准备 LightGBM 数据集
# ==========================================
# 注意：在 lgb.Dataset 中不需要传 group，group 信息在 cv 函数中通过 folds 隐式传递
lgb_train = lgb.Dataset(X, y, group=group_sizes)

# ==========================================
# 5. 配置参数与执行 CV
# ==========================================
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 5, 10, 20],
    'learning_rate': 0.01,
    'num_leaves': 32,
    'max_depth': 5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'seed': 42,

    "min_data_in_leaf": 1, 
    "lambdarank_truncation_level": 200, 
}

print("开始 5-Fold 交叉验证...")
cv_results = lgb.cv(
    params,
    lgb_train,
    num_boost_round=1000,
    folds=folds,            # 传入刚才生成的 folds
    seed=42,
    callbacks=[
        early_stopping(stopping_rounds=50),  # 替代 early_stopping_rounds=50
        log_evaluation(period=10)           # 替代 verbose_eval=100
    ],
)

# ==========================================
# 6. 输出结果
# ==========================================
import pprint
pprint.pprint(cv_results)

best_iter = len(cv_results['ndcg@5-mean'])
final_score = cv_results['ndcg@5-mean'][-1]
print(f"\n✅ 训练结束！")
print(f"最佳迭代轮数: {best_iter}")
print(f"最终 NDCG@5 均值: {final_score:.4f}")

