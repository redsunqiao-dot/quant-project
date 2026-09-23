"""
线性模型加权 (Ridge / Lasso)。

知识点 (Knowledge Points):
1.  **线性模型加权 (Linear Model Weighting)**:
    *   **原理**: 将资产收益率 (Return) 作为因变量 (Y)，各个因子暴露度 (Factor Exposures) 作为自变量 (X)，通过多元线性回归模型 (OLS) 来确定每个因子的最佳权重 (Coefficients)。
    *   **公式**: Returns = w1*F1 + w2*F2 + ... + wn*Fn + Error
    *   **目标**: 找到一组权重 w，使得预测误差平方和最小。

2.  **多重共线性与正则化 (Multicollinearity & Regularization)**:
    *   **问题**: 金融因子之间往往存在高度相关性（如不同期限的动量因子、不同方法的估值因子）。直接使用 OLS 会导致参数估计极不稳定，方差过大（过拟合）。
    *   **正则化**: 在损失函数中增加一个“惩罚项”，约束系数的大小，从而提高模型的泛化能力。

3.  **Ridge Regression (岭回归, L2 Regularization)**:
    *   **惩罚项**: 系数的平方和 (alpha * ||w||^2)。
    *   **特性**: 倾向于让系数均匀地变小，但不为零。适合处理多重共线性问题，保留所有因子，但降低其权重，起到“平滑”作用。
    *   **闭式解**: Ridge 回归可以通过矩阵运算直接求出解析解: w = (X'X + alpha*I)^(-1) X'Y，计算效率极高。

4.  **Lasso Regression (套索回归, L1 Regularization)**:
    *   **惩罚项**: 系数的绝对值和 (alpha * ||w||)。
    *   **特性**: 倾向于让部分系数直接变为零。具有**特征选择 (Feature Selection)** 的功能，能自动剔除无效或冗余的因子，生成稀疏模型。
    *   **应用**: 当因子数量众多且含有大量噪音时，Lasso 能帮助筛选出真正的有效因子。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from src.common.utils import (  # type: ignore
    ensure_dir,
    infer_factor_columns,
    load_real_panel,
    normalize_weights,
    zscore_by_date,
)

from sklearn.linear_model import Lasso, Ridge

def ridge_closed_form(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ridge 回归的闭式解 (不依赖 sklearn)。
    w = (X^T X + alpha * I)^(-1) X^T y
    
    数学推导教学:
    1. 普通最小二乘法 (OLS) 试图最小化 ||y - Xw||^2，解为 w = (X^T X)^(-1) X^T y。
    2. 当 X 存在多重共线性（因子高度相关）时，X^T X 接近奇异矩阵（不可逆），导致 w 数值极不稳定。
    3. Ridge 引入惩罚项 alpha * ||w||^2。
    4. 对损失函数求导后，解变为 w = (X^T X + alpha * I)^(-1) X^T y。
    5. "alpha * I" 相当于给矩阵的对角线加上一个正数，强制使其满秩（可逆），从而稳定了数值解。
    
    Args:
        x: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        alpha: 正则化强度 (L2 惩罚系数)
        
    Returns:
        权重向量 (n_features,)
    """
    # 1. 计算协方差矩阵 X^T X (Gram Matrix)
    xtx = x.T @ x
    
    # 2. 添加 L2 正则项 (alpha * 单位矩阵) 到对角线
    # 这就像给协方差矩阵“加了点料”，使其变为满秩矩阵，从而可逆
    ridge = xtx + alpha * np.eye(x.shape[1])
    
    # 3. 求解线性方程组 (计算逆矩阵并求解权重)
    # 使用 pinv (伪逆) 比 inv 更稳健
    return np.linalg.pinv(ridge) @ x.T @ y


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    使用 Ridge 回归拟合因子权重。
    
    教学点:
    Ridge 适合处理“谁也不服谁”的情况。如果你有一堆相关的因子（比如估值因子 BP, EP, SP），
    Ridge 会倾向于给它们每个人都分配一点权重，而不是只选一个扔掉其他的。
    这有助于分散风险，降低模型方差。
    
    Args:
        alpha: 控制正则化强度。alpha 越大，惩罚越重，权重越趋近于 0，模型越简单（欠拟合风险）；
               alpha 越小，越接近普通 OLS（过拟合风险）。
    """
    if Ridge is None:
        print("Warning: sklearn not found, using numpy closed-form solution for Ridge.")
        return ridge_closed_form(x, y, alpha)
    
    # fit_intercept=True: 允许模型拟合截距（即市场平均收益 Alpha），
    # 这样因子权重解释的是超额收益部分
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, y)
    return model.coef_


def fit_lasso(x: np.ndarray, y: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    """
    使用 Lasso 回归拟合因子权重。
    
    教学点:
    Lasso 是一个严厉的“面试官”。它使用 L1 正则化 (菱形约束区域)，这在几何上更容易在坐标轴上获得最优解。
    这意味着它会把那些贡献不大的因子的权重直接砍成 0。
    如果你有 100 个因子但只想选出最有效的 5 个，用 Lasso。
    """
    if Lasso is None:
        print("Warning: sklearn not found, cannot run Lasso.")
        return None
    
    # Lasso 没有闭式解，需要迭代求解 (Coordinate Descent 等算法)
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
    model.fit(x, y)
    return model.coef_


def run(
    output_dir: str = "./outputs/day13_multifactor",
    data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dates: Optional[int] = 240,
    alpha: float = 1.0, # 正则化系数
) -> None:
    """
    比较 Ridge 和 Lasso 两种线性模型产生的因子权重。
    
    本示例使用的是“全样本静态回归”(Static Regression)，即用过去所有的历史数据算出一组固定的权重。
    在真实实战中，通常会使用“滚动窗口回归”(Rolling Regression)，例如每个月用过去 24 个月的数据重算一次权重，
    以适应市场风格的切换。
    """
    # 1. 准备真实数据
    panel = load_real_panel(
        data_dir=data_dir,
        ret_col=ret_horizon,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )
    factor_cols = infer_factor_columns(panel)
    if not factor_cols:
        raise ValueError("No factor columns found in the real panel data.")
    
    # 2. 预处理：标准化 (Z-Score)
    # 教学重点: 为什么必须做标准化？
    # 线性回归试图通过系数 w 来解释 y 的变化。如果因子 A 的波动范围是 [0, 1000]，因子 B 是 [0, 0.01]。
    # 即使两个因子同等重要，模型为了平衡数量级，会给 A 一个极小的权重，给 B 一个极大的权重。
    # 这会导致正则化项 (alpha * w^2) 失效，因为它主要惩罚了大权重的 B，而忽略了 A。
    # Z-Score 将所有因子拉回同一起跑线 (均值 0，标准差 1)，让权重的数值真正反映因子的重要性。
    panel = zscore_by_date(panel, factor_cols)
    panel = panel.dropna(subset=factor_cols + ["ret"])  # 丢弃缺失值样本

    # --- 新增步骤：自动校正因子方向 (Factor Sign Adjustment) ---
    # 教学重点: 为什么要做因子翻转？
    # 线性模型本身可以处理负系数（代表负相关）。但在后续的“权重归一化”步骤中，
    # 为了构建纯多头组合，我们通常会强制要求权重非负 (clip(0))。
    # 这会导致那些有效的“反向因子”（如高波动率通常预示低收益）被误杀，权重变为 0。
    # 解决方案：在回归前，检测因子与收益的相关性 (IC)，如果是负相关，就提前把因子值乘以 -1。
    # 这样负负得正，反向因子就变成了正向因子，模型就能给它分配正权重了。
    
    print("\n[因子方向自动校正]")
    flipped_factors = []
    for col in factor_cols:
        # 计算该因子的全局 IC (简单相关系数)
        ic = panel[col].corr(panel["ret"])
        if ic < 0:
            panel[col] = -panel[col]
            flipped_factors.append(col)
            print(f"  -> {col}: IC={ic:.4f} < 0, 已执行翻转 (x -1)")
        else:
            print(f"  -> {col}: IC={ic:.4f} >= 0, 保持原样")
            
    if not flipped_factors:
        print("  没有因子需要翻转。")
    # -----------------------------------------------------

    x = panel[factor_cols].values
    y = panel["ret"].values

    # 3. Ridge 回归
    # Ridge 倾向于保留所有特征，权重分布相对均匀
    ridge_w_raw = fit_ridge(x, y, alpha)
    print(f"\n[Ridge Raw Coefficients (alpha={alpha})]:\n", 
          pd.Series(ridge_w_raw, index=factor_cols).round(6))
    
    # 归一化权重以便比较（使其和为1，便于直观理解相对重要性）
    # 注意：normalize_weights 会将负系数截断为 0 (Long-only)
    ridge_w = normalize_weights(pd.Series(ridge_w_raw, index=factor_cols))

    records = [
        pd.DataFrame({"method": "ridge", "factor": factor_cols, "weight": ridge_w.values})
    ]

    # 4. Lasso 回归
    # Lasso 倾向于产生稀疏解，某些权重会变为 0
    # Lasso 对 alpha 非常敏感，如果 alpha 过大，所有系数都会变为 0
    lasso_w_raw = fit_lasso(x, y, alpha)
    if lasso_w_raw is not None:
        print(f"\n[Lasso Raw Coefficients (alpha={alpha})]:\n", 
              pd.Series(lasso_w_raw, index=factor_cols).round(6))
        
        lasso_w = normalize_weights(pd.Series(lasso_w_raw, index=factor_cols))
        records.append(
            pd.DataFrame(
                {"method": "lasso", "factor": factor_cols, "weight": lasso_w.values}
            )
        )

    # 5. 保存结果
    weights_df = pd.concat(records, ignore_index=True)
    out = ensure_dir(output_dir)
    weights_df.to_csv(out / "weights_linear.csv", index=False)
    
    print(f"线性模型权重计算完成。结果已保存至 {out}/weights_linear.csv")
    print(weights_df)


if __name__ == "__main__":
    # 使用更小的 alpha，因为收益率数据 (y) 的数值通常很小 (e.g., 0.001)
    run(alpha=1e-4)
