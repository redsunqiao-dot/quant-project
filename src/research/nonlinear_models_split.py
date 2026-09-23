'''
非线性模型与时间序列划分 (Non-Linear Models with Time Split)。

知识点 (Knowledge Points):
1.  **非线性模型 (Non-Linear Models)**: 许多因子与收益的关系并非线性的（例如市值因子可能呈现 U 型关系）。随机森林 (Random Forest)、XGBoost 和 LightGBM 等树模型能自动捕捉这些非线性关系和因子间的交互作用 (Interaction Effects)。
2.  **时间序列划分 (Time Series Split)**: 金融数据具有时间依赖性。为了避免**未来函数 (Look-ahead Bias)**，我们必须严格按时间顺序划分训练集、验证集和测试集（例如前 60% 时间训练，20% 验证，20% 测试），严禁随机打乱 (Shuffle)。
3.  **集成学习 (Ensemble Methods)**:
    *   **Random Forest (Bagging)**: 并行训练多棵树，取平均预测。能有效降低方差，防止过拟合。
    *   **XGBoost (Boosting)**: 串行训练，每一棵树都试图修正前一棵树的错误。预测精度通常更高，但需要精细调参。
    *   **LightGBM (Boosting)**: 基于直方图的决策树算法，训练速度更快，内存占用更少。
4.  **特征重要性 (Feature Importance)**: 树模型能输出每个因子的重要性评分（通常基于分裂时带来的纯度提升），有助于筛选最有效的 Alpha 因子。
5.  **SHAP 值 (SHAP Values)**: 基于博弈论的模型解释方法，能精确计算每个特征对每个预测的贡献值，提供更细粒度的模型解释（正向/负向影响）。
6.  **滚动窗口 (Rolling Window)**: 在时间轴上滚动推进训练-验证-测试窗口，模拟真实的“伴随式”投资过程，确保样本外 (Out-of-Sample, OOS) 评估的有效性。
'''

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.utils import ensure_dir, infer_factor_columns, load_real_panel, zscore_by_date  # type: ignore
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import shap

def fit_model(x_train: np.ndarray, y_train: np.ndarray, model_type: str = "auto"):
    """
    拟合非线性模型。支持 XGBoost、LightGBM 和 Random Forest。

    Args:
        x_train: 训练特征矩阵 (Sample x Features)
        y_train: 训练标签向量 (Sample x 1)
        model_type: 模型类型 ("auto", "xgboost", "lightgbm", "random_forest")

    Returns:
        (model_name, model) 元组
    """
    # -------------------------------------------------------------------------
    # 1. XGBoost (eXtreme Gradient Boosting)
    # -------------------------------------------------------------------------
    if model_type == "xgboost" and xgb is not None:
        # XGBoost 参数详解:
        # n_estimators: 迭代次数（树的数量）。越多越拟合，但也越容易过拟合。
        # max_depth: 树深。金融数据通常信噪比低，不宜过深（3-5层即可），防止记住噪声。
        # learning_rate: 学习率。越小模型越稳健，但收敛越慢，需配合较大的 n_estimators。
        # subsample: 样本采样率。每棵树只用 80% 的样本，增加随机性，防过拟合。
        # colsample_bytree: 特征采样率。每棵树只用 80% 的特征，类似 Random Forest。
        model = xgb.XGBRegressor(
            n_estimators=200,    
            max_depth=4,         
            learning_rate=0.05,  
            subsample=0.8,       
            colsample_bytree=0.8,
            objective="reg:squarederror", # 损失函数: 均方误差 (MSE)
            n_jobs=-1,           # 并行计算：使用所有 CPU 核心
            random_state=42
        )
        model.fit(x_train, y_train)
        return "xgboost", model

    # -------------------------------------------------------------------------
    # 2. LightGBM (Light Gradient Boosting Machine)
    # -------------------------------------------------------------------------
    if model_type == "lightgbm" and lgb is not None:
        # LightGBM 特点:
        # 速度快：使用直方图算法和单边梯度采样 (GOSS)。
        # 内存省：对类别特征支持更好。
        # 参数含义大致同 XGBoost。
        model = lgb.LGBMRegressor(
            n_estimators=200,    
            max_depth=4,         
            learning_rate=0.05,  
            subsample=0.8,       
            colsample_bytree=0.8,
            objective="regression",
            n_jobs=-1,           
            random_state=42,
            verbose=-1           # 静默模式，减少日志输出
        )
        model.fit(x_train, y_train)
        return "lightgbm", model

    # -------------------------------------------------------------------------
    # 3. Random Forest (随机森林)
    # -------------------------------------------------------------------------
    if model_type == "random_forest" and RandomForestRegressor is not None:
        # Random Forest 特点:
        # Bagging 算法：所有树并行训练，最后取平均。
        # 鲁棒性强：对参数不敏感，不易过拟合，适合作为 Baseline。
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,         # 限制深度，避免模型文件过大
            random_state=42,
            n_jobs=-1            
        )
        model.fit(x_train, y_train)
        return "random_forest", model

    # -------------------------------------------------------------------------
    # 4. 自动选择 (Auto Selection)
    # -------------------------------------------------------------------------
    # 优先级: LightGBM (最快) > XGBoost (最准) > Random Forest (最稳)
    if model_type == "auto":
        if lgb is not None:
            return fit_model(x_train, y_train, "lightgbm")
        if xgb is not None:
            return fit_model(x_train, y_train, "xgboost")
        if RandomForestRegressor is not None:
            return fit_model(x_train, y_train, "random_forest")

    # 如果所有库都未安装
    return "none", None


def compute_shap_values(model, x_sample: np.ndarray, model_name: str):
    """
    计算 SHAP 值，用于解释模型预测。

    知识点:
    SHAP (SHapley Additive exPlanations) 基于博弈论，将预测值分解为每个特征的贡献之和。
    相比 Feature Importance，SHAP 能告诉我们特征是正向还是负向影响结果。

    Args:
        model: 训练好的模型对象
        x_sample: 用于计算的样本矩阵 (Sample x Features)
        model_name: 模型名称字符串

    Returns:
        shap_values: SHAP 值矩阵
        explainer: 解释器对象
    """
    if shap is None:
        return None, None

    try:
        # 针对树模型 (Tree-based Models)，使用 TreeExplainer
        # TreeExplainer 经过高度优化，速度远快于通用的 KernelExplainer
        if model_name in ["xgboost", "lightgbm", "random_forest"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_sample)
        else:
            # 兜底逻辑：通用解释器
            explainer = shap.Explainer(model, x_sample)
            shap_values = explainer(x_sample)

        return shap_values, explainer
    except Exception as e:
        print(f"Warning: Failed to compute SHAP values: {e}")
        return None, None


def run(
    output_dir: str = "./outputs/day13_multifactor",
    data_dir: str = "./data",
    ret_horizon: str = "1vwap_pct",
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = 240,
    model_type: str = "auto",
    enable_rolling: bool = True,
    enable_shap: bool = True,
) -> None:
    """
    主流程：演示非线性模型在时序划分数据上的训练与特征重要性分析。
    """
    print("=" * 80)
    print("非线性模型与时间序列划分演示 (Non-Linear Models Demo)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. 准备数据
    # -------------------------------------------------------------------------
    print("\n[1/6] 加载数据...")
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

    print(f"   因子数量: {len(factor_cols)}")
    print(f"   数据形状: {panel.shape}")

    # -------------------------------------------------------------------------
    # 2. 数据预处理
    # -------------------------------------------------------------------------
    print("\n[2/6] 数据预处理 (Preprocessing)...")
    # Z-Score 标准化：非常重要！
    # 作用：消除不同因子的量纲差异（如市值是千亿级，换手率是0.01级），
    # 并让数据在横截面上分布更均匀。
    panel = zscore_by_date(panel, factor_cols)
    
    # 去除缺失值：
    # 大多数 Scikit-Learn 模型不支持 NaN 输入。
    # 虽然 XGBoost/LightGBM 可以处理 NaN，但预先清洗更安全。
    panel = panel.dropna(subset=factor_cols + ["ret"])
    print(f"   清洗后数据形状: {panel.shape}")

    # -------------------------------------------------------------------------
    # 3. 时间序列切分 (Time Series Split)
    # -------------------------------------------------------------------------
    # 核心原则: 严禁使用未来数据预测过去 (No Look-ahead Bias)。
    # 必须按时间轴切分，不能随机 Shuffle。
    print("\n[3/6] 时间序列切分策略...")
    dates = sorted(panel["date"].unique())
    total_dates = len(dates)

    # 划分比例: 60% 训练 / 20% 验证 / 20% 测试
    # 验证集 (Validation): 用于调参、Early Stopping。
    # 测试集 (Test): 这里的“测试集”指最后一段保留数据，用于一次性评估。
    train_end_idx = int(total_dates * 0.6)
    val_end_idx = int(total_dates * 0.8)

    train_dates = set(dates[:train_end_idx])
    val_dates = set(dates[train_end_idx:val_end_idx])
    test_dates = set(dates[val_end_idx:])

    print(f"   总日期数: {total_dates}")
    print(f"   训练集: {len(train_dates)} 天 ({dates[0]} ~ {dates[train_end_idx-1]})")
    print(f"   验证集: {len(val_dates)} 天 ({dates[train_end_idx]} ~ {dates[val_end_idx-1]})")
    print(f"   测试集: {len(test_dates)} 天 ({dates[val_end_idx]} ~ {dates[-1]})")

    # 保存切分信息供后续分析
    split_info = pd.DataFrame({
        "split": ["train", "validate", "test"],
        "start_date": [dates[0], dates[train_end_idx], dates[val_end_idx]],
        "end_date": [dates[train_end_idx-1], dates[val_end_idx-1], dates[-1]],
        "num_dates": [len(train_dates), len(val_dates), len(test_dates)],
    })

    # 根据日期筛选 DataFrame
    train = panel[panel["date"].isin(train_dates)]
    val = panel[panel["date"].isin(val_dates)]
    test = panel[panel["date"].isin(test_dates)]

    # 转换为 NumPy 数组 (加速训练)
    x_train, y_train = train[factor_cols].values, train["ret"].values
    x_val, y_val = val[factor_cols].values, val["ret"].values
    x_test, y_test = test[factor_cols].values, test["ret"].values

    if train.empty:
        raise ValueError("Training data is empty after filtering.")

    # -------------------------------------------------------------------------
    # 4. 模型训练
    # -------------------------------------------------------------------------
    print(f"\n[4/6] 模型训练 (类型: {model_type})...")
    model_name, model = fit_model(x_train, y_train, model_type)

    if model is None:
        print("ERROR: 无法训练模型，请安装 xgboost、lightgbm 或 scikit-learn")
        return

    print(f"   使用模型: {model_name}")

    # -------------------------------------------------------------------------
    # 5. 特征重要性分析
    # -------------------------------------------------------------------------
    # Tree Models 自带的 Feature Importance 通常基于 "Split Gain" (分裂增益)
    # 即该特征在所有树的所有分裂中，带来的纯度提升总和。
    print("\n[5/6] 提取特征重要性...")
    try:
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "factor": factor_cols,
            "importance": importance,
            "model": model_name
        }).sort_values("importance", ascending=False)

        print(f"   Top 5 重要特征 (Based on Split Gain):")
        for idx, row in feature_importance_df.head(5).iterrows():
            print(f"      {row['factor']}: {row['importance']:.4f}")
    except AttributeError:
        print("   该模型不支持 feature_importances_ 属性。")
        feature_importance_df = pd.DataFrame()

    # -------------------------------------------------------------------------
    # 6. SHAP 值分析
    # -------------------------------------------------------------------------
    shap_summary_df = None
    if enable_shap and shap is not None:
        print("\n[6/6] 计算 SHAP 值 (Model Interpretation)...")
        # 采样: 如果验证集太大，计算 SHAP 会很慢，这里采样 1000 个样本
        shap_sample_size = min(1000, len(x_val))
        shap_sample_indices = np.random.choice(len(x_val), shap_sample_size, replace=False)
        x_shap_sample = x_val[shap_sample_indices]

        shap_values, explainer = compute_shap_values(model, x_shap_sample, model_name)

        if shap_values is not None:
            # 计算平均绝对 SHAP 值 (Mean Absolute SHAP Value) 作为特征整体重要性
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_summary_df = pd.DataFrame({
                "factor": factor_cols,
                "mean_abs_shap": mean_shap,
                "model": model_name
            }).sort_values("mean_abs_shap", ascending=False)

            print(f"   Top 5 SHAP 特征:")
            for idx, row in shap_summary_df.head(5).iterrows():
                print(f"      {row['factor']}: {row['mean_abs_shap']:.4f}")
        else:
            print("   SHAP 计算失败")
    else:
        print("\n[6/6] 跳过 SHAP 值计算 (需要安装 shap 库)")

    # -------------------------------------------------------------------------
    # 7. 滚动窗口评估 (Rolling Window / Walk-Forward)
    # -------------------------------------------------------------------------
    # 这是一个可选的高级步骤，更贴近实战。
    rolling_results = None
    if enable_rolling:
        print("\n[额外] 滚动窗口评估 (Walk-Forward Validation)...")
        rolling_results = rolling_window_evaluation(
            panel, factor_cols, dates,
            train_window=120,  # 训练 120 天
            val_window=40,     # 验证 40 天
            test_window=40,    # 测试 40 天
            step_size=20,      # 每次向前滚动 20 天
            model_type=model_type
        )
        if rolling_results is not None:
            print(f"   完成 {len(rolling_results)} 个滚动窗口的评估")

    # -------------------------------------------------------------------------
    # 8. 保存结果
    # -------------------------------------------------------------------------
    print("\n保存结果...")
    out = ensure_dir(output_dir)

    split_info.to_csv(out / "time_split_info.csv", index=False)
    if not feature_importance_df.empty:
        feature_importance_df.to_csv(out / "feature_importance_nonlinear.csv", index=False)

    if shap_summary_df is not None:
        shap_summary_df.to_csv(out / "shap_summary.csv", index=False)

    if rolling_results is not None:
        rolling_results.to_csv(out / "rolling_window_results.csv", index=False)

    print(f"\n✓ 非线性模型分析完成!")
    print(f"  模型: {model_name}")
    print(f"  结果已保存至: {out}")
    print("=" * 80)


def rolling_window_evaluation(
    panel: pd.DataFrame,
    factor_cols: list,
    dates: list,
    train_window: int = 120,
    val_window: int = 40,
    test_window: int = 40,
    step_size: int = 20,
    model_type: str = "auto",
):
    """
    滚动窗口评估 (Walk-Forward Validation)。

    逻辑:
    1.  从 start_idx 开始，选取 train_window 长度的数据作为训练集。
    2.  紧接着选取 val_window 作为验证集 (可选，用于早停)。
    3.  最后选取 test_window 作为测试集，计算样本外预测表现。
    4.  向前移动 step_size，重复上述步骤。

    这种方法能检测模型在不同市场环境下的稳定性。

    Args:
        panel: 面板数据
        factor_cols: 因子列名
        dates: 日期列表
        train_window: 训练窗口大小（天数）
        val_window: 验证窗口大小（天数）
        test_window: 测试窗口大小（天数）
        step_size: 滚动步长（天数）
        model_type: 模型类型

    Returns:
        滚动窗口评估结果 DataFrame
    """
    results = []
    total_dates = len(dates)
    window_size = train_window + val_window + test_window

    print(f"   开始滚动回测: 总天数={total_dates}, 窗口总长={window_size}, 步长={step_size}")

    # 循环滚动
    # range(start, stop, step)
    for start_idx in range(0, total_dates - window_size + 1, step_size):
        # 1. 定义当前窗口的边界索引
        train_end_idx = start_idx + train_window
        val_end_idx = train_end_idx + val_window
        test_end_idx = val_end_idx + test_window

        # 2. 获取对应的日期集合
        train_dates_window = set(dates[start_idx:train_end_idx])
        val_dates_window = set(dates[train_end_idx:val_end_idx])
        test_dates_window = set(dates[val_end_idx:test_end_idx])

        # 3. 根据日期筛选数据
        train_data = panel[panel["date"].isin(train_dates_window)]
        val_data = panel[panel["date"].isin(val_dates_window)]
        test_data = panel[panel["date"].isin(test_dates_window)]

        # 如果某一段数据为空（例如停牌或数据缺失），则跳过
        if train_data.empty or val_data.empty or test_data.empty:
            continue

        x_train = train_data[factor_cols].values
        y_train = train_data["ret"].values
        x_val = val_data[factor_cols].values
        y_val = val_data["ret"].values
        x_test = test_data[factor_cols].values
        y_test = test_data["ret"].values

        # 4. 训练模型 (只在当前窗口的训练集上)
        model_name, model = fit_model(x_train, y_train, model_type)
        if model is None:
            continue

        # 5. 评估模型性能
        from sklearn.metrics import mean_squared_error
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        # 6. 计算 IC (Information Coefficient)
        # IC = 预测值与真实值的皮尔逊相关系数
        # IC > 0.05 通常被认为是有效的 Alpha 信号
        train_ic = np.corrcoef(y_train, train_pred)[0, 1]
        val_ic = np.corrcoef(y_val, val_pred)[0, 1]
        test_ic = np.corrcoef(y_test, test_pred)[0, 1]

        results.append({
            "window_id": len(results) + 1,
            "train_start": dates[start_idx],
            "train_end": dates[train_end_idx - 1],
            "val_start": dates[train_end_idx],
            "val_end": dates[val_end_idx - 1],
            "test_start": dates[val_end_idx],
            "test_end": dates[test_end_idx - 1],
            "model": model_name,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "test_mse": test_mse,
            "train_ic": train_ic,
            "val_ic": val_ic,
            "test_ic": test_ic,
        })

    if not results:
        print("   Warning: 没有完成任何滚动窗口的评估。")
        return None

    return pd.DataFrame(results)


if __name__ == "__main__":
    run(model_type='random_forest')