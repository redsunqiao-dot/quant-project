"""
因子风险暴露分析 (Factor Risk Exposure Analysis)

功能说明：
    通过横截面回归分析，检测目标因子在已知风险因子（如市值、行业、Beta）上的“暴露程度”。
    输出每一天的回归系数（Beta）和拟合优度（R²）。

核心知识点讲解：
    1. 什么是“暴露” (Exposure)？
       - 在多因子模型中，我们想知道一个新的 Alpha 因子是否只是某些已知风格因子的“马甲”。
       - 例如：如果不做中性化，“高价股”因子的走势可能跟“大盘股”(Size)因子一模一样。
       - 此时，我们说该因子对 Size 因子有高暴露。这种暴露通常是我们要剔除的（中性化），或者至少是需要被认知的。

    2. 诊断工具：回归分析 (Regression)
       - 模型：Target_Factor = β1 * Size + β2 * Industry + ... + ε
       - R-squared (R方)：衡量 Target_Factor 能被风险因子解释的比例。
         - R2 > 0.6：说明你的因子大部分波动都是由风格/行业决定的，本身包含的独特信息很少。
         - R2 < 0.1：说明你的因子非常独特（Idiosyncratic），与主流风格无关，可能是一个纯粹的 Alpha。
       - Beta (系数)：暴露的方向和强度。
         - Beta_Size > 0：正向暴露，偏好大盘股。
         - Beta_Size < 0：负向暴露，偏好小盘股。
         
    3. 行业哑变量 (Dummy Variables)：
       - 将分类的行业变量转换为多个 0/1 变量，用于剔除行业板块带来的系统性影响。
"""

import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

# 常见的市值因子列名候选，用于自动识别
SIZE_COL_CANDIDATES = ["size", "ln_mktcap", "log_mktcap", "mktcap"]


class ExposureAnalyzer:
    """
    因子风险暴露分析器
    
    该类负责执行每日的横截面回归，并汇总统计因子的风格暴露情况。
    """
    
    def __init__(
        self,
        size_col: Optional[str] = None,
        extra_barra_cols: Optional[Iterable[str]] = None,
        industry_col: str = "industry",
    ):
        """
        初始化分析器
        
        Parameters:
        -----------
        size_col : str, optional
            市值因子列名。
        extra_barra_cols : list, optional
            额外的风格因子列名，例如 ['beta', 'momentum', 'volatility']。
        industry_col : str, default 'industry'
            行业分类列名。
        """
        self.size_col = size_col
        self.extra_barra_cols = list(extra_barra_cols) if extra_barra_cols else None
        self.industry_col = industry_col

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """读取交易日列表并根据起止日期筛选。"""
        with open(date_path, "rb") as f:
            dates = pickle.load(f)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def load_csv_with_index(self, path: Path, index_col: str = "code") -> pd.DataFrame:
        """通用表格读取并设置索引（兼容 CSV / Parquet）。"""
        from src.common.utils import read_frame

        df = read_frame(path)
        if index_col in df.columns:
            df = df.set_index(index_col)
        return df

    def infer_size_col(self, barra_df: pd.DataFrame) -> str:
        """自动在数据列中搜索可能的市值列名。"""
        for col in SIZE_COL_CANDIDATES:
            if col in barra_df.columns:
                return col
        raise ValueError("No size column found in Barra data")

    def build_risk_matrix(
        self,
        barra_df: pd.DataFrame,
        industry_series: Optional[pd.Series] = None,
        size_col: Optional[str] = None,
        extra_barra_cols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        构造回归用风险矩阵（Design Matrix）
        
        X = [Intercept, Size, Other_Factors, Industry_Dummies...]
        """
        risk = pd.DataFrame(index=barra_df.index)
        
        # 1. 处理市值因子 (Size)
        size_col_name = size_col or self.size_col or self.infer_size_col(barra_df)
        size_series = barra_df[size_col_name].astype(float)
        if size_col_name == "mktcap":
            # 市值必须取对数，否则长尾分布会破坏线性回归的假设
            size_series = np.log(size_series.replace(0, np.nan))
        risk["size"] = size_series

        # 2. 处理其他风格因子 (如 Beta, Momentum)
        extra_cols = list(extra_barra_cols) if extra_barra_cols is not None else self.extra_barra_cols
        if extra_cols:
            for col in extra_cols:
                if col in barra_df.columns:
                    risk[col] = barra_df[col]

        # 3. 处理行业因子 (One-Hot Encoding)
        if industry_series is not None:
            # drop_first=True: 避免完全多重共线性 (Dummy Trap)
            industry_dummies = pd.get_dummies(industry_series, prefix="ind", drop_first=True)
            risk = risk.join(industry_dummies, how="left")

        # 数据清洗：处理无穷大和非数值
        risk = risk.replace([np.inf, -np.inf], np.nan)
        risk = risk.astype(float)
        
        # 4. 添加截距项 (Intercept)
        # 必须添加截距，否则 R2 计算会失效，且残差均值不为 0
        risk = sm.add_constant(risk, has_constant="add")
        return risk

    def exposure_for_date(
        self,
        date: str,
        factor_dir: str,
        barra_dir: str,
        industry_dir: Optional[str] = None,
        factor_cols: Optional[list] = None,
        size_col: Optional[str] = None,
        extra_barra_cols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        核心计算逻辑：计算某一天的因子暴露回归结果。
        
        Returns:
        --------
        pd.DataFrame
            包含每个因子的回归统计信息：date, factor_name, R2, beta coefficients...
        """
        from src.common.utils import resolve_dated_file
        from src.research.factor_neutralization import FactorNeutralizer

        factor_file = resolve_dated_file(factor_dir, date)
        if factor_file is None:
            return pd.DataFrame()

        # 加载因子；市值优先 Barra，缺失时回退日行情估算（与 FactorNeutralizer 一致）
        factor_df = self.load_csv_with_index(factor_file)
        barra_file = resolve_dated_file(barra_dir, date) if barra_dir else None
        if barra_file is not None:
            barra_df = self.load_csv_with_index(barra_file)
        else:
            # data_barra 未覆盖近期时，用 money/turnover 构造 ln_mktcap
            size_df = FactorNeutralizer().load_size_frame(
                date, barra_dir=barra_dir, daily_dir="./data/data_daily"
            )
            if size_df is None or size_df.empty:
                return pd.DataFrame()
            barra_df = size_df

        # 加载行业数据 (可选)
        industry_series = None
        if industry_dir:
            industry_file = resolve_dated_file(industry_dir, date)
            if industry_file is not None:
                industry_df = self.load_csv_with_index(industry_file)
                industry_col_name = self.industry_col
                if industry_col_name not in industry_df.columns:
                    industry_col_name = industry_df.columns[0]
                industry_series = industry_df[industry_col_name]

        # 构建自变量矩阵 X
        risk_df = self.build_risk_matrix(
            barra_df,
            industry_series=industry_series,
            size_col=size_col,
            extra_barra_cols=extra_barra_cols,
        )

        # 确定需要分析的因子列
        if factor_cols is None:
            factor_cols = [col for col in factor_df.columns if col not in {"date"}]

        records = []
        for col in factor_cols:
            # 拼接 y (因子) 和 X (风险矩阵)，并去除空值行
            data = pd.concat([factor_df[col], risk_df], axis=1).dropna()
            
            # 样本量检查：样本数必须大于特征数
            if data.shape[0] <= risk_df.shape[1]:
                continue
            
            # 确保全部转为数值类型，避免 object 类型导致回归报错
            y = pd.to_numeric(data[col], errors='coerce')
            x = data[risk_df.columns] # x 已经在 build_risk_matrix 中转为 float
            
            # 再次清洗转换后的 NaN
            combined = pd.concat([y, x], axis=1).dropna()
            if combined.empty:
                continue
                
            y = combined[col]
            x = combined[risk_df.columns]
            
            # 执行 OLS 回归
            model = sm.OLS(y, x).fit()
            
            # 记录结果
            record = {"date": date, "factor": col, "r2": model.rsquared}
            # 提取所有回归系数 (Beta)
            for key, val in model.params.items():
                record[f"beta_{key}"] = val
            records.append(record)

        return pd.DataFrame(records)

    def run_exposure_analysis(
        self,
        date_path: str,
        factor_dir: str,
        barra_dir: str,
        industry_dir: Optional[str] = None,
        size_col: Optional[str] = None,
        extra_barra_cols: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        批量执行暴露分析流程。
        
        Process:
        1. 遍历日期列表。
        2. 调用 exposure_for_date 计算每日暴露。
        3. 汇总所有日期的结果。
        4. 保存详细结果和汇总摘要。
        """
        dates = self.load_dates(date_path, start_date=start_date, end_date=end_date)
        frames = []

        print(f"Starting exposure analysis for {len(dates)} dates...")
        for date in dates:
            df = self.exposure_for_date(
                date,
                factor_dir=factor_dir,
                barra_dir=barra_dir,
                industry_dir=industry_dir,
                size_col=size_col,
                extra_barra_cols=extra_barra_cols,
            )
            if not df.empty:
                frames.append(df)

        if not frames:
            print("No valid exposure data generated.")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            
            # 1. 保存每日详细系数表
            result.to_csv(out / "exposure_coefficients.csv", index=False)

            # 2. 保存因子维度的汇总摘要 (平均 R2，平均暴露)
            # 重点关注 R2 的均值，判断因子是否主要被风格解释
            r2_summary = result.groupby("factor")["r2"].mean().reset_index()
            r2_summary.to_csv(out / "exposure_r2_summary.csv", index=False)
            
            print(f"Exposure analysis done. Results saved to {output_dir}")

        return result


if __name__ == "__main__":
    # 使用示例
    analyzer = ExposureAnalyzer(
        size_col="size",
        extra_barra_cols=["beta"],  # 检查对 Beta 的暴露
        industry_col="industry",
    )
    
    analyzer.run_exposure_analysis(
        date_path="./data/date.pkl",
        factor_dir="./factors/preprocessed",  # 建议分析未经中性化的因子
        barra_dir="./data/data_barra",
        industry_dir="./data/data_industry",
        size_col="size",
        extra_barra_cols=["beta"],
        start_date="2020-01-02",
        end_date="2020-12-31",
        output_dir="./outputs/exposure_analysis",
    )