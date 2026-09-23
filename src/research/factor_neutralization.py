"""
因子中性化（Factor Neutralization） - 横截面回归法

功能说明：
    读取每日因子与 Barra 风险暴露（如市值 Size、Beta、行业 Industry）数据，
    对因子做横截面回归，取残差作为“中性化后因子”。

核心知识点讲解：
    1. 中性化的数学本质：正交化 (Orthogonalization)
       - 我们的目标是提取因子中"独有"的信息，剔除它与已知风险因子（如市值、行业）"重叠"的部分。
       - 数学上，这等价于计算回归残差：
         Factor_Raw = β1 * Size + β2 * Industry + ... + ε (残差)
         Factor_Neutral = ε
       - 此时，Factor_Neutral 与 Size, Industry 的相关性理论上为 0。

    2. 为什么要剔除风格暴露？
       - 避免"伪Alpha"：很多因子看似有效，其实是因为它选中了小盘股（Size暴露）。如果小盘股崩盘，该因子也会失效。
       - 风险控制：我们希望策略的收益来自于我们的选股逻辑，而不是被动的市场风格波动。

    3. 行业处理技巧：哑变量 (Dummy Variables)
       - 行业是分类变量（Categorical），不能直接回归。
       - 需要转换为 One-Hot 编码（即 Dummy 矩阵）。
       - drop_first=True：为了避免"完全多重共线性"（Dummy Trap），通常会去掉一个基准行业，或者在包含截距项时注意处理。

    4. 数据要求：
       - 样本量 > 特征数：横截面上的股票数量（N）必须远大于风险因子数量（K），否则回归过拟合或无解。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional

from src.common.utils import (
    resolve_dated_file,
    read_frame,
    write_factor_frame,
    list_dated_stems,
    estimate_ln_circ_mktcap,
)

# 定义可能的市值列名，用于自动推断
SIZE_COL_CANDIDATES = ["size", "ln_mktcap", "log_mktcap", "mktcap"]


class FactorNeutralizer:
    """
    因子中性化处理器
    
    该类负责执行因子的中性化操作，主要流程包括：
    1. 加载因子数据和风险因子数据（市值、行业等）。
    2. 构建风险因子矩阵（Design Matrix），包括对分类变量（行业）进行哑变量编码。
    3. 使用 OLS (最小二乘法) 对因子进行横截面回归。
    4. 提取回归残差作为中性化后的因子值。
    """
    
    def __init__(
        self,
        size_col: Optional[str] = None,
        extra_barra_cols: Optional[Iterable[str]] = None,
        industry_col: str = "industry",
    ):
        """
        初始化中性化处理器
        
        Parameters:
        -----------
        size_col : str, optional
            指定市值因子的列名。如果不指定，会根据 SIZE_COL_CANDIDATES 自动推断。
        extra_barra_cols : Iterable[str], optional
            额外的 Barra 风格因子列名列表，例如 ['beta', 'momentum', 'volatility']。
            这些因子也会被放入回归方程的自变量中剔除。
        industry_col : str, default "industry"
            行业分类数据的列名。
        """
        self.size_col = size_col
        self.extra_barra_cols = list(extra_barra_cols) if extra_barra_cols else None
        self.industry_col = industry_col

    def load_csv_with_index(self, path: Path, index_col: str = "code") -> pd.DataFrame:
        """
        通用表格读取并设置股票代码为索引（兼容 CSV / Parquet）。
        
        Parameters:
        -----------
        path : Path
            文件路径
        index_col : str
            用作索引的列名，通常是股票代码列。
            
        Returns:
        --------
        pd.DataFrame
            设置了正确索引的 DataFrame。
        """
        df = read_frame(path)
        if index_col in df.columns:
            df = df.set_index(index_col)
        return df

    def infer_size_col(self, barra_df: pd.DataFrame) -> str:
        """
        自动推断市值列名，兼容不同数据源命名。
        
        会在 SIZE_COL_CANDIDATES 定义的候选列表中查找是否存在于 barra_df 的列名中。
        """
        for col in SIZE_COL_CANDIDATES:
            if col in barra_df.columns:
                return col
        raise ValueError(f"No size column found in {barra_df.columns.tolist()}")

    def build_size_series(self, barra_df: pd.DataFrame, size_col: Optional[str] = None) -> pd.Series:
        """
        构造标准化后的 size 序列 (通常取对数)
        
        注意：
        市值（Market Cap）通常呈长尾分布（Lognormal），直接用于线性回归效果不佳，
        且大盘股数值过大，会主导回归结果（Leverage Point）。
        必须取对数 (Log Market Cap) 将其转化为接近正态分布的形式，更适合线性回归。
        
        Parameters:
        -----------
        barra_df : pd.DataFrame
            包含市值数据的 DataFrame
        size_col : str, optional
            强制指定的市值列名
            
        Returns:
        --------
        pd.Series
            处理后的对数市值序列，名称统一为 "size"
        """
        col = size_col or self.size_col or self.infer_size_col(barra_df)
        series = barra_df[col].astype(float)
        
        # 如果列名包含 'mktcap' 且未被明确指出是 log 值，则认为它是原始市值，需要取对数
        if col in {"mktcap"}:
            # 市值常用对数处理，缓解长尾分布
            # replace(0, np.nan) 避免 log(0) = -inf 的情况
            series = np.log(series.replace(0, np.nan))
            
        return series.rename("size")

    def build_industry_dummies(self, industry_series: pd.Series, drop_first: bool = True) -> pd.DataFrame:
        """
        构建行业哑变量矩阵 (One-Hot Encoding)
        
        Pandas 的 get_dummies 可以自动将分类列转为 0/1 矩阵。
        例如：
           Stock | Industry
           -------------
           A     | IT
           B     | Bank
           C     | IT
        转为：
           Stock | ind_Bank | ind_IT
           ---------------------
           A     | 0        | 1
           B     | 1        | 0
           C     | 0        | 1
           
        Parameters:
        -----------
        industry_series : pd.Series
            行业分类序列
        drop_first : bool, default True
            是否丢弃第一个类别。
            统计学上，如果回归模型包含截距项（Intercept/Constant），
            全量的哑变量矩阵会导致“完全多重共线性”（Dummy Variable Trap），
            导致矩阵不可逆。因此通常去掉一个基准行业。
        """
        return pd.get_dummies(industry_series, prefix="ind", drop_first=drop_first)

    def build_risk_matrix(
        self,
        barra_df: pd.DataFrame,
        industry_series: Optional[pd.Series] = None,
        size_col: Optional[str] = None,
        extra_barra_cols: Optional[Iterable[str]] = None,
        add_const: bool = True,
    ) -> pd.DataFrame:
        """
        拼装回归用风险暴露矩阵 X (Design Matrix)
        
        该矩阵将作为线性回归的自变量 (Independent Variables)。
        X = [Constant, Size, Industry_Dummies, Other_Risk_Factors...]
        
        Parameters:
        -----------
        barra_df : pd.DataFrame
            基础 Barra 因子数据（如市值、Beta等）
        industry_series : pd.Series, optional
            行业分类数据。如果提供，将生成哑变量并拼接到 X 中。
        size_col : str, optional
            指定的市值列名
        extra_barra_cols : list, optional
            其他需要剔除的风格因子列名
        add_const : bool, default True
            是否添加截距项（常量列，全为1）。
            
        Returns:
        --------
        pd.DataFrame
            构建好的风险因子矩阵 X，索引与 barra_df 一致。
        """
        risk = pd.DataFrame(index=barra_df.index)
        
        # 1. 添加市值因子 (Size)
        risk["size"] = self.build_size_series(barra_df, size_col=size_col)

        # 2. 添加其他风格因子 (如 Beta, Momentum)
        extra_cols = list(extra_barra_cols) if extra_barra_cols is not None else self.extra_barra_cols
        if extra_cols:
            for col in extra_cols:
                if col in barra_df.columns:
                    risk[col] = barra_df[col]

        # 3. 添加行业哑变量 (Industry Dummies)
        if industry_series is not None:
            industry_dummies = self.build_industry_dummies(industry_series)
            risk = risk.join(industry_dummies, how="left")

        # 数据清洗：转数值，处理无穷大
        risk = risk.apply(pd.to_numeric, errors="coerce")
        risk = risk.replace([np.inf, -np.inf], np.nan)
        
        # 4. 添加截距项 (Intercept/Constant)
        if add_const:
            # 统计学意义：当所有风险暴露为0时，因子的基准水平。
            # 如果不加截距，强制过原点，可能会扭曲残差，导致回归均值不为0。
            if "const" not in risk.columns:
                risk.insert(0, "const", 1.0)
            
        return risk

    def neutralize_factor_df(
        self,
        factor_df: pd.DataFrame,
        risk_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        核心方法：对因子做横截面回归，返回残差矩阵。
        
        Process:
        1. 对齐数据：取 Factor 和 Risk Matrix 的交集部分（inner join）。
        2. 逐列回归：对 factor_df 中的每一列（每一个因子）单独进行回归。
        3. 计算残差：Residual = Actual - Predicted
        
        Equation: 
            y (Factor) = X (Risk Matrix) * β + ε (Residual)
        Target: 
            ε = y - X * β_hat
        
        Parameters:
        -----------
        factor_df : pd.DataFrame
            原始因子数据（可能包含多个因子列）
        risk_df : pd.DataFrame
            风险因子矩阵 X (Design Matrix)
            
        Returns:
        --------
        pd.DataFrame
            中性化后的因子矩阵（即残差 ε），形状与 factor_df 相同。
        """
        # 确保因子数据为数值型
        numeric_factors = factor_df.apply(pd.to_numeric, errors="coerce")
        numeric_factors = numeric_factors.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        numeric_factor_cols = list(numeric_factors.columns)
        
        if not numeric_factor_cols:
            return pd.DataFrame(index=factor_df.index)

        # 仅对齐两个矩阵都有数据的行（股票）
        aligned = numeric_factors.join(risk_df, how="inner")
        if aligned.empty:
            return pd.DataFrame(index=factor_df.index, columns=numeric_factor_cols)

        # 初始化结果矩阵
        result = pd.DataFrame(index=aligned.index, columns=numeric_factor_cols, dtype=float)
        risk_cols = list(risk_df.columns)

        for col in numeric_factor_cols:
            # 准备当前因子的回归数据，去除空值
            data = aligned[[col] + risk_cols].dropna()
            
            # 检查样本量是否充足
            # 统计学要求：样本量 N 必须大于 特征数 K (包括截距)
            # 最好 N >> K，否则自由度过低，回归结果无意义
            if data.shape[0] <= len(risk_cols) + 1:
                # 样本不足时跳过，保留为 NaN 或默认值
                continue
                
            y = data[col].to_numpy(dtype=float)
            x = data[risk_cols].to_numpy(dtype=float)
            
            # 普通最小二乘：用 numpy 解 β，残差即中性化因子
            beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ beta
            
            # 将残差填回结果表
            result.loc[data.index, col] = resid

        return result

    def load_size_frame(
        self,
        date: str,
        barra_dir: Optional[str] = None,
        daily_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载单日市值暴露：优先 Barra size；否则用日频成交额/换手率估算 ln_mktcap。
        """
        if barra_dir:
            barra_file = resolve_dated_file(barra_dir, date)
            if barra_file is not None:
                barra_df = self.load_csv_with_index(barra_file)
                size_col = self.size_col
                try:
                    if size_col is None:
                        size_col = self.infer_size_col(barra_df)
                    if size_col in barra_df.columns:
                        return pd.DataFrame({"size": self.build_size_series(barra_df, size_col=size_col)})
                except ValueError:
                    pass

        if daily_dir:
            daily_file = Path(daily_dir) / f"{date}.csv"
            if daily_file.exists():
                daily = pd.read_csv(daily_file)
                ln_mkt = estimate_ln_circ_mktcap(daily)
                return pd.DataFrame({"ln_mktcap": ln_mkt})

        return pd.DataFrame()

    def neutralize_folder(
        self,
        factor_dir: str,
        barra_dir: Optional[str] = None,
        output_dir: str = "./factors/neutralized",
        industry_dir: Optional[str] = None,
        daily_dir: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """
        遍历文件夹进行批量中性化处理。
        
        假设文件夹结构为：
        factors/
          ├── 2020-01-01.parquet
          ├── 2020-01-02.parquet
          ...
          
        Processing Logic:
        1. 遍历 factor_dir 下按日截面。
        2. 市值：Barra size（若有）或日频估算 ln_mktcap。
        3. 可选拼接行业哑变量，OLS 取残差后写入 Parquet。
        
        Parameters:
        -----------
        factor_dir : str
            输入因子文件目录
        barra_dir : str, optional
            Barra 风格因子文件目录
        output_dir : str
            结果输出目录
        industry_dir : str, optional
            行业分类文件目录
        daily_dir : str, optional
            日频行情目录，用于在无 Barra 时估算流通市值
        start_date / end_date : str, optional
            处理区间
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dates = list_dated_stems(factor_dir)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        if not dates:
            print(f"No factor files found in {factor_dir}")
            return

        print(f"Starting neutralization for {len(dates)} files...")
        written = 0
        for date in dates:
            factor_file = resolve_dated_file(factor_dir, date)
            if factor_file is None:
                continue

            size_df = self.load_size_frame(date, barra_dir=barra_dir, daily_dir=daily_dir)
            if size_df.empty:
                continue

            factor_df = self.load_csv_with_index(factor_file)
            drop_cols = [c for c in factor_df.columns if str(c).lower() in {"date", "datetime"}]
            if drop_cols:
                factor_df = factor_df.drop(columns=drop_cols)

            industry_series = None
            if industry_dir:
                industry_file = Path(industry_dir) / f"{date}.csv"
                if industry_file.exists():
                    industry_df = self.load_csv_with_index(industry_file)
                    industry_col_name = self.industry_col
                    if industry_col_name not in industry_df.columns:
                        industry_col_name = industry_df.columns[0]
                    industry_series = industry_df[industry_col_name]

            size_col = "size" if "size" in size_df.columns else "ln_mktcap"
            risk_df = self.build_risk_matrix(
                size_df,
                industry_series=industry_series,
                size_col=size_col,
                extra_barra_cols=self.extra_barra_cols,
            )

            neutralized = self.neutralize_factor_df(factor_df, risk_df)
            if neutralized.empty:
                continue
            write_factor_frame(neutralized, output_path, date, index_as_code=True)
            written += 1

        print(f"Neutralization done. Written {written} days -> {output_path}")


if __name__ == "__main__":
    # 使用示例：市值 + 行业 OLS 中性化（第九课经典流程）
    neutralizer = FactorNeutralizer(
        size_col=None,
        industry_col="industry",
    )
    
    # 批量处理文件夹
    neutralizer.neutralize_folder(
        factor_dir="./factors/preprocessed",
        barra_dir="./data/data_barra",
        daily_dir="./data/data_daily",
        output_dir="./factors/neutralized",
        industry_dir="./data/data_industry",
    )