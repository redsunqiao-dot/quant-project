import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.common.utils import read_frame, write_factor_frame, list_dated_stems

# 默认跳过的非因子列名集合
DEFAULT_SKIP_COLS = {"code", "date", "industry"}


class FactorPreprocessor:
    """
    因子预处理类 (Factor Preprocessor)
    
    该类负责对原始因子数据进行一系列清洗和处理，使其符合量化模型的输入要求。
    标准的预处理流程通常包括：
    1. 缺失值处理 (Filling Missing Values)
    2. 去极值 (Winsorization)
    3. 标准化 (Standardization)
    
    知识点：
    - **去极值**: 防止异常值（Outliers）对统计特性（如均值、方差）产生过大影响，进而影响模型训练。
    - **标准化**: 将不同量纲或分布的因子转换到统一的尺度（如 N(0,1)），以便于比较和线性组合。
    - **缺失值填充**: 保证数据的完整性，避免因个别数据缺失导致整行数据不可用。
    """
    def __init__(
        self,
        factor_cols=None,
        fill_method: str = "median",
        standardize: str = "zscore",
        winsorize: bool = True,
        n_sigma: float = 3.0,
        industry_col: str = "industry",
        winsor_method: str = "mad",
        n_mad: float = 5.0,
    ):
        """
        初始化预处理器
        
        Args:
            factor_cols (list, optional): 显式指定需要处理的因子列名列表。如果不指定，会自动推断。
            fill_method (str): 缺失值填充方法。
                               - 'median': 使用全市场中位数填充
                               - 'zero': 填充为0
                               - 'industry_median': 使用行业中位数填充（更精细）
                               - 'drop': 直接删除包含缺失值的行
            standardize (str): 标准化方法。
                               - 'zscore': Z-Score标准化 (x - mean) / std
                               - 'rank': 排序分位数 (0~1)
                               - 'minmax': 归一化到 [0, 1]
                               - 'robust': 使用中位数和MAD（绝对中位差）进行标准化，对异常值更鲁棒
            winsorize (bool): 是否进行去极值处理。默认为 True。
            n_sigma (float): 标准差去极值倍数（winsor_method='sigma'）。
            industry_col (str): 行业分类的列名，用于行业中位数填充。
            winsor_method (str): 去极值方法 'mad'（默认）或 'sigma'。
            n_mad (float): MAD 去极值倍数，常用 5。
        """
        self.factor_cols = list(factor_cols) if factor_cols is not None else None
        self.fill_method = fill_method
        self.standardize = standardize
        self.winsorize = winsorize
        self.n_sigma = n_sigma
        self.industry_col = industry_col
        self.winsor_method = winsor_method
        self.n_mad = n_mad

    def load_factor_file(self, path: Path) -> pd.DataFrame:
        """加载因子数据文件（CSV / Parquet），包含 date, code 和因子值"""
        df = read_frame(path)
        if "code" in df.columns:
            df = df.set_index("code")
        return df

    def load_industry_series(self, path: Path, industry_col: Optional[str] = None) -> pd.Series:
        """加载行业分类数据，用于行业中位数填充"""
        df = read_frame(path)
        if "code" in df.columns:
            df = df.set_index("code")
        col = industry_col or self.industry_col
        if col not in df.columns:
            # 如果指定的行业列不存在，默认取第一列
            col = df.columns[0]
        return df[col]

    def infer_factor_cols(self, df: pd.DataFrame, factor_cols=None) -> list:
        """
        推断需要处理的因子列。
        排除掉 code, date, industry 等非因子数据列。
        """
        if factor_cols is not None:
            return list(factor_cols)
        if self.factor_cols is not None:
            return list(self.factor_cols)
        # 排除默认列和行业列
        skip_cols = DEFAULT_SKIP_COLS | {self.industry_col}
        return [col for col in df.columns if col not in skip_cols]

    def winsorize_series(self, series: pd.Series, n_sigma: Optional[float] = None) -> pd.Series:
        """
        去极值处理 (Winsorization)

        - sigma: mean ± n_sigma * std
        - mad: median ± n_mad * MAD（对异常值更稳健，默认）
        """
        method = (self.winsor_method or "mad").lower()
        if method == "mad":
            median = series.median()
            mad = (series - median).abs().median()
            if pd.isna(mad) or mad == 0:
                return series
            # 与常见实现一致：用 1.4826 将 MAD 校准到近似标准差尺度
            scale = 1.4826 * mad
            n = self.n_mad if n_sigma is None else float(n_sigma)
            lower = median - n * scale
            upper = median + n * scale
            return series.clip(lower, upper)

        mean = series.mean()
        std = series.std()
        if pd.isna(std) or std == 0:
            return series
        sigma = self.n_sigma if n_sigma is None else n_sigma
        lower = mean - sigma * std
        upper = mean + sigma * std
        return series.clip(lower, upper)

    def standardize_series(self, series: pd.Series, method: Optional[str] = None) -> pd.Series:
        """
        标准化处理 (Standardization)
        
        将因子转化为特定的分布，消除量纲影响。
        """
        method = method or self.standardize
        if method == "zscore":
            # Z-Score: (x - μ) / σ
            # 结果均值为0，标准差为1。假设数据近似正态分布时效果好。
            std = series.std()
            return (series - series.mean()) / std if std != 0 else series - series.mean()
        if method == "rank":
            # Rank: 将数值转化为排名百分比 (0.0 ~ 1.0)
            # 对分布不敏感，消除异常值影响，但会丢失数值间的相对距离信息。
            return series.rank(pct=True)
        if method == "minmax":
            # Min-Max: (x - min) / (max - min)
            # 将数据缩放到 [0, 1] 区间。对异常值非常敏感。
            denom = series.max() - series.min()
            return (series - series.min()) / denom if denom != 0 else series * 0
        if method == "robust":
            # Robust Scaler: 使用中位数和MAD (Median Absolute Deviation)
            # (x - median) / MAD
            # 相比 Z-Score，中位数和MAD受异常值影响更小。
            median = series.median()
            mad = (series - median).abs().median()
            return (series - median) / mad if mad != 0 else series - median
        raise ValueError(f"Unknown standardize method: {method}")

    def fill_missing_series(
        self,
        series: pd.Series,
        method: Optional[str] = None,
        industry_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        缺失值填充 (Imputation)
        """
        method = method or self.fill_method
        if method == "zero":
            return series.fillna(0)
        if method == "median":
            # 全市场中位数填充，简单有效
            return series.fillna(series.median())
        if method == "industry_median":
            # 行业中位数填充
            # 逻辑：先尝试用同行业的非空数据的中位数填充。
            # 如果该行业全是 NaN，则回退到使用全市场中位数填充。
            if industry_series is None:
                return series.fillna(series.median())
            aligned = pd.DataFrame({"value": series, "industry": industry_series})
            # transform('median') 会计算每个分组的中位数，并广播回原索引
            group_median = aligned.groupby("industry")["value"].transform("median")
            filled = aligned["value"].fillna(group_median)
            # 兜底：如果某行业全是NaN，group_median也是NaN，再用全局中位数填
            return filled.fillna(aligned["value"].median())
        raise ValueError(f"Unknown fill method: {method}")

    def preprocess_factor_df(
        self,
        df: pd.DataFrame,
        factor_cols=None,
        industry_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        对单截面（单日）的所有因子数据进行预处理的主流程。
        
        流程顺序：
        1. 识别因子列
        2. (可选) 删除缺失值
        3. 遍历每个因子列：
           a. 处理 Inf / -Inf 为 NaN
           b. Winsorize (去极值)
           c. 缺失值填充 (Filling)
           d. 标准化 (Standardization)
        """
        cols = self.infer_factor_cols(df, factor_cols=factor_cols)
        if self.fill_method == "drop":
            df = df.dropna(subset=cols)
        out = df.copy()
        for col in cols:
            # 预清洗：将无穷大视为缺失值
            series = out[col].replace([np.inf, -np.inf], np.nan)
            
            # 1. 去极值 (Winsorization)
            if self.winsorize:
                series = self.winsorize_series(series, n_sigma=self.n_sigma)
            
            # 2. 缺失值填充 (Filling)
            if self.fill_method != "drop":
                series = self.fill_missing_series(series, method=self.fill_method, industry_series=industry_series)
            
            # 3. 标准化 (Standardization)
            out[col] = self.standardize_series(series, method=self.standardize)
        return out

    def preprocess_folder(
        self,
        factor_dir: str,
        output_dir: str,
        industry_dir: Optional[str] = None,
    ) -> None:
        """
        批量处理文件夹中的因子文件
        
        假设：
        1. factor_dir 下是按日期命名的截面文件 (Parquet 优先，兼容 CSV)。
        2. industry_dir 下有同日的行业分类数据。
        """
        from src.common.utils import resolve_dated_file

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dates = list_dated_stems(factor_dir)
        if not dates:
            print(f"No factor files found in {factor_dir}")
            return

        for date in dates:
            file = resolve_dated_file(factor_dir, date)
            if file is None:
                continue
            df = self.load_factor_file(file)
            industry_series = None
            if industry_dir:
                industry_file = Path(industry_dir) / f"{date}.csv"
                if industry_file.exists():
                    industry_series = self.load_industry_series(industry_file, industry_col=self.industry_col)

            processed = self.preprocess_factor_df(
                df,
                factor_cols=self.factor_cols,
                industry_series=industry_series,
            )
            write_factor_frame(processed, output_path, date, index_as_code=True)

        print(f"Preprocess done. Output -> {output_path}")


if __name__ == "__main__":
    # 使用示例
    preprocessor = FactorPreprocessor(
        factor_cols=None,       # 自动推断因子列
        fill_method="median",   # 中位数填充缺失
        standardize="zscore",   # Z-Score 标准化
        winsorize=True,         # 开启去极值
        n_sigma=3.0,            # 3倍标准差
        industry_col="industry",
    )
    preprocessor.preprocess_folder(
        factor_dir="./factors/raw",
        output_dir="./factors/preprocessed",
        industry_dir="./data/data_industry",
    )