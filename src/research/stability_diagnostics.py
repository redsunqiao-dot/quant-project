"""
因子稳定性诊断

功能说明：
    汇总因子相关性、VIF（共线性）与滚动 IC 等稳定性指标，
    评估多因子组合是否存在冗余与波动。

知识点讲解：
    1) 相关矩阵：用于发现因子高度相似或重复。
    2) VIF：衡量多重共线性，数值越高越不稳定。
    3) Rolling IC：观察预测力是否在时间上稳定。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FactorDiagnostics:
    def __init__(self, factor_cols: Optional[list] = None, max_dates: int = 20):
        # factor_cols 为 None 时会自动推断因子列
        self.factor_cols = list(factor_cols) if factor_cols is not None else None
        self.max_dates = max_dates

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """加载交易日列表并按区间过滤。"""
        with open(date_path, "rb") as f:
            dates = pickle.load(f)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def load_factor_panel(
        self,
        factor_dir: str,
        dates,
        factor_cols: Optional[list] = None,
        max_dates: Optional[int] = None,
        barra_dir: Optional[str] = None,
        barra_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """拼接多日因子为面板数据（date 作为列；兼容 Parquet / CSV）。"""
        from src.common.utils import resolve_dated_file, read_frame

        max_dates = self.max_dates if max_dates is None else max_dates
        if max_dates:
            dates = dates[-max_dates:]
        frames = []

        # 确定要读取的基础因子列（如果预先指定了）
        base_cols = factor_cols if factor_cols is not None else self.factor_cols

        for date in dates:
            file = resolve_dated_file(factor_dir, date)
            if file is None:
                continue
            df = read_frame(file)
            if "code" in df.columns:
                df = df.set_index("code")

            # --- 新增：读取并合并 Barra 因子 ---
            if barra_dir:
                barra_file = resolve_dated_file(barra_dir, date)
                if barra_file is not None:
                    df_barra = read_frame(barra_file)
                    if "code" in df_barra.columns:
                        df_barra = df_barra.set_index("code")

                    # 仅保留需要的 Barra 因子
                    target_barra_cols = barra_cols if barra_cols else df_barra.columns.tolist()
                    # 取交集，防止文件缺少某些列报错
                    valid_barra_cols = [c for c in target_barra_cols if c in df_barra.columns]

                    if valid_barra_cols:
                        # Inner Join: 仅保留两边都有的股票
                        df = df.join(df_barra[valid_barra_cols], how="inner")
            # -----------------------------------

            # 动态确定最终要保留的列（基础因子 + Barra 因子）
            if base_cols is None:
                # 默认保留所有非 date 列
                current_cols = [col for col in df.columns if col not in {"date"}]
            else:
                # 基础因子 + Barra 因子
                extra = barra_cols if barra_cols else []
                current_cols = base_cols + extra
                # 过滤掉 dataframe 里不存在的列
                current_cols = [c for c in current_cols if c in df.columns]

            df = df[current_cols].copy()
            df["date"] = date
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)

    @staticmethod
    def rolling_ic(ic_series: pd.Series, window: int = 20) -> pd.Series:
        """滚动平均 IC，用于观测稳定性趋势。"""
        return ic_series.rolling(window).mean()

    @staticmethod
    def calc_correlation_matrix(factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子相关性矩阵 (Spearman Rank Correlation)
        
        意义：
        - 检查因子之间是否高度相似。
        - 如果两个因子相关性 > 0.8，说明它们几乎在描述同一件事（例如 "5日涨幅" 和 "10日涨幅"）。
        - 在多因子合成时，保留高度相关的因子不会增加太多信息量，反而可能引入噪音。
        """
        return factor_df.corr(method="spearman")

    @staticmethod
    def calc_vif(factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 VIF (Variance Inflation Factor, 方差膨胀因子)
        
        核心知识点：
        1. 什么是多重共线性 (Multicollinearity)？
           - 在线性回归模型中，如果解释变量（因子）之间存在高度线性相关，会导致系数估计极其不稳定。
           - 表现为：因子稍微变动一点数据，回归系数（权重）就大幅跳动，甚至正负号翻转。
           
        2. VIF 的含义：
           - VIF = 1 / (1 - R^2_i)，其中 R^2_i 是第 i 个因子对其他所有因子回归的 R^2。
           - VIF 越大，说明该因子能被其他因子线性表示的程度越高。
           
        3. 阈值：
           - VIF < 5: 正常。
           - VIF > 10: 存在严重共线性，建议剔除该因子或进行正交化处理。
        """
        clean = factor_df.dropna()
        # 样本数必须大于特征数，否则不可解
        if clean.shape[1] < 2 or clean.shape[0] <= clean.shape[1]:
            return pd.DataFrame(columns=["factor", "vif"])
        x = clean.values
        vifs = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
        return pd.DataFrame({"factor": clean.columns, "vif": vifs})

    def run_diagnostics(
        self,
        factor_dir: str,
        dates,
        factor_cols: Optional[list] = None,
        max_dates: Optional[int] = None,
        output_dir: Optional[str] = None,
        barra_dir: Optional[str] = None,
        barra_cols: Optional[list] = None,
    ):
        """输出相关矩阵与 VIF 结果。"""
        panel = self.load_factor_panel(
            factor_dir=factor_dir,
            dates=dates,
            factor_cols=factor_cols,
            max_dates=max_dates,
            barra_dir=barra_dir,
            barra_cols=barra_cols,
        )
        if panel.empty:
            print("No factor data found for diagnostics")
            return None, None

        # 仅保留因子列，去掉 date 字段
        factor_only = panel.drop(columns=["date"], errors="ignore")
        
        print(f"参与诊断的因子: {factor_only.columns.tolist()}")
        
        corr = self.calc_correlation_matrix(factor_only)
        vif = self.calc_vif(factor_only)
        
        print("\n=== Factor Correlation Matrix ===")
        print(corr)
        print("\n=== Factor VIF ===")
        print(vif)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            corr.to_csv(out / "factor_corr.csv")
            vif.to_csv(out / "factor_vif.csv", index=False)

        return corr, vif

    def export_rolling_ic(self, ic_path: str, window: int = 20, output_path: Optional[str] = None):
        """对 IC 时序做滚动平滑并保存。"""
        ic_df = pd.read_csv(ic_path, index_col=0)
        rolling_df = pd.DataFrame(index=ic_df.index)
        for col in ic_df.columns:
            rolling_df[f"{col}_roll{window}"] = self.rolling_ic(ic_df[col], window=window)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            rolling_df.to_csv(output_path)
        return rolling_df


if __name__ == "__main__":
    diagnostics = FactorDiagnostics(max_dates=30)
    dates = diagnostics.load_dates("./data/date.pkl", start_date="2020-01-02", end_date="2020-12-31")

    # 运行诊断：加入 Barra 的 size 因子作为对照
    diagnostics.run_diagnostics(
        factor_dir="./factors/neutralized",
        dates=dates,
        max_dates=30,
        output_dir="./outputs/stability_diagnostics",
        barra_dir="./data/data_barra",  # Barra 数据路径
        barra_cols=["size"]             # 选取的对照因子
    )

    ic_path = "./outputs/pre_post_cmp/ic_compare.csv"
    if Path(ic_path).exists():
        diagnostics.export_rolling_ic(
            ic_path,
            window=20,
            output_path="./outputs/stability_diagnostics/ic_rolling.csv",
        )
