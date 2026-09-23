import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.utils import resolve_dated_file, read_frame, write_factor_frame
from src.data.list_date_cache import load_list_dates


class UniverseFilter:
    """
    股票池过滤器类
    
    Attributes:
        min_price (float): 最低股价阈值，低于此价格的股票将被剔除。
                          (低价股往往波动大，且容易有退市风险或流动性问题)
        min_volume (float): 最低成交量阈值，低于此成交量的股票将被剔除。
        min_turnover (float): 最低换手率阈值。
        remove_st (bool): 是否剔除ST/Star/PT等风险警示股票。
        min_list_days (int): 上市满多少个自然日才可交易；0 表示不限制。
    """
    def __init__(
        self,
        min_price: float = 2.0,
        min_volume: float = 1e5,
        min_turnover: float = 0.0005,
        remove_st: bool = True,
        min_list_days: int = 365,
        list_date_cache: str = "./data/_cache/list_dates.parquet",
        data_daily_dir: str = "./data/data_daily",
    ):
        self.min_price = min_price
        self.min_volume = min_volume
        self.min_turnover = min_turnover
        self.remove_st = remove_st
        self.min_list_days = min_list_days
        self.list_date_cache = list_date_cache
        self._data_daily_dir = data_daily_dir
        self._list_dates: Optional[pd.Series] = None

    def load_dates(self, date_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """加载交易日期列表。"""
        with open(date_path, "rb") as f:
            dates = pickle.load(f)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def load_csv_with_index(self, path: Path, index_col: str = "code") -> pd.DataFrame:
        """通用表格读取，自动设置索引（兼容 CSV / Parquet）。"""
        df = read_frame(path)
        if index_col in df.columns:
            df = df.set_index(index_col)
        return df

    def _ensure_list_dates(self) -> Optional[pd.Series]:
        """懒加载上市日缓存。"""
        if self.min_list_days <= 0:
            return None
        if self._list_dates is None:
            self._list_dates = load_list_dates(
                cache_path=self.list_date_cache,
                daily_dir=self._data_daily_dir,
            )
        return self._list_dates

    def build_tradeable_mask(
        self,
        merged: pd.DataFrame,
        trade_date: Optional[str] = None,
    ) -> pd.Series:
        """
        核心逻辑：构建可交易股票的布尔掩码 (Boolean Mask)
        
        Args:
            merged (pd.DataFrame): 包含行情数据(close, volume等)和状态数据(paused, st等)的大表。
            trade_date: 交易日，用于上市天数过滤。
            
        Returns:
            pd.Series:索引为股票代码，值为True(保留)或False(剔除)的Series。
        """
        # 初始默认全为 True (全部保留)
        mask = pd.Series(True, index=merged.index)

        # 1. 状态过滤
        # paused: 1表示停牌，0表示正常。我们需要 paused == 0
        if "paused" in merged.columns:
            mask &= merged["paused"] == 0
        
        # zt (涨停) / dt (跌停): 通常为1表示涨跌停。
        # 这里一刀切剔除涨跌停，因为涨停买不进，跌停卖不出，且容易产生极端价格。
        if "zt" in merged.columns:
            mask &= merged["zt"] == 0
        if "dt" in merged.columns:
            mask &= merged["dt"] == 0

        # 2. 流动性与价格过滤
        # 剔除低价股 (e.g. < 2.0 元)
        if "close" in merged.columns:
            mask &= merged["close"] >= self.min_price
        
        # 剔除成交量过低的股票 (e.g. < 100,000 股)
        if "volume" in merged.columns:
            mask &= merged["volume"] >= self.min_volume
            
        # 剔除换手率过低的股票 (e.g. < 0.05%)
        # 换手率 = 成交量 / 流通股本，反映交易活跃度
        if "turnover_ratio" in merged.columns:
            mask &= merged["turnover_ratio"] >= self.min_turnover

        # 3. 风险警示过滤 (ST股)
        if self.remove_st:
            # 不同的数据源可能有不同的字段名来标识ST
            if "is_st" in merged.columns:
                mask &= merged["is_st"] == 0
            elif "st" in merged.columns:
                mask &= merged["st"] == 0
            elif "name" in merged.columns:
                # 如果只有名称，通过字符串匹配包含 "ST" 的
                mask &= ~merged["name"].astype(str).str.contains("ST", case=False, na=False)

        # 4. 上市不满 min_list_days 自然日剔除（默认 1 年）
        if self.min_list_days > 0 and trade_date:
            list_dates = self._ensure_list_dates()
            if list_dates is not None and len(list_dates) > 0:
                asof = pd.Timestamp(trade_date)
                ld = list_dates.reindex(merged.index)
                # 无上市日记录：保守剔除（避免次新漏网）
                age_ok = pd.Series(False, index=merged.index)
                valid = ld.notna()
                ages = (asof - pd.to_datetime(ld[valid])).dt.days
                age_ok.loc[valid] = ages >= self.min_list_days
                mask &= age_ok

        return mask

    def filter_factors_for_date(
        self,
        date: str,
        factor_dir: str,
        data_daily_dir: str,
        data_ud_dir: str,
        output_dir: str,
    ) -> dict:
        """
        处理单日的因子数据：应用过滤规则并保存结果
        """
        self._data_daily_dir = data_daily_dir
        factor_file = resolve_dated_file(factor_dir, date)
        daily_file = Path(data_daily_dir) / f"{date}.csv"
        status_file = Path(data_ud_dir) / f"{date}.csv"  # 包含ST、停牌、涨跌停状态

        # 完整性检查：如果任一数据文件缺失，则无法进行准确过滤，直接跳过该日期
        # 这就是为什么有时候输出会“少几天”的原因
        if not (factor_file is not None and daily_file.exists() and status_file.exists()):
            return {"date": date, "status": "missing"}

        # 加载三个数据源
        factor_df = self.load_csv_with_index(factor_file)
        daily_df = self.load_csv_with_index(daily_file)   # 包含 price, volume
        status_df = self.load_csv_with_index(status_file) # 包含 paused, is_st, zt, dt

        # 数据清洗：移除 status_df 中可能与 daily_df 重复的列 (如 open, close 等)，避免 join 时报错或产生 _x, _y 列
        overlapping_cols = status_df.columns.intersection(daily_df.columns)
        if len(overlapping_cols) > 0:
            status_df = status_df.drop(columns=overlapping_cols)

        # 合并每日行情和状态数据，用于计算 Mask
        # 使用 left join 以 daily_df 为准 (通常 daily_df 包含所有上市股票)
        merged = daily_df.join(status_df, how="left")
        
        # 计算过滤掩码
        mask = self.build_tradeable_mask(merged, trade_date=date)

        # 对齐索引：因子数据 和 Mask 的股票代码取交集
        common_index = factor_df.index.intersection(mask.index)
        
        before_count = len(common_index)
        
        # 应用过滤：
        # 1. .loc[common_index]: 确保只取存在的股票
        # 2. [mask.loc[common_index]]: 应用布尔筛选
        filtered = factor_df.loc[common_index][mask.loc[common_index]]
        after_count = filtered.shape[0]

        # 保存结果（Parquet）
        write_factor_frame(filtered, output_dir, date, index_as_code=True)

        return {
            "date": date,
            "status": "ok",
            "before": before_count,
            "after": after_count,
            "drop_rate": 1 - (after_count / before_count) if before_count else None,
        }

    def filter_folder(
        self,
        date_path: str,
        factor_dir: str,
        data_daily_dir: str,
        data_ud_dir: str,
        output_dir: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """
        批量处理：遍历指定日期范围内的所有文件进行过滤
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取需要处理的日期列表
        dates = self.load_dates(date_path, start_date=start_date, end_date=end_date)
        records = []

        print(f"开始过滤... 计划处理 {len(dates)} 天")
        for date in dates:
            record = self.filter_factors_for_date(
                date,
                factor_dir=factor_dir,
                data_daily_dir=data_daily_dir,
                data_ud_dir=data_ud_dir,
                output_dir=output_dir,
            )
            records.append(record)

        # 生成汇总报告，方便检查每一天的过滤情况 (包括缺失的天数)
        summary = pd.DataFrame(records)
        summary.to_csv(output_path / "universe_filter_summary.csv", index=False)
        print(f"Universe filter 完成. 结果保存在 -> {output_dir}")
        print(f"汇总报告 -> {output_path / 'universe_filter_summary.csv'}")


if __name__ == "__main__":
    # 实例化过滤器
    universe_filter = UniverseFilter(
        min_price=2.0,      # 剔除股价小于 2 元的
        min_volume=1e5,     # 剔除成交量小于 10万股 的
        min_turnover=0.0005,# 剔除换手率小于 0.05% 的
        remove_st=True,     # 剔除 ST 股
    )
    
    # 执行批量过滤
    universe_filter.filter_folder(
        date_path="./data/date.pkl",
        factor_dir="./factors/raw",         # 原始因子输入目录
        data_daily_dir="./data/data_daily", # 日行情数据目录
        data_ud_dir="./data/data_ud_new",   # 涨跌停/状态数据目录
        output_dir="./factors/filtered",    # 过滤后因子输出目录
        start_date="2020-01-02",
        end_date="2020-12-31",
    )