"""
因子计算器 - Version 3.0
对应教案 Day 7 第三步
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.common.utils import resolve_dated_file, read_frame, write_factor_frame
from src.data.data_loader import DataLoader


class FactorCalculator:
    """因子计算器 - Version 1.0"""

    @staticmethod
    def momentum_col(period: int) -> str:
        """动量因子列名"""
        return f"momentum_{period}"

    def __init__(self, data_loader: DataLoader):
        """
        初始化因子计算器

        Args:
            data_loader: 数据加载器实例
        """
        self.loader = data_loader
        self.factor_dir = Path(data_loader.data_dir).parent / 'factors' / 'raw'
        self.factor_dir.mkdir(parents=True, exist_ok=True)

        print("✅ 因子计算器初始化成功")

    def calculate_momentum_daily(self, date: str, period: int = 20) -> pd.DataFrame:
        """
        计算单日的动量因子

        Args:
            date: 当前日期（字符串，如 '2020-02-04'）
            period: 回看周期（天数）

        Returns:
            DataFrame: 当日各股票的动量因子值（列：code, date, momentum_{period}）
        """
        # 1. 找到 period 天前的日期
        trade_dates = self.loader.get_all_dates()
        if date not in trade_dates:
            return pd.DataFrame()

        current_idx = trade_dates.index(date)
        if current_idx < period:
            # 数据不足，返回空
            return pd.DataFrame()

        past_date = trade_dates[current_idx - period]

        # 2. 读取当前日期和过去日期的数据
        current_data = self.loader.get_daily_data(date)
        past_data = self.loader.get_daily_data(past_date)

        if current_data.empty or past_data.empty:
            return pd.DataFrame()

        # 3. 合并数据（按 code）
        merged = pd.merge(
            current_data[['code', 'close']],
            past_data[['code', 'close']],
            on='code',
            suffixes=('_now', '_past')
        )

        # 4. 计算动量因子
        factor_col = self.momentum_col(period)
        merged[factor_col] = (merged['close_now'] / merged['close_past']) - 1
        merged['date'] = date

        # 5. 只保留需要的列
        result = merged[['code', 'date', factor_col]]

        return result

    def calculate_and_save_all(self, period: int = 20):
        """
        批量计算并保存所有交易日的因子

        Args:
            period: 回看周期
        """
        factor_col = self.momentum_col(period)
        trade_dates = self.loader.get_all_dates()
        total = len(trade_dates)

        print(f"📊 开始计算因子: {factor_col}")
        print(f"📊 回看周期: {period} 天")
        print(f"📊 总交易日: {total}")

        success_count = 0

        for i, date in enumerate(trade_dates):
            # 计算单日因子
            factor_df = self.calculate_momentum_daily(date, period)

            if not factor_df.empty:
                # 保存到文件（Parquet）
                write_factor_frame(factor_df, self.factor_dir, date)
                success_count += 1

            # 每 100 天打印一次进度
            if (i + 1) % 100 == 0:
                print(f"进度: {i + 1}/{total} ({(i+1)/total*100:.1f}%)")

        print(f"✅ 因子计算完成！成功: {success_count}/{total}")
        print(f"💾 因子已保存至: {self.factor_dir}")

    def calculate_range_and_save(
        self,
        start_date: str,
        end_date: str,
        momentum_period: int = 20,
        reversal_period: int = 5,
        turnover_window: int = 20,
        factor_names: Optional[List[str]] = None,
        ubl_z_window: int = 10,
        ubl_feat_window: int = 20,
        ubl_wms_n: int = 20,
    ) -> list:
        """
        计算区间内多个价量因子，按日写入 factors/raw。
        在原有单因子动量接口之外新增，不改 calculate_momentum_daily。

        因子方向已统一为「越大越好」，方便后续等权和 IC 加权：
        - momentum_{n}: n 日涨幅
        - reversal_{n}: n 日跌幅（即收益取负）
        - turn_stable_{n}: n 日换手率标准差取负（越稳越大）
        - ubl: 东吴 UBL（蜡烛上_std + 威廉下_mean，取负）

        factor_names:
            若给定，则只输出名单内且已实现的列；未知名字直接报错。
            若为 None，保持原行为，写出三个默认列。
        """
        from src.research.factor_ops import assert_ops_implemented
        from src.research.panel_dispatch import build_series_map, required_lookback

        default_names = [
            self.momentum_col(momentum_period),
            f"reversal_{reversal_period}",
            f"turn_stable_{turnover_window}",
        ]
        if factor_names is None:
            selected = default_names
        else:
            selected = list(factor_names)
            assert_ops_implemented(selected)

        lookback, need_ohlc = required_lookback(
            selected,
            momentum_period=momentum_period,
            reversal_period=reversal_period,
            turnover_window=turnover_window,
            ubl_z_window=ubl_z_window,
            ubl_feat_window=ubl_feat_window,
            ubl_wms_n=ubl_wms_n,
        )
        from src.research.factor_ops import resolve_op
        from src.research.panel_dispatch import M6_OPS
        from src.research.impl.module6_factors import (
            compute_module6_panels,
            load_fundamental_long,
            load_valuation_long,
        )

        m6_names = [n for n in selected if resolve_op(n) in M6_OPS]

        trade_dates = self.loader.get_all_dates()
        target = [d for d in trade_dates if start_date <= d <= end_date]
        if not target:
            return []

        start_idx = trade_dates.index(target[0])
        end_idx = trade_dates.index(target[-1])
        hist_dates = trade_dates[max(0, start_idx - lookback): end_idx + 1]

        frames = []
        for date in hist_dates:
            daily = self.loader.get_daily_data(date)
            if daily.empty:
                continue
            cols = ["code", "close", "turnover_ratio"]
            if need_ohlc:
                cols = [
                    "code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "money",
                    "turnover_ratio",
                ]
            part = daily[[c for c in cols if c in daily.columns]].copy()
            part["date"] = date
            frames.append(part)
        if not frames and not m6_names:
            return []

        series_map = {}
        if frames:
            panel = pd.concat(frames, ignore_index=True)
            close = panel.pivot_table(
                index="date", columns="code", values="close"
            ).sort_index()
            turn = panel.pivot_table(
                index="date", columns="code", values="turnover_ratio"
            ).sort_index()
            series_map = build_series_map(
                selected,
                panel=panel,
                close=close,
                turn=turn,
                target_dates=target,
                momentum_period=momentum_period,
                reversal_period=reversal_period,
                turnover_window=turnover_window,
                ubl_z_window=ubl_z_window,
                ubl_feat_window=ubl_feat_window,
                ubl_wms_n=ubl_wms_n,
            )

        if m6_names:
            val_long = load_valuation_long(self.loader.data_dir, hist_dates)
            fund_long = load_fundamental_long(self.loader.data_dir, hist_dates)
            div_events = None
            close_for_dy = None
            if any(resolve_op(n) == "m6_dy" for n in m6_names):
                from src.data.update_dividend_data import load_all_dividend_events

                div_events = load_all_dividend_events(
                    Path(self.loader.data_dir) / "_cache" / "dividend"
                )
                if frames:
                    close_for_dy = close
            m6_panels = compute_module6_panels(
                val_long,
                fund_long,
                close=close_for_dy,
                dividend_events=div_events,
            )
            missing = [n for n in m6_names if n not in m6_panels]
            if missing:
                print(
                    f"警告: 模块6因子数据不足，暂跳过: {missing}；"
                    "请续拉 data_valuation / data_fundamental / 分红缓存"
                )
            for name in m6_names:
                if name in m6_panels:
                    series_map[name] = m6_panels[name]

        written = []
        self.factor_dir.mkdir(parents=True, exist_ok=True)
        for date in target:
            cols = {}
            for name, mat in series_map.items():
                if date not in mat.index:
                    continue
                cols[name] = mat.loc[date]
            if not cols:
                continue
            out = pd.DataFrame(cols)
            out = out.dropna(how="all")
            if out.empty:
                continue
            out = out.reset_index()
            if "code" not in out.columns:
                out = out.rename(columns={out.columns[0]: "code"})
            out["date"] = date

            # 与已有 raw 合并，避免只算子集时覆盖掉其它因子列
            existing_path = resolve_dated_file(self.factor_dir, date)
            if existing_path is not None:
                old = read_frame(existing_path)
                if "code" in old.columns:
                    old_only = [c for c in old.columns if c not in out.columns]
                    if old_only:
                        out = out.merge(old[["code"] + old_only], on="code", how="outer")
                out["date"] = date

            preferred = ["code", "date"] + [c for c in selected if c in out.columns]
            preferred += [c for c in out.columns if c not in preferred]
            out = out[preferred]
            write_factor_frame(out, self.factor_dir, date)
            written.append(date)

        print(f"原始因子已写入 {self.factor_dir}，共 {len(written)} 天，列={selected}")
        return written

    def calculate_from_registry(
        self,
        start_date: str,
        end_date: str,
        registry_path: str = "./factors/registry.json",
        statuses: Tuple[str, ...] = ("candidate", "active"),
        momentum_period: int = 20,
        reversal_period: int = 5,
        turnover_window: int = 20,
    ) -> list:
        """
        读取 registry 中 candidate/active 因子并计算。
        未实现算子的因子（如仅占位的 CPV）会直接报错。
        """
        from src.research.factor_library import FactorLibrary
        from src.research.factor_ops import assert_ops_implemented

        library = FactorLibrary(registry_path)
        names = library.names_for_compute(statuses=statuses)
        if not names:
            raise RuntimeError("registry 中没有 candidate/active 因子可计算")
        assert_ops_implemented(names, impl_by_name=library.impl_map(names))
        print(f"按 registry 计算因子: {names}")
        return self.calculate_range_and_save(
            start_date=start_date,
            end_date=end_date,
            momentum_period=momentum_period,
            reversal_period=reversal_period,
            turnover_window=turnover_window,
            factor_names=names,
        )
    def load_factor(self, date: str) -> pd.DataFrame:
        """
        读取单日因子数据

        Args:
            date: 日期字符串

        Returns:
            DataFrame: 因子数据（列名取决于计算时的 period，例如 momentum_20）
        """
        file_path = resolve_dated_file(self.factor_dir, date)
        if file_path is None:
            return pd.DataFrame()

        return read_frame(file_path)


# ========== 测试代码 ==========
if __name__ == '__main__':
    # 初始化
    loader = DataLoader('./data')
    calculator = FactorCalculator(loader)

    # 测试单日计算
    if len(loader.trade_dates) > 30:
        test_date = loader.trade_dates[30]  # 选择第 31 天（确保有 20 天历史）
        factor_df = calculator.calculate_momentum_daily(test_date, period=20)

        print(f"\n📊 {test_date} 的动量因子（前 5 只股票）:")
        print(factor_df.head())
        print(f"\n因子统计:")
        factor_col = calculator.momentum_col(20)
        print(factor_df[factor_col].describe())

        # 批量计算并保存（取消注释可运行，但会花费较长时间）
        calculator.calculate_and_save_all(period=20)
    else:
        print("交易日数据不足，无法进行测试")
