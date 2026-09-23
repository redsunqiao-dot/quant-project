import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.strategy.strategy_base import Strategy


class CPVStrategy(Strategy):
    """
    CPV策略（Candle-Price-Volume Strategy）是基于K线形态（Candle）、价格（Price）和成交量（Volume）信息构建的日频量化投资策略。
    该策略旨在通过分析股票的K线结构、威廉指标（Williams %R）和趋势等多个维度，来计算一个综合因子，并根据该因子进行股票选择。

    策略核心逻辑：
    - 在 `calculate_factor` 方法中，完成股票池的筛选、原始因子的计算、去极值、标准化以及市值中性化、行业中性化等预处理步骤。
    - `generate_signal` 方法则根据中性化后的因子值，选取排名靠前的股票作为交易信号。

    因子组成：
    1.  U因子 (Upper Shadow Ratio)：衡量K线上影线长度占K线总长度的比例，反映上涨动能。
    2.  B因子 (Body Ratio)：衡量K线实体长度占K线总长度的比例，反映多空力量。
    3.  L因子 (Lower Shadow Ratio)：衡量K线下影线长度占K线总长度的比例，反映下跌动能。
    4.  WR因子 (Williams %R)：威廉指标，衡量当前收盘价在过去一段时间内的相对位置。
    5.  TREND因子：WR指标的短期均线减去长期均线，捕捉WR指标的趋势。

    中性化处理：
    - 支持市值中性化，以消除市值大小对因子表现的影响。
    - 支持行业中性化，以消除行业因素对因子表现的影响。

    股票池筛选：
    - 过滤ST股票。
    - 过滤上市天数不足的股票。
    - 过滤流动性不足的股票（通过平均成交量）。

    数据缓存：
    - 内部使用缓存机制，避免在回测过程中重复加载历史数据，提高效率。
    """

    def __init__(
        self,
        data_dir: str = './data',  # 数据文件存储的根目录
        candle_window_short: int = 5,  # K线相关因子（U/B/L）的短期时间窗口
        candle_window_long: int = 20,  # K线相关因子（U/B/L）的长期时间窗口
        wr_window_short: int = 5,  # 威廉指标（WR）的短期时间窗口
        wr_window_long: int = 20,  # 威廉指标（WR）的长期时间窗口
        min_avg_volume: float = 5e5,  # 股票池筛选：最小日均成交量
        liquidity_window: int = 20,  # 股票池筛选：计算日均成交量的历史窗口
        min_stock_count: int = 200,  # 股票池筛选：每日股票数量的最小阈值
        weights: Optional[Dict[str, float]] = None,  # 各子因子（U, B, L, WR, TREND）的组合权重
        outlier_method: str = 'sigma',  # 去极值方法，可选 'sigma' (标准差) 或 'quantile' (分位数)
        outlier_param: float = 3.0,  # 去极值参数，若为 'sigma' 则代表几倍标准差，若为 'quantile' 则代表分位数比例
        neutralize_industry: bool = True,  # 是否进行行业中性化
        min_listed_days: int = 252,  # 股票池筛选：最小上市天数
        min_listed_coverage: float = 0.8,  # 股票池筛选：上市天数覆盖率（例如0.8表示在min_listed_days内至少有80%的交易日数据）
        use_long_candle: bool = True,  # 是否使用长期K线指标（即U/B/L因子是否结合短期和长期窗口）
    ):
        super().__init__(name='CPV_UBL')
        self.data_dir = Path(data_dir)  # 数据目录路径
        self.candle_window_short = candle_window_short  # K线短期窗口
        self.candle_window_long = candle_window_long  # K线长期窗口
        self.wr_window_short = wr_window_short  # WR短期窗口
        self.wr_window_long = wr_window_long  # WR长期窗口
        self.min_avg_volume = min_avg_volume  # 最小平均成交量
        self.liquidity_window = liquidity_window  # 流动性窗口
        self.min_stock_count = min_stock_count  # 最小股票数量
        # 各子因子权重，默认为等权重
        self.weights = weights or {'U': 1, 'B': 1, 'L': 1, 'WR': 1, 'TREND': 1}
        self.outlier_method = outlier_method  # 去极值方法
        self.outlier_param = outlier_param  # 去极值参数
        self.neutralize_industry = neutralize_industry  # 是否行业中性化
        self.min_listed_days = min_listed_days  # 最小上市天数
        self.min_listed_coverage = min_listed_coverage  # 上市覆盖率
        self.use_long_candle = use_long_candle  # 是否使用长期K线

        # 缓存，避免重复 IO，提高数据读取效率
        self.trade_dates: Optional[List[str]] = None  # 所有交易日列表
        self.date_to_idx: Dict[str, int] = {}  # 交易日期到其在交易日列表中的索引的映射
        self.daily_cache: Dict[str, pd.DataFrame] = {}  # 每日行情数据缓存
        self.status_cache: Dict[str, pd.DataFrame] = {}  # 每日股票交易状态数据缓存
        self.barra_cache: Dict[str, pd.DataFrame] = {}  # 每日 Barra 风险因子数据缓存（用于市值中性化）
        self.industry_cache: Dict[str, pd.DataFrame] = {}  # 每日行业数据缓存（用于行业中性化）

    # ====== 工具函数 ======
    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """
        计算 Series 的 Z-score 标准化。
        如果标准差为0或NaN，则返回全为0的Series，避免除以零错误。

        Args:
            series (pd.Series): 输入的 Series 数据。

        Returns:
            pd.Series: Z-score 标准化后的 Series。
        """
        s = series.astype(float)
        std = s.std()
        if std == 0 or np.isnan(std):
            return s * 0.0
        return (s - s.mean()) / (std + 1e-8)  # 加一个很小的常数防止除以零

    @staticmethod
    def _mad_clip(series: pd.Series, n: int = 5) -> pd.Series:
        """
        基于中位数绝对偏差 (MAD) 对 Series 进行去极值处理。
        将超出中位数 ± n * MAD 范围的值截断到边界上。

        Args:
            series (pd.Series): 输入的 Series 数据。
            n (int): MAD 的倍数，用于确定截断边界。

        Returns:
            pd.Series: 去极值处理后的 Series。
        """
        med = series.median()
        # 计算 MAD，并加一个很小的常数防止为零
        mad = np.median(np.abs(series - med)) + 1e-8
        return series.clip(lower=med - n * mad, upper=med + n * mad)

    def _clip_outliers(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        根据指定的去极值方法（sigma或quantile）对DataFrame的指定列进行去极值处理。

        Args:
            df (pd.DataFrame): 待处理的 DataFrame。
            cols (List[str]): 需要进行去极值处理的列名列表。

        Returns:
            pd.DataFrame: 去极值处理后的 DataFrame。
        """
        if df.empty:
            return df

        method = (self.outlier_method or '').lower()  # 获取去极值方法，转为小写
        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].astype(float)
            if method == 'quantile':  # 分位数去极值
                q = float(self.outlier_param)  # 分位数参数，例如0.01表示截去上下1%的极端值
                lower = series.quantile(q)  # 下限
                upper = series.quantile(1 - q)  # 上限
                df[col] = series.clip(lower=lower, upper=upper)  # 截断
            else:  # 默认使用 Sigma (标准差) 去极值
                n_sigma = float(self.outlier_param)  # Sigma 参数，例如3.0表示截去正负3个标准差以外的值
                mean = series.mean()
                std = series.std()
                if std == 0 or np.isnan(std):  # 如果标准差为0或NaN，则无法去极值
                    continue
                df[col] = series.clip(lower=mean - n_sigma * std, upper=mean + n_sigma * std)  # 截断
        return df

    def _ensure_trade_dates(self, data_loader) -> None:
        """
        确保交易日期列表已加载并缓存。
        如果未加载，则从数据加载器获取所有交易日期并建立日期到索引的映射。

        Args:
            data_loader: 数据加载器实例，用于获取交易日期。
        """
        if self.trade_dates is None:
            self.trade_dates = data_loader.get_all_dates()  # 从数据加载器获取所有交易日期
            # 创建日期到索引的映射，方便通过日期查找其在交易日列表中的位置
            self.date_to_idx = {d: i for i, d in enumerate(self.trade_dates)}


    def _get_daily(self, data_loader, date: str) -> pd.DataFrame:
        """
        获取指定日期的每日行情数据（K线、成交量、市值等）。
        数据会进行缓存，避免重复加载。

        Args:
            data_loader: 数据加载器实例。
            date (str): 交易日期。

        Returns:
            pd.DataFrame: 指定日期的行情数据。
        """
        if date not in self.daily_cache:  # 如果数据不在缓存中
            df = data_loader.get_daily_data(date)  # 从数据加载器获取每日数据
            if not df.empty:
                # 定义需要的基础列和可能的市值列
                cols = ['code', 'open', 'high', 'low', 'close', 'volume', 'date']
                extra_cols = [
                    'market_cap', 'total_mv', 'mkt_cap', 'mv', 'circ_mv', 'float_mv'
                ]
                # 筛选出实际存在的列
                cols += [c for c in extra_cols if c in df.columns]
                available = [c for c in cols if c in df.columns]
                df = df[available].copy()  # 仅保留需要的列并复制，防止SettingWithCopyWarning
                if 'date' in df.columns:
                    df['time'] = pd.to_datetime(df['date'])  # 将日期列转换为datetime类型
                else:
                    df['time'] = pd.to_datetime(date)  # 如果没有date列，使用传入的date作为时间
            self.daily_cache[date] = df  # 缓存数据
        return self.daily_cache[date]  # 返回缓存的数据

    def _get_status(self, data_loader, date: str) -> pd.DataFrame:
        """
        获取指定日期的股票交易状态数据（如是否ST、是否停牌、涨跌停状态）。
        数据会进行缓存，避免重复加载。

        Args:
            data_loader: 数据加载器实例。
            date (str): 交易日期。

        Returns:
            pd.DataFrame: 指定日期的股票交易状态数据。
        """
        if date not in self.status_cache:  # 如果数据不在缓存中
            self.status_cache[date] = data_loader.get_daily_status(date)  # 从数据加载器获取每日状态数据
        return self.status_cache[date]  # 返回缓存的数据

    def _get_barra(self, date: str) -> pd.DataFrame:
        """
        获取指定日期的Barra风格因子数据（例如size因子）。
        数据会进行缓存，避免重复加载。

        Args:
            date (str): 交易日期。

        Returns:
            pd.DataFrame: 指定日期的Barra因子数据。
        """
        if date not in self.barra_cache:  # 如果数据不在缓存中
            path = self.data_dir / 'data_barra' / f'{date}.csv'  # 构建Barra数据文件路径
            if path.exists():
                self.barra_cache[date] = pd.read_csv(path)  # 读取CSV文件并缓存
            else:
                self.barra_cache[date] = pd.DataFrame()  # 如果文件不存在，缓存空DataFrame
        return self.barra_cache[date]  # 返回缓存的数据

    def _get_industry(self, date: str) -> pd.DataFrame:
        """
        获取指定日期的行业分类数据。
        数据会进行缓存，避免重复加载。

        Args:
            date (str): 交易日期。

        Returns:
            pd.DataFrame: 指定日期的行业分类数据。
        """
        if date not in self.industry_cache:  # 如果数据不在缓存中
            path = self.data_dir / 'data_industry' / f'{date}.csv'  # 构建行业数据文件路径
            if path.exists():
                self.industry_cache[date] = pd.read_csv(path)  # 读取CSV文件并缓存
            else:
                self.industry_cache[date] = pd.DataFrame()  # 如果文件不存在，缓存空DataFrame
        return self.industry_cache[date]  # 返回缓存的数据

    def _get_stock_pool(self, date: str, data_loader) -> List[str]:
        """
        根据一系列筛选条件获取指定日期的股票池。

        筛选条件包括：
        1. 排除ST股票。
        2. 排除停牌、涨停、跌停的股票。
        3. 排除上市天数不足的股票。
        4. 排除流动性（日均成交量）不足的股票。

        Args:
            date (str): 交易日期。
            data_loader: 数据加载器实例。

        Returns:
            List[str]: 符合筛选条件的股票代码列表。
        """
        daily = self._get_daily(data_loader, date)
        if daily is None or daily.empty:
            return []

        stocks = daily['code'].tolist()  # 获取当日所有股票代码

        # ST 过滤（依赖本地 data_ud_new 的 dt/zt 信息，此处假设 zt/dt=1 代表涨跌停/异常）
        status = self._get_status(data_loader, date)
        if status is not None and not status.empty:
            status = status.copy()
            if 'st' in status.columns:
                status = status[status['st'] == 0]  # 过滤掉ST股票
            # 过滤掉停牌、涨停（zt=1）、跌停（dt=1）的股票
            tradable = status[
                (status['paused'] == 0) &  # 未停牌
                (status['zt'] == 0) &     # 未涨停
                (status['dt'] == 0)       # 未跌停
            ]
            stocks = [s for s in stocks if s in set(tradable['code'])]  # 更新股票池

        idx = self.date_to_idx.get(date)
        if idx is None:
            return []

        # 上市天数过滤
        # 考虑K线和WR指标的最大窗口以及最小上市天数，取其中最大值作为历史数据窗口需求
        min_days = max(self.candle_window_long, self.wr_window_long, self.min_listed_days)
        if idx >= min_days:  # 确保当前日期有足够的历史数据进行判断
            eligible_dates = self.trade_dates[idx - min_days: idx]  # 获取需要检查的过去日期
            counts = {}  # 记录每只股票在 eligible_dates 中出现的次数
            for d in eligible_dates:
                df = self._get_daily(data_loader, d)
                if df is None or df.empty:
                    continue
                for code in df['code'].tolist():
                    counts[code] = counts.get(code, 0) + 1
            # 计算最小需要的交易日数量 (例如 252 * 0.8)
            min_required = int(min_days * self.min_listed_coverage)
            # 筛选出上市天数满足要求的股票
            stocks = [s for s in stocks if counts.get(s, 0) >= min_required]

        if idx <= self.liquidity_window:  # 如果当前日期没有足够的历史数据来计算流动性，则跳过流动性过滤
            return stocks

        # 流动性过滤 (日均成交量)
        hist_dates = self.trade_dates[idx - self.liquidity_window: idx]  # 获取用于计算流动性的历史日期
        vol_frames = []
        stock_set = set(stocks)  # 将当前股票池转为集合，加快查找速度
        for d in hist_dates:
            df = self._get_daily(data_loader, d)
            if df is None or df.empty:
                continue
            # 筛选出当前股票池中的股票及其成交量
            sub = df[df['code'].isin(stock_set)][['code', 'volume']]
            if not sub.empty:
                vol_frames.append(sub)

        if vol_frames:
            vol_df = pd.concat(vol_frames)  # 合并历史成交量数据
            avg_vol = vol_df.groupby('code')['volume'].mean()  # 计算每只股票的平均成交量
            # 筛选出日均成交量大于最小阈值的股票
            stocks = avg_vol[avg_vol >= self.min_avg_volume].index.tolist()

        return stocks

    def _load_panel(self, date: str, stocks: List[str], data_loader, window: int) -> pd.DataFrame:
        """
        加载指定日期和股票列表的 K 线历史面板数据。

        Args:
            date (str): 当前交易日期。
            stocks (List[str]): 需要加载数据的股票代码列表。
            data_loader: 数据加载器实例。
            window (int): 需要加载的历史天数窗口。

        Returns:
            pd.DataFrame: 包含指定股票在过去 window 天内的 K 线历史数据面板。
        """
        idx = self.date_to_idx.get(date)
        # 如果当前日期不在交易日列表中，或者历史窗口不足，则返回空DataFrame
        if idx is None or idx < window:
            return pd.DataFrame()

        # 获取需要加载数据的历史日期列表
        use_dates = self.trade_dates[idx - window + 1: idx + 1]
        frames = []
        stock_set = set(stocks)  # 将股票列表转为集合，加快查找速度
        for d in use_dates:
            df = self._get_daily(data_loader, d)
            if df is None or df.empty:
                continue
            # 筛选出当前股票池中的股票数据
            sub = df[df['code'].isin(stock_set)].copy()
            if not sub.empty:
                # 确保 'time' 列存在，方便后续处理
                if 'time' not in sub.columns:
                    sub['time'] = pd.to_datetime(d)
                frames.append(sub)

        if not frames:  # 如果没有获取到任何数据，返回空DataFrame
            return pd.DataFrame()

        # 合并所有历史数据，并按股票代码和时间排序
        panel = pd.concat(frames).sort_values(['code', 'time'])
        # 统计每只股票在面板中出现的次数，确保数据完整性
        counts = panel.groupby('code').size()
        # 筛选出在整个窗口期内都有完整数据的股票
        valid = counts[counts >= window].index
        return panel[panel['code'].isin(valid)]

    def _neutralize(self, factor_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        对因子进行中性化处理，包括市值中性化和行业中性化。
        使用线性回归方法，将因子暴露在市值和行业上的部分剥离。

        Args:
            factor_df (pd.DataFrame): 包含待中性化因子的 DataFrame，至少包含 'CPV' 列。
            date (str): 交易日期。

        Returns:
            pd.DataFrame: 中性化后的因子 DataFrame，新增 'CPV_neu' 列。
        """
        if factor_df.empty:
            return factor_df

        barra = self._get_barra(date)
        # 检查是否存在Barra的size因子数据
        has_size = not barra.empty and 'size' in barra.columns
        size = barra.set_index('code')['size'] if has_size else None

        df = factor_df.copy()
        # 如果存在Barra的size因子，则合并到factor_df中
        if has_size:
            df = df.join(size.rename('barra_size'))

        # 优先使用日线数据中的市值字段（如market_cap），如果没有则使用Barra的size因子
        cap_col = None
        for c in ['market_cap', 'total_mv', 'mkt_cap', 'mv', 'circ_mv', 'float_mv']:
            if c in df.columns:
                cap_col = c
                break
        if cap_col:
            # 市值取对数作为市值因子，避免数据过大，并处理0值
            df['size_factor'] = np.log(df[cap_col].astype(float).replace(0, np.nan))
        elif has_size:
            df['size_factor'] = df['barra_size']

        # 如果需要进行行业中性化
        if self.neutralize_industry:
            industry = self._get_industry(date)
            if not industry.empty and 'industry' in industry.columns:
                # 合并行业数据
                industry = industry.set_index('code')['industry']
                df = df.join(industry.rename('industry'))

        # 用于回归的自变量列，确保这些列不包含NaN
        drop_cols = ['CPV']
        if 'size_factor' in df.columns:
            drop_cols.append('size_factor')
        if self.neutralize_industry and 'industry' in df.columns:
            drop_cols.append('industry')

        df = df.dropna(subset=drop_cols)  # 删除包含NaN的行
        if df.empty:
            return pd.DataFrame()

        # 构建回归矩阵 X
        X_parts = [np.ones(len(df))]  # 截距项
        if 'size_factor' in df.columns:
            X_parts.append(df['size_factor'].values)  # 添加市值因子

        if self.neutralize_industry and 'industry' in df.columns:
            # 将行业类别转换为独热编码（Dummy Variables），并去除第一个，避免多重共线性
            industry_dummies = pd.get_dummies(df['industry'], drop_first=True)
            if not industry_dummies.empty:
                X_parts.append(industry_dummies.values)  # 添加行业虚拟变量

        X = np.column_stack(X_parts)  # 将所有自变量堆叠成矩阵 X
        y = df['CPV'].values  # 因变量为原始 CPV 因子

        try:
            # 使用最小二乘法进行回归
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            # 计算回归残差，即中性化后的因子值
            resid = y - X.dot(coef)
        except Exception:
            # 如果回归失败，则简单地用因子值减去其均值作为残差
            # 这是一种简单的替代方法，但不如完整的回归中性化有效
            resid = y - y.mean()

        # 对中性化后的残差进行 Z-score 标准化
        df['CPV_neu'] = self._zscore(pd.Series(resid, index=df.index))
        return df

    # ====== 策略接口实现 ======
    def calculate_factor(self, date: str, data_loader, **kwargs) -> pd.DataFrame:
        """
        计算指定日期的CPV因子。
        整个过程包括：股票池筛选 -> K线特征计算 -> WR指标计算 -> 趋势指标计算 ->
        子因子组合 -> 去极值 -> 截面标准化 -> 市值/行业中性化。

        Args:
            date (str): 当前交易日期。
            data_loader: 数据加载器实例。
            **kwargs: 其他可选参数。

        Returns:
            pd.DataFrame: 包含股票代码、日期和因子值的DataFrame。
                          列为 ['code', 'date', 'factor_value']。
        """
        self._ensure_trade_dates(data_loader)  # 确保交易日期列表已加载

        stocks = self._get_stock_pool(date, data_loader)  # 获取当日股票池
        if len(stocks) < self.min_stock_count:
            return pd.DataFrame()  # 如果股票数量不足，返回空DataFrame

        # 计算K线和WR指标所需的最大历史窗口
        max_window = max(self.candle_window_long, self.wr_window_long)
        # 加载股票池内股票在过去max_window天内的K线面板数据
        panel = self._load_panel(date, stocks, data_loader, max_window)
        if panel.empty:
            return pd.DataFrame()

        df = panel.copy()
        # 计算K线基本特征
        df['hl'] = (df['high'] - df['low']).replace(0, 0.001)  # 最高价-最低价，防止除以零
        df['upper'] = df['high'] - df[['open', 'close']].max(axis=1)  # 上影线长度
        df['lower'] = df[['open', 'close']].min(axis=1) - df['low']  # 下影线长度
        df['body'] = (df['close'] - df['open']).abs()  # K线实体长度

        grouped = df.groupby('code', group_keys=False)
        # 计算短期和长期窗口内的K线特征均值
        for window in [self.candle_window_short, self.candle_window_long]:
            # 计算上影线、下影线、实体和高低价差的滑动平均
            df[f'upper_mean_{window}'] = grouped['upper'].rolling(window).mean().reset_index(level=0, drop=True)
            df[f'lower_mean_{window}'] = grouped['lower'].rolling(window).mean().reset_index(level=0, drop=True)
            df[f'body_mean_{window}'] = grouped['body'].rolling(window).mean().reset_index(level=0, drop=True)
            df[f'hl_mean_{window}'] = grouped['hl'].rolling(window).mean().reset_index(level=0, drop=True)
            # 计算U, B, L因子（标准化处理）
            df[f'U_{window}'] = df[f'upper_mean_{window}'] / df[f'hl_mean_{window}']
            df[f'B_{window}'] = df[f'body_mean_{window}'] / df[f'hl_mean_{window}']
            df[f'L_{window}'] = df[f'lower_mean_{window}'] / df[f'hl_mean_{window}']

        # 计算威廉指标（Williams %R）
        df['wr'] = (df['close'] - df['low']) / df['hl'] * 100
        # 计算威廉指标的短期和长期滑动平均
        df['wr_s'] = grouped['wr'].rolling(self.wr_window_short).mean().reset_index(level=0, drop=True)
        df['wr_l'] = grouped['wr'].rolling(self.wr_window_long).mean().reset_index(level=0, drop=True)
        # 计算WR指标的趋势因子 (短期WR - 长期WR)
        df['wr_trend'] = df['wr_s'] - df['wr_l']

        # 提取最新一期（当前日期）的因子值
        latest = df.groupby('code').tail(1).set_index('code')
        factor_raw = pd.DataFrame(index=latest.index)
        # 收集原始因子值
        factor_raw['U5'] = latest[f'U_{self.candle_window_short}']
        factor_raw['B5'] = latest[f'B_{self.candle_window_short}']
        factor_raw['L5'] = latest[f'L_{self.candle_window_short}']
        factor_raw['U20'] = latest[f'U_{self.candle_window_long}']
        factor_raw['B20'] = latest[f'B_{self.candle_window_long}']
        factor_raw['L20'] = latest[f'L_{self.candle_window_long}']
        factor_raw['WR'] = latest['wr_l']
        factor_raw['TREND'] = latest['wr_trend']
        # 将市值字段带入，用于后续的中性化处理
        for cap_col in ['market_cap', 'total_mv', 'mkt_cap', 'mv', 'circ_mv', 'float_mv']:
            if cap_col in latest.columns:
                factor_raw[cap_col] = latest[cap_col]
                break

        # 对原始因子进行去极值处理
        factor_raw = self._clip_outliers(
            factor_raw,
            cols=['U5', 'B5', 'L5', 'U20', 'B20', 'L20', 'WR', 'TREND']
        )

        # 对去极值后的因子进行截面标准化（Z-score）
        z = {}
        for col in ['U5', 'B5', 'L5', 'U20', 'B20', 'L20', 'WR', 'TREND']:
            if col in factor_raw.columns:
                z[col] = self._zscore(factor_raw[col])

        # 组合因子
        factor = pd.DataFrame(index=latest.index)
        if self.use_long_candle:
            # 如果使用长期K线，则短期和长期因子取平均
            factor['U'] = (z['U5'] + z['U20']) / 2
            factor['B'] = (z['B5'] + z['B20']) / 2
            factor['L'] = (z['L5'] + z['L20']) / 2
        else:
            # 否则只使用短期K线因子
            factor['U'] = z['U5']
            factor['B'] = z['B5']
            factor['L'] = z['L5']
        factor['WR'] = z['WR']
        factor['TREND'] = z['TREND']

        # 根据权重合并子因子，得到综合CPV因子
        w = self.weights
        total_w = sum(w.values()) or 1  # 权重归一化分母
        factor['CPV'] = (
            w['U'] * factor['U'] +
            w['B'] * factor['B'] +
            w['L'] * factor['L'] +
            w['WR'] * factor['WR'] +
            w['TREND'] * factor['TREND']
        ) / total_w
        # 对综合CPV因子再次去极值
        factor = self._clip_outliers(factor, cols=['CPV'])

        # 对CPV因子进行市值和行业中性化
        factor = self._neutralize(factor, date)
        # 如果中性化后结果为空或不包含中性化因子，则返回空DataFrame
        if factor.empty or 'CPV_neu' not in factor.columns:
            return pd.DataFrame()

        # 整理结果，重命名中性化因子列为 'factor_value'
        result = factor[['CPV_neu']].rename(columns={'CPV_neu': 'factor_value'})
        # 删除因子值为空的行，并重置索引
        result = result.dropna(subset=['factor_value']).reset_index()
        result['date'] = date  # 添加日期列
        return result[['code', 'date', 'factor_value']]  # 返回最终因子DataFrame

    def generate_signal(self, factor_df: pd.DataFrame, top_n: int = 10) -> list:
        if factor_df.empty or 'factor_value' not in factor_df.columns:
            return []

        sorted_df = factor_df.sort_values('factor_value', ascending=False)
        return sorted_df.head(top_n)['code'].tolist()
