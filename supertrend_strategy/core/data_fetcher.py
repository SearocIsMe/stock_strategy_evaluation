import os
import pandas as pd
import numpy as np
import tushare as ts
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataFetcher')

class DataFetcher:
    """
    数据获取和处理类
    负责从tushare获取A股数据并计算技术指标
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        初始化数据获取器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化tushare
        self._init_tushare()
        
        # 缓存数据
        self.stock_data = {}  # 股票数据缓存
        self.stock_list = []  # 股票列表缓存
        self.trade_dates = [] # 交易日期缓存
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def _init_tushare(self):
        """初始化tushare API"""
        try:
            token = self.config['data']['tushare_token']
            ts.set_token(token)
            self.pro = ts.pro_api()
            logger.info("Tushare API初始化成功")
        except Exception as e:
            logger.error(f"Tushare API初始化失败: {e}")
            raise
    
    def get_stock_list(self, refresh: bool = False) -> List[str]:
        """
        获取A股股票列表
        
        Args:
            refresh: 是否强制刷新缓存
            
        Returns:
            股票代码列表
        """
        if not self.stock_list or refresh:
            try:
                # 获取所有A股列表
                data = self.pro.stock_basic(
                    exchange='',
                    list_status='L',
                    fields='ts_code,symbol,name,area,industry,list_date'
                )
                # 过滤科创板、创业板等特殊板块（如有需要）
                # data = data[~data['ts_code'].str.startswith(('688', '300', '301'))]
                
                self.stock_list = data['ts_code'].tolist()
                logger.info(f"获取股票列表成功，共{len(self.stock_list)}只股票")
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
                raise
        
        return self.stock_list
    
    def get_trade_dates(self, start_date: str = None, end_date: str = None, refresh: bool = False) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期，默认使用配置文件中的日期
            end_date: 结束日期，默认使用配置文件中的日期
            refresh: 是否强制刷新缓存
            
        Returns:
            交易日期列表
        """
        if not self.trade_dates or refresh:
            start = start_date or self.config['data']['start_date']
            end = end_date or self.config['data']['end_date']
            
            try:
                # 获取交易日历
                df = self.pro.trade_cal(
                    exchange='SSE',
                    start_date=start.replace('-', ''),
                    end_date=end.replace('-', ''),
                    is_open='1'
                )
                self.trade_dates = df['cal_date'].tolist()
                logger.info(f"获取交易日历成功，从{start}到{end}共{len(self.trade_dates)}个交易日")
            except Exception as e:
                logger.error(f"获取交易日历失败: {e}")
                raise
        
        return self.trade_dates
    
    def get_stock_data(self, ts_code: str, start_date: str = None, end_date: str = None, 
                       refresh: bool = False) -> pd.DataFrame:
        """
        获取单只股票的日线数据并计算技术指标
        
        Args:
            ts_code: 股票代码（tushare格式，如：000001.SZ）
            start_date: 开始日期，默认使用配置文件中的日期
            end_date: 结束日期，默认使用配置文件中的日期
            refresh: 是否强制刷新缓存
            
        Returns:
            包含价格和技术指标的DataFrame
        """
        # 生成缓存键
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self.stock_data and not refresh:
            return self.stock_data[cache_key]
        
        # 设置日期范围
        start = start_date or self.config['data']['start_date']
        end = end_date or self.config['data']['end_date']
        
        try:
            # 获取日线数据
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=start.replace('-', ''),
                end_date=end.replace('-', ''),
                adj='qfq',  # 前复权
                freq='D'    # 日线
            )
            
            # 检查数据是否为空
            if df is None or df.empty:
                logger.warning(f"股票{ts_code}在{start}至{end}期间无数据")
                return pd.DataFrame()
            
            # 按日期升序排序
            df = df.sort_values('trade_date')
            
            # 计算技术指标
            df = self._calculate_indicators(df)
            
            # 缓存数据
            self.stock_data[cache_key] = df
            
            logger.info(f"获取股票{ts_code}数据成功，从{start}到{end}共{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取股票{ts_code}数据失败: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, ts_code: str) -> pd.DataFrame:
        """
        获取股票基本面数据
        
        Args:
            ts_code: 股票代码（tushare格式）
            
        Returns:
            基本面数据DataFrame
        """
        try:
            # 获取最新财务指标
            df_basic = self.pro.daily_basic(
                ts_code=ts_code,
                trade_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv'
            )
            
            # 获取最新财务报表
            df_finance = self.pro.fina_indicator(
                ts_code=ts_code,
                period=self._get_latest_quarter(),
                fields='ts_code,ann_date,end_date,grossprofit_margin,netprofit_margin,roe,roa'
            )
            
            # 合并数据
            if not df_basic.empty and not df_finance.empty:
                result = pd.merge(df_basic, df_finance, on='ts_code', how='left')
                return result
            elif not df_basic.empty:
                return df_basic
            elif not df_finance.empty:
                return df_finance
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取股票{ts_code}基本面数据失败: {e}")
            return pd.DataFrame()
    
    def _get_latest_quarter(self) -> str:
        """获取最近的财报季度"""
        now = datetime.now()
        year = now.year
        month = now.month
        
        if month < 4:
            return f"{year-1}1231"
        elif month < 7:
            return f"{year}0331"
        elif month < 10:
            return f"{year}0630"
        else:
            return f"{year}0930"
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 原始价格数据
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 确保数据不为空
        if df.empty:
            return df
        
        # 复制数据，避免修改原始数据
        df = df.copy()
        
        # 计算Supertrend指标
        self._calculate_supertrend(df)
        
        # 计算EMA指标
        self._calculate_ema(df)
        
        # 计算Ichimoku云层指标
        self._calculate_ichimoku(df)
        
        # 计算周线均线
        self._calculate_weekly_ma(df)
        
        # 计算量能指标
        self._calculate_volume_indicators(df)
        
        # 计算筹码集中度
        self._calculate_chip_concentration(df)
        
        # 计算RSI指标
        self._calculate_rsi(df)
        
        return df
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> None:
        """计算Supertrend指标"""
        # 获取参数
        st1_factor = self.config['supertrend']['st1']['factor']
        st1_period = self.config['supertrend']['st1']['period']
        st2_factor = self.config['supertrend']['st2']['factor']
        st2_period = self.config['supertrend']['st2']['period']
        st3_factor = self.config['supertrend']['st3']['factor']
        st3_period = self.config['supertrend']['st3']['period']
        
        # 计算Supertrend 1
        df['st1'], df['st1_dir'] = self._supertrend(df, st1_factor, st1_period)
        df['uptrend1'] = np.where(df['st1_dir'] < 0, 1, 0)
        
        # 计算Supertrend 2
        df['st2'], df['st2_dir'] = self._supertrend(df, st2_factor, st2_period)
        df['uptrend2'] = np.where(df['st2_dir'] < 0, 1, 0)
        
        # 计算Supertrend 3
        df['st3'], df['st3_dir'] = self._supertrend(df, st3_factor, st3_period)
        df['uptrend3'] = np.where(df['st3_dir'] < 0, 1, 0)
        
        # 计算综合上升趋势
        df['all_uptrend'] = np.where(
            (df['uptrend1'] == 1) & (df['uptrend2'] == 1) & (df['uptrend3'] == 1),
            1, 0
        )
    
    def _supertrend(self, df: pd.DataFrame, factor: float, period: int) -> Tuple[pd.Series, pd.Series]:
        """
        计算Supertrend指标
        
        Args:
            df: 价格数据
            factor: 乘数因子
            period: 周期
            
        Returns:
            (supertrend, direction)
        """
        # 计算ATR
        high, low, close = df['high'], df['low'], df['close']
        
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # 计算上下轨
        hl2 = (high + low) / 2
        upperband = hl2 + (factor * atr)
        lowerband = hl2 - (factor * atr)
        
        # 计算Supertrend
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = -1  # 上升趋势
            else:
                supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = 1   # 下降趋势
        
        return supertrend, direction
    
    def _calculate_ema(self, df: pd.DataFrame) -> None:
        """计算EMA指标"""
        ema_length = self.config['ema']['length']
        df['ema144'] = df['close'].ewm(span=ema_length, adjust=False).mean()
        df['price_above_ema'] = np.where(df['close'] > df['ema144'], 1, 0)
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> None:
        """计算Ichimoku云层指标"""
        # 获取参数
        conversion_periods = self.config['ichimoku']['conversion_periods']
        base_periods = self.config['ichimoku']['base_periods']
        lagging_span_periods = self.config['ichimoku']['lagging_span_periods']
        
        # 计算转换线 (Conversion Line)
        high9 = df['high'].rolling(window=conversion_periods).max()
        low9 = df['low'].rolling(window=conversion_periods).min()
        df['conversion_line'] = (high9 + low9) / 2
        
        # 计算基准线 (Base Line)
        high26 = df['high'].rolling(window=base_periods).max()
        low26 = df['low'].rolling(window=base_periods).min()
        df['base_line'] = (high26 + low26) / 2
        
        # 计算先行带1 (Leading Span A)
        df['lead_line1'] = ((df['conversion_line'] + df['base_line']) / 2).shift(base_periods)
        
        # 计算先行带2 (Leading Span B)
        high52 = df['high'].rolling(window=lagging_span_periods).max()
        low52 = df['low'].rolling(window=lagging_span_periods).min()
        df['lead_line2'] = ((high52 + low52) / 2).shift(base_periods)
        
        # 计算延迟线 (Lagging Span)
        df['lagging_span'] = df['close'].shift(-base_periods)
        
        # 价格是否在云层上方
        df['price_above_cloud'] = np.where(
            (df['close'] > df['lead_line1']) & (df['close'] > df['lead_line2']),
            1, 0
        )
    
    def _calculate_weekly_ma(self, df: pd.DataFrame) -> None:
        """计算周线均线"""
        # 获取参数
        ma1_period = self.config['weekly_ma']['ma1_period']
        ma2_period = self.config['weekly_ma']['ma2_period']
        
        # 将日线数据转换为周线数据
        df['date'] = pd.to_datetime(df['trade_date'])
        df.set_index('date', inplace=True)
        
        # 计算周收盘价
        weekly_close = df['close'].resample('W').last()
        
        # 计算周线EMA
        weekly_ma13 = weekly_close.ewm(span=ma1_period, adjust=False).mean()
        weekly_ma34 = weekly_close.ewm(span=ma2_period, adjust=False).mean()
        
        # 将周线数据映射回日线数据
        weekly_ma13 = weekly_ma13.reindex(df.index, method='ffill')
        weekly_ma34 = weekly_ma34.reindex(df.index, method='ffill')
        
        df['weekly_ma13'] = weekly_ma13
        df['weekly_ma34'] = weekly_ma34
        
        # 计算周线条件
        df['weekly_condition'] = np.where(
            (df['close'] > df['weekly_ma13']) & (df['weekly_ma13'] > df['weekly_ma34']),
            1, 0
        )
        
        # 重置索引
        df.reset_index(inplace=True)
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> None:
        """计算量能指标"""
        # 获取参数
        vol_multiplier = self.config['volume']['multiplier']
        turnover_threshold = self.config['volume']['turnover_threshold']
        ma_period = self.config['volume']['ma_period']
        
        # 计算成交量均线
        df['volume_ma'] = df['vol'].rolling(window=ma_period).mean()
        
        # 计算放量信号
        df['volume_signal'] = np.where(df['vol'] >= df['volume_ma'] * vol_multiplier, 1, 0)
        
        # 计算换手率（如果数据中没有）
        if 'turnover_rate' not in df.columns:
            # 这里使用简化计算，实际应从tushare获取
            df['turnover_rate'] = df['vol'] / df['vol'].rolling(window=20).mean() * 5
        
        # 计算换手率信号
        df['turnover_signal'] = np.where(df['turnover_rate'] >= turnover_threshold, 1, 0)
    
    def _calculate_chip_concentration(self, df: pd.DataFrame) -> None:
        """计算筹码集中度"""
        # 获取参数
        threshold = self.config['chip_concentration']['threshold']
        
        # 初始化筹码集中度列
        df['concentration_ratio'] = np.nan
        df['dominant_price'] = np.nan
        
        # 计算筹码集中度（简化版本）
        window_size = 30  # 使用30天窗口
        
        for i in range(window_size, len(df)):
            # 获取窗口数据
            window = df.iloc[i-window_size:i]
            
            # 计算成交量加权平均价格
            total_vol = window['vol'].sum()
            if total_vol > 0:
                weighted_sum = (window['close'] * window['vol']).sum()
                dominant_price = weighted_sum / total_vol
                
                # 计算价格在主力成本附近的成交量占比
                price_range = 0.05  # 5%范围内
                in_range_vol = window.loc[
                    (window['close'] >= dominant_price * (1 - price_range)) &
                    (window['close'] <= dominant_price * (1 + price_range)),
                    'vol'
                ].sum()
                
                concentration_ratio = (in_range_vol / total_vol) * 100
                
                df.loc[df.index[i], 'concentration_ratio'] = concentration_ratio
                df.loc[df.index[i], 'dominant_price'] = dominant_price
        
        # 计算筹码集中度信号
        df['cost_valid'] = np.where(
            (~df['concentration_ratio'].isna()) & 
            (df['concentration_ratio'] >= threshold) &
            (~df['dominant_price'].isna()) &
            (abs(df['close'] - df['dominant_price']) / df['dominant_price'] * 100 < (100 - threshold)),
            1, 0
        )
    
    def _calculate_rsi(self, df: pd.DataFrame) -> None:
        """计算RSI指标"""
        # 获取参数
        period = self.config['rsi']['period']
        overbought = self.config['rsi']['overbought']
        
        # 计算价格变化
        delta = df['close'].diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算超买信号
        df['overbought'] = np.where(df['rsi'] > overbought, 1, 0)

    def get_batch_stock_data(self, ts_codes: List[str], start_date: str = None, 
                            end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据字典 {股票代码: 数据DataFrame}
        """
        result = {}
        for ts_code in ts_codes:
            df = self.get_stock_data(ts_code, start_date, end_date)
            if not df.empty:
                result[ts_code] = df
        
        logger.info(f"批量获取{len(ts_codes)}只股票数据，成功获取{len(result)}只")
        return result
