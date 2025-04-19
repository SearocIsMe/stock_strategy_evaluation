import os
import pandas as pd
import numpy as np
import tushare as ts
import yaml
import logging
import warnings
import redis
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional

# Suppress tushare warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
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
        
        # 初始化Redis连接
        self._init_redis()
        
        # 缓存数据
        self.stock_data = {}  # 本地股票数据缓存
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
            # 优先使用环境变量中的token，如果不存在则使用配置文件中的token
            token = os.environ.get('TUSHARE_TOKEN') or self.config['data']['tushare_token']
            ts.set_token(token)
            self.pro = ts.pro_api()
            logger.info("Tushare API初始化成功")
        except Exception as e:
            logger.error(f"Tushare API初始化失败: {e}")
            raise
    
    def _init_redis(self):
        """初始化Redis连接"""
        try:
            # 获取Redis配置
            redis_config = self.config.get('redis', {})
            host = redis_config.get('host', 'localhost')
            port = redis_config.get('port', 6379)
            db = redis_config.get('db', 0)
            password = redis_config.get('password', None)
            self.redis_ttl = redis_config.get('ttl', 86400)  # 默认缓存1天
            
            # 创建Redis连接
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password if password else None,
                decode_responses=False  # 不自动解码，因为我们存储的是二进制数据
            )
            
            # 测试连接
            self.redis.ping()
            logger.info(f"Redis连接成功: {host}:{port}/{db}")
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis连接失败: {e}，将使用本地缓存")
            self.redis_available = False
    
    def get_stock_list(self, refresh: bool = False) -> List[str]:
        """
        获取A股股票列表
        
        Args:
            refresh: 是否强制刷新缓存
            
        Returns:
            股票代码列表
        """
        # 生成缓存键
        cache_key = "stock_list"
        
        # 如果不强制刷新，先检查本地缓存
        if self.stock_list and not refresh:
            logger.info(f"从本地缓存获取股票列表")
            return self.stock_list
        
        # 如果不强制刷新，再检查Redis缓存
        if not refresh:
            redis_data = self._get_from_redis(cache_key)
            if redis_data is not None:
                # 缓存到本地
                self.stock_list = redis_data
                logger.info(f"从Redis获取股票列表成功，共{len(self.stock_list)}只股票")
                return self.stock_list
        
        try:
            # 从Tushare获取所有A股列表
            logger.info(f"从Tushare获取股票列表")
            data = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,list_date'
            )
            # 过滤科创板、创业板等特殊板块（如有需要）
            # data = data[~data['ts_code'].str.startswith(('688', '300', '301'))]
            
            self.stock_list = data['ts_code'].tolist()
            
            # 缓存数据到Redis
            self._save_to_redis(cache_key, self.stock_list)
            
            logger.info(f"从Tushare获取股票列表成功，共{len(self.stock_list)}只股票")
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
        # 设置日期范围
        start = start_date or self.config['data']['start_date']
        end = end_date or self.config['data']['end_date']
        
        # 生成缓存键
        cache_key = f"trade_dates:{start}:{end}"
        
        # 如果不强制刷新，先检查本地缓存
        if self.trade_dates and not refresh:
            logger.info(f"从本地缓存获取交易日历")
            return self.trade_dates
        
        # 如果不强制刷新，再检查Redis缓存
        if not refresh:
            redis_data = self._get_from_redis(cache_key)
            if redis_data is not None:
                # 缓存到本地
                self.trade_dates = redis_data
                logger.info(f"从Redis获取交易日历成功，从{start}到{end}共{len(self.trade_dates)}个交易日")
                return self.trade_dates
        
        try:
            # 从Tushare获取交易日历
            logger.info(f"从Tushare获取交易日历，日期范围: {start}至{end}")
            df = self.pro.trade_cal(
                exchange='SSE',
                start_date=start.replace('-', ''),
                end_date=end.replace('-', ''),
                is_open='1'
            )
            self.trade_dates = df['cal_date'].tolist()
            
            # 缓存数据到Redis
            self._save_to_redis(cache_key, self.trade_dates)
            
            logger.info(f"从Tushare获取交易日历成功，从{start}到{end}共{len(self.trade_dates)}个交易日")
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise
        
        return self.trade_dates
    
    def _save_to_redis(self, key: str, data) -> bool:
        """
        将数据保存到Redis
        
        Args:
            key: Redis键
            data: 要保存的数据
            
        Returns:
            是否保存成功
        """
        if not self.redis_available:
            return False
            
        try:
            # 使用pickle序列化DataFrame
            serialized_data = pickle.dumps(data)
            self.redis.setex(key, self.redis_ttl, serialized_data)
            logger.debug(f"数据已成功保存到Redis: {key}, TTL: {self.redis_ttl}秒")
            return True
        except Exception as e:
            logger.warning(f"保存数据到Redis失败: {e}")
            return False
    
    def _get_from_redis(self, key: str):
        """
        从Redis获取数据
        
        Args:
            key: Redis键
            
        Returns:
            获取的数据，如果不存在则返回None
        """
        if not self.redis_available:
            return None
            
        try:
            data = self.redis.get(key)
            if data:
                # 使用pickle反序列化数据
                deserialized_data = pickle.loads(data)
                logger.debug(f"Redis缓存命中: {key}, 成功获取数据")
                return deserialized_data
            return None
        except Exception as e:
            logger.warning(f"从Redis获取数据失败: {e}")
            return None
    
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
        # 设置日期范围
        start = start_date or self.config['data']['start_date']
        end = end_date or self.config['data']['end_date']
        
        # 生成缓存键
        cache_key = f"stock_data:{ts_code}:{start}:{end}"
        
        # 如果不强制刷新，先检查本地缓存
        if cache_key in self.stock_data and not refresh:
            logger.info(f"从本地缓存获取股票{ts_code}数据")
            return self.stock_data[cache_key]
        
        # 如果不强制刷新，再检查Redis缓存
        if not refresh:
            redis_data = self._get_from_redis(cache_key)
            if redis_data is not None:
                # 缓存到本地
                self.stock_data[cache_key] = redis_data
                logger.info(f"从Redis获取股票{ts_code}数据成功，从{start}到{end}共{len(redis_data)}条记录")
                return redis_data
        
        try:
            # 从Tushare获取日线数据
            logger.debug(f"Redis缓存未命中，从Tushare获取股票{ts_code}数据，日期范围: {start}至{end}")
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
            
            # 缓存数据到本地
            self.stock_data[cache_key] = df
            
            # 缓存数据到Redis
            self._save_to_redis(cache_key, df)
            
            logger.debug(f"从Tushare获取股票{ts_code}数据成功，从{start}到{end}共{len(df)}条记录，已保存到Redis")
            return df
            
        except Exception as e:
            logger.error(f"获取股票{ts_code}数据失败: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, ts_code: str, refresh: bool = False) -> pd.DataFrame:
        """
        获取股票基本面数据
        
        Args:
            ts_code: 股票代码（tushare格式）
            refresh: 是否强制刷新缓存
            
        Returns:
            基本面数据DataFrame
        """
        # 生成缓存键
        today = datetime.now().strftime('%Y-%m-%d')
        quarter = self._get_latest_quarter()
        cache_key = f"fundamental:{ts_code}:{today}:{quarter}"
        
        # 如果不强制刷新，检查Redis缓存
        if not refresh:
            redis_data = self._get_from_redis(cache_key)
            if redis_data is not None:
                logger.info(f"从Redis获取股票{ts_code}基本面数据成功")
                return redis_data
        
        try:
            # 从Tushare获取最新财务指标
            logger.debug(f"Redis缓存未命中，从Tushare获取股票{ts_code}基本面数据")
            df_basic = self.pro.daily_basic(
                ts_code=ts_code,
                trade_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv'
            )
            
            # 获取最新财务报表
            df_finance = self.pro.fina_indicator(
                ts_code=ts_code,
                period=quarter,
                fields='ts_code,ann_date,end_date,grossprofit_margin,netprofit_margin,roe,roa'
            )
            
            # 合并数据
            result = pd.DataFrame()
            if not df_basic.empty and not df_finance.empty:
                result = pd.merge(df_basic, df_finance, on='ts_code', how='left')
            elif not df_basic.empty:
                result = df_basic
            elif not df_finance.empty:
                result = df_finance
            
            # 如果获取到数据，缓存到Redis
            if not result.empty:
                self._save_to_redis(cache_key, result)
                logger.debug(f"从Tushare获取股票{ts_code}基本面数据成功，已保存到Redis")
            
            return result
                
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
        """
        计算筹码集中度
        基于Pine脚本实现的改进版本，使用动态窗口大小和尾部密集区计算
        """
        # 获取参数
        threshold = self.config['chip_concentration']['threshold']
        
        # 初始化筹码集中度列
        df['concentration_ratio'] = np.nan
        df['dominant_price'] = np.nan
        
        # 计算成交量加权平均价格（VWAP）作为主力成本
        for i in range(10, len(df)):  # 至少需要10天数据
            # 获取历史成交量数据
            volume_array = df.iloc[:i]['vol'].values
            
            # 前置条件校验：至少10天数据且所有成交量非负
            if len(volume_array) >= 10 and np.min(volume_array) >= 0:
                # 计算动态窗口大小
                window_size = max(3, int(np.sqrt(len(volume_array))))
                
                # 对成交量数组排序（升序）
                sorted_volumes = np.sort(volume_array)
                
                # 计算尾部密集区（最近的高成交量区域）
                start_idx = max(0, len(sorted_volumes) - window_size)
                
                # 计算总成交量和密集区成交量
                sum_total = np.sum(sorted_volumes)
                sum_dense = np.sum(sorted_volumes[start_idx:])
                
                # 计算筹码集中度
                if sum_total > 0 and sum_dense > 0:
                    concentration_ratio = min(100, (sum_dense / (sum_total + 1e-6)) * 100)
                    df.loc[df.index[i], 'concentration_ratio'] = concentration_ratio
                
                # 计算主力成本价（VWAP）
                window = df.iloc[:i]
                total_vol = window['vol'].sum()
                if total_vol > 0:
                    weighted_sum = (window['close'] * window['vol']).sum()
                    dominant_price = weighted_sum / total_vol
                    df.loc[df.index[i], 'dominant_price'] = dominant_price
        
        # 计算筹码集中度信号
        # 根据Pine脚本实现：
        # costValid = not na(dominantPrice) ?
        #   (math.abs(close[1] - dominantPrice)/dominantPrice*100 < (100-chipConcentration)) and
        #   (concentrationRatio >= chipConcentration) : false
        
        # 使用前一天收盘价计算偏离度
        df['prev_close'] = df['close'].shift(1)
        
        df['cost_valid'] = np.where(
            (~df['dominant_price'].isna()) &
            (~df['concentration_ratio'].isna()) &
            (df['concentration_ratio'] >= threshold) &  # 筹码集中度高于阈值
            (abs(df['prev_close'] - df['dominant_price']) / df['dominant_price'] * 100 < (100 - threshold)),  # 价格偏离主力成本小于(100-threshold)%
            1, 0
        )
        
        # 记录日志
        logger.info(f"筹码集中度计算完成，阈值: {threshold}%")
    
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
                            end_date: str = None, refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            refresh: 是否强制刷新缓存
            
        Returns:
            股票数据字典 {股票代码: 数据DataFrame}
        """
        # 设置日期范围
        start = start_date or self.config['data']['start_date']
        end = end_date or self.config['data']['end_date']
        
        # 生成批量缓存键
        batch_cache_key = f"batch_stock_data:{','.join(ts_codes)}:{start}:{end}"
        
        # 如果不强制刷新，检查Redis缓存
        if not refresh:
            redis_data = self._get_from_redis(batch_cache_key)
            if redis_data is not None:
                logger.info(f"从Redis获取批量股票数据成功，共{len(redis_data)}只")
                return redis_data
        
        # 逐个获取股票数据
        result = {}
        for ts_code in ts_codes:
            df = self.get_stock_data(ts_code, start_date, end_date, refresh)
            if not df.empty:
                result[ts_code] = df
        
        # 缓存批量结果到Redis
        if result:
            self._save_to_redis(batch_cache_key, result)
        
        logger.info(f"批量获取{len(ts_codes)}只股票数据，成功获取{len(result)}只，已保存到Redis")
        return result
