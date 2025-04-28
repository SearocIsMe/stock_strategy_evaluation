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
import traceback
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
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml', mode: str = 'backtest'):
        """
        初始化数据获取器
        
        Args:
            config_path: 配置文件路径
            mode: 运行模式，'backtest'或'live'
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置运行模式
        self.mode = mode
        logger.info(f"数据获取器初始化，运行模式: {mode}")
        
        # 初始化tushare
        self._init_tushare()
        
        # 初始化Redis连接
        self._init_redis()
        
        # 缓存数据
        self.stock_data = {}       # 本地股票数据缓存
        self.stock_list = []       # 股票列表缓存
        self.all_stock_list = []   # 未过滤的股票列表缓存
        self.trade_dates = []      # 交易日期缓存
        self.fundamental_data = {} # 基本面数据缓存
        
        # 回测模式下的日期范围
        if self.mode == 'backtest':
            self.start_date = self.config['data']['start_date']
            self.end_date = self.config['data']['end_date']
            logger.info(f"回测模式，日期范围: {self.start_date} 至 {self.end_date}")
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查是否使用新的策略配置格式
            if 'strategies' in config:
                # 获取默认策略或第一个可用策略
                strategy_id = config.get('active_strategy', 'default')
                if strategy_id not in config['strategies']:
                    available_strategies = list(config['strategies'].keys())
                    if available_strategies:
                        strategy_id = available_strategies[0]
                        logger.warning(f"策略 '{strategy_id}' 不存在，使用 '{available_strategies[0]}'")
                    else:
                        logger.warning("配置文件中没有定义任何策略，使用原始配置")
                        return config
                
                # 合并策略配置到主配置
                strategy_config = config['strategies'][strategy_id]
                for key, value in strategy_config.items():
                    config[key] = value
                
                logger.info(f"使用策略配置: {strategy_id}")
            
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
            
            # 获取当前日期
            today = datetime.now().strftime('%Y%m%d')
            
            # 应用股票筛选条件
            filtered_data = data.copy()
            
            # 1. 排除ST股票
            if self.config['data'].get('stock_filters', {}).get('exclude_st', True):
                filtered_data = filtered_data[~filtered_data['name'].str.contains('ST|退')]
                logger.info(f"已排除ST股票，剩余{len(filtered_data)}只")
            
            # 2. 排除新上市股票
            min_listing_days = self.config['data'].get('stock_filters', {}).get('min_listing_days', 60)
            if min_listing_days > 0:
                # 计算上市天数
                filtered_data['listing_days'] = filtered_data['list_date'].apply(
                    lambda x: (pd.to_datetime(today) - pd.to_datetime(x)).days
                    if pd.notna(x) else 0
                )
                filtered_data = filtered_data[filtered_data['listing_days'] >= min_listing_days]
                logger.info(f"已排除上市不足{min_listing_days}天的股票，剩余{len(filtered_data)}只")
            
            # 保存原始和过滤后的股票列表
            self.all_stock_list = data['ts_code'].tolist()
            self.stock_list = filtered_data['ts_code'].tolist()
            
            # 缓存数据到Redis
            self._save_to_redis(cache_key, self.stock_list)
            
            logger.info(f"从Tushare获取股票列表成功，原始{len(self.all_stock_list)}只，过滤后{len(self.stock_list)}只")
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
            # 确保交易日期按升序排序（从旧到新）
            df = df.sort_values('cal_date', ascending=True)
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
        original_start = start_date or self.config['data']['start_date']
        end = end_date or self.config['data']['end_date']
        
        # 获取回溯期天数
        lookback_period = self.config['data'].get('lookback_period', 10)
        
        # 调整开始日期，向前推移回溯期天数
        start = self._adjust_date_by_days(original_start, -lookback_period)
        logger.debug(f"调整开始日期: 从 {original_start} 到 {start} (回溯{lookback_period}天)")
        
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
    
    def get_fundamental_data(self, ts_code: str, refresh: bool = False, date: str = None) -> pd.DataFrame:
        """
        获取股票基本面数据
        
        Args:
            ts_code: 股票代码（tushare格式）
            refresh: 是否强制刷新缓存
            date: 指定日期，回测模式下使用
            
        Returns:
            基本面数据DataFrame
        """
        # 根据模式确定日期
        if self.mode == 'backtest':
            # 回测模式：使用指定日期或回测结束日期
            ref_date = date or self.end_date
            
            # 更健壮的日期解析：处理不同格式的日期
            if '-' in ref_date:
                # 格式为 YYYY-MM-DD
                ref_date_obj = datetime.strptime(ref_date, '%Y-%m-%d')
            else:
                # 格式为 YYYYMMDD
                ref_date_obj = datetime.strptime(ref_date, '%Y%m%d')
            quarter = self._get_quarter_for_date(ref_date_obj)
            logger.debug(f"回测模式，使用日期: {ref_date}, 季度: {quarter}")
        else:
            # 实盘模式：使用当前日期
            ref_date = datetime.now().strftime('%Y-%m-%d')
            quarter = self._get_latest_quarter()
            logger.debug(f"实盘模式，使用当前日期: {ref_date}, 季度: {quarter}")
        
        # 生成缓存键
        cache_key = f"fundamental:{ts_code}:{ref_date}:{quarter}"
        
        # 如果不强制刷新，检查Redis缓存
        if not refresh:
            redis_data = self._get_from_redis(cache_key)
            if redis_data is not None:
                logger.debug(f"从Redis获取股票{ts_code}基本面数据成功")
                return redis_data
        
        try:
            # 根据模式获取交易日期
            if self.mode == 'backtest':
                # 回测模式：使用指定日期或回测结束日期的最近交易日
                if date:
                    trade_date = self._get_nearest_trade_date(date)
                else:
                    trade_date = self._get_nearest_trade_date(self.end_date)
                logger.debug(f"回测模式，使用交易日期: {trade_date}")
            else:
                # 实盘模式：使用最近的交易日期
                trade_date = self._get_recent_trade_date()
                logger.debug(f"实盘模式，使用最近交易日期: {trade_date}")
            
            logger.debug(f"Redis缓存未命中，从Tushare获取股票{ts_code}基本面数据，使用交易日期: {trade_date}")
            
            # 从Tushare获取财务指标
            df_basic = None
            try:
                if self.mode == 'backtest':
                    # 回测模式：使用回测日期的财务指标
                    logger.debug(f"回测模式，获取{trade_date}的财务指标")
                    df_basic = self.pro.daily_basic(
                        ts_code=ts_code,
                        trade_date=trade_date,
                        fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv'
                    )
                    logger.debug(f"获取daily_basic数据(日期={trade_date}): {len(df_basic) if df_basic is not None else 0}行")
                else:
                    # 实盘模式：使用最近交易日的财务指标
                    df_basic = self.pro.daily_basic(
                        ts_code=ts_code,
                        trade_date=trade_date,
                        fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv'
                    )
                    logger.debug(f"获取daily_basic数据(日期={trade_date}): {len(df_basic) if df_basic is not None else 0}行")
            except Exception as e:
                logger.error(f"获取daily_basic数据失败: {e}")
                df_basic = pd.DataFrame()
            
            # 获取财务报表
            df_finance = None
            try:
                if self.mode == 'backtest':
                    # 回测模式：只获取在回测日期前已经公布的财务数据
                    # 使用ann_date_lte参数确保只获取在指定日期前已公布的财务数据
                    logger.debug(f"回测模式，获取{trade_date}前已公布的财务数据")
                    
                    # 尝试获取最新季度的财务数据
                    df_finance = self.pro.fina_indicator(
                        ts_code=ts_code,
                        period=quarter,
                        ann_date_lte=trade_date,  # 只获取在回测日期前已公布的数据
                        fields='ts_code,ann_date,end_date,grossprofit_margin,netprofit_margin,roe,roa'
                    )
                    logger.debug(f"获取fina_indicator数据(季度={quarter}, 公布日期<={trade_date}): {len(df_finance) if df_finance is not None else 0}行")
                    
                    # 如果最新季度没有数据，尝试获取上一季度的数据
                    if df_finance is None or df_finance.empty:
                        prev_quarter = self._get_previous_quarter(quarter)
                        logger.debug(f"最新季度{quarter}无数据，尝试获取上一季度{prev_quarter}数据")
                        df_finance = self.pro.fina_indicator(
                            ts_code=ts_code,
                            period=prev_quarter,
                            ann_date_lte=trade_date,  # 只获取在回测日期前已公布的数据
                            fields='ts_code,ann_date,end_date,grossprofit_margin,netprofit_margin,roe,roa'
                        )
                        logger.debug(f"获取fina_indicator数据(季度={prev_quarter}, 公布日期<={trade_date}): {len(df_finance) if df_finance is not None else 0}行")
                else:
                    # 实盘模式：获取最新的财务数据
                    # 尝试获取最新季度的财务数据
                    df_finance = self.pro.fina_indicator(
                        ts_code=ts_code,
                        period=quarter,
                        fields='ts_code,ann_date,end_date,grossprofit_margin,netprofit_margin,roe,roa'
                    )
                    logger.debug(f"获取fina_indicator数据(季度={quarter}): {len(df_finance) if df_finance is not None else 0}行")
                    
                    # 如果最新季度没有数据，尝试获取上一季度的数据
                    if df_finance is None or df_finance.empty:
                        prev_quarter = self._get_previous_quarter(quarter)
                        logger.debug(f"最新季度{quarter}无数据，尝试获取上一季度{prev_quarter}数据")
                        df_finance = self.pro.fina_indicator(
                            ts_code=ts_code,
                            period=prev_quarter,
                            fields='ts_code,ann_date,end_date,grossprofit_margin,netprofit_margin,roe,roa'
                        )
                        logger.debug(f"获取fina_indicator数据(季度={prev_quarter}): {len(df_finance) if df_finance is not None else 0}行")
            except Exception as e:
                logger.error(f"获取fina_indicator数据失败: {e}")
                df_finance = pd.DataFrame()
            
            # 合并数据
            result = pd.DataFrame()
            if df_basic is not None and df_finance is not None and not df_basic.empty and not df_finance.empty:
                # 两个数据源都有数据，合并它们
                result = pd.merge(df_basic, df_finance, on='ts_code', how='left')
                logger.debug(f"合并daily_basic和fina_indicator数据，结果: {len(result)}行")
            elif df_basic is not None and not df_basic.empty:
                # 只有daily_basic有数据
                result = df_basic
                logger.debug(f"只使用daily_basic数据，结果: {len(result)}行")
            elif df_finance is not None and not df_finance.empty:
                # 只有fina_indicator有数据
                result = df_finance
                logger.debug(f"只使用fina_indicator数据，结果: {len(result)}行")
            else:
                logger.warning(f"股票{ts_code}没有获取到任何基本面数据")
            
            # 如果获取到数据，缓存到Redis
            if not result.empty:
                self._save_to_redis(cache_key, result)
                logger.debug(f"从Tushare获取股票{ts_code}基本面数据成功，已保存到Redis")
            else:
                logger.warning(f"股票{ts_code}基本面数据为空，不缓存")
            
            return result
                
        except Exception as e:
            logger.error(f"获取股票{ts_code}基本面数据失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def filter_stocks_by_fundamental(self, stock_list: List[str] = None, date: str = None) -> List[str]:
        """
        根据基本面数据筛选股票
        
        Args:
            stock_list: 待筛选的股票列表，默认使用self.stock_list
            date: 指定日期，回测模式下使用
            
        Returns:
            筛选后的股票列表
        """
        if stock_list is None:
            stock_list = self.stock_list
        
        if not stock_list:
            logger.warning("股票列表为空，无法进行基本面筛选")
            return []
        
        # 获取基本面筛选参数
        fundamental_params = self.config.get('fundamental', {})
        min_gross_margin = fundamental_params.get('min_gross_margin', 0)
        min_roe = fundamental_params.get('min_roe', 0)
        min_roa = fundamental_params.get('min_roa', 0)
        max_pe = fundamental_params.get('max_pe', float('inf'))
        max_pb = fundamental_params.get('max_pb', float('inf'))
        
        # 记录筛选条件
        filter_conditions = []
        if min_gross_margin > 0:
            filter_conditions.append(f"毛利率>={min_gross_margin}%")
        if min_roe > 0:
            filter_conditions.append(f"ROE>={min_roe}%")
        if min_roa > 0:
            filter_conditions.append(f"ROA>={min_roa}%")
        if max_pe < float('inf'):
            filter_conditions.append(f"PE<={max_pe}")
        if max_pb < float('inf'):
            filter_conditions.append(f"PB<={max_pb}")
            
        if not filter_conditions:
            logger.warning("未设置任何基本面筛选条件，将返回原始股票列表")
            return stock_list
            
        logger.info(f"开始基本面筛选，条件: {', '.join(filter_conditions)}")
        
        # 批量获取基本面数据
        filtered_stocks = []
        batch_size = 50
        total_stocks = len(stock_list)
        processed = 0
        
        # 统计筛选结果
        stats = {
            'total': total_stocks,
            'no_data': 0,
            'failed_gross_margin': 0,
            'failed_roe': 0,
            'failed_roa': 0,
            'failed_pe': 0,
            'failed_pb': 0,
            'passed': 0
        }
        
        for i in range(0, total_stocks, batch_size):
            batch = stock_list[i:i+batch_size]
            processed += len(batch)
            logger.info(f"正在获取基本面数据 {processed}/{total_stocks}")

            # 使用指定日期或回测结束日期
            ref_date = date or self.end_date
            # 检查日期是否为交易日，如果不是则跳过
            if not self._is_trade_date(ref_date):
                logger.debug(f"日期 {ref_date} 不是交易日，跳过")
                stats['no_data'] += 1
                continue

            for ts_code in batch:
                # 获取基本面数据
                if ts_code in self.fundamental_data:
                    df = self.fundamental_data[ts_code]
                else:
                    # 在回测模式下，传递日期参数
                    if self.mode == 'backtest':
                        df = self.get_fundamental_data(ts_code, date=ref_date)
                    else:
                        df = self.get_fundamental_data(ts_code)
                    if not df.empty:
                        self.fundamental_data[ts_code] = df
                
                if df.empty:
                    stats['no_data'] += 1
                    continue
                
                # 检查基本面指标
                meets_criteria = True
                failure_reason = ""
                
                # 检查毛利率
                if 'grossprofit_margin' in df.columns and min_gross_margin > 0:
                    gross_margin = df['grossprofit_margin'].iloc[0]
                    if pd.isna(gross_margin) or gross_margin is None or gross_margin < min_gross_margin:
                        meets_criteria = False
                        failure_reason = "毛利率"
                        stats['failed_gross_margin'] += 1
                        logger.debug(f"股票{ts_code}毛利率不符合条件: {gross_margin if not pd.isna(gross_margin) else 'N/A'} < {min_gross_margin}")
                
                # 检查ROE
                if meets_criteria and 'roe' in df.columns and min_roe > 0:
                    roe = df['roe'].iloc[0]
                    if pd.isna(roe) or roe is None or roe < min_roe:
                        meets_criteria = False
                        failure_reason = "ROE"
                        stats['failed_roe'] += 1
                        logger.debug(f"股票{ts_code}ROE不符合条件: {roe if not pd.isna(roe) else 'N/A'} < {min_roe}")
                
                # 检查ROA
                if meets_criteria and 'roa' in df.columns and min_roa > 0:
                    roa = df['roa'].iloc[0]
                    if pd.isna(roa) or roa is None or roa < min_roa:
                        meets_criteria = False
                        failure_reason = "ROA"
                        stats['failed_roa'] += 1
                        logger.debug(f"股票{ts_code}ROA不符合条件: {roa if not pd.isna(roa) else 'N/A'} < {min_roa}")
                
                # 检查PE
                if meets_criteria and 'pe' in df.columns and max_pe < float('inf'):
                    pe = df['pe'].iloc[0]
                    if pd.isna(pe) or pe is None or pe > max_pe or pe <= 0:
                        meets_criteria = False
                        failure_reason = "PE"
                        stats['failed_pe'] += 1
                        logger.debug(f"股票{ts_code}PE不符合条件: {pe if not pd.isna(pe) else 'N/A'} > {max_pe} 或 <= 0")
                
                # 检查PB
                if meets_criteria and 'pb' in df.columns and max_pb < float('inf'):
                    pb = df['pb'].iloc[0]
                    if pd.isna(pb) or pb is None or pb > max_pb or pb <= 0:
                        meets_criteria = False
                        failure_reason = "PB"
                        stats['failed_pb'] += 1
                        logger.debug(f"股票{ts_code}PB不符合条件: {pb if not pd.isna(pb) else 'N/A'} > {max_pb} 或 <= 0")
                
                if meets_criteria:
                    filtered_stocks.append(ts_code)
                    stats['passed'] += 1
                    logger.debug(f"股票{ts_code}通过基本面筛选")
                else:
                    logger.debug(f"股票{ts_code}未通过基本面筛选，失败原因: {failure_reason}")
        
        # 输出详细的筛选统计
        logger.info(f"基本面筛选完成，从{stats['total']}只股票中筛选出{stats['passed']}只符合条件的股票")
        logger.info(f"筛选统计: 无数据={stats['no_data']}，毛利率不符={stats['failed_gross_margin']}，"
                  f"ROE不符={stats['failed_roe']}，ROA不符={stats['failed_roa']}，"
                  f"PE不符={stats['failed_pe']}，PB不符={stats['failed_pb']}")
        
        # 如果筛选后股票数量太少，给出警告
        if len(filtered_stocks) < 10:
            logger.warning(f"筛选后股票数量过少({len(filtered_stocks)}只)，可能会影响回测结果，建议放宽筛选条件")
            
        return filtered_stocks
    
    def _get_latest_quarter(self) -> str:
        """获取最近的财报季度"""
        now = datetime.now()
        return self._get_quarter_for_date(now)
    
    def _get_quarter_for_date(self, date_obj: datetime) -> str:
        """
        根据日期获取对应的财报季度
        
        Args:
            date_obj: 日期对象
            
        Returns:
            财报季度字符串 (YYYYMMDD格式)
        """
        year = date_obj.year
        month = date_obj.month
        
        if month < 4:
            return f"{year-1}1231"
        elif month < 7:
            return f"{year}0331"
        elif month < 10:
            return f"{year}0630"
        else:
            return f"{year}0930"
    
    def _get_previous_quarter(self, quarter: str) -> str:
        """获取上一个财报季度"""
        year = int(quarter[:4])
        month = int(quarter[4:6])
        
        if month == 3:  # 如果是Q1 (0331)
            return f"{year-1}1231"
        elif month == 6:  # 如果是Q2 (0630)
            return f"{year}0331"
        elif month == 9:  # 如果是Q3 (0930)
            return f"{year}0630"
        else:  # 如果是Q4 (1231)
            return f"{year}0930"
    
    def _get_recent_trade_date(self) -> str:
        """获取最近的交易日期"""
        try:
            # 获取当前日期
            now = datetime.now()
            
            # 尝试获取最近30天的交易数据
            end_date = now.strftime('%Y%m%d')
            start_date = (now - timedelta(days=30)).strftime('%Y%m%d')
            
            # 使用上证指数（000001.SH）作为参考获取交易日
            # 使用daily API而不是trade_cal
            df_daily = self.pro.daily(
                ts_code='000001.SH',
                start_date=start_date,
                end_date=end_date
            )
            
            if not df_daily.empty:
                # 按日期降序排序，获取最近的交易日
                df_daily = df_daily.sort_values('trade_date', ascending=False)
                return df_daily['trade_date'].iloc[0]
            else:
                # 如果获取失败，返回昨天的日期
                yesterday = (now - timedelta(days=1)).strftime('%Y%m%d')
                logger.warning(f"无法获取最近交易日，使用昨天日期: {yesterday}")
                return yesterday
                
        except Exception as e:
            # 如果出错，返回昨天的日期
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            logger.error(f"获取最近交易日出错: {e}，使用昨天日期: {yesterday}")
            return yesterday
    
    def _adjust_date_by_days(self, date_str: str, days: int) -> str:
        """
        调整日期，向前或向后推移指定天数
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD 或 YYYYMMDD格式)
            days: 调整天数，正数为向后，负数为向前
            
        Returns:
            调整后的日期 (与输入格式相同)
        """
        # 标准化日期格式
        has_hyphen = '-' in date_str
        if has_hyphen:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
        
        # 调整日期
        adjusted_date = date_obj + timedelta(days=days)
        
        # 返回与输入格式相同的日期字符串
        if has_hyphen:
            return adjusted_date.strftime('%Y-%m-%d')
        else:
            return adjusted_date.strftime('%Y%m%d')
    
    def _get_nearest_trade_date(self, date_str: str) -> str:
        """
        获取指定日期最近的交易日（回测模式）
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD 或 YYYYMMDD格式)
            
        Returns:
            最近的交易日 (YYYYMMDD格式)
        """
        try:
            # 标准化日期格式
            if '-' in date_str:
                date_str = date_str.replace('-', '')
            
            # 向前后各查找15天，以找到最近的交易日
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            start_date = (date_obj - timedelta(days=15)).strftime('%Y%m%d')
            end_date = (date_obj + timedelta(days=15)).strftime('%Y%m%d')
            
            # 使用上证指数（000001.SH）作为参考获取交易日
            # 使用daily API而不是trade_cal
            df_daily = self.pro.daily(
                ts_code='000001.SH',
                start_date=start_date,
                end_date=end_date
            )
            
            if not df_daily.empty:
                # 将日期转换为数值进行比较
                target_date = int(date_str)
                df_daily['date_diff'] = abs(df_daily['trade_date'].astype(int) - target_date)
                
                # 按日期差排序，获取最近的交易日
                df_daily = df_daily.sort_values('date_diff')
                nearest_date = df_daily['trade_date'].iloc[0]
                
                logger.debug(f"找到日期 {date_str} 最近的交易日: {nearest_date}")
                return nearest_date
            else:
                logger.warning(f"无法获取日期 {date_str} 附近的交易数据，使用原始日期")
                return date_str
                
        except Exception as e:
            logger.error(f"获取最近交易日出错: {e}，使用原始日期: {date_str}")
            return date_str
    
    def _is_trade_date(self, date_str: str) -> bool:
        """
        检查指定日期是否为交易日
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD 或 YYYYMMDD格式)
            
        Returns:
            是否为交易日
        """
        # 标准化日期格式
        if '-' in date_str:
            date_str = date_str.replace('-', '')
        
        # 获取交易日历（如果尚未获取）
        if not self.trade_dates:
            self.get_trade_dates()
        
        # 检查日期是否在交易日列表中
        return date_str in self.trade_dates
    
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
        
        # 获取回溯期天数
        lookback_period = self.config['data'].get('lookback_period', 10)
        
        # 计算成交量加权平均价格（VWAP）作为主力成本
        for i in range(lookback_period, len(df)):  # 至少需要lookback_period天数据
            # 获取历史成交量数据
            volume_array = df.iloc[:i]['vol'].values
            
            # 前置条件校验：至少lookback_period天数据且所有成交量非负
            if len(volume_array) >= lookback_period and np.min(volume_array) >= 0:
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
        
        # 计算筹码集中度比率 (concentration_ratio / threshold)
        df['cost_valid_ratio'] = np.where(
            (~df['dominant_price'].isna()) &
            (~df['concentration_ratio'].isna()),
            df['concentration_ratio'] / threshold,  # 筹码集中度与阈值的比率
            0
        )
        
        # 计算价格偏离度比率
        df['price_deviation_ratio'] = np.where(
            (~df['dominant_price'].isna()) &
            (~df['prev_close'].isna()),
            1 - (abs(df['prev_close'] - df['dominant_price']) / df['dominant_price'] * 100) / (100 - threshold),  # 价格偏离度与阈值的比率
            0
        )
        
        # 保留原始的二元信号，但同时提供比率值
        df['cost_valid'] = np.where(
            (~df['dominant_price'].isna()) &
            (~df['concentration_ratio'].isna()) &
            (df['concentration_ratio'] >= threshold) &  # 筹码集中度高于阈值
            (abs(df['prev_close'] - df['dominant_price']) / df['dominant_price'] * 100 < (100 - threshold)),  # 价格偏离主力成本小于(100-threshold)%
            1, 0
        )
        
        # 记录日志
        logger.debug(f"筹码集中度计算完成，阈值: {threshold}%")
    
    def _calculate_rsi(self, df: pd.DataFrame) -> None:
        """计算RSI指标"""
        # 获取参数
        period = self.config['rsi']['period']
        overbought = self.config['rsi']['overbought']
        
        # 确保数据足够计算RSI
        if len(df) < period + 1:
            logger.warning(f"数据长度不足以计算RSI，需要至少{period+1}条数据，但只有{len(df)}条")
            df['rsi'] = np.nan
            df['overbought'] = 0
            return
        
        # 计算价格变化
        delta = df['close'].diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 使用指数移动平均计算平均上涨和下跌（更稳定）
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        # 计算相对强度，避免除以零
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)  # 当avg_loss为0时，将rs设为100（相当于RSI=99）
        
        # 计算RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 处理极端情况
        df['rsi'] = np.where(avg_gain == 0, 0, df['rsi'])  # 如果没有上涨，RSI=0
        df['rsi'] = np.where(avg_loss == 0, 100, df['rsi'])  # 如果没有下跌，RSI=100
        
        # 计算超买信号
        df['overbought'] = np.where(df['rsi'] > overbought, 1, 0)
        
        # 填充前period个值为NaN
        df.loc[:period, 'rsi'] = np.nan
        df.loc[:period, 'overbought'] = 0

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
        original_start = start_date or self.config['data']['start_date']
        end = end_date or self.config['data']['end_date']
        
        # 获取回溯期天数
        lookback_period = self.config['data'].get('lookback_period', 10)
        
        # 调整开始日期，向前推移回溯期天数
        start = self._adjust_date_by_days(original_start, -lookback_period)
        logger.debug(f"批量获取数据：调整开始日期: 从 {original_start} 到 {start} (回溯{lookback_period}天)")
        
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
