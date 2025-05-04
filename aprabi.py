import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tushare as ts
from typing import Dict, List, Tuple
import warnings
import logging
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('APRABI')

class AprabiCalculator:
    """
    计算A股市场交易量集中度的类
    计算每月交易量前5%的股票的交易量占比
    """
    
    def __init__(self, tushare_token: str = None):
        """
        初始化计算器
        
        Args:
            tushare_token: Tushare API token
        """
        # 初始化tushare
        self._init_tushare(tushare_token)
        
        # 数据存储
        self.monthly_data = {}  # 存储每个月的所有股票数据
        self.aprabi_results = []  # 存储每个月的aprabi结果
        
    def _init_tushare(self, token: str = None):
        """初始化tushare API"""
        try:
            # 优先使用参数中的token，如果不存在则使用环境变量中的token
            token = token or os.environ.get('TUSHARE_TOKEN')
            if not token:
                raise ValueError("Tushare token not provided. Please provide a token or set TUSHARE_TOKEN environment variable.")
            
            ts.set_token(token)
            self.pro = ts.pro_api()
            logger.info("Tushare API初始化成功")
        except Exception as e:
            logger.error(f"Tushare API初始化失败: {e}")
            raise
    
    def get_stock_list(self) -> List[str]:
        """
        获取A股股票列表
        
        Returns:
            股票代码列表
        """
        try:
            # 从Tushare获取所有A股列表
            logger.info("从Tushare获取股票列表")
            data = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,list_date'
            )
            
            # 排除ST股票
            filtered_data = data[~data['name'].str.contains('ST|退')]
            
            stock_list = filtered_data['ts_code'].tolist()
            logger.info(f"获取股票列表成功，共{len(stock_list)}只股票")
            return stock_list
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise
    
    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            交易日期列表
        """
        try:
            # 从Tushare获取交易日历
            logger.info(f"从Tushare获取交易日历，日期范围: {start_date}至{end_date}")
            df = self.pro.trade_cal(
                exchange='SSE',
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                is_open='1'
            )
            # 确保交易日期按升序排序（从旧到新）
            df = df.sort_values('cal_date', ascending=True)
            trade_dates = df['cal_date'].tolist()
            
            logger.info(f"从Tushare获取交易日历成功，从{start_date}到{end_date}共{len(trade_dates)}个交易日")
            return trade_dates
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise
    
    def get_monthly_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用tushare月线接口获取月度数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            月线数据DataFrame
        """
        try:
            # 确保日期格式正确
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            
            # 使用tushare的月线接口获取数据
            monthly_df = self.pro.monthly(
                ts_code=ts_code,
                start_date=start,
                end_date=end,
                fields='ts_code,trade_date,open,high,low,close,vol,amount'
            )
            
            # 按日期排序
            if not monthly_df.empty:
                monthly_df.sort_values('trade_date', inplace=True)
                logger.debug(f"获取到{ts_code}的月线数据，共{len(monthly_df)}条记录")
            else:
                logger.warning(f"未获取到{ts_code}的月线数据")
                
            return monthly_df
            
        except Exception as e:
            logger.error(f"获取月线数据失败: {e}")
            return pd.DataFrame()
    
    def collect_all_monthly_data(self, start_date: str, end_date: str):
        """
        收集所有股票的月度数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        logger.info(f"开始收集所有股票的月度数据，日期范围: {start_date}至{end_date}")
        
        # 获取股票列表
        stock_list = self.get_stock_list()
        
        # 创建进度条
        pbar = tqdm(total=len(stock_list), desc="收集股票数据")
        
        # 收集每只股票的月度数据
        all_monthly_data = []
        for ts_code in stock_list:
            df = self.get_monthly_data(ts_code, start_date, end_date)
            if not df.empty:
                all_monthly_data.append(df)
            pbar.update(1)
        
        pbar.close()
        
        # 合并所有数据
        if all_monthly_data:
            combined_df = pd.concat(all_monthly_data, ignore_index=True)
            logger.info(f"成功收集{len(all_monthly_data)}只股票的月度数据，共{len(combined_df)}条记录")
            
            # 按月份分组
            combined_df['year_month'] = combined_df['trade_date'].astype(str).str[:6]
            monthly_groups = combined_df.groupby('year_month')
            
            # 存储每个月的数据
            for year_month, group in monthly_groups:
                self.monthly_data[year_month] = group
            
            logger.info(f"数据已按月份分组，共{len(self.monthly_data)}个月")
        else:
            logger.warning("未收集到任何股票的月度数据")
    
    def calculate_aprabi(self, top_percent: float = 0.05):
        """
        计算每个月交易量前5%的股票的交易量占比
        
        Args:
            top_percent: 前多少比例的股票，默认为0.05（5%）
        """
        logger.info(f"开始计算每个月交易量前{top_percent*100}%的股票的交易量占比")
        
        results = []
        
        # 按月份计算
        for year_month, monthly_df in sorted(self.monthly_data.items()):
            # 计算该月所有股票的总交易量
            total_volume = monthly_df['vol'].sum()
            
            # 按交易量排序
            sorted_df = monthly_df.sort_values('vol', ascending=False)
            
            # 计算前5%的股票数量
            top_n = max(1, int(len(sorted_df) * top_percent))
            
            # 获取前5%的股票
            top_stocks = sorted_df.head(top_n)
            
            # 计算前5%股票的总交易量
            top_volume = top_stocks['vol'].sum()
            
            # 计算占比
            aprabi = top_volume / total_volume if total_volume > 0 else 0
            
            # 格式化日期为YYYY-MM
            formatted_date = f"{year_month[:4]}-{year_month[4:6]}"
            
            # 存储结果
            results.append({
                'year_month': year_month,
                'formatted_date': formatted_date,
                'total_volume': total_volume,
                'top_volume': top_volume,
                'aprabi': aprabi,
                'stock_count': len(monthly_df),
                'top_stock_count': top_n
            })
            
            logger.debug(f"{formatted_date}: 总交易量={total_volume:.2f}, 前{top_n}只股票交易量={top_volume:.2f}, 占比={aprabi:.4f}")
        
        # 转换为DataFrame
        self.aprabi_results = pd.DataFrame(results)
        logger.info(f"计算完成，共{len(self.aprabi_results)}个月的数据")
    
    def plot_aprabi(self, save_path: str = None):
        """
        绘制aprabi曲线
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
        """
        if self.aprabi_results.empty:
            logger.warning("没有可绘制的数据")
            return
        
        logger.info("开始绘制aprabi曲线")
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 转换日期为datetime对象
        dates = [datetime.strptime(d, '%Y-%m') for d in self.aprabi_results['formatted_date']]
        
        # 绘制曲线
        plt.plot(dates, self.aprabi_results['aprabi'], marker='o', linestyle='-', linewidth=1, markersize=3)
        
        # 设置标题和标签
        plt.title('Monthly Trading Volume Concentration (Top 5% Stocks)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Concentration Ratio (APRABI)', fontsize=12)
        
        # 设置y轴范围
        plt.ylim(0, min(1.0, self.aprabi_results['aprabi'].max() * 1.1))
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        # 添加平均线
        avg_aprabi = self.aprabi_results['aprabi'].mean()
        plt.axhline(y=avg_aprabi, color='r', linestyle='--', alpha=0.7)
        plt.text(dates[0], avg_aprabi, f'Avg: {avg_aprabi:.4f}', 
                 verticalalignment='bottom', horizontalalignment='left', color='r')
        
        # 紧凑布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存至: {save_path}")
        else:
            plt.show()
            logger.info("图表已显示")
    
    def save_results_to_csv(self, file_path: str):
        """
        将结果保存为CSV文件
        
        Args:
            file_path: 文件保存路径
        """
        if self.aprabi_results.empty:
            logger.warning("没有可保存的数据")
            return
        
        self.aprabi_results.to_csv(file_path, index=False)
        logger.info(f"结果已保存至: {file_path}")


def main():
    """主函数"""
    # 获取Tushare token
    tushare_token = os.environ.get('TUSHARE_TOKEN')
    if not tushare_token:
        tushare_token = input("请输入Tushare API token: ")
    
    # 设置日期范围
    start_date = "2010-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 初始化计算器
    calculator = AprabiCalculator(tushare_token)
    
    # 收集数据
    calculator.collect_all_monthly_data(start_date, end_date)
    
    # 计算aprabi
    calculator.calculate_aprabi(top_percent=0.05)
    
    # 保存结果
    calculator.save_results_to_csv("aprabi_results.csv")
    
    # 绘制曲线
    calculator.plot_aprabi("aprabi_curve.png")
    
    logger.info("计算完成")


if __name__ == "__main__":
    main()