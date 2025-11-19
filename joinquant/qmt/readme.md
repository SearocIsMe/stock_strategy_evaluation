import myqmt_sql

# ⭐ 在这里设置这个策略的分类标签（写入 trade.fenlei）
myqmt_sql.FENLEI = '日内短线策略A'      # 或 '中长线趋势策略B' 等

from myqmt_sql import (
    order_zzy as order,
    order_target_zzy as order_target,
    order_value_zzy as order_value,
    order_target_value_zzy as order_target_value
)