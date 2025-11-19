#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码
# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入 

#####################只需要修改自己的用户名和密码，别的无需更改#######################################
from kuanke.user_space_api import *
import pymssql
from typing import Optional, Union

# ================== 这里是全局配置 ==================
# 默认分类，可以在策略文件中修改这个变量
FENLEI = '未分类'
# ==================================================

class MyTrade():
    
    def __init__(self): 
        
        self.conn = pymssql.connect('8.138.38.43', 'sa', 'Just4Jhp@QmtJoinQuant3333', 'touzi')  # 建立连接

        
    def update(self, code, quantity, types):
        conn = self.conn
        cursor = conn.cursor()

        # 代码转换
        if code.endswith('XSHE'):
            code1 = code[:-4] + "SZ"
        else:
            code1 = code[:-4] + "SH"

        name1 = get_security_info(code).display_name  # 平安银行 之类
        now = datetime.datetime.now()                 # 直接用 datetime 对象即可

        sql = """
        INSERT INTO trade (name, code, type, num, date, fenlei)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        # ⭐ 这里改成用全局变量 FENLEI，而不是写死 '未分类'
        params = (
            name1,        # name  -> NVARCHAR
            code1,        # code
            types,        # type
            quantity,     # num
            now,          # date
            FENLEI,       # fenlei，从全局变量读取
        )

        cursor.execute(sql, params)
        conn.commit()
        cursor.close()
        conn.close()

    

def order_zzy(security: str, quantity: int,style = None,pindex=0):  #按股数下单.
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order')
    _order = order(security, quantity, style=style,pindex=pindex)  
    return _order

def order_target_zzy(security: str, quantity: int,style = None,pindex=0):
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order_target')
    _order = order_target(security, quantity,style=style,pindex=pindex)
    return _order

def order_value_zzy(security: str, quantity: int,style = None,pindex=0):  #按价值下单
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order_value')
    _order = order_value(security, quantity, style=style,pindex=pindex)
    return _order

def order_target_value_zzy(security: str, quantity: int,style = None,pindex=0):  #目标价值下单
    mytrade = MyTrade()
    mytrade.update(code=security,quantity=quantity,types='order_target_value')
    _order = order_target_value(security, quantity, style=style,pindex=pindex)
    return _order
