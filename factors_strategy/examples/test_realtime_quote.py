import tushare as ts

#设置你的token，登录tushare在个人用户中心里拷贝
ts.set_token('a56e2085d63e65ca75096d2b1e014fb7e767abb40fa4a6f9a1416e39')

#sina数据
df = ts.realtime_quote(ts_code='600000.SH,000001.SZ,000001.SH')


#东财数据
df = ts.realtime_quote(ts_code='600000.SH', src='dc')

print(df)