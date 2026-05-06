from jqdatasdk import *


auth('13146215928','only4Jhp!Q@W') #ID是申请时所填写的手机号；Password为聚宽官网登录密码

count = get_query_count()
print(count)

#查询账号信息
infos = get_account_info()
print(infos)