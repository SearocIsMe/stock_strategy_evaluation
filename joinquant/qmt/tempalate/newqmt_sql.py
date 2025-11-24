from kuanke.user_space_api import *
import datetime
import requests

TABLE = "trade"
FENLEI = "未分类"
API_BASE_URL = "http://8.163.19.106:30212"
API_KEY = "CHANGE_ME_TO_A_SECURE_RANDOM_STRING"

class MyTrade():
    def __init__(self):
        self.api_base = API_BASE_URL.rstrip("/")

    def update(self, code, quantity, types):
        # 代码转换
        if code.endswith('XSHE'):
            code1 = code[:-4] + "SZ"
        else:
            code1 = code[:-4] + "SH"

        try:
            name1 = get_security_info(code).display_name
        except Exception as e:
            log.error("MyTrade 获取证券信息失败 code=%s, error=%s" % (code, e))
            name1 = code  # 兜底

        now = datetime.datetime.now().replace(microsecond=0)

        payload = {
            "table": TABLE,
            "name": name1,
            "code": code1,
            "type": types,
            "num": int(quantity) if quantity is not None else 0,
            "date": now.isoformat(),
            "fenlei": FENLEI,
        }

        headers = {"X-API-Key": API_KEY}
        url = self.api_base + "/api/trade"

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
        except Exception as e:
            log.error("MyTrade REST 请求异常: url=%s, payload=%s, error=%s" % (url, payload, e))
            return

        # 尝试解析服务端返回 JSON
        resp_text = resp.text
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = None

        if resp.status_code != 200:
            log.error(
                "MyTrade REST 调用失败: HTTP %s, 响应文本=%s, 请求payload=%s" %
                (resp.status_code, resp_text, payload)
            )
            if resp_json is not None:
                log.error("MyTrade REST 失败 JSON 详情: %s" % resp_json)
        else:
            # 可选：调试成功记录（量大时可以关掉）
            log.info("MyTrade REST 成功: code=%s, type=%s, num=%s" % (code1, types, quantity))


def order_zzy(security: str, quantity: int, style=None, pindex=0):
    mytrade = MyTrade()
    mytrade.update(code=security, quantity=quantity, types='order')
    _order = order(security, quantity, style=style, pindex=pindex)
    return _order

def order_target_zzy(security: str, quantity: int, style=None, pindex=0):
    mytrade = MyTrade()
    mytrade.update(code=security, quantity=quantity, types='order_target')
    _order = order_target(security, quantity, style=style, pindex=pindex)
    return _order

def order_value_zzy(security: str, quantity: int, style=None, pindex=0):
    mytrade = MyTrade()
    mytrade.update(code=security, quantity=quantity, types='order_value')
    _order = order_value(security, quantity, style=style, pindex=pindex)
    return _order

def order_target_value_zzy(security: str, quantity: int, style=None, pindex=0):
    mytrade = MyTrade()
    mytrade.update(code=security, quantity=quantity, types='order_target_value')
    _order = order_target_value(security, quantity, style=style, pindex=pindex)
    return _order
