'''
uvicorn server:app --host 0.0.0.0 --port 8000
'''
# server.py
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import pymssql
from datetime import datetime
from collections import deque
import time
import logging

# ==============================
# Logging 配置：保存到 server.log
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("server.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("trade_api")

# ==============================
# DB Config
# ==============================
DB_HOST = "172.25.180.214"
DB_USER = "sa"
DB_PASSWORD = "Just4Jhp@QmtJoinQuant"
DB_NAME = "touzi"

ALLOWED_TABLES = {"trade"}

# ==============================
# API KEY 设置
# ==============================
API_KEY = "CHANGE_ME_TO_A_SECURE_RANDOM_STRING"

# ==============================
# 限流：按 API Key 100 次 / 60 秒
# ==============================
RATE_LIMIT = 100
WINDOW_SECONDS = 60
rate_limit_store = {}  # {api_key: deque[timestamps]}

app = FastAPI(title="JoinQuant Trade Logger API")


class TradeIn(BaseModel):
    table: str
    name: str
    code: str
    type: str
    num: int
    date: datetime
    fenlei: str


def get_db_conn():
    return pymssql.connect(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)


# ========== API Key 校验 ==========
def verify_api_key(request: Request):
    client_key = request.headers.get("X-API-Key")
    client_ip = request.client.host

    if client_key != API_KEY:
        logger.warning(f"[AUTH FAIL] IP={client_ip}  API Key invalid")
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 记录合法访问
    logger.info(f"[AUTH OK] IP={client_ip}")
    return client_key


# ========== 限流（按 API Key） ==========
def rate_limiter(api_key: str = Depends(verify_api_key)):
    now = time.time()
    dq = rate_limit_store.get(api_key)

    if dq is None:
        dq = deque()
        rate_limit_store[api_key] = dq

    # 清除窗口外的访问
    while dq and dq[0] <= now - WINDOW_SECONDS:
        dq.popleft()

    # 超过限制
    if len(dq) >= RATE_LIMIT:
        logger.warning(f"[RATE LIMIT] API Key={api_key} 已超过 {RATE_LIMIT}/min 限制")
        raise HTTPException(status_code=429, detail="Too many requests for this API key.")

    dq.append(now)
    return True


@app.post("/api/trade")
async def create_trade(
    request: Request,
    trade: TradeIn,
    _authorized: str = Depends(verify_api_key),
    _limited: bool = Depends(rate_limiter),
):
    client_ip = request.client.host

    logger.info(
        f"[REQUEST] IP={client_ip}  "
        f"Insert -> table={trade.table}, code={trade.code}, type={trade.type}, num={trade.num}"
    )

    # 表名检查（防止 SQL 注入）
    if trade.table not in ALLOWED_TABLES:
        logger.error(f"[INVALID TABLE] IP={client_ip} table={trade.table}")
        raise HTTPException(status_code=400, detail="Invalid table name")

    table = trade.table

    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        sql = f"""
        INSERT INTO {table} (name, code, type, num, date, fenlei)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (
            trade.name,
            trade.code,
            trade.type,
            trade.num,
            trade.date,
            trade.fenlei,
        )

        cursor.execute(sql, params)
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"[INSERT OK] IP={client_ip}  code={trade.code} type={trade.type}")
        return {"status": "ok"}

    except Exception as e:
        logger.exception(f"[DB ERROR] IP={client_ip}  Error={e}")
        raise HTTPException(status_code=500, detail=f"DB insert error: {e}")


@app.get("/health")
def health_check(request: Request):
    client_ip = request.client.host
    logger.info(f"[HEALTH] IP={client_ip}")
    return {"status": "alive"}
