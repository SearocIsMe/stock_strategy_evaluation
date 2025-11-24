# server.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import pymssql
from collections import deque
import time
import logging
import traceback
import json

# ==============================
# Logging 配置：文件 + 控制台
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
# API KEY & 限流
# ==============================
API_KEY = "CHANGE_ME_TO_A_SECURE_RANDOM_STRING"

RATE_LIMIT = 100
WINDOW_SECONDS = 60
rate_limit_store = {}  # {api_key: deque[timestamps]}

app = FastAPI(title="JoinQuant Trade Logger API")


class TradeIn(BaseModel):
    table: str
    name: str
    code: str
    type: str
    num: float
    # 允许 date 为空，为空则用当前时间
    date: Optional[datetime] = None
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

    logger.info(f"[AUTH OK] IP={client_ip}")
    return client_key


# ========== 限流（按 API Key） ==========
def rate_limiter(api_key: str = Depends(verify_api_key)):
    now = time.time()
    dq = rate_limit_store.get(api_key)

    if dq is None:
        dq = deque()
        rate_limit_store[api_key] = dq

    while dq and dq[0] <= now - WINDOW_SECONDS:
        dq.popleft()

    if len(dq) >= RATE_LIMIT:
        logger.warning(f"[RATE LIMIT] API Key={api_key} 已超过 {RATE_LIMIT}/min 限制")
        raise HTTPException(status_code=429, detail="Too many requests for this API key.")

    dq.append(now)
    return True


# ========== 全局：请求体验证失败（422）处理 ==========
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    client_ip = request.client.host
    try:
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8", errors="ignore")
    except Exception:
        body_str = "<cannot decode body>"

    logger.error(
        "[422 ValidationError] IP=%s PATH=%s ERRORS=%s BODY=%s",
        client_ip,
        request.url.path,
        exc.errors(),
        body_str,
    )

    # 返回更详细的信息给客户端
    return JSONResponse(
        status_code=422,
        content={
            "error": "Request validation failed",
            "path": str(request.url.path),
            "client_ip": client_ip,
            "errors": exc.errors(),
            "body": body_str,
        },
    )


@app.post("/api/trade")
async def create_trade(
    request: Request,
    trade: TradeIn,
    _authorized: str = Depends(verify_api_key),
    _limited: bool = Depends(rate_limiter),
):
    client_ip = request.client.host

    logger.info(
        "[REQUEST] IP=%s Insert -> table=%s code=%s type=%s num=%s",
        client_ip,
        trade.table,
        trade.code,
        trade.type,
        trade.num,
    )

    # 表名检查
    if trade.table not in ALLOWED_TABLES:
        logger.error("[INVALID TABLE] IP=%s table=%s", client_ip, trade.table)
        raise HTTPException(status_code=400, detail="Invalid table name")

    # date 允许为空时，使用当前时间
    if trade.date is None:
        trade_date = datetime.now()
    else:
        trade_date = trade.date

    table = trade.table

    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        # num 允许 float，后端自动 round
        safe_num = int(round(trade.num))
        sql = f"""
        INSERT INTO {table} (name, code, type, num, date, fenlei)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (
            trade.name,
            trade.code,
            trade.type,
            safe_num,   # 再次强制转 int，避免 weird 类型
            trade_date,
            trade.fenlei,
        )

        cursor.execute(sql, params)
        conn.commit()
        cursor.close()
        conn.close()

        logger.info("[INSERT OK] IP=%s code=%s type=%s", client_ip, trade.code, trade.type)
        return {"status": "ok"}

    except Exception as e:
        # 提取最后一帧的代码行信息
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last_frame = tb[-1]
            err_file = last_frame.filename
            err_line = last_frame.lineno
            err_func = last_frame.name
        else:
            err_file = "<unknown>"
            err_line = -1
            err_func = "<unknown>"

        logger.exception(
            "[DB ERROR] IP=%s file=%s line=%s func=%s error=%s",
            client_ip,
            err_file,
            err_line,
            err_func,
            repr(e),
        )

        # 返回给客户端详细错误（仅内部使用时 OK）
        return JSONResponse(
            status_code=500,
            content={
                "error": "DB insert error",
                "exception": repr(e),
                "file": err_file,
                "line": err_line,
                "function": err_func,
            },
        )


@app.get("/health")
async def health_check(request: Request):
    client_ip = request.client.host
    logger.info("[HEALTH] IP=%s", client_ip)
    return {"status": "alive"}
