# database/src/connection.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

def build_database_url() -> str:
    # Если есть полный URL в env, используем его (удобно)
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    dialect = os.getenv("DB_DIALECT", "mysql")
    driver = os.getenv("DB_DRIVER", "pymysql")                # например pymysql
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "investment_portfolio")

    dialect_driver = f"{dialect}+{driver}" if driver else dialect
    return f"{dialect_driver}://{user}:{password}@{host}:{port}/{name}"

_engine: Engine | None = None

def setup_engine(echo: bool | None = None, pool_size: int | None = None, max_overflow: int | None = None) -> Engine:
    global _engine
    if _engine is not None:
        return _engine

    database_url = build_database_url()
    if echo is None:
        echo = os.getenv("SQLALCHEMY_ECHO", "0").lower() in ("1", "true", "yes")
    pool_size = int(pool_size or os.getenv("SQLALCHEMY_POOL_SIZE", "5"))
    max_overflow = int(max_overflow or os.getenv("SQLALCHEMY_MAX_OVERFLOW", "10"))

    _engine = create_engine(database_url, echo=echo, pool_size=pool_size, max_overflow=max_overflow)
    return _engine

def get_engine() -> Engine:
    if _engine is None:
        return setup_engine()
    return _engine


    