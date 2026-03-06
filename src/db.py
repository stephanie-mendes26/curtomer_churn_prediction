import os # permite a leitura de variaveis no sistema, como as do .env
import pandas as pd
from dotenv import load_dotenv # carrega as variáveis do .env para o ambiente
import pyodbc # biblioteca para conectar ao SQL Server
load_dotenv()

def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variável {name} não encontrada no .env")
    return value

def make_conn_str() -> str:
    server = _get_env("DB_SERVER")
    database = _get_env("DB_DATABASE")
    user = _get_env("DB_USER")
    password = _get_env("DB_PASSWORD")

    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )

def test_connection() -> tuple:
    conn_str = make_conn_str()
    with pyodbc.connect(conn_str) as cn:
        cur = cn.cursor()
        cur.execute("SELECT 1")
        return cur.fetchone()

def get_data(sql: str, params=None) -> pd.DataFrame:
    conn_str = make_conn_str()
    with pyodbc.connect(conn_str) as cn:
        return pd.read_sql(sql, cn, params=params)