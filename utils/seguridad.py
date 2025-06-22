from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import Header, HTTPException
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ADMIN_ID = os.getenv("ADMIN_ID")

def crear_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verificar_token(authorization: str = Header(...)):
    try:
        esquema, token = authorization.split()
        if esquema.lower() != "bearer":
            raise ValueError("Formato inválido")

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")

        if user_id != ADMIN_ID:
            raise HTTPException(status_code=403, detail="❌ Acceso no autorizado: no eres el administrador.")

        return user_id

    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="❌ Token inválido o no proporcionado.")


def verificar_token_general(authorization: str = Header(...)):
    try:
        esquema, token = authorization.split()
        if esquema.lower() != "bearer":
            raise ValueError("Formato inválido")

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        return user_id

    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="❌ Token inválido o no proporcionado.")
