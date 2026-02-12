from fastapi import Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name=settings.token_header_name, auto_error=False)


async def require_api_token(api_token: str | None = Depends(api_key_header)) -> str:
    if not api_token or api_token != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )
    return api_token
