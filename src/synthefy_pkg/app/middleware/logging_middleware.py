import traceback

from fastapi import HTTPException, Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as http_exc:
            raise HTTPException(
                status_code=http_exc.status_code, detail=str(http_exc.detail)
            )
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # You can also return a generic error response here
            return Response("Internal Server Error", status_code=500)
