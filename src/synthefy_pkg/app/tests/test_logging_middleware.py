from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from synthefy_pkg.app.middleware.logging_middleware import LoggingMiddleware


@pytest.fixture
def app_with_logging():
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/ok")
    async def ok():
        return {"msg": "ok"}

    @app.get("/fail")
    async def fail():
        raise RuntimeError("fail!")

    return app


def test_logging_middleware_pass_through(app_with_logging):
    client = TestClient(app_with_logging)
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert resp.json() == {"msg": "ok"}


def test_logging_middleware_catches_exception(app_with_logging):
    client = TestClient(app_with_logging)
    with patch(
        "synthefy_pkg.app.middleware.logging_middleware.logger.error"
    ) as mock_log:
        resp = client.get("/fail")
        assert resp.status_code == 500
        assert resp.text == "Internal Server Error"
        # The middleware logs twice: once with exc_info and once with traceback
        assert mock_log.call_count == 2
        # Check that the first call contains "Unhandled error"
        assert "Unhandled error" in mock_log.call_args_list[0][0][0]
        # Check that the second call contains "Stack trace"
        assert "Stack trace" in mock_log.call_args_list[1][0][0]
