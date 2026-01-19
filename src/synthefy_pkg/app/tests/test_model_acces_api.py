from typing import Optional

import pytest
from fastapi import HTTPException

from synthefy_pkg.app.routers.access_model_api import determine_task


class TestDetermineTask:
    @pytest.mark.parametrize(
        "task,syn_id,for_id,expected",
        [
            ("synthesis", "syn_123", None, "synthesis"),
            ("forecast", None, "for_123", "forecast"),
            ("synthesis", None, "for_123", "forecast"),
            ("forecast", "syn_123", None, "synthesis"),
            # New test cases for when both IDs are provided
            ("synthesis", "syn_123", "for_123", "synthesis"),
            ("forecast", "syn_123", "for_123", "forecast"),
        ],
    )
    def test_task_determination(
        self,
        task: str,
        syn_id: Optional[str],
        for_id: Optional[str],
        expected: str,
    ) -> None:
        """Test various task determination scenarios, including when both IDs are provided."""
        assert determine_task(task, syn_id, for_id) == expected

    @pytest.mark.parametrize(
        "task,syn_id,for_id,error_msg",
        [
            ("invalid", "syn_123", None, "Invalid task provided"),
            ("synthesis", None, None, "Synthesis training job ID is required"),
            ("forecast", None, None, "Forecast training job ID is required"),
        ],
    )
    def test_task_determination_errors(
        self,
        task: str,
        syn_id: Optional[str],
        for_id: Optional[str],
        error_msg: str,
    ) -> None:
        """Test error cases for task determination."""
        with pytest.raises(HTTPException) as exc:
            determine_task(task, syn_id, for_id)
        assert exc.value.status_code == 400
        assert exc.value.detail == error_msg
