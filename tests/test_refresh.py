from datetime import datetime
from zoneinfo import ZoneInfo

from refresh import forecast_due


def test_friday_job_honors_local_time_and_dst():
    utc = ZoneInfo("UTC")
    assert not forecast_due(datetime(2026, 9, 18, 7, 59, tzinfo=utc))
    assert forecast_due(datetime(2026, 9, 18, 8, 0, tzinfo=utc))
    assert not forecast_due(datetime(2026, 12, 18, 8, 59, tzinfo=utc))
    assert forecast_due(datetime(2026, 12, 18, 9, 0, tzinfo=utc))
    assert not forecast_due(datetime(2026, 9, 17, 10, 0, tzinfo=utc))
