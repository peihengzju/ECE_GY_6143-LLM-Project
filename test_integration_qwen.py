"""Integration test that hits the real Qwen endpoint.

Enable by setting env var QWEN_TEST_ENABLED=1 and ensuring the local Qwen API is reachable.
By default the test is skipped to avoid network/model dependency in CI.
"""

import os

import pytest

from qwen_client import call_qwen


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("QWEN_TEST_ENABLED") != "1",
    reason="Set QWEN_TEST_ENABLED=1 to run real Qwen call.",
)
def test_call_qwen_round_trip():
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with the single word 'pong'."},
    ]

    resp = call_qwen(messages, max_tokens=8, temperature=0.0, top_p=0.8)

    # Allow minor formatting, but the keyword should appear.
    assert "pong" in resp.lower()
