"""Tests for the conditional (304) HTTP pre-check."""

import os

import pytest

from docforge.conditional import ConditionalResponse, http_conditional_get

_NETWORK = os.getenv("DOCFORGE_NETWORK_TESTS")


def test_response_properties() -> None:
    assert ConditionalResponse(304, '"E"', None).not_modified
    assert not ConditionalResponse(200, None, None).not_modified
    assert ConditionalResponse(200, '"E"', None).has_validators
    assert ConditionalResponse(200, None, "Mon").has_validators
    assert not ConditionalResponse(200, None, None).has_validators


@pytest.mark.skipif(not _NETWORK, reason="set DOCFORGE_NETWORK_TESTS=1 to run network tests")
def test_conditional_round_trip_against_a_real_server() -> None:
    first = http_conditional_get("https://example.com", None, None)
    assert first.status == 200
    if not first.has_validators:
        pytest.skip("server sent no validators; 304 not possible for this URL")

    # Send the validators back -> the server should say 304 Not Modified.
    second = http_conditional_get("https://example.com", first.etag, first.last_modified)
    assert second.not_modified
