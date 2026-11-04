import pytest
import requests

from serein import http


def _response(status, body, content_range=None):
    response = requests.Response()
    response.status_code = status
    response._content = body
    response._content_consumed = True
    if content_range is not None:
        response.headers["Content-Range"] = content_range
    return response


@pytest.mark.parametrize("end, header", [(12, "bytes 10-12/20"),
                                        (None, "bytes 10-12/13"),
                                        (20, "bytes 10-12/13")])
def test_get_range_returns_requested_bytes(monkeypatch, end, header):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        return _response(206, b"abc", header)

    monkeypatch.setattr(http, "get", get)
    assert http.get_range("https://example.test/data", 10, end) == b"abc"
    suffix = "" if end is None else str(end)
    assert calls == [{"headers": {"Range": f"bytes=10-{suffix}"}, "timeout": 120}]


@pytest.mark.parametrize("status, body, header", [
    (200, b"whole file", None),
    (206, b"abc", None),
    (206, b"abc", "bytes 0-2/20"),
    (206, b"ab", "bytes 10-12/20"),
    (206, b"abc", "bytes 10-12/30"),
    (206, b"abcde", "bytes 10-14/20"),
])
def test_get_range_rejects_incorrect_response(monkeypatch, status, body, header):
    monkeypatch.setattr(http, "get", lambda *args, **kwargs: _response(status, body, header))
    with pytest.raises(requests.RequestException):
        http.get_range("https://example.test/data", 10, 13)


@pytest.mark.parametrize("start, end", [
    (-1, 2), (5, 4), (1.5, 3), (1, 3.5), (True, 3), (1, False),
    ("1", 3), (1, "3"), (None, 3), (float("nan"), 3), (1, float("inf")),
])
def test_get_range_rejects_invalid_bounds_before_request(monkeypatch, start, end):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid bounds must not make a request")

    monkeypatch.setattr(http, "get", unexpected)
    with pytest.raises(ValueError):
        http.get_range("https://example.test/data", start, end)


def test_get_range_rejects_unverifiable_open_ended_response(monkeypatch):
    monkeypatch.setattr(http, "get", lambda *args, **kwargs:
                        _response(206, b"abc", "bytes 10-12/*"))
    with pytest.raises(requests.RequestException, match="byte range"):
        http.get_range("https://example.test/data", 10, None)


def test_get_range_accepts_bounded_response_with_unknown_total(monkeypatch):
    monkeypatch.setattr(http, "get", lambda *args, **kwargs:
                        _response(206, b"abc", "bytes 10-12/*"))
    assert http.get_range("https://example.test/data", 10, 12) == b"abc"


@pytest.mark.parametrize("header", [
    "bytes 10-12/0",
    "bytes 10-12/10",
    "bytes 10-12/12",
    "bytes 12-10/20",
    "bytes */20",
    "bytes 10-12/-1",
    "bytes 10-12/20 trailing",
    "bytes 10-12/20\n",
    "bytes 10-12/20, bytes 13-15/20",
    "items 10-12/20",
])
def test_get_range_rejects_impossible_or_malformed_content_range(monkeypatch, header):
    response = _response(206, b"abc", header)
    closed = []
    monkeypatch.setattr(response, "close", lambda: closed.append(True))
    monkeypatch.setattr(http, "get", lambda *args, **kwargs: response)

    with pytest.raises(requests.RequestException) as error:
        http.get_range("https://example.test/data", 10, 12)

    assert error.value.response is response
    assert closed == [True]


@pytest.mark.parametrize("status, body, header", [
    (206, b"abc", "bytes 10-12/20"),
    (200, b"whole file", None),
    (206, b"abc", None),
    (206, b"ab", "bytes 10-12/20"),
])
def test_get_range_closes_consumed_and_rejected_responses(monkeypatch, status, body, header):
    response = _response(status, body, header)
    closed = []
    monkeypatch.setattr(response, "close", lambda: closed.append(True))
    monkeypatch.setattr(http, "get", lambda *args, **kwargs: response)
    if status == 206 and header is not None and body == b"abc":
        assert http.get_range("https://example.test/data", 10, 12) == b"abc"
    else:
        with pytest.raises(requests.RequestException) as error:
            http.get_range("https://example.test/data", 10, 12)
        assert error.value.response is response
    assert closed == [True]


@pytest.mark.parametrize("status, allow_404", [(404, True), (404, False), (500, False)])
def test_get_closes_discarded_responses(monkeypatch, status, allow_404):
    response = _response(status, b"error")
    closed = []
    monkeypatch.setattr(response, "close", lambda: closed.append(True))
    client = requests.Session()
    monkeypatch.setattr(client, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(http, "session", lambda: client)
    if allow_404:
        assert http.get("https://example.test/data", allow_404=True) is None
    else:
        with pytest.raises(requests.HTTPError) as error:
            http.get("https://example.test/data")
        assert error.value.response is response
    assert closed == [True]


def test_get_retries_connection_failures_and_preserves_request_options(monkeypatch):
    response = _response(200, b"ok")
    monkeypatch.setattr(response, "close", lambda: pytest.fail("caller owns successful response"))
    calls, sleeps = [], []
    client = requests.Session()

    def request(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) < 3:
            raise requests.ConnectionError("offline")
        return response

    monkeypatch.setattr(client, "get", request)
    monkeypatch.setattr(http, "session", lambda: client)
    monkeypatch.setattr(http.time, "sleep", sleeps.append)
    options = {"headers": {"Range": "bytes=0-2"}, "params": {"day": "2026-11-04"},
               "timeout": 12}
    assert http.get("https://example.test/data", **options) is response
    assert calls == [("https://example.test/data", options)] * 3
    assert sleeps == [1, 2]


def test_get_raises_last_connection_failure_after_retry_limit(monkeypatch):
    failures = [requests.Timeout(str(index)) for index in range(3)]
    calls, sleeps = [], []
    client = requests.Session()

    def request(*args, **kwargs):
        calls.append(True)
        raise failures[len(calls) - 1]

    monkeypatch.setattr(client, "get", request)
    monkeypatch.setattr(http, "session", lambda: client)
    monkeypatch.setattr(http.time, "sleep", sleeps.append)
    with pytest.raises(requests.Timeout) as error:
        http.get("https://example.test/data")
    assert error.value is failures[-1]
    assert len(calls) == 3
    assert sleeps == [1, 2]


@pytest.mark.parametrize("failure_type", [requests.exceptions.ChunkedEncodingError,
                                        requests.exceptions.ContentDecodingError])
def test_get_range_closes_response_when_body_read_fails(monkeypatch, failure_type):
    failure = failure_type("interrupted response body")
    events = []

    class InterruptedBody:
        def stream(self, chunk_size, decode_content):
            events.append("read")
            yield b"a"
            raise failure

        def close(self):
            events.append("close")

        def release_conn(self):
            events.append("release")

    response = requests.Response()
    response.status_code = 206
    response.headers["Content-Range"] = "bytes 10-12/20"
    response.raw = InterruptedBody()
    monkeypatch.setattr(http, "get", lambda *args, **kwargs: response)

    with pytest.raises(failure_type) as error:
        http.get_range("https://example.test/data", 10, 12)

    assert error.value is failure
    assert events == ["read", "close", "release"]
