import pytest
import requests

from serein import http


def _response(status, body, content_range=None):
    response = requests.Response()
    response.status_code = status
    response._content = body
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
