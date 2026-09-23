"""Guards on the Drive download.

A partial download used to stay in data/raw/ after a failure or a Ctrl-C. The
next run saw the file, printed "already downloaded", and make_excerpt.py then
cut an excerpt out of a truncated recording — measured against a gold file, that
is a silent wrong number rather than an error.
"""

import fetch


def test_the_partial_name_stays_out_of_the_recording_glob(tmp_path):
    # make_excerpt.py locates a recording with RAW.glob(stem + ".*"), so the
    # temp name must not match it — ".part" would.
    out = tmp_path / "s1-261.m4a"
    tmp = fetch.partial_path(out)
    assert tmp not in set(tmp_path.glob("s1-261.*"))
    assert tmp.parent == out.parent


def test_a_short_download_leaves_nothing_behind(tmp_path, monkeypatch):
    out = tmp_path / "s1-261.m4a"
    monkeypatch.setattr(fetch, "RAW", tmp_path)

    def truncated(args, cwd=None):
        fetch.partial_path(out).write_bytes(b"only the first bytes")
        return ""

    monkeypatch.setattr(fetch, "gws", truncated)
    try:
        fetch.download({"id": "x", "size": "9999"}, out)
    except RuntimeError as e:
        assert "불완전" in str(e)
    else:
        raise AssertionError("short download was accepted")
    assert not out.exists()
    assert not fetch.partial_path(out).exists()


def test_a_complete_download_moves_into_place(tmp_path, monkeypatch):
    out = tmp_path / "s1-261.m4a"
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    payload = b"0123456789"
    monkeypatch.setattr(
        fetch, "gws", lambda args, cwd=None: fetch.partial_path(out).write_bytes(payload)
    )
    fetch.download({"id": "x", "size": str(len(payload))}, out)
    assert out.read_bytes() == payload
    assert not fetch.partial_path(out).exists()
