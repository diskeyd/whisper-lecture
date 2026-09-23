"""A differing span must map back to the segment — and the file — it was spoken in.

Synthetic sidecars only. The second device is split across two files, as the
2026-09-22 iPhone recording was, so the file index has to survive concatenation.
"""

import json

import device_diff


def sidecar(tmp_path, name, segs):
    p = tmp_path / name
    p.write_text(
        "\n".join(json.dumps({"start": s, "end": e, "text": t}, ensure_ascii=False) for s, e, t in segs),
        encoding="utf-8",
    )
    return str(p)


def test_span_maps_to_segment_and_file(tmp_path):
    a = sidecar(tmp_path, "a.jsonl", [
        (0.0, 4.0, "오늘은 힙 정렬을 배웁니다"),
        (4.0, 9.0, "완전 이진 트리를 배열로 표현하면"),
        (9.0, 14.0, "부모 노드는 인덱스 나누기 이입니다"),
    ])
    b1 = sidecar(tmp_path, "b1.jsonl", [(0.0, 4.0, "오늘은 힙 정렬을 배웁니다")])
    b2 = sidecar(tmp_path, "b2.jsonl", [
        (0.0, 5.0, "완전 이진 트리를 배열로 표현하면"),
        (5.0, 10.0, "부모 노드는 루트에서부터 차례대로 번호를 매깁니다"),  # 뒷부분이 다르게 들림
    ])
    rows, _, _ = device_diff.build("ipad", [a], "iphone", [b1, b2])
    assert len(rows) == 1
    assert rows[0]["ipad"]["start"] == 9.0 and rows[0]["ipad"]["file"] == 0
    assert rows[0]["iphone"]["start"] == 5.0 and rows[0]["iphone"]["file"] == 1
    assert "나누기" in rows[0]["ipad"]["span"] and "차례대로" in rows[0]["iphone"]["span"]
