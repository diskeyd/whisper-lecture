"""Self-checks for post_edit's edit application. Synthetic text only — no lecture content."""

import post_edit

TEXT = "**0:00 - 1:00**\n스택 하나리를 봅니다. 스택은 스택\n\n**1:00 - 2:00**\n힙 정렬\n"


def test_only_unique_substrings_in_the_named_block_change():
    blocks = post_edit.split_blocks(TEXT)
    rows = post_edit.apply(blocks, [
        {"at": "0:00", "from": "하나리를", "to": "카나리를", "why": "용어"},
        {"at": "0:00", "from": "스택", "to": "큐", "why": "두 번 나옴"},
        {"at": "1:00", "from": "하나리", "to": "카나리", "why": "다른 블록"},
        {"at": "9:00", "from": "힙", "to": "힙", "why": "없는 블록"},
    ])
    assert [r[4] for r in rows] == ["applied", "skipped", "skipped", "skipped"]
    out = "".join(h + b for _, h, b in blocks)
    assert out == TEXT.replace("하나리를", "카나리를")
