import json

from dllama.proxy import UsageParser


def parse_in_pieces(payload: bytes, size: int = 7) -> dict:
    parser = UsageParser()
    for i in range(0, len(payload), size):
        parser.feed(payload[i : i + size])
    parser.close()
    return parser.usage


def ndjson(*objs: dict) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def sse(*objs: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs) + b"data: [DONE]\n\n"


def test_native_stream():
    payload = ndjson(
        {"message": {"role": "assistant", "content": "hi"}, "done": False},
        {"done": True, "prompt_eval_count": 12, "eval_count": 34, "eval_duration": 2_000_000_000},
    )
    assert parse_in_pieces(payload) == {"prompt_tok": 12, "out_tok": 34, "eval_ns": 2_000_000_000}


def test_native_non_streaming_without_trailing_newline():
    payload = json.dumps({"response": "hi", "done": True, "prompt_eval_count": 3, "eval_count": 4}).encode()
    assert parse_in_pieces(payload) == {"prompt_tok": 3, "out_tok": 4}


def test_openai_stream_with_usage():
    payload = sse(
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}},
    )
    assert parse_in_pieces(payload) == {"prompt_tok": 5, "out_tok": 9}


def test_openai_stream_without_usage():
    assert parse_in_pieces(sse({"choices": [{"delta": {"content": "hi"}}]})) == {}


def test_anthropic_stream():
    payload = (
        b"event: message_start\n"
        + sse({"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}})
        + b"event: content_block_delta\n"
        + sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}})
        + b"event: message_delta\n"
        + sse({"type": "message_delta", "usage": {"output_tokens": 42}})
    )
    assert parse_in_pieces(payload) == {"prompt_tok": 20, "out_tok": 42}
