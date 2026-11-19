import socket

from dllama.tui import Confirm, Dashboard


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_quit_asks_first_when_dashboard_owns_server():
    app = Dashboard("127.0.0.1", _closed_port(), "token", owns_server=True)
    async with app.run_test() as pilot:
        await pilot.press("q")
        assert isinstance(app.screen, Confirm)
        await pilot.press("n")
        assert not isinstance(app.screen, Confirm) and app.is_running
        await pilot.press("q", "y")
    assert not app.is_running


async def test_quit_is_immediate_for_remote_server():
    app = Dashboard("127.0.0.1", _closed_port(), "token")
    async with app.run_test() as pilot:
        await pilot.press("q")
    assert not app.is_running
