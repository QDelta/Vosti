"""Fail early when the selected suite cannot run its local HTTP fixtures."""

import socket

import pytest


@pytest.fixture(scope="session", autouse=True)
def check_test_capabilities(request):
    if not any(item.get_closest_marker("loopback") for item in request.session.items):
        return
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            probe.listen(1)
    except OSError as error:
        pytest.exit(
            "Selected tests require loopback TCP access, but socket setup failed: "
            f"{error}. Run with local socket permission; this is an environment "
            "failure, not a kernel-verification result.",
            returncode=2,
        )
