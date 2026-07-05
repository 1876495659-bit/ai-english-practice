"""
Pytest fixtures shared across all test modules.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_modules():
    """
    在每个测试前后重置单例状态，防止测试间互相污染。
    """
    # Reset graph builder singletons
    from agents import graph_builder
    graph_builder.reset_graph()
    graph_builder.reset_checkpointer()

    yield

    # Cleanup after test
    graph_builder.reset_graph()
    graph_builder.reset_checkpointer()
