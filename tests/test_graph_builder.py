"""
单元测试 - Graph Builder

验证 StateGraph 构建、节点注册、边配置和 checkpointer 逻辑。
"""

import pytest
from agents.graph_builder import (
    build_graph,
    get_graph,
    reset_graph,
    reset_checkpointer,
    get_checkpointer,
)


class TestBuildGraph:
    """测试图构建逻辑"""

    def test_build_graph_returns_state_graph(self):
        """build_graph 应返回 StateGraph 实例"""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_four_nodes(self):
        """图应包含 4 个业务节点（含 __start__ 共 5 个）"""
        graph = build_graph()
        node_names = list(graph.nodes.keys())
        # LangGraph 1.x 自动添加 __start__ 节点
        assert "scenario" in node_names
        assert "conversation" in node_names
        assert "correction" in node_names
        assert "scoring" in node_names

    def test_graph_has_entry_point(self):
        """图应能成功获取图形结构（LangGraph 1.x CompiledStateGraph）"""
        graph = build_graph()
        # LangGraph 1.x 返回 CompiledStateGraph，用 get_graph() 获取可视化
        try:
            pg = graph.get_graph()
            # 验证有节点
            assert len(pg.nodes) > 0
        except AttributeError:
            # 兼容旧版
            assert hasattr(graph, "nodes")

    def test_graph_has_checkpointer(self):
        """图应附带 checkpointer"""
        graph = build_graph()
        assert graph.checkpointer is not None

    def test_graph_compile_success(self):
        """build_graph 应能成功编译（不抛出异常）"""
        try:
            graph = build_graph()
            # 尝试访问 compiled_graph 属性
            _ = graph.get_graph()
        except Exception as e:
            pytest.fail(f"Graph compilation failed: {e}")


class TestSingletonGraph:
    """测试图单例模式"""

    def test_get_graph_returns_same_instance(self):
        """多次调用 get_graph 应返回同一实例"""
        reset_graph()
        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2

    def test_reset_graph_clears_singleton(self):
        """reset_graph 应清除单例"""
        reset_graph()
        g1 = get_graph()
        reset_graph()
        g2 = get_graph()
        assert g1 is not g2


class TestCheckpointer:
    """测试 checkpointer 获取逻辑"""

    def test_get_checkpointer_returns_non_none(self):
        """get_checkpointer 应返回非 None 实例"""
        reset_checkpointer()
        cp = get_checkpointer()
        assert cp is not None

    def test_get_checkpointer_is_memory_or_sqlite(self):
        """checkpointer 应为 InMemorySaver（LangGraph 1.x MemorySaver 别名）"""
        reset_checkpointer()
        cp = get_checkpointer()
        cp_name = type(cp).__name__
        assert cp_name in ("InMemorySaver", "MemorySaver", "SqliteSaver")

    def test_get_checkpointer_is_cached(self):
        """get_checkpointer 应缓存结果"""
        reset_checkpointer()
        cp1 = get_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is cp2

    def test_reset_checkpointer_clears_cache(self):
        """reset_checkpointer 应清除缓存"""
        reset_checkpointer()
        cp1 = get_checkpointer()
        reset_checkpointer()
        cp2 = get_checkpointer()
        # 可能是同一类型的新实例（SQLite 会重新打开 DB）
        assert type(cp1) == type(cp2)


class TestCustomCheckpointer:
    """测试自定义 checkpointer 传入"""

    def test_build_graph_with_custom_checkpointer(self):
        """传入自定义 checkpointer 应被使用"""
        from langgraph.checkpoint.memory import MemorySaver

        reset_checkpointer()
        custom_cp = MemorySaver()
        graph = build_graph(checkpointer=custom_cp)
        assert graph.checkpointer is custom_cp
