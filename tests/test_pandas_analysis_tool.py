"""Tests for the PandasAnalysisTool (data_analyst scenario)."""

import json

import pytest

from scripts.tools.pandas_analysis_tool import PandasAnalysisTool


@pytest.fixture
def workspace(tmp_path):
    """Create a sample sales CSV in a temporary workspace."""
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "region,product,units,price\n"
        "EU,Widget,10,9.5\n"
        "US,Gadget,3,25.0\n"
        "EU,Gadget,7,25.0\n"
        "ASIA,Widget,15,9.5\n"
        "US,Widget,2,9.5\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _result_json(out: str) -> dict:
    return json.loads(out)


class TestPandasAnalysisTool:
    @pytest.mark.asyncio
    async def test_head_operation(self, workspace):
        tool = PandasAnalysisTool(csv_path="sales.csv", operation="head", n=2)
        out = await tool(context=None, config=None, workspace_path=workspace)
        data = _result_json(out)
        assert data["operation"] == "head"
        assert data["shape"] == [5, 4]
        assert len(data["result"]) == 2
        assert data["result"][0]["region"] == "EU"

    @pytest.mark.asyncio
    async def test_summary_operation(self, workspace):
        tool = PandasAnalysisTool(csv_path="sales.csv", operation="summary")
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert data["result"]["rows"] == 5
        assert data["result"]["columns"] == 4
        assert "price" in data["result"]["dtypes"]

    @pytest.mark.asyncio
    async def test_value_counts_operation(self, workspace):
        tool = PandasAnalysisTool(csv_path="sales.csv", operation="value_counts", column="region")
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert data["result"]["EU"] == 2

    @pytest.mark.asyncio
    async def test_groupby_agg_operation(self, workspace):
        tool = PandasAnalysisTool(
            csv_path="sales.csv", operation="groupby_agg", by="region", column="units", agg_func="sum"
        )
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert data["result"]["EU"] == 17  # 10 + 7

    @pytest.mark.asyncio
    async def test_filter_rows_operation(self, workspace):
        tool = PandasAnalysisTool(csv_path="sales.csv", operation="filter_rows", query="region == 'US'")
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert len(data["result"]) == 2
        assert all(r["region"] == "US" for r in data["result"])

    @pytest.mark.asyncio
    async def test_sort_by_operation(self, workspace):
        tool = PandasAnalysisTool(csv_path="sales.csv", operation="sort_by", by="units", ascending=False, n=2)
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert data["result"][0]["units"] == 15

    @pytest.mark.asyncio
    async def test_missing_required_field_returns_error(self, workspace):
        tool = PandasAnalysisTool(csv_path="sales.csv", operation="value_counts")  # no column
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert "error" in data
        assert "column" in data["error"]

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, workspace):
        tool = PandasAnalysisTool(csv_path="../../../etc/passwd", operation="head")
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert "error" in data
        assert "escapes workspace" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self, workspace):
        tool = PandasAnalysisTool(csv_path="nonexistent.csv", operation="head")
        data = _result_json(await tool(context=None, config=None, workspace_path=workspace))
        assert "error" in data

    def test_tool_is_registered(self):
        from sgr_agent_core.services.registry import ToolRegistry

        assert ToolRegistry.get("pandas_analysis_tool") is PandasAnalysisTool
