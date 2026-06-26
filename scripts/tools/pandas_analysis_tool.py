"""Safe tabular-data analysis tool backed by pandas.

This tool exposes a *constrained* set of analytical operations (no arbitrary
code execution, no shell) so it can be used as a function-calling tool by the
``data_analyst`` agent. Each call loads a CSV from a sandboxed ``workspace_path``
and returns a JSON-serialized, size-limited result.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, ClassVar, Literal

from pydantic import Field

from sgr_agent_core.base_tool import BaseTool

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 8000

Operation = Literal[
    "head",
    "summary",
    "describe",
    "columns",
    "value_counts",
    "groupby_agg",
    "filter_rows",
    "sort_by",
    "correlation",
]


class PandasAnalysisTool(BaseTool):
    """Analyze a CSV file with a constrained set of pandas operations.

    Usage: Provide the dataset (csv_path relative to workspace) and one operation
    plus its parameters. Returns a JSON string with the result (truncated).
    """

    tool_name: ClassVar[str] = "pandas_analysis_tool"

    csv_path: str = Field(description="CSV file path relative to the workspace directory")
    operation: Operation = Field(description="Analytical operation to perform on the dataset")
    column: str | None = Field(default=None, description="Target column (for value_counts, groupby_agg)")
    by: str | None = Field(default=None, description="Group/sort key column (for groupby_agg, sort_by)")
    agg_func: str | None = Field(
        default=None, description="Aggregation for groupby_agg: sum, mean, median, min, max, count, std"
    )
    query: str | None = Field(default=None, description="pandas query expression (for filter_rows), e.g. 'price > 100'")
    ascending: bool = Field(default=True, description="Sort order (for sort_by)")
    n: int = Field(default=10, description="Number of rows to return (for head, sort_by)")

    async def __call__(self, context: Any, config: Any, workspace_path: str | None = None, **kwargs: Any) -> str:
        """Run the requested operation and return a JSON result string.

        Args:
            context: Agent context (unused).
            config: Agent config (unused).
            workspace_path: Root directory holding the CSV files (from tool config).

        Returns:
            A JSON-encoded result (``{"operation": ..., "result": ...}``) or an
            ``{"error": ...}`` object, truncated to ``MAX_RESULT_CHARS``.
        """
        import pandas as pd

        workspace = self._resolve_workspace(workspace_path)
        full_path = self._safe_path(workspace, self.csv_path)
        if full_path is None:
            return json.dumps({"operation": self.operation, "error": f"csv_path escapes workspace: {self.csv_path}"})

        try:
            df = pd.read_csv(full_path)
        except Exception as e:  # noqa: BLE001 - surface any read error to the model
            return json.dumps({"operation": self.operation, "error": f"failed to read CSV: {e}"})

        try:
            result = self._apply_operation(df)
        except Exception as e:  # noqa: BLE001 - surface analysis errors to the model
            return json.dumps({"operation": self.operation, "error": f"{type(e).__name__}: {e}"})

        payload = {"operation": self.operation, "shape": list(df.shape), "result": result}
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        if len(encoded) > MAX_RESULT_CHARS:
            encoded = encoded[:MAX_RESULT_CHARS] + "...<truncated>"
        return encoded

    @staticmethod
    def _resolve_workspace(workspace_path: str | None) -> str:
        return os.path.abspath(workspace_path or ".")

    @staticmethod
    def _safe_path(workspace: str, csv_path: str) -> str | None:
        """Return the absolute path if it stays inside ``workspace``, else None."""
        full = os.path.abspath(os.path.join(workspace, csv_path))
        workspace_real = os.path.realpath(workspace)
        full_real = os.path.realpath(full)
        if full_real == workspace_real or full_real.startswith(workspace_real + os.sep):
            return full
        return None

    def _apply_operation(self, df: Any) -> Any:
        op = self.operation
        if op == "head":
            return df.head(self.n).to_dict(orient="records")
        if op == "summary":
            return {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "missing": df.isna().sum().to_dict(),
            }
        if op == "describe":
            return json.loads(df.describe(include="all").to_json(default_handler=str))
        if op == "columns":
            return [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()]
        if op == "value_counts":
            self._require("column")
            return df[self.column].astype(str).value_counts().head(self.n).to_dict()
        if op == "groupby_agg":
            self._require("by", "column", "agg_func")
            grouped = df.groupby(self.by)[self.column].agg(self.agg_func)
            return grouped.to_dict()
        if op == "filter_rows":
            self._require("query")
            return df.query(self.query).head(self.n).to_dict(orient="records")
        if op == "sort_by":
            self._require("by")
            return df.sort_values(self.by, ascending=self.ascending).head(self.n).to_dict(orient="records")
        if op == "correlation":
            return json.loads(df.select_dtypes("number").corr().to_json(default_handler=str))
        raise ValueError(f"Unknown operation: {op}")

    def _require(self, *fields: str) -> None:
        missing = [f for f in fields if getattr(self, f) is None]
        if missing:
            raise ValueError(f"operation '{self.operation}' requires: {', '.join(missing)}")
