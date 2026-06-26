"""Custom tools for dataset-generation scenarios (imported via config base_class strings).

These live outside the core package to avoid adding heavy optional dependencies
(e.g. pandas) to ``sgr-agent-core`` itself. The config loader adds the config
directory to ``sys.path``, so ``base_class: tools.pandas_analysis_tool.PandasAnalysisTool``
resolves correctly and the tool auto-registers in ``ToolRegistry``.
"""

from .pandas_analysis_tool import PandasAnalysisTool

__all__ = ["PandasAnalysisTool"]
