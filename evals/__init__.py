"""On-demand real-LLM eval harness for topic_watch (dev-only, never CI).

Run as ``python -m evals <command>``. Adds input control + total observability
around the real LLM stages with zero production-code changes — the recorder hooks
the existing ``app.analysis.llm._get_client`` patch seam from the outside.

Not shipped in the wheel (``packages = ["app"]``) and not measured by the app
coverage gate; ``tests/test_evals.py`` exercises it offline.
"""
