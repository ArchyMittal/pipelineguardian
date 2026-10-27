"""
pipelineguardian — PipelineGuardian evaluation and audit framework.
"""
try:
    import langchain
    for _attr in ("verbose", "debug"):
        if not hasattr(langchain, _attr):
            setattr(langchain, _attr, False)
    if not hasattr(langchain, "llm_cache"):
        setattr(langchain, "llm_cache", None)
except ImportError:
    pass
