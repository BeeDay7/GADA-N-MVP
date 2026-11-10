"""gadanplus — GADA-N+ evaluation harness (reference implementation).

NOTE: the environment dynamics here are a simplified reference stand-in for
building and testing the pipeline. They are NOT the validated GADA-N model.
Numbers from this harness must not be reported as paper results.
"""
__all__ = ["env", "attacker", "metrics", "stats", "integrity", "figures",
           "run", "run_conditions", "gde", "data", "adapter", "gadan_env"]
