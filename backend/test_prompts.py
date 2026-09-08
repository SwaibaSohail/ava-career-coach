import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import ava_system_prompt


def test_prompt_has_core_rules_and_interpolates_context():
    p = ava_system_prompt("CVSTATE_SENTINEL")
    assert "You are Ava" in p
    assert "FAITHFULNESS" in p
    assert "PRESENTING JOB RESULTS" in p
    assert p.rstrip().endswith("CONTEXT: CVSTATE_SENTINEL")
