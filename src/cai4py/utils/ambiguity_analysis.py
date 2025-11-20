import subprocess
import os
from typing import Optional

path_to_analyser = os.path.dirname(os.path.abspath(__file__)) + "/../../../static-analyser/"
file_name = path_to_analyser + "processed/analysed_regex"

"""
    Takes a regex and determines if it contains counter ambiguity. Counter ambiguity is an NP-hard problem, thus
    the algorithm does an over-approximation, where if 'unambiguous' is returned we can be sure that it is unambiguous.
    If 'ambiguous' is returned then we are relatively certain the counter is ambiguous but the algorithm could be incorrect.
        TRUE: ambiguous
        FALSE: unambiguous
"""
def is_ambiguous(regex: str) -> Optional[bool]:

    try:
        java_checker = subprocess.run([f"java -jar {path_to_analyser}checker.jar --cambiguity '{regex}'"], shell=True, capture_output=True, text=True)

        if "is counter-ambiguous = false" in java_checker.stdout:
            return False
        elif "is counter-ambiguous = true" in java_checker.stdout: 
            return True
        else:
            return None
    except:
        return None
