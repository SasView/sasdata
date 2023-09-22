from typing import Any

MINIMUM_REQUIREMENTS = []
OPTIONAL_REQUIREMENTS = []


class BaseConditionView:
    name = "base condition view"

    def __init__(self):
        self.minimum_reqs = MINIMUM_REQUIREMENTS
        self.optional_reqs = OPTIONAL_REQUIREMENTS

    def percent_compatible(self, obj: Any) -> (float, float):
        min_pct = 0.0
        min_step = 1.0 / float(len(self.minimum_reqs))
        opt_pct = 0.0
        opt_step = 1.0 / float(len(self.optional_reqs))
        for param in self.minimum_reqs:
            if hasattr(obj, param):
                min_pct += min_step
        for param in self.optional_reqs:
            if hasattr(obj, param):
                opt_pct += opt_step
        return min_pct, opt_pct
