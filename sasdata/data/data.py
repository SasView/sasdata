"""
Class to hold all data. All data coming in should be unmodified in this class.
"""
from typing import Optional


class Data:
    """"""
    def __init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)
        self.view_likely = self.find_all_views()
        self._conditions = {}

    def __setattr__(self, key, value):
        pass

    def find_all_views(self) -> Optional[str]:

        return None
