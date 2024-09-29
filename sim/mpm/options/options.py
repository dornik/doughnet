import numpy as np
import mpm as us
from pydantic import BaseModel
from mpm.utils.repr import _repr, _simple_repr


class Options(BaseModel):
    def __init__(self, **data):
        # enforce parameters are supported
        for key in data.keys():
            if key not in self.__annotations__.keys():
                us.raise_exception(f'Unrecognized attribute: {key}')

        super().__init__(**data)

    def copy_attributes_from(self, options, override=False):
        for field in options.__fields__:
            if field in self.__fields__:
                if override or getattr(self, field) is None:
                    setattr(self, field, getattr(options, field))

    def __repr__(self):
        return _simple_repr(self)
