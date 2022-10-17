"""
This class converts things to other things. It can take a in value of a in type and convert it to an out value of an out type.

"""

from abc import ABC, abstractmethod

class CCConverter(ABC):
    def __init__(self, defaultOutType=None, defaultInType=None):
        self.defaultOutType = defaultOutType
        self.defaultInType = defaultInType

    def setDefaultTarget(self, defaultTarget):
        self.defaultTarget = defaultTarget

    @abstractmethod
    def convert(self, inValue, inType=None, outType=None):
        """
        Converts inValue of type inType to the type outType.

        Args:
            inValue - The value which needs converting
            inType:str - The type of the inValue. If None, the type is resolved.
            outType:str - The desired type to convert inType to. If None,
                should use self.defaultTarget. If self.defaultTarget not
                specified, it's recommended to perform no conversion.
        """
        pass

