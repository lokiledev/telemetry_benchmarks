from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class JointState(_message.Message):
    __slots__ = ("position", "velocity", "effort")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FIELD_NUMBER: _ClassVar[int]
    EFFORT_FIELD_NUMBER: _ClassVar[int]
    position: float
    velocity: float
    effort: float
    def __init__(self, position: _Optional[float] = ..., velocity: _Optional[float] = ..., effort: _Optional[float] = ...) -> None: ...

class JointPositions(_message.Message):
    __slots__ = ("angles",)
    ANGLES_FIELD_NUMBER: _ClassVar[int]
    angles: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, angles: _Optional[_Iterable[float]] = ...) -> None: ...
