# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DataModel

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

# Point Coordinate in R3
class P(object):
    __slots__ = ['_tab']

    # P
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # P
    def X(self): return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))
    # P
    def Y(self): return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))
    # P
    def Z(self): return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(8))

def CreateP(builder, x, y, z):
    builder.Prep(4, 12)
    builder.PrependFloat32(z)
    builder.PrependFloat32(y)
    builder.PrependFloat32(x)
    return builder.Offset()


class PT(object):

    # PT
    def __init__(self):
        self.x = 0.0  # type: float
        self.y = 0.0  # type: float
        self.z = 0.0  # type: float

    @classmethod
    def InitFromBuf(cls, buf, pos):
        p = P()
        p.Init(buf, pos)
        return cls.InitFromObj(p)

    @classmethod
    def InitFromObj(cls, p):
        x = PT()
        x._UnPack(p)
        return x

    # PT
    def _UnPack(self, p):
        if p is None:
            return
        self.x = p.X()
        self.y = p.Y()
        self.z = p.Z()

    # PT
    def Pack(self, builder):
        return CreateP(builder, self.x, self.y, self.z)
