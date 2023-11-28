# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DataModel

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


# Quaternion h = a + bi + cj + dk
class Quaternion(object):
    __slots__ = ["_tab"]

    # Quaternion
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # a
    # Quaternion
    def A(self):
        return self._tab.Get(
            flatbuffers.number_types.Float32Flags,
            self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0),
        )

    # bi
    # Quaternion
    def B(self):
        return self._tab.Get(
            flatbuffers.number_types.Float32Flags,
            self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4),
        )

    # cj
    # Quaternion
    def C(self):
        return self._tab.Get(
            flatbuffers.number_types.Float32Flags,
            self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(8),
        )

    # dk
    # Quaternion
    def D(self):
        return self._tab.Get(
            flatbuffers.number_types.Float32Flags,
            self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(12),
        )


def CreateQuaternion(builder, a, b, c, d):
    builder.Prep(4, 16)
    builder.PrependFloat32(d)
    builder.PrependFloat32(c)
    builder.PrependFloat32(b)
    builder.PrependFloat32(a)
    return builder.Offset()


class QuaternionT(object):
    # QuaternionT
    def __init__(self):
        self.a = 0.0  # type: float
        self.b = 0.0  # type: float
        self.c = 0.0  # type: float
        self.d = 0.0  # type: float

    @classmethod
    def InitFromBuf(cls, buf, pos):
        quaternion = Quaternion()
        quaternion.Init(buf, pos)
        return cls.InitFromObj(quaternion)

    @classmethod
    def InitFromObj(cls, quaternion):
        x = QuaternionT()
        x._UnPack(quaternion)
        return x

    # QuaternionT
    def _UnPack(self, quaternion):
        if quaternion is None:
            return
        self.a = quaternion.A()
        self.b = quaternion.B()
        self.c = quaternion.C()
        self.d = quaternion.D()

    # QuaternionT
    def Pack(self, builder):
        return CreateQuaternion(builder, self.a, self.b, self.c, self.d)