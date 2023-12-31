# automatically generated by the FlatBuffers compiler, do not modify

# namespace:

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class data(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsdata(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = data()
        x.Init(buf, n + offset)
        return x

    # data
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # data
    def M(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Image import Image

            obj = Image()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # data
    def MLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # data
    def MIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def dataStart(builder):
    builder.StartObject(1)


def dataAddM(builder, M):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(M), 0
    )


def dataStartMVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def dataEnd(builder):
    return builder.EndObject()
