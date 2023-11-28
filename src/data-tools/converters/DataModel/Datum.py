# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DataModel

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class Datum(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsDatum(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Datum()
        x.Init(buf, n + offset)
        return x

    # Datum
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Datum
    def M1(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from DataModel.Matrix import Matrix

            obj = Matrix()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Datum
    def M2(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from DataModel.Matrix import Matrix

            obj = Matrix()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None


def DatumStart(builder):
    builder.StartObject(2)


def DatumAddM1(builder, m1):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(m1), 0
    )


def DatumAddM2(builder, m2):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(m2), 0
    )


def DatumEnd(builder):
    return builder.EndObject()


import DataModel.Matrix

try:
    from typing import Optional
except:
    pass


class DatumT(object):
    # DatumT
    def __init__(self):
        self.m1 = None  # type: Optional[DataModel.Matrix.MatrixT]
        self.m2 = None  # type: Optional[DataModel.Matrix.MatrixT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        datum = Datum()
        datum.Init(buf, pos)
        return cls.InitFromObj(datum)

    @classmethod
    def InitFromObj(cls, datum):
        x = DatumT()
        x._UnPack(datum)
        return x

    # DatumT
    def _UnPack(self, datum):
        if datum is None:
            return
        if datum.M1() is not None:
            self.m1 = DataModel.Matrix.MatrixT.InitFromObj(datum.M1())
        if datum.M2() is not None:
            self.m2 = DataModel.Matrix.MatrixT.InitFromObj(datum.M2())

    # DatumT
    def Pack(self, builder):
        if self.m1 is not None:
            m1 = self.m1.Pack(builder)
        if self.m2 is not None:
            m2 = self.m2.Pack(builder)
        DatumStart(builder)
        if self.m1 is not None:
            DatumAddM1(builder, m1)
        if self.m2 is not None:
            DatumAddM2(builder, m2)
        datum = DatumEnd(builder)
        return datum