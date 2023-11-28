# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DataModel

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


# Atom
class Atom(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsAtom(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Atom()
        x.Init(buf, n + offset)
        return x

    # Atom
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Position point
    # Atom
    def Pos(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = o + self._tab.Pos
            from DataModel.P import P

            obj = P()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Atomic mass
    # Atom
    def Mass(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(
                flatbuffers.number_types.Float32Flags, o + self._tab.Pos
            )
        return 0.0

    # Element number
    # Atom
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Atomic model (for rendering)
    # Atom
    def Model(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0


def AtomStart(builder):
    builder.StartObject(4)


def AtomAddPos(builder, pos):
    builder.PrependStructSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(pos), 0)


def AtomAddMass(builder, mass):
    builder.PrependFloat32Slot(1, mass, 0.0)


def AtomAddName(builder, name):
    builder.PrependInt32Slot(2, name, 0)


def AtomAddModel(builder, model):
    builder.PrependInt8Slot(3, model, 0)


def AtomEnd(builder):
    return builder.EndObject()


import DataModel.P

try:
    from typing import Optional
except:
    pass


class AtomT(object):
    # AtomT
    def __init__(self):
        self.pos = None  # type: Optional[DataModel.P.PT]
        self.mass = 0.0  # type: float
        self.name = 0  # type: int
        self.model = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        atom = Atom()
        atom.Init(buf, pos)
        return cls.InitFromObj(atom)

    @classmethod
    def InitFromObj(cls, atom):
        x = AtomT()
        x._UnPack(atom)
        return x

    # AtomT
    def _UnPack(self, atom):
        if atom is None:
            return
        if atom.Pos() is not None:
            self.pos = DataModel.P.PT.InitFromObj(atom.Pos())
        self.mass = atom.Mass()
        self.name = atom.Name()
        self.model = atom.Model()

    # AtomT
    def Pack(self, builder):
        AtomStart(builder)
        if self.pos is not None:
            pos = self.pos.Pack(builder)
            AtomAddPos(builder, pos)
        AtomAddMass(builder, self.mass)
        AtomAddName(builder, self.name)
        AtomAddModel(builder, self.model)
        atom = AtomEnd(builder)
        return atom