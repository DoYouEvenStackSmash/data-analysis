# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DataModel

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


# A group of associated atoms corresponding to a conformation with at least one (implicit) orientation
class Structure(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsStructure(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Structure()
        x.Init(buf, n + offset)
        return x

    # Structure
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Points in the cloud
    # Structure
    def Atoms(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from DataModel.Atom import Atom

            obj = Atom()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Structure
    def AtomsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Structure
    def AtomsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Quaternion orientations
    # Structure
    def Orientations(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 16
            from DataModel.Quaternion import Quaternion

            obj = Quaternion()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Structure
    def OrientationsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Structure
    def OrientationsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0


def StructureStart(builder):
    builder.StartObject(2)


def StructureAddAtoms(builder, atoms):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(atoms), 0
    )


def StructureStartAtomsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)


def StructureAddOrientations(builder, orientations):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(orientations), 0
    )


def StructureStartOrientationsVector(builder, numElems):
    return builder.StartVector(16, numElems, 4)


def StructureEnd(builder):
    return builder.EndObject()


import DataModel.Atom
import DataModel.Quaternion

try:
    from typing import List
except:
    pass


class StructureT(object):
    # StructureT
    def __init__(self):
        self.atoms = None  # type: List[DataModel.Atom.AtomT]
        self.orientations = None  # type: List[DataModel.Quaternion.QuaternionT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        structure = Structure()
        structure.Init(buf, pos)
        return cls.InitFromObj(structure)

    @classmethod
    def InitFromObj(cls, structure):
        x = StructureT()
        x._UnPack(structure)
        return x

    # StructureT
    def _UnPack(self, structure):
        if structure is None:
            return
        if not structure.AtomsIsNone():
            self.atoms = []
            for i in range(structure.AtomsLength()):
                if structure.Atoms(i) is None:
                    self.atoms.append(None)
                else:
                    atom_ = DataModel.Atom.AtomT.InitFromObj(structure.Atoms(i))
                    self.atoms.append(atom_)
        if not structure.OrientationsIsNone():
            self.orientations = []
            for i in range(structure.OrientationsLength()):
                if structure.Orientations(i) is None:
                    self.orientations.append(None)
                else:
                    quaternion_ = DataModel.Quaternion.QuaternionT.InitFromObj(
                        structure.Orientations(i)
                    )
                    self.orientations.append(quaternion_)

    # StructureT
    def Pack(self, builder):
        if self.atoms is not None:
            atomslist = []
            for i in range(len(self.atoms)):
                atomslist.append(self.atoms[i].Pack(builder))
            StructureStartAtomsVector(builder, len(self.atoms))
            for i in reversed(range(len(self.atoms))):
                builder.PrependUOffsetTRelative(atomslist[i])
            atoms = builder.EndVector(len(self.atoms))
        if self.orientations is not None:
            StructureStartOrientationsVector(builder, len(self.orientations))
            for i in reversed(range(len(self.orientations))):
                self.orientations[i].Pack(builder)
            orientations = builder.EndVector(len(self.orientations))
        StructureStart(builder)
        if self.atoms is not None:
            StructureAddAtoms(builder, atoms)
        if self.orientations is not None:
            StructureAddOrientations(builder, orientations)
        structure = StructureEnd(builder)
        return structure