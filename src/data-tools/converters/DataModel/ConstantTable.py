# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DataModel

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

# table for looking up constants
class ConstantTable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConstantTable(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConstantTable()
        x.Init(buf, n + offset)
        return x

    # ConstantTable
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConstantTable
    def Amplitude(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # ConstantTable
    def Defocus(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # ConstantTable
    def BFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # ConstantTable
    def ImgDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # ConstantTable
    def NumPixels(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # ConstantTable
    def PixelWidth(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # ConstantTable
    def Elecwavel(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # ConstantTable
    def Snr(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # ConstantTable
    def ExperimentParameters(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # random seed
    # ConstantTable
    def Seed(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # placeholder for structures
    # ConstantTable
    def Structures(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # placeholder for coordinates
    # ConstantTable
    def Coordinates(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

def ConstantTableStart(builder): builder.StartObject(12)
def ConstantTableAddAmplitude(builder, amplitude): builder.PrependBoolSlot(0, amplitude, 1)
def ConstantTableAddDefocus(builder, defocus): builder.PrependBoolSlot(1, defocus, 0)
def ConstantTableAddBFactor(builder, bFactor): builder.PrependBoolSlot(2, bFactor, 1)
def ConstantTableAddImgDims(builder, imgDims): builder.PrependBoolSlot(3, imgDims, 1)
def ConstantTableAddNumPixels(builder, numPixels): builder.PrependBoolSlot(4, numPixels, 1)
def ConstantTableAddPixelWidth(builder, pixelWidth): builder.PrependBoolSlot(5, pixelWidth, 1)
def ConstantTableAddElecwavel(builder, elecwavel): builder.PrependBoolSlot(6, elecwavel, 1)
def ConstantTableAddSnr(builder, snr): builder.PrependBoolSlot(7, snr, 1)
def ConstantTableAddExperimentParameters(builder, experimentParameters): builder.PrependBoolSlot(8, experimentParameters, 1)
def ConstantTableAddSeed(builder, seed): builder.PrependBoolSlot(9, seed, 1)
def ConstantTableAddStructures(builder, structures): builder.PrependBoolSlot(10, structures, 1)
def ConstantTableAddCoordinates(builder, coordinates): builder.PrependBoolSlot(11, coordinates, 1)
def ConstantTableEnd(builder): return builder.EndObject()


class ConstantTableT(object):

    # ConstantTableT
    def __init__(self):
        self.amplitude = True  # type: bool
        self.defocus = False  # type: bool
        self.bFactor = True  # type: bool
        self.imgDims = True  # type: bool
        self.numPixels = True  # type: bool
        self.pixelWidth = True  # type: bool
        self.elecwavel = True  # type: bool
        self.snr = True  # type: bool
        self.experimentParameters = True  # type: bool
        self.seed = True  # type: bool
        self.structures = True  # type: bool
        self.coordinates = True  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        constantTable = ConstantTable()
        constantTable.Init(buf, pos)
        return cls.InitFromObj(constantTable)

    @classmethod
    def InitFromObj(cls, constantTable):
        x = ConstantTableT()
        x._UnPack(constantTable)
        return x

    # ConstantTableT
    def _UnPack(self, constantTable):
        if constantTable is None:
            return
        self.amplitude = constantTable.Amplitude()
        self.defocus = constantTable.Defocus()
        self.bFactor = constantTable.BFactor()
        self.imgDims = constantTable.ImgDims()
        self.numPixels = constantTable.NumPixels()
        self.pixelWidth = constantTable.PixelWidth()
        self.elecwavel = constantTable.Elecwavel()
        self.snr = constantTable.Snr()
        self.experimentParameters = constantTable.ExperimentParameters()
        self.seed = constantTable.Seed()
        self.structures = constantTable.Structures()
        self.coordinates = constantTable.Coordinates()

    # ConstantTableT
    def Pack(self, builder):
        ConstantTableStart(builder)
        ConstantTableAddAmplitude(builder, self.amplitude)
        ConstantTableAddDefocus(builder, self.defocus)
        ConstantTableAddBFactor(builder, self.bFactor)
        ConstantTableAddImgDims(builder, self.imgDims)
        ConstantTableAddNumPixels(builder, self.numPixels)
        ConstantTableAddPixelWidth(builder, self.pixelWidth)
        ConstantTableAddElecwavel(builder, self.elecwavel)
        ConstantTableAddSnr(builder, self.snr)
        ConstantTableAddExperimentParameters(builder, self.experimentParameters)
        ConstantTableAddSeed(builder, self.seed)
        ConstantTableAddStructures(builder, self.structures)
        ConstantTableAddCoordinates(builder, self.coordinates)
        constantTable = ConstantTableEnd(builder)
        return constantTable
