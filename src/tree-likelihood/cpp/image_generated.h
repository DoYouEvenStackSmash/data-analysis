// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_IMAGE_MYDATA_H_
#define FLATBUFFERS_GENERATED_IMAGE_MYDATA_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 23 &&
              FLATBUFFERS_VERSION_MINOR == 5 &&
              FLATBUFFERS_VERSION_REVISION == 26,
             "Non-compatible flatbuffers version included");

namespace MyData {

struct Image;
struct ImageBuilder;
struct ImageT;

struct Data;
struct DataBuilder;
struct DataT;

struct ImageT : public ::flatbuffers::NativeTable {
  typedef Image TableType;
  int32_t shape = 0;
  std::vector<uint8_t> data{};
};

struct Image FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef ImageT NativeTableType;
  typedef ImageBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_SHAPE = 4,
    VT_DATA = 6
  };
  int32_t shape() const {
    return GetField<int32_t>(VT_SHAPE, 0);
  }
  bool mutate_shape(int32_t _shape = 0) {
    return SetField<int32_t>(VT_SHAPE, _shape, 0);
  }
  const ::flatbuffers::Vector<uint8_t> *data() const {
    return GetPointer<const ::flatbuffers::Vector<uint8_t> *>(VT_DATA);
  }
  ::flatbuffers::Vector<uint8_t> *mutable_data() {
    return GetPointer<::flatbuffers::Vector<uint8_t> *>(VT_DATA);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_SHAPE, 4) &&
           VerifyOffset(verifier, VT_DATA) &&
           verifier.VerifyVector(data()) &&
           verifier.EndTable();
  }
  ImageT *UnPack(const ::flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ImageT *_o, const ::flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static ::flatbuffers::Offset<Image> Pack(::flatbuffers::FlatBufferBuilder &_fbb, const ImageT* _o, const ::flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct ImageBuilder {
  typedef Image Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_shape(int32_t shape) {
    fbb_.AddElement<int32_t>(Image::VT_SHAPE, shape, 0);
  }
  void add_data(::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> data) {
    fbb_.AddOffset(Image::VT_DATA, data);
  }
  explicit ImageBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<Image> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Image>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<Image> CreateImage(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    int32_t shape = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> data = 0) {
  ImageBuilder builder_(_fbb);
  builder_.add_data(data);
  builder_.add_shape(shape);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<Image> CreateImageDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    int32_t shape = 0,
    const std::vector<uint8_t> *data = nullptr) {
  auto data__ = data ? _fbb.CreateVector<uint8_t>(*data) : 0;
  return MyData::CreateImage(
      _fbb,
      shape,
      data__);
}

::flatbuffers::Offset<Image> CreateImage(::flatbuffers::FlatBufferBuilder &_fbb, const ImageT *_o, const ::flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct DataT : public ::flatbuffers::NativeTable {
  typedef Data TableType;
  std::vector<std::unique_ptr<MyData::ImageT>> M{};
  DataT() = default;
  DataT(const DataT &o);
  DataT(DataT&&) FLATBUFFERS_NOEXCEPT = default;
  DataT &operator=(DataT o) FLATBUFFERS_NOEXCEPT;
};

struct Data FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef DataT NativeTableType;
  typedef DataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_M = 4
  };
  const ::flatbuffers::Vector<::flatbuffers::Offset<MyData::Image>> *M() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<MyData::Image>> *>(VT_M);
  }
  ::flatbuffers::Vector<::flatbuffers::Offset<MyData::Image>> *mutable_M() {
    return GetPointer<::flatbuffers::Vector<::flatbuffers::Offset<MyData::Image>> *>(VT_M);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_M) &&
           verifier.VerifyVector(M()) &&
           verifier.VerifyVectorOfTables(M()) &&
           verifier.EndTable();
  }
  DataT *UnPack(const ::flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(DataT *_o, const ::flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static ::flatbuffers::Offset<Data> Pack(::flatbuffers::FlatBufferBuilder &_fbb, const DataT* _o, const ::flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct DataBuilder {
  typedef Data Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_M(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<MyData::Image>>> M) {
    fbb_.AddOffset(Data::VT_M, M);
  }
  explicit DataBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<Data> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Data>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<Data> CreateData(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<MyData::Image>>> M = 0) {
  DataBuilder builder_(_fbb);
  builder_.add_M(M);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<Data> CreateDataDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<::flatbuffers::Offset<MyData::Image>> *M = nullptr) {
  auto M__ = M ? _fbb.CreateVector<::flatbuffers::Offset<MyData::Image>>(*M) : 0;
  return MyData::CreateData(
      _fbb,
      M__);
}

::flatbuffers::Offset<Data> CreateData(::flatbuffers::FlatBufferBuilder &_fbb, const DataT *_o, const ::flatbuffers::rehasher_function_t *_rehasher = nullptr);

inline ImageT *Image::UnPack(const ::flatbuffers::resolver_function_t *_resolver) const {
  auto _o = std::unique_ptr<ImageT>(new ImageT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void Image::UnPackTo(ImageT *_o, const ::flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = shape(); _o->shape = _e; }
  { auto _e = data(); if (_e) { _o->data.resize(_e->size()); std::copy(_e->begin(), _e->end(), _o->data.begin()); } }
}

inline ::flatbuffers::Offset<Image> Image::Pack(::flatbuffers::FlatBufferBuilder &_fbb, const ImageT* _o, const ::flatbuffers::rehasher_function_t *_rehasher) {
  return CreateImage(_fbb, _o, _rehasher);
}

inline ::flatbuffers::Offset<Image> CreateImage(::flatbuffers::FlatBufferBuilder &_fbb, const ImageT *_o, const ::flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { ::flatbuffers::FlatBufferBuilder *__fbb; const ImageT* __o; const ::flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _shape = _o->shape;
  auto _data = _o->data.size() ? _fbb.CreateVector(_o->data) : 0;
  return MyData::CreateImage(
      _fbb,
      _shape,
      _data);
}

inline DataT::DataT(const DataT &o) {
  M.reserve(o.M.size());
  for (const auto &M_ : o.M) { M.emplace_back((M_) ? new MyData::ImageT(*M_) : nullptr); }
}

inline DataT &DataT::operator=(DataT o) FLATBUFFERS_NOEXCEPT {
  std::swap(M, o.M);
  return *this;
}

inline DataT *Data::UnPack(const ::flatbuffers::resolver_function_t *_resolver) const {
  auto _o = std::unique_ptr<DataT>(new DataT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void Data::UnPackTo(DataT *_o, const ::flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = M(); if (_e) { _o->M.resize(_e->size()); for (::flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { if(_o->M[_i]) { _e->Get(_i)->UnPackTo(_o->M[_i].get(), _resolver); } else { _o->M[_i] = std::unique_ptr<MyData::ImageT>(_e->Get(_i)->UnPack(_resolver)); }; } } else { _o->M.resize(0); } }
}

inline ::flatbuffers::Offset<Data> Data::Pack(::flatbuffers::FlatBufferBuilder &_fbb, const DataT* _o, const ::flatbuffers::rehasher_function_t *_rehasher) {
  return CreateData(_fbb, _o, _rehasher);
}

inline ::flatbuffers::Offset<Data> CreateData(::flatbuffers::FlatBufferBuilder &_fbb, const DataT *_o, const ::flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { ::flatbuffers::FlatBufferBuilder *__fbb; const DataT* __o; const ::flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _M = _o->M.size() ? _fbb.CreateVector<::flatbuffers::Offset<MyData::Image>> (_o->M.size(), [](size_t i, _VectorArgs *__va) { return CreateImage(*__va->__fbb, __va->__o->M[i].get(), __va->__rehasher); }, &_va ) : 0;
  return MyData::CreateData(
      _fbb,
      _M);
}

inline const MyData::Data *GetData(const void *buf) {
  return ::flatbuffers::GetRoot<MyData::Data>(buf);
}

inline const MyData::Data *GetSizePrefixedData(const void *buf) {
  return ::flatbuffers::GetSizePrefixedRoot<MyData::Data>(buf);
}

inline Data *GetMutableData(void *buf) {
  return ::flatbuffers::GetMutableRoot<Data>(buf);
}

inline MyData::Data *GetMutableSizePrefixedData(void *buf) {
  return ::flatbuffers::GetMutableSizePrefixedRoot<MyData::Data>(buf);
}

inline bool VerifyDataBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<MyData::Data>(nullptr);
}

inline bool VerifySizePrefixedDataBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<MyData::Data>(nullptr);
}

inline void FinishDataBuffer(
    ::flatbuffers::FlatBufferBuilder &fbb,
    ::flatbuffers::Offset<MyData::Data> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedDataBuffer(
    ::flatbuffers::FlatBufferBuilder &fbb,
    ::flatbuffers::Offset<MyData::Data> root) {
  fbb.FinishSizePrefixed(root);
}

inline std::unique_ptr<MyData::DataT> UnPackData(
    const void *buf,
    const ::flatbuffers::resolver_function_t *res = nullptr) {
  return std::unique_ptr<MyData::DataT>(GetData(buf)->UnPack(res));
}

inline std::unique_ptr<MyData::DataT> UnPackSizePrefixedData(
    const void *buf,
    const ::flatbuffers::resolver_function_t *res = nullptr) {
  return std::unique_ptr<MyData::DataT>(GetSizePrefixedData(buf)->UnPack(res));
}

}  // namespace MyData

#endif  // FLATBUFFERS_GENERATED_IMAGE_MYDATA_H_
