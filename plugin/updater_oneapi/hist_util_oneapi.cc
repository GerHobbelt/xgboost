/*!
 * Copyright 2017-2023 by Contributors
 * \file hist_util_oneapi.cc
 */
#include <vector>
#include <limits>

#include "hist_util_oneapi.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace common {

uint32_t SearchBin(const bst_float* cut_values, const uint32_t* cut_ptrs, Entry const& e) {
  auto beg = cut_ptrs[e.index];
  auto end = cut_ptrs[e.index + 1];
  const auto &values = cut_values;
  auto it = std::upper_bound(cut_values + beg, cut_values + end, e.fvalue);
  uint32_t idx = it - cut_values;
  if (idx == end) {
    idx -= 1;
  }
  return idx;
}

template <typename BinIdxType>
void mergeSort(BinIdxType* begin, BinIdxType* end, BinIdxType* buf) {
  const size_t total_len = end - begin;
  for (size_t block_len = 1; block_len < total_len; block_len <<= 1) {
    for (size_t cur_block = 0; cur_block + block_len < total_len; cur_block += 2 * block_len) {
      size_t start = cur_block;
      size_t mid = start + block_len;
      size_t finish = mid + block_len < total_len ? mid + block_len : total_len;
      size_t left_pos = start;
      size_t right_pos = mid;
      size_t pos = start;
      while (left_pos < mid || right_pos < finish) {
        if (left_pos < mid && (right_pos == finish || begin[left_pos] < begin[right_pos])) {
          buf[pos++] = begin[left_pos++];
        } else {
          buf[pos++] = begin[right_pos++];
        }
      }
      for (size_t i = start; i < finish; i++) begin[i] = buf[i];
    }
  }
}

template <typename BinIdxType>
void GHistIndexMatrixOneAPI::SetIndexData(sycl::queue qu,
                                          common::Span<BinIdxType> index_data_span,
                                          const DeviceMatrixOneAPI &dmat_device,
                                          size_t nbins,
                                          size_t row_stride,
                                          uint32_t* offsets) {
  const xgboost::Entry *data_ptr = dmat_device.data.DataConst();
  const bst_row_t *offset_vec = dmat_device.row_ptr.DataConst();
  const size_t num_rows = dmat_device.row_ptr.Size() - 1;
  BinIdxType* index_data = index_data_span.data();
  const bst_float* cut_values = cut_device.Values().DataConst();
  const uint32_t* cut_ptrs = cut_device.Ptrs().DataConst();
  sycl::buffer<size_t, 1> hit_count_buf(hit_count.data(), hit_count.size());

  USMVector<BinIdxType> sort_buf(qu, num_rows * row_stride);
  BinIdxType* sort_data = sort_buf.Data();

  qu.submit([&](sycl::handler& cgh) {
    auto hit_count_acc = hit_count_buf.template get_access<sycl::access::mode::atomic>(cgh);
    cgh.parallel_for<>(sycl::range<1>(num_rows), [=](sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      const size_t ibegin = offset_vec[i];
      const size_t iend = offset_vec[i + 1];
      const size_t size = iend - ibegin;
      const size_t start = i * row_stride;
      for (bst_uint j = 0; j < size; ++j) {
        uint32_t idx = SearchBin(cut_values, cut_ptrs, data_ptr[ibegin + j]);
        index_data[start + j] = offsets ? idx - offsets[j] : idx;
        sycl::atomic_fetch_add<size_t>(hit_count_acc[idx], 1);
      }
      if (!offsets) {
        // Sparse case only
        mergeSort<BinIdxType>(index_data + start, index_data + start + size, sort_data + start);
        for (bst_uint j = size; j < row_stride; ++j) {
          index_data[start + j] = nbins;
        }
      }
    });
  }).wait();
}

void GHistIndexMatrixOneAPI::ResizeIndex(const size_t n_offsets,
                                         const size_t n_index,
                                         const bool isDense) {
  if ((max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint8_t>::max())) && isDense) {
    index.SetBinTypeSize(kUint8BinsTypeSize);
    index.Resize((sizeof(uint8_t)) * n_index);
  } else if ((max_num_bins - 1 > static_cast<int>(std::numeric_limits<uint8_t>::max())  &&
    max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint16_t>::max())) && isDense) {
    index.SetBinTypeSize(kUint16BinsTypeSize);
    index.Resize((sizeof(uint16_t)) * n_index);
  } else {
    index.SetBinTypeSize(kUint32BinsTypeSize);
    index.Resize((sizeof(uint32_t)) * n_index);
  }
}

void GHistIndexMatrixOneAPI::Init(sycl::queue qu,
                                  const DeviceMatrixOneAPI& p_fmat_device,
                                  int max_bins) {
  nfeatures = p_fmat_device.p_mat->Info().num_col_;

  cut = SketchOnDMatrix(p_fmat_device.p_mat, max_bins);
  cut_device.Init(qu, cut);

  max_num_bins = max_bins;
  const uint32_t nbins = cut.Ptrs().back();
  this->nbins = nbins;
  hit_count.resize(nbins, 0);

  this->p_fmat = p_fmat_device.p_mat;
  const bool isDense = p_fmat_device.p_mat->IsDense();
  this->isDense_ = isDense;

  row_ptr = std::vector<size_t>(p_fmat_device.row_ptr.Begin(), p_fmat_device.row_ptr.End());
  row_ptr_device = p_fmat_device.row_ptr;

  index.setQueue(qu);

  row_stride = 0;
  for (const auto& batch : p_fmat_device.p_mat->GetBatches<SparsePage>()) {
    const auto& row_offset = batch.offset.ConstHostVector();
    for (auto i = 1ull; i < row_offset.size(); i++) {
      row_stride = std::max(row_stride, static_cast<size_t>(row_offset[i] - row_offset[i - 1]));
    }
  }

  const size_t n_offsets = cut.Ptrs().size() - 1;
  const size_t n_rows = p_fmat_device.row_ptr.Size() - 1;
  const size_t n_index = n_rows * row_stride;
  ResizeIndex(n_offsets, n_index, isDense);

  CHECK_GT(cut.Values().size(), 0U);

  uint32_t* offsets = nullptr;
  if (isDense) {
    index.ResizeOffset(n_offsets);
    offsets = index.Offset();
    for (size_t i = 0; i < n_offsets; ++i) {
      offsets[i] = cut.Ptrs()[i];
    }
  }

  if (isDense) {
    BinTypeSize curent_bin_size = index.GetBinTypeSize();
    if (curent_bin_size == kUint8BinsTypeSize) {
      common::Span<uint8_t> index_data_span = {index.data<uint8_t>(),
                                               n_index};
      SetIndexData(qu, index_data_span, p_fmat_device, nbins, row_stride, offsets);

    } else if (curent_bin_size == kUint16BinsTypeSize) {
      common::Span<uint16_t> index_data_span = {index.data<uint16_t>(),
                                                n_index};
      SetIndexData(qu, index_data_span, p_fmat_device, nbins, row_stride, offsets);
    } else {
      CHECK_EQ(curent_bin_size, kUint32BinsTypeSize);
      common::Span<uint32_t> index_data_span = {index.data<uint32_t>(),
                                                n_index};
      SetIndexData(qu, index_data_span, p_fmat_device, nbins, row_stride, offsets);
    }
  /* For sparse DMatrix we have to store index of feature for each bin
     in index field to chose right offset. So offset is nullptr and index is not reduced */
  } else {
    common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
    SetIndexData(qu, index_data_span, p_fmat_device, nbins, row_stride, offsets);
  }
}

/*!
 * \brief Fill histogram with zeroes
 */
template<typename GradientSumT>
void InitHist(sycl::queue qu, GHistRowOneAPI<GradientSumT>& hist, size_t size) {
  qu.fill(hist.Begin(), xgboost::detail::GradientPairInternal<GradientSumT>(), size);
}
template void InitHist(sycl::queue qu, GHistRowOneAPI<float>& hist, size_t size);
template void InitHist(sycl::queue qu, GHistRowOneAPI<double>& hist, size_t size);

/*!
 * \brief Copy histogram from src to dst
 */
template<typename GradientSumT>
void CopyHist(sycl::queue qu,
              GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& src,
              size_t size) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.Data());
  const GradientSumT* psrc = reinterpret_cast<const GradientSumT*>(src.DataConst());

  qu.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::range<1>(2 * size), [=](sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc[i];
    });
  }).wait();  
}
template void CopyHist(sycl::queue qu,
                       GHistRowOneAPI<float>& dst, const GHistRowOneAPI<float>& src,
                       size_t size);
template void CopyHist(sycl::queue qu,
                       GHistRowOneAPI<double>& dst, const GHistRowOneAPI<double>& src,
                       size_t size);

/*!
 * \brief Compute Subtraction: dst = src1 - src2
 */
template<typename GradientSumT>
void SubtractionHist(sycl::queue qu,
                     GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& src1,
                     const GHistRowOneAPI<GradientSumT>& src2,
                     size_t size) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.Data());
  const GradientSumT* psrc1 = reinterpret_cast<const GradientSumT*>(src1.DataConst());
  const GradientSumT* psrc2 = reinterpret_cast<const GradientSumT*>(src2.DataConst());

  qu.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::range<1>(2 * size), [=](sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc1[i] - psrc2[i];
    });
  }).wait();
}
template void SubtractionHist(sycl::queue qu,
                              GHistRowOneAPI<float>& dst, const GHistRowOneAPI<float>& src1,
                              const GHistRowOneAPI<float>& src2,
                              size_t size);
template void SubtractionHist(sycl::queue qu,
                              GHistRowOneAPI<double>& dst, const GHistRowOneAPI<double>& src1,
                              const GHistRowOneAPI<double>& src2,
                              size_t size);

template<typename FPType, typename BinIdxType>
void BuildHistDenseKernel(sycl::queue qu,
                          const std::vector<GradientPair>& gpair,
                          const USMVector<GradientPair>& gpair_device,
                          const RowSetCollectionOneAPI::Elem& row_indices,
                          const GHistIndexMatrixOneAPI& gmat,
                          const size_t n_features,
                          GHistRowOneAPI<FPType>& hist,
                          GHistRowOneAPI<FPType>& hist_buffer) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair_device.DataConst());
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  FPType* hist_data = reinterpret_cast<FPType*>(hist.Data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nbins = gmat.nbins;

  const size_t max_nblocks = hist_buffer.Size() / (nbins * two);
  const size_t min_block_size = 128;
  const size_t blocks_local = 1;
  const size_t max_feat_local = qu.get_device().get_info<sycl::info::device::max_work_group_size>();
  const size_t feat_local = n_features < max_feat_local ? n_features : max_feat_local;
  size_t nblocks = std::min(max_nblocks, size / min_block_size + !!(size % min_block_size));
  if (nblocks % blocks_local != 0) nblocks += blocks_local - nblocks % blocks_local;
  const size_t block_size = size / nblocks + !!(size % nblocks);
  FPType* hist_buffer_data = reinterpret_cast<FPType*>(hist_buffer.Data());

  qu.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::nd_range<2>(sycl::range<2>(nblocks, feat_local), sycl::range<2>(blocks_local, feat_local)), [=](sycl::nd_item<2> pid) {
      size_t block = pid.get_global_id(0);
      size_t feat = pid.get_global_id(1);

      FPType* hist_local = hist_buffer_data + block * nbins * two;

      for (size_t j = feat; j < 2 * nbins; j += feat_local) {
        hist_local[j] = 0.0f;
      }

      pid.barrier(sycl::access::fence_space::local_space);

      for (size_t i = block; i < size; i += nblocks) {
        const size_t icol_start = n_features * rid[i];
        const size_t idx_gh = two * rid[i];

        const BinIdxType* gr_index_local = gradient_index + icol_start;

        for (size_t j = feat; j < n_features; j += feat_local) {
          const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      offsets[j]);
          hist_local[idx_bin]   += pgh[idx_gh];
          hist_local[idx_bin+1] += pgh[idx_gh+1];
        }
      }
    });
  }).wait();

  qu.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::range<1>(nbins), [=](sycl::item<1> pid) {
      size_t i = pid.get_id(0);
      size_t j = pid.get_id(0);

      const size_t idx_bin = two * i;

      FPType gsum = 0.0f;
      FPType hsum = 0.0f;

      for (size_t j = 0; j < nblocks; ++j) {
        gsum += hist_buffer_data[j * nbins * two + idx_bin];
        hsum += hist_buffer_data[j * nbins * two + idx_bin + 1];
      }

      hist_data[idx_bin] = gsum;
      hist_data[idx_bin + 1] = hsum;
    });
  }).wait();
}

template<typename FPType>
void BuildHistSparseKernel(sycl::queue qu,
                           const std::vector<GradientPair>& gpair,
                           const USMVector<GradientPair>& gpair_device,
                           const RowSetCollectionOneAPI::Elem& row_indices,
                           const GHistIndexMatrixOneAPI& gmat,
                           GHistRowOneAPI<FPType>& hist,
                           GHistRowOneAPI<FPType>& hist_buffer) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const size_t row_stride = gmat.row_stride;
  const float* pgh = reinterpret_cast<const float*>(gpair_device.DataConst());
  const uint32_t* gradient_index = gmat.index.data<uint32_t>();
  FPType* hist_data = reinterpret_cast<FPType*>(hist.Data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  const size_t nbins = gmat.nbins;

  const size_t max_nblocks = hist_buffer.Size() / (nbins * two);
  const size_t min_block_size = 128;
  const size_t nblocks = std::min(max_nblocks, size / min_block_size + !!(size % min_block_size));
  const size_t block_size = size / nblocks + !!(size % nblocks);

  FPType* hist_buffer_data = reinterpret_cast<FPType*>(hist_buffer.Data());

  qu.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::range<2>(nblocks, nbins), [=](sycl::item<2> pid) {
      size_t i = pid.get_id(0);
      size_t j = pid.get_id(1);
      hist_buffer_data[two * (i * nbins + j)] = 0.0f;
      hist_buffer_data[two * (i * nbins + j) + 1] = 0.0f;
    });
  }).wait();

  const size_t max_feat_local = qu.get_device().get_info<sycl::info::device::max_work_group_size>();
  const size_t feat_local = row_stride < max_feat_local ? row_stride : max_feat_local;

  qu.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::nd_range<2>(sycl::range<2>(nblocks, feat_local),
                                         sycl::range<2>(1, feat_local)), [=](sycl::nd_item<2> pid) {
      size_t block = pid.get_global_id(0);
      size_t col_id = pid.get_global_id(1);

      size_t start = block * block_size;
      size_t end = (block + 1) * block_size;
      if (end > size) {
        end = size;
      }

      FPType* hist_local = hist_buffer_data + block * nbins * two;

      for (size_t i = start; i < end; ++i) {
        const size_t icol_start = row_stride * rid[i];
        const size_t idx_gh = two * rid[i];

        pid.barrier(sycl::access::fence_space::local_space);

        for (size_t j = col_id; j < row_stride; j += feat_local) {
          uint32_t idx_bin = gradient_index[icol_start + j];
          if (idx_bin < nbins) {
            idx_bin *= 2;
            hist_local[idx_bin]     += pgh[idx_gh];
            hist_local[idx_bin + 1] += pgh[idx_gh + 1];
          }
        }
      }
    });
  }).wait();

  qu.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<>(sycl::range<1>(nbins), [=](sycl::item<1> pid) {
      const size_t i = pid.get_id(0);

      const size_t idx_bin = two * i;

      FPType gsum = 0.0f;
      FPType hsum = 0.0f;

      for (size_t j = 0; j < nblocks; j++) {
        gsum += hist_buffer_data[j * nbins * two + idx_bin];
        hsum += hist_buffer_data[j * nbins * two + idx_bin + 1];
      }

      hist_data[idx_bin] = gsum;
      hist_data[idx_bin + 1] = hsum;
    });
  }).wait();
}

template<typename FPType, typename BinIdxType>
void BuildHistDispatchKernel(sycl::queue qu,
                             const std::vector<GradientPair>& gpair,
                             const USMVector<GradientPair>& gpair_device,
                             const RowSetCollectionOneAPI::Elem& row_indices,
                             const GHistIndexMatrixOneAPI& gmat, GHistRowOneAPI<FPType>& hist, bool isDense,
                             GHistRowOneAPI<FPType>& hist_buffer) {
  if (isDense) {
    BuildHistDenseKernel<FPType, BinIdxType>(qu, gpair, gpair_device, row_indices,
                                             gmat, gmat.nfeatures, hist, hist_buffer);
  } else {
    BuildHistSparseKernel<FPType>(qu, gpair, gpair_device, row_indices,
                                  gmat, hist, hist_buffer);
  }
}

template<typename FPType>
void BuildHistKernel(sycl::queue qu,
                     const std::vector<GradientPair>& gpair,
                     const USMVector<GradientPair>& gpair_device,
                     const RowSetCollectionOneAPI::Elem& row_indices,
                     const GHistIndexMatrixOneAPI& gmat, const bool isDense, GHistRowOneAPI<FPType>& hist,
                     GHistRowOneAPI<FPType>& hist_buffer) {
  const bool is_dense = isDense;
  switch (gmat.index.GetBinTypeSize()) {
    case kUint8BinsTypeSize:
      BuildHistDispatchKernel<FPType, uint8_t>(qu, gpair, gpair_device, row_indices,
                                               gmat, hist, is_dense, hist_buffer);
      break;
    case kUint16BinsTypeSize:
      BuildHistDispatchKernel<FPType, uint16_t>(qu, gpair, gpair_device, row_indices,
                                                gmat, hist, is_dense, hist_buffer);
      break;
    case kUint32BinsTypeSize:
      BuildHistDispatchKernel<FPType, uint32_t>(qu, gpair, gpair_device, row_indices,
                                                gmat, hist, is_dense, hist_buffer);
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

template <typename GradientSumT>
void GHistBuilderOneAPI<GradientSumT>::BuildHist(const std::vector<GradientPair> &gpair,
                                                 const USMVector<GradientPair>& gpair_device,
                                                 const RowSetCollectionOneAPI::Elem& row_indices,
                                                 const GHistIndexMatrixOneAPI &gmat,
                                                 GHistRowT& hist,
                                                 bool isDense,
                                                 GHistRowT& hist_buffer) {
  BuildHistKernel<GradientSumT>(qu_, gpair, gpair_device, row_indices, gmat, isDense, hist, hist_buffer);
}

template
void GHistBuilderOneAPI<float>::BuildHist(const std::vector<GradientPair>& gpair,
                                          const USMVector<GradientPair>& gpair_device,
                                          const RowSetCollectionOneAPI::Elem& row_indices,
                                          const GHistIndexMatrixOneAPI& gmat,
                                          GHistRowOneAPI<float>& hist,
                                          bool isDense,
                                          GHistRowOneAPI<float>& hist_buffer);
template
void GHistBuilderOneAPI<double>::BuildHist(const std::vector<GradientPair>& gpair,
                                           const USMVector<GradientPair>& gpair_device,
                                           const RowSetCollectionOneAPI::Elem& row_indices,
                                           const GHistIndexMatrixOneAPI& gmat,
                                           GHistRowOneAPI<double>& hist,
                                           bool isDense,
                                           GHistRowOneAPI<double>& hist_buffer);

template<typename GradientSumT>
void GHistBuilderOneAPI<GradientSumT>::SubtractionTrick(GHistRowT& self,
                                                        GHistRowT& sibling,
                                                        GHistRowT& parent) {
  const size_t size = self.Size();
  CHECK_EQ(sibling.Size(), size);
  CHECK_EQ(parent.Size(), size);

  SubtractionHist(qu_, self, parent, sibling, size);
}
template
void GHistBuilderOneAPI<float>::SubtractionTrick(GHistRowOneAPI<float>& self,
                                                 GHistRowOneAPI<float>& sibling,
                                                 GHistRowOneAPI<float>& parent);
template
void GHistBuilderOneAPI<double>::SubtractionTrick(GHistRowOneAPI<double>& self,
                                                  GHistRowOneAPI<double>& sibling,
                                                  GHistRowOneAPI<double>& parent);
}  // namespace common
}  // namespace xgboost