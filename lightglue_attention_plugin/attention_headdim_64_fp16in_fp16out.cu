//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include <stdio.h>
#include <cassert>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include "./attention_headdim_64_fp16in_fp16out.cuh"


using namespace cute;
using namespace torch::indexing;

namespace AttentionHeadDim64 {

__global__ void attention_kernel_headdim_64_no_remainder_fp16in_fp16out(
    const FP16* Qptr, 
    const FP16* Kptr, 
    const FP16* Vptr, 
    FP16* Optr, 
    int batch, 
    int num_heads, 
    int num_Q_padded, 
    int num_KV_padded, 
    int num_Q_real, 
    int num_KV_real
);


struct KernelConfigHeadDim64_FP16IN_FP16OUT {
    static constexpr int NUM_THREADS_PER_BLOCK = 128;
    static constexpr int BLOCK_SEQ_LEN_Q       =  64;
    static constexpr int BLOCK_SEQ_LEN_K       =  64;
    static constexpr int BLOCK_SEQ_LEN_V       =  64;
    static constexpr int BLOCK_SEQ_LEN_KV      =  64;
    static constexpr int HEAD_DIM              =  64;
    static constexpr int TC_SIZE               =  64;
    static constexpr int NUM_STAGES            =   2;

    using SmemLayoutAtomQKVOWithSwizzle = decltype(
        composition(
            Swizzle<3, 3, 3>{},
            make_layout(
                make_shape(
                    Int<64>{}, 
                    Int<64>{}
                ),
                make_stride(
                    Int<64>{}, 
                    Int<1>{}
                )
            )
        )
    );
    using SmemLayoutQ = decltype(
        tile_to_shape(
            SmemLayoutAtomQKVOWithSwizzle{},
            make_shape(
                Int<BLOCK_SEQ_LEN_Q>{},
                Int<TC_SIZE>{}
            )
        )
    );
    using SmemLayoutK = decltype(
        tile_to_shape(
            SmemLayoutAtomQKVOWithSwizzle{},
            make_shape(
                Int<BLOCK_SEQ_LEN_K>{},
                Int<TC_SIZE>{},
                Int<NUM_STAGES>{}
            )
        )
    );
    using SmemLayoutV = decltype(
        tile_to_shape(
            SmemLayoutAtomQKVOWithSwizzle{},
            make_shape(
                Int<BLOCK_SEQ_LEN_V>{}, 
                Int<TC_SIZE>{},
                Int<NUM_STAGES>{}
            )
        )
    );
    using SmemLayoutO = decltype(
        tile_to_shape(
            SmemLayoutAtomQKVOWithSwizzle{},
            make_shape(
                Int<BLOCK_SEQ_LEN_Q>{},
                Int<TC_SIZE>{}
            )
        )
    );

    using SmemLayoutAtomONoSwizzle = Layout<Shape<Int<64>, Int<64>>, Stride< Int<64>, _1>>;
    using SmemLayoutONoSwizzle = decltype(
        tile_to_shape(
            SmemLayoutAtomONoSwizzle{},
            make_shape(
                Int<BLOCK_SEQ_LEN_Q>{},
                Int<TC_SIZE>{}
            )
        )
    );

    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<64>, Int<64>>, Stride<_1, Int<64>>>;
    using SmemLayoutAtomVtransposedWithSwizzle = decltype(
        composition(
            Swizzle<3, 3, 3>{}, 
            SmemLayoutAtomVtransposedNoSwizzle{}
        )
    );
    using SmemLayoutVtransposedNoSwizzle = decltype(
        tile_to_shape(
            SmemLayoutAtomVtransposedNoSwizzle{},
            make_shape(
                Int<TC_SIZE>{}, 
                Int<BLOCK_SEQ_LEN_V>{},
                Int<NUM_STAGES>{}
            )
        )
    );
    using SmemLayoutVtransposedWithSwizzle = decltype(
        tile_to_shape(
            SmemLayoutAtomVtransposedWithSwizzle{},
            make_shape(
                Int<TC_SIZE>{}, 
                Int<BLOCK_SEQ_LEN_V>{},
                Int<NUM_STAGES>{}
            )
        )
    );

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, FP16>;
    using G2SCopy = decltype(
        make_tiled_copy(
            g2s_copy_atom{},
            make_layout(
                make_shape(
                    Int<BLOCK_SEQ_LEN_Q>{},
                    Int<NUM_THREADS_PER_BLOCK / BLOCK_SEQ_LEN_Q>{}
                ),
                make_stride(
                    Int<NUM_THREADS_PER_BLOCK / BLOCK_SEQ_LEN_Q>{},
                    Int<1>{}
                )
            ),
            make_layout(
                make_shape(
                    Int<1>{}, 
                    Int<TC_SIZE / (NUM_THREADS_PER_BLOCK / BLOCK_SEQ_LEN_Q)>{}
                )
            )
        )
    );
    using G2SCopyQ = G2SCopy;
    using G2SCopyK = G2SCopy;
    using G2SCopyV = G2SCopy;

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int L1RepeatM = 2;
    static constexpr int L1RepeatN = 2;
    static constexpr int L1RepeatK = 1;
    static constexpr int L2RepeatM = 1;
    static constexpr int L2RepeatN = 1;
    static constexpr int L2RepeatK = 1;
    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int L2TileM = L2RepeatM * L1RepeatM * get<0>(mma_atom_shape{});
    static constexpr int L2TileN = L2RepeatN * L1RepeatN * get<1>(mma_atom_shape{});
    static constexpr int L2TileK = L2RepeatK * L1RepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(
        make_layout(
            make_shape(
                Int<L1RepeatM>{}, 
                Int<L1RepeatN>{}, 
                Int<L1RepeatK>{}
            )
        )
    );
    using MMA_P_T = Tile<Int<L2TileM>, Int<L2TileN>, Int<L2TileK>>;
    using MMA = decltype(
        make_tiled_mma(
            mma_atom{}, 
            MMA_EU_RepeatT{}, 
            MMA_P_T{}
        )
    );

    using R2SCopyAtomO = Copy_Atom<UniversalCopy<cute::uint32_t>, FP16>;

    using S2GCopyAtomO = Copy_Atom<UniversalCopy<cute::uint128_t>, FP16>;
    using S2GCopyO = decltype(
        make_tiled_copy(
            S2GCopyAtomO{},
            make_layout(
                make_shape(
                    Int<BLOCK_SEQ_LEN_Q>{}, 
                    Int<NUM_THREADS_PER_BLOCK / BLOCK_SEQ_LEN_Q>{}
                ),
                make_stride(
                    Int<NUM_THREADS_PER_BLOCK / BLOCK_SEQ_LEN_Q>{}, 
                    Int<1>{}
                )
            ),
            make_layout(
                make_shape(
                    Int<1>{}, 
                    Int<TC_SIZE / (NUM_THREADS_PER_BLOCK / BLOCK_SEQ_LEN_Q)>{}
                )
            )
        )
    );

    static_assert(NUM_THREADS_PER_BLOCK == cute::size(MMA{}));

    static constexpr int shm_elem_size_Q = cute::cosize(SmemLayoutQ{});
    static constexpr int shm_byte_size_Q = shm_elem_size_Q * sizeof(FP16);
    static constexpr int shm_elem_size_K = cute::cosize(SmemLayoutK{});
    static constexpr int shm_byte_size_K = shm_elem_size_K * sizeof(FP16);
    static constexpr int shm_elem_size_V = cute::cosize(SmemLayoutV{});
    static constexpr int shm_byte_size_V = shm_elem_size_V * sizeof(FP16);
    static constexpr int shm_elem_size   = shm_elem_size_Q + shm_elem_size_K + shm_elem_size_V;
    static constexpr int shm_byte_size   = shm_byte_size_Q + shm_byte_size_K + shm_byte_size_V;

    static constexpr int get_num_threads_per_block() { return NUM_THREADS_PER_BLOCK; }
    static int get_num_blocks_per_grid(int batch, int num_heads, int num_q) {
        return batch * num_heads * ((num_q + BLOCK_SEQ_LEN_Q - 1) / BLOCK_SEQ_LEN_Q);
    }

    KernelConfigHeadDim64_FP16IN_FP16OUT() {
        cudaFuncSetAttribute(
            attention_kernel_headdim_64_no_remainder_fp16in_fp16out,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            shm_byte_size
        );
    }
};
static KernelConfigHeadDim64_FP16IN_FP16OUT kernel_config_headdim_64_fp16in_fp16out;


__global__ void attention_kernel_headdim_64_no_remainder_fp16in_fp16out(    // "no_remainder" means can not deal with sequence_length % 64 != 0
    const FP16* Qptr, 
    const FP16* Kptr, 
    const FP16* Vptr, 
    FP16* Optr, 
    int batch, 
    int num_heads, 
    int num_Q_padded, 
    int num_KV_padded, 
    int num_Q_real, 
    int num_KV_real
) {
    extern __shared__ uint8_t smem[];

    FP16* smem_Q_ptr_fp16 = reinterpret_cast<FP16*>(smem);
    FP16* smem_K_ptr_fp16 = reinterpret_cast<FP16*>(smem + cute::cosize(KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutQ{}) * sizeof(FP16));
    FP16* smem_V_ptr_fp16 = reinterpret_cast<FP16*>(smem + cute::cosize(KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutQ{}) * sizeof(FP16) + cute::cosize(KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutK{}) * sizeof(FP16));

    const int  batch_idx = (blockIdx.x / (num_Q_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q)) / num_heads;
    const int  head_idx  = (blockIdx.x / (num_Q_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q)) % num_heads;
    const int  block_idx = (blockIdx.x % (num_Q_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q));

    const bool this_block_is_full = \
    !(block_idx == (((num_Q_padded + KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q - 1) / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q) - 1)) \
    || ((num_Q_real % KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q) == 0);

    auto gmem_Q_whole_fp16 = make_tensor(
        make_gmem_ptr(Qptr + (num_Q_padded * KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM) * (num_heads * batch_idx + head_idx)), 
        make_shape(num_Q_padded, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}), 
        make_stride(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}, Int<1>{})
    );
    auto gmem_Q_tile_fp16  = local_tile(gmem_Q_whole_fp16, make_tile(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q>{}, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::TC_SIZE>{}), make_coord(block_idx, 0));

    auto gmem_K_whole_fp16 = make_tensor(
        make_gmem_ptr(Kptr + (num_KV_padded * KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM) * (num_heads * batch_idx + head_idx)), 
        make_shape(num_KV_padded, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}), 
        make_stride(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}, Int<1>{})
    );
    auto gmem_K_tile_fp16  = local_tile(gmem_K_whole_fp16, make_tile(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV>{}, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::TC_SIZE>{}), make_coord(_, 0));

    auto gmem_V_whole_fp16 = make_tensor(
        make_gmem_ptr(Vptr + (num_KV_padded * KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM) * (num_heads * batch_idx + head_idx)), 
        make_shape(num_KV_padded, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}), 
        make_stride(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}, Int<1>{})
    );
    auto gmem_V_tile_fp16  = local_tile(gmem_V_whole_fp16, make_tile(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV>{}, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::TC_SIZE>{}), make_coord(_, 0));

    auto gmem_O_whole_fp16 = make_tensor(
        make_gmem_ptr(Optr + (num_Q_padded * KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM) * (num_heads * batch_idx + head_idx)), 
        make_shape(num_Q_padded, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}), 
        make_stride(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::HEAD_DIM>{}, Int<1>{})
    );
    auto gmem_O_tile_fp16  = local_tile(gmem_O_whole_fp16, make_tile(Int<KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q>{}, Int<KernelConfigHeadDim64_FP16IN_FP16OUT::TC_SIZE>{}), make_coord(block_idx, 0));

    auto smem_Q_tile_fp16             = make_tensor(make_smem_ptr(smem_Q_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutQ{});
    auto smem_K_tile_fp16             = make_tensor(make_smem_ptr(smem_K_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutK{});
    auto smem_V_tile_fp16             = make_tensor(make_smem_ptr(smem_V_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutV{});
    auto smem_VtNoSwizzle_tile_fp16   = make_tensor(make_smem_ptr(smem_V_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutVtransposedNoSwizzle{});
    auto smem_VtWithSwizzle_tile_fp16 = make_tensor(make_smem_ptr(smem_V_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutVtransposedWithSwizzle{});
    auto smem_S_tile_fp16             = make_tensor(make_smem_ptr(smem_K_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutK{});
    auto smem_P_tile_fp16             = make_tensor(make_smem_ptr(smem_K_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutK{});
    auto smem_O_tile_fp16             = make_tensor(make_smem_ptr(smem_Q_ptr_fp16), KernelConfigHeadDim64_FP16IN_FP16OUT::SmemLayoutO{});

    KernelConfigHeadDim64_FP16IN_FP16OUT::G2SCopyQ g2s_tiled_copy_Q;
    auto g2s_thr_copy_Q = g2s_tiled_copy_Q.get_slice(threadIdx.x);
    auto g2s_copy_gmem_Q_tile_fp16_view = g2s_thr_copy_Q.partition_S(gmem_Q_tile_fp16);
    auto g2s_copy_smem_Q_tile_fp16_view = g2s_thr_copy_Q.partition_D(smem_Q_tile_fp16);

    KernelConfigHeadDim64_FP16IN_FP16OUT::G2SCopyK g2s_tiled_copy_K;
    auto g2s_thr_copy_K = g2s_tiled_copy_K.get_slice(threadIdx.x);
    auto g2s_copy_gmem_K_tile_fp16_view = g2s_thr_copy_K.partition_S(gmem_K_tile_fp16);
    auto g2s_copy_smem_K_tile_fp16_view = g2s_thr_copy_K.partition_D(smem_K_tile_fp16);

    KernelConfigHeadDim64_FP16IN_FP16OUT::G2SCopyV g2s_tiled_copy_V;
    auto g2s_thr_copy_V = g2s_tiled_copy_V.get_slice(threadIdx.x);
    auto g2s_copy_gmem_V_tile_fp16_view = g2s_thr_copy_K.partition_S(gmem_V_tile_fp16);
    auto g2s_copy_smem_V_tile_fp16_view = g2s_thr_copy_K.partition_D(smem_V_tile_fp16);

    KernelConfigHeadDim64_FP16IN_FP16OUT::MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto mma_smem_S_tile_fp16_view = thr_mma.partition_C(smem_S_tile_fp16);
    auto mma_rmem_Q_tile_fp16_frag = thr_mma.partition_fragment_A(smem_Q_tile_fp16);
    auto mma_rmem_K_tile_fp16_frag = thr_mma.partition_fragment_B(smem_K_tile_fp16(_, _, 0));
    auto mma_rmem_V_tile_fp16_frag = thr_mma.partition_fragment_B(smem_VtNoSwizzle_tile_fp16(_, _, 0));
    auto mma_rmem_S_tile_fp16_frag = thr_mma.partition_fragment_C(smem_S_tile_fp16(_, _, 0));
    auto mma_rmem_P_tile_fp16_frag = thr_mma.partition_fragment_A(smem_P_tile_fp16(_, _, 0));
    auto mma_rmem_O_tile_fp16_frag = thr_mma.partition_fragment_C(smem_O_tile_fp16);

    // DefaultCopy       : yes
    // SM75_U32x1_LDSM_N :  no
    // SM75_U32x2_LDSM_N :  no
    // SM75_U32x4_LDSM_N : yes
    // SM75_U16x2_LDSM_T :  no
    // SM75_U16x4_LDSM_T :  no
    // SM75_U16x8_LDSM_T :  no
    auto s2r_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, FP16>{}, tiled_mma);
    auto s2r_thr_copy_Q   = s2r_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto s2r_smem_Q_tile_fp16_view  = s2r_thr_copy_Q.partition_S(smem_Q_tile_fp16);
    // DefaultCopy       : yes
    // SM75_U32x1_LDSM_N : yes
    // SM75_U32x2_LDSM_N : yes
    // SM75_U32x4_LDSM_N :  no
    // SM75_U16x2_LDSM_T :  no
    // SM75_U16x4_LDSM_T :  no
    // SM75_U16x8_LDSM_T :  no
    auto s2r_tiled_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, FP16>{}, tiled_mma);
    auto s2r_thr_copy_K   = s2r_tiled_copy_K.get_thread_slice(threadIdx.x);
    auto s2r_smem_K_tile_fp16_view  = s2r_thr_copy_K.partition_S(smem_K_tile_fp16);
    // DefaultCopy       : yes
    // SM75_U32x1_LDSM_N :  no
    // SM75_U32x2_LDSM_N :  no
    // SM75_U32x4_LDSM_N :  no
    // SM75_U16x2_LDSM_T : yes
    // SM75_U16x4_LDSM_T : yes
    // SM75_U16x8_LDSM_T :  no
    auto s2r_tiled_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, FP16>{}, tiled_mma);
    auto s2r_thr_copy_V   = s2r_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto s2r_smem_V_tile_fp16_view  = s2r_thr_copy_V.partition_S(smem_VtWithSwizzle_tile_fp16);
    // DefaultCopy       : yes
    // SM75_U32x1_LDSM_N :  no
    // SM75_U32x2_LDSM_N :  no
    // SM75_U32x4_LDSM_N : yes
    // SM75_U16x2_LDSM_T :  no
    // SM75_U16x4_LDSM_T :  no
    // SM75_U16x8_LDSM_T :  no
    auto s2r_tiled_copy_P = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, FP16>{}, tiled_mma);
    auto s2r_thr_copy_P   = s2r_tiled_copy_P.get_thread_slice(threadIdx.x);
    auto s2r_smem_P_tile_fp16_view  = s2r_thr_copy_P.partition_S(smem_P_tile_fp16);

    __shared__ FP16 smem_row_max[2][KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q]; // 2是因为横向有2个warp，BLOCK_SEQ_LEN_Q是因为纵向有BLOCK_SEQ_LEN_Q个元素
    __shared__ FP16 smem_row_sum[2][KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q]; // 2是因为横向有2个warp，BLOCK_SEQ_LEN_Q是因为纵向有BLOCK_SEQ_LEN_Q个元素
    __shared__ FP16 smem_row_max_prev[KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q];
    __shared__ FP16 smem_l[KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q];
    if (threadIdx.x < KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q) {
        smem_row_max_prev[threadIdx.x] = __float2half(-INFINITY);
        smem_l[threadIdx.x] = __float2half(0.0f);
    }

    const int warp_id = threadIdx.x / 32;
    const int thread_id_in_warp = threadIdx.x % 32;
    const int warp_id_row = warp_id % 2; // 2 is L1RepeatM along M
    const int warp_id_col = warp_id / 2; // 2 is L1RepeatM along M

    clear(mma_rmem_S_tile_fp16_frag);

    #pragma unroll
    for (int ver_idx = 0; ver_idx < (num_KV_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV); ver_idx++) {
        clear(mma_rmem_S_tile_fp16_frag);
        
        if (ver_idx == 0) {
            cute::copy(
                g2s_tiled_copy_Q, 
                g2s_copy_gmem_Q_tile_fp16_view, 
                g2s_copy_smem_Q_tile_fp16_view
            );
            cp_async_fence();
            cute::copy(
                g2s_tiled_copy_K, 
                g2s_copy_gmem_K_tile_fp16_view(_, _, _, ver_idx),
                g2s_copy_smem_K_tile_fp16_view(_, _, _, (ver_idx % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES))
            );
            cp_async_fence();
            cute::copy(
                g2s_tiled_copy_V, 
                g2s_copy_gmem_V_tile_fp16_view(_, _, _, ver_idx),
                g2s_copy_smem_V_tile_fp16_view(_, _, _, (ver_idx % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES))
            );
            cp_async_fence();
        }
        cp_async_wait<1>();
        __syncthreads();
        if (ver_idx < ((num_KV_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV) - 1)) {
            cute::copy(
                g2s_tiled_copy_K, 
                g2s_copy_gmem_K_tile_fp16_view(_, _, _, (ver_idx + 1)),
                g2s_copy_smem_K_tile_fp16_view(_, _, _, ((ver_idx + 1) % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES))
            );
            cp_async_fence();
        }
        if (ver_idx == 0) {
            cute::copy(
                s2r_tiled_copy_Q, 
                s2r_smem_Q_tile_fp16_view, 
                mma_rmem_Q_tile_fp16_frag
            );
        }
        cute::copy(
            s2r_tiled_copy_K, 
            s2r_smem_K_tile_fp16_view(_, _, _, (ver_idx % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES)), 
            mma_rmem_K_tile_fp16_frag
        );
        cute::gemm(
            tiled_mma, 
            mma_rmem_S_tile_fp16_frag, 
            mma_rmem_Q_tile_fp16_frag, 
            mma_rmem_K_tile_fp16_frag, 
            mma_rmem_S_tile_fp16_frag
        );

        if (ver_idx < ((num_KV_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV) - 1)) {
            cute::copy(
                g2s_tiled_copy_V, 
                g2s_copy_gmem_V_tile_fp16_view(_, _, _, (ver_idx + 1)),
                g2s_copy_smem_V_tile_fp16_view(_, _, _, ((ver_idx + 1) % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES))
            );
            cp_async_fence();
        }

        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            #pragma unroll
            for (int j = 0; j < cute::size<2>(mma_rmem_S_tile_fp16_frag); j++) {
                #pragma unroll
                for (int k = 0; k < cute::size<0>(mma_rmem_S_tile_fp16_frag); k += 2) {
                    (*(reinterpret_cast<__half2*>(&(mma_rmem_S_tile_fp16_frag(k, i, j))))) = __hmul2(
                        __float2half2_rn(0.125f),
                        (*(reinterpret_cast<__half2*>(&(mma_rmem_S_tile_fp16_frag(k, i, j)))))
                    );
                }
            }
        }

        if ((ver_idx == ((num_KV_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV) - 1)) && ((num_KV_real % KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV) != 0)) {
            #pragma unroll
            for (int idx_0 = 0; idx_0 < cute::size<0>(mma_rmem_S_tile_fp16_frag); idx_0++) {
                #pragma unroll
                for (int idx_1 = 0; idx_1 < cute::size<1>(mma_rmem_S_tile_fp16_frag); idx_1++) {
                    #pragma unroll
                    for (int idx_2 = 0; idx_2 < cute::size<2>(mma_rmem_S_tile_fp16_frag); idx_2++) {
                        int row_id_in_S = idx_1 * (16 * KernelConfigHeadDim64_FP16IN_FP16OUT::L1RepeatM) + (warp_id_row * 16) + (idx_0 / 2) * 8 + (thread_id_in_warp / 4);
                        int col_id_in_S = idx_2 * ( 8 * KernelConfigHeadDim64_FP16IN_FP16OUT::L1RepeatN) + (warp_id_col *  8) + (thread_id_in_warp % 4) * 2 + (idx_0 % 2);
                        bool set_to_neg_inf = col_id_in_S >= (num_KV_real % KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV);
                        if (!this_block_is_full) {
                            set_to_neg_inf = set_to_neg_inf && (row_id_in_S < (num_Q_real % KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q));
                        }
                        if (set_to_neg_inf) {
                            mma_rmem_S_tile_fp16_frag(idx_0, idx_1, idx_2) = __float2half(-INFINITY);
                        }
                    }
                }
            }
        }

        // 2是因为：首维的4个元素分布在两行上，以threadIdx == 0为例，前2个元素在row_0，后2个元素在row_8，
        FP16 thread_row_sum[cute::size<1>(mma_rmem_S_tile_fp16_frag)][2];
        FP16 thread_row_max[cute::size<1>(mma_rmem_S_tile_fp16_frag)][2];
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            (*(reinterpret_cast<__half2*>(&(thread_row_max[i][0])))) = __float2half2_rn(-INFINITY);
        }
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            (*(reinterpret_cast<__half2*>(&(thread_row_sum[i][0])))) = __float2half2_rn(0.0f);
        }
        // __syncthreads();

        // compute row_max : starts
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            #pragma unroll
            for (int j = 0; j < cute::size<2>(mma_rmem_S_tile_fp16_frag); j++) {
                (*(reinterpret_cast<__half2*>(&(thread_row_max[i][0])))) = __hmax2(
                    (*(reinterpret_cast<__half2*>(&thread_row_max[i][0]))),
                    __halves2half2(
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(0, i, j))))),
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(2, i, j)))))
                    )
                );
                (*(reinterpret_cast<__half2*>(&thread_row_max[i][0]))) = __hmax2(
                    (*(reinterpret_cast<__half2*>(&thread_row_max[i][0]))),
                    __halves2half2(
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(1, i, j))))),
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(3, i, j)))))
                    )
                );
            }
        }
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            thread_row_max[i][0] = __hmax((*(reinterpret_cast<__half*>(&(thread_row_max[i][0])))), __half(__shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_max[i][0])))), 2, 4)));
            thread_row_max[i][1] = __hmax((*(reinterpret_cast<__half*>(&(thread_row_max[i][1])))), __half(__shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_max[i][1])))), 2, 4)));
            thread_row_max[i][0] = __hmax((*(reinterpret_cast<__half*>(&(thread_row_max[i][0])))), __half(__shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_max[i][0])))), 1, 4)));
            thread_row_max[i][1] = __hmax((*(reinterpret_cast<__half*>(&(thread_row_max[i][1])))), __half(__shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_max[i][1])))), 1, 4)));
        }
        // __syncthreads();
        if (threadIdx.x % 4 == 0) {
            #pragma unroll
            for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int row_id_in_tile = i * (16 * KernelConfigHeadDim64_FP16IN_FP16OUT::L1RepeatM) + (warp_id_row * 16) + j * 8 + (thread_id_in_warp / 4);
                    smem_row_max[warp_id_col][row_id_in_tile] = thread_row_max[i][j];
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q / 2) {
            (*(reinterpret_cast<__half2*>(&(smem_row_max[0][threadIdx.x * 2])))) = __hmax2(
                (*(reinterpret_cast<__half2*>(&(smem_row_max[0][threadIdx.x * 2])))),
                (*(reinterpret_cast<__half2*>(&(smem_row_max[1][threadIdx.x * 2]))))
            );
            (*(reinterpret_cast<__half2*>(&(smem_row_max[0][threadIdx.x * 2])))) = __hmax2(
                (*(reinterpret_cast<__half2*>(&(smem_row_max[0][threadIdx.x * 2])))),
                (*(reinterpret_cast<__half2*>(&(smem_row_max_prev[threadIdx.x * 2]))))
            );
        }
        __syncthreads();
        // compute row_max : ends
        // to this point : row_max is stored in smem_row_max

        // compute exp(elem - row_max) : starts
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            int row_id_in_tile = i * (16 * KernelConfigHeadDim64_FP16IN_FP16OUT::L1RepeatM) + (warp_id_row * 16) + (thread_id_in_warp / 4);
            thread_row_max[i][0] = smem_row_max[0][row_id_in_tile];
            thread_row_max[i][1] = smem_row_max[0][row_id_in_tile + 8];
        }
        // __syncthreads();
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            __half2 thread_row_max_half2_0 = __half2half2(__half(thread_row_max[i][0]));
            __half2 thread_row_max_half2_1 = __half2half2(__half(thread_row_max[i][1]));
            #pragma unroll
            for (int j = 0; j < cute::size<2>(mma_rmem_S_tile_fp16_frag); j++) {
                (*(reinterpret_cast<__half2*>(&(mma_rmem_S_tile_fp16_frag(0, i, j))))) = h2exp(__hsub2((*(reinterpret_cast<__half2*>(&(mma_rmem_S_tile_fp16_frag(0, i, j))))), thread_row_max_half2_0));
                (*(reinterpret_cast<__half2*>(&(mma_rmem_S_tile_fp16_frag(2, i, j))))) = h2exp(__hsub2((*(reinterpret_cast<__half2*>(&(mma_rmem_S_tile_fp16_frag(2, i, j))))), thread_row_max_half2_1));
            }
        }
        // __syncthreads();
        // compute exp(elem - row_max) : ends
        // to this point : exp(elem - row_max) is stored in mma_rmem_S_tile_fp16_frag

        // compute row_sum : starts
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            #pragma unroll
            for (int j = 0; j < cute::size<2>(mma_rmem_S_tile_fp16_frag); j++) {
                (*(reinterpret_cast<__half2*>(&(thread_row_sum[i][0])))) = __hadd2(
                    (*(reinterpret_cast<__half2*>(&thread_row_sum[i][0]))),
                    __halves2half2(
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(0, i, j))))),
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(2, i, j)))))
                    )
                );
                (*(reinterpret_cast<__half2*>(&thread_row_sum[i][0]))) = __hadd2(
                    (*(reinterpret_cast<__half2*>(&thread_row_sum[i][0]))),
                    __halves2half2(
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(1, i, j))))),
                        (*(reinterpret_cast<__half*>(&(mma_rmem_S_tile_fp16_frag(3, i, j)))))
                    )
                );
            }
        }
        #pragma unroll
        for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
            thread_row_sum[i][0] = __hadd((*(reinterpret_cast<__half*>(&(thread_row_sum[i][0])))), __shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_sum[i][0])))), 2, 4));
            thread_row_sum[i][1] = __hadd((*(reinterpret_cast<__half*>(&(thread_row_sum[i][1])))), __shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_sum[i][1])))), 2, 4));
            thread_row_sum[i][0] = __hadd((*(reinterpret_cast<__half*>(&(thread_row_sum[i][0])))), __shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_sum[i][0])))), 1, 4));
            thread_row_sum[i][1] = __hadd((*(reinterpret_cast<__half*>(&(thread_row_sum[i][1])))), __shfl_xor_sync(0xffffffff, (*(reinterpret_cast<__half*>(&(thread_row_sum[i][1])))), 1, 4));
        }
        // __syncthreads();
        if (threadIdx.x % 4 == 0) {
            #pragma unroll
            for (int i = 0; i < cute::size<1>(mma_rmem_S_tile_fp16_frag); i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int row_id_in_tile = i * (16 * KernelConfigHeadDim64_FP16IN_FP16OUT::L1RepeatM) + (warp_id_row * 16) + j * 8 + (thread_id_in_warp / 4);
                    smem_row_sum[warp_id_col][row_id_in_tile] = thread_row_sum[i][j];
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q / 2) {
            __half2 temp = ver_idx == 0 ? __float2half2_rn(0.0f) : h2exp(
                __hsub2(
                    (*(reinterpret_cast<__half2*>(&(smem_row_max_prev[threadIdx.x * 2])))), 
                    (*(reinterpret_cast<__half2*>(&(smem_row_max[0][threadIdx.x * 2]))))
                )
            );
            (*(reinterpret_cast<__half2*>(&(smem_row_max_prev[threadIdx.x * 2])))) = (*(reinterpret_cast<__half2*>(&(smem_row_max[0][threadIdx.x * 2]))));
            (*(reinterpret_cast<__half2*>(&(smem_row_max[1][threadIdx.x * 2])))) = temp;
            (*(reinterpret_cast<__half2*>(&(smem_l[threadIdx.x * 2])))) = __hadd2(
                __hmul2(
                    temp, 
                    (*(reinterpret_cast<__half2*>(&(smem_l[threadIdx.x * 2]))))
                ),
                __hadd2(
                    (*(reinterpret_cast<__half2*>(&(smem_row_sum[0][threadIdx.x * 2])))), 
                    (*(reinterpret_cast<__half2*>(&(smem_row_sum[1][threadIdx.x * 2]))))
                )
            );
        }
        // __syncthreads();
        // compute row_sum : ends
        // to this point : 
        // smem_row_max[0][:]                      is now row_max aka m
        // smem_row_max[1][:]                      is now exp(row_max_prev - row_max), in first iter, it should be 0
        // smem_row_sum[0][:] + smem_row_sum[1][:] is now row_sum
        // smem_l[:]                               is now lij

        cute::copy(mma_rmem_S_tile_fp16_frag, mma_smem_S_tile_fp16_view(_, _, _, (ver_idx % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES)));
        __syncthreads();
        cute::copy(s2r_tiled_copy_P, s2r_smem_P_tile_fp16_view(_, _, _, (ver_idx % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES)), mma_rmem_P_tile_fp16_frag);

        // ----------------------------------------------------------------------------------------------------------------------------------------------------------------

        if (ver_idx < ((num_KV_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV) - 1)) {
            cp_async_wait<2>();
        } else if (ver_idx == ((num_KV_padded / KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_KV) - 1)) {
            cp_async_wait<0>();
        }
        __syncthreads();
        cute::copy(
            s2r_tiled_copy_V, 
            s2r_smem_V_tile_fp16_view(_, _, _, (ver_idx % KernelConfigHeadDim64_FP16IN_FP16OUT::NUM_STAGES)), 
            mma_rmem_V_tile_fp16_frag
        );

        #pragma unroll
        for (int idx_1 = 0; idx_1 < cute::size<1>(mma_rmem_O_tile_fp16_frag); idx_1++) {
            #pragma unroll
            for (int idx_0 = 0; idx_0 < cute::size<0>(mma_rmem_O_tile_fp16_frag); idx_0 += 2) {
                int row_id_in_S = idx_1 * (16 * KernelConfigHeadDim64_FP16IN_FP16OUT::L1RepeatM) + (warp_id_row * 16) + (idx_0 / 2) * 8 + (thread_id_in_warp / 4);
                FP16 coeff = smem_row_max[1][row_id_in_S];
                #pragma unroll
                for (int idx_2 = 0; idx_2 < cute::size<2>(mma_rmem_O_tile_fp16_frag); idx_2++) {
                    (*(reinterpret_cast<__half2*>(&(mma_rmem_O_tile_fp16_frag(idx_0, idx_1, idx_2))))) = __hmul2(
                        (*(reinterpret_cast<__half2*>(&(mma_rmem_O_tile_fp16_frag(idx_0, idx_1, idx_2))))), 
                        __half2half2(coeff)
                    );
                }
            }
        }

        cute::gemm(
            tiled_mma, 
            mma_rmem_O_tile_fp16_frag, 
            mma_rmem_P_tile_fp16_frag, 
            mma_rmem_V_tile_fp16_frag, 
            mma_rmem_O_tile_fp16_frag
        );
    }

    if (threadIdx.x < KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q) {
        if (this_block_is_full) {
            smem_l[threadIdx.x] = hrcp(smem_l[threadIdx.x]);
        } else {
            smem_l[threadIdx.x] = (threadIdx.x < (num_Q_real % KernelConfigHeadDim64_FP16IN_FP16OUT::BLOCK_SEQ_LEN_Q)) ? hrcp(smem_l[threadIdx.x]) : __float2half(0.0f);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int idx_1 = 0; idx_1 < cute::size<1>(mma_rmem_O_tile_fp16_frag); idx_1++) {
        #pragma unroll
        for (int idx_0 = 0; idx_0 < cute::size<0>(mma_rmem_O_tile_fp16_frag); idx_0 += 2) {
            int row_id_in_O = idx_1 * (16 * KernelConfigHeadDim64_FP16IN_FP16OUT::L1RepeatM) + (warp_id_row * 16) + (idx_0 / 2) * 8 + (thread_id_in_warp / 4);
            FP16 coeff = smem_l[row_id_in_O];
            #pragma unroll
            for (int idx_2 = 0; idx_2 < cute::size<2>(mma_rmem_O_tile_fp16_frag); idx_2++) {
                (*(reinterpret_cast<__half2*>(&(mma_rmem_O_tile_fp16_frag(idx_0, idx_1, idx_2))))) = __hmul2(
                    (*(reinterpret_cast<__half2*>(&(mma_rmem_O_tile_fp16_frag(idx_0, idx_1, idx_2))))), 
                    __half2half2(coeff)
                );
            }
        }
    }

    auto r2s_tiled_copy_O = make_tiled_copy_C(KernelConfigHeadDim64_FP16IN_FP16OUT::R2SCopyAtomO{}, tiled_mma);
    auto r2s_thr_copy_O = r2s_tiled_copy_O.get_slice(threadIdx.x);
    auto r2s_copy_rmem_O_tile_view = r2s_thr_copy_O.retile_S(mma_rmem_O_tile_fp16_frag);
    auto r2s_copy_smem_O_tile_view = r2s_thr_copy_O.partition_D(smem_O_tile_fp16);

    KernelConfigHeadDim64_FP16IN_FP16OUT::S2GCopyO s2g_tiled_copy_O;
    auto s2g_thr_copy_O = s2g_tiled_copy_O.get_thread_slice(threadIdx.x);
    auto s2g_copy_smem_O_tile_view = s2g_thr_copy_O.partition_S(smem_O_tile_fp16);
    auto s2g_copy_gmem_O_tile_view = s2g_thr_copy_O.partition_D(gmem_O_tile_fp16);

    cute::copy(r2s_tiled_copy_O, r2s_copy_rmem_O_tile_view, r2s_copy_smem_O_tile_view);
    __syncthreads();
    cute::copy(s2g_tiled_copy_O, s2g_copy_smem_O_tile_view, s2g_copy_gmem_O_tile_view);
}


__global__ void QKV_move_from_unpadded_to_padded_fp16(
    const FP16* Q_ptr_fp16, 
    const FP16* K_ptr_fp16, 
    const FP16* V_ptr_fp16, 
    FP16* Q_ptr_fp16_padded, 
    FP16* K_ptr_fp16_padded, 
    FP16* V_ptr_fp16_padded, 
    int batch, 
    int num_heads, 
    int num_Q_real, 
    int num_KV_real, 
    int num_Q_padded, 
    int num_KV_padded
) {
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    int idx_v = thread_idx / (64 / 8);  // 8 means : each thread move 8 elements
    int idx_h = thread_idx % (64 / 8);

    const FP16* src_ptr = nullptr;
    FP16* dst_ptr = nullptr;
    int offset_src = -1;
    int offset_dst = -1;
    int batch_idx = -1;
    int head_idx  = -1;
    int row_idx   = -1;
    if (0 <= idx_v && idx_v < batch * num_heads * num_Q_padded) {
        src_ptr = Q_ptr_fp16;
        dst_ptr = Q_ptr_fp16_padded;
        offset_dst = idx_v - 0;
        batch_idx = offset_dst / (num_heads * num_Q_padded);
        head_idx  = (offset_dst % (num_heads * num_Q_padded)) / num_Q_padded;
        row_idx   = offset_dst % num_Q_padded;
        offset_src = (row_idx < num_Q_real) ? (batch_idx * (num_heads * num_Q_real) + head_idx * num_Q_real + row_idx) : -1;
    } else if (batch * num_heads * num_Q_padded <= idx_v && idx_v < batch * num_heads * (num_Q_padded + num_KV_padded)) {
        src_ptr = K_ptr_fp16;
        dst_ptr = K_ptr_fp16_padded;
        offset_dst = idx_v - batch * num_heads * num_Q_padded;
        batch_idx = offset_dst / (num_heads * num_KV_padded);
        head_idx  = (offset_dst % (num_heads * num_KV_padded)) / num_KV_padded;
        row_idx   = offset_dst % num_KV_padded;
        offset_src = (row_idx < num_KV_real) ? (batch_idx * (num_heads * num_KV_real) + head_idx * num_KV_real + row_idx) : -1;
    } else if (batch * num_heads * (num_Q_padded + num_KV_padded) <= idx_v && idx_v < batch * num_heads * (num_Q_padded + num_KV_padded + num_KV_padded)) {
        src_ptr = V_ptr_fp16;
        dst_ptr = V_ptr_fp16_padded;
        offset_dst = idx_v - batch * num_heads * (num_Q_padded + num_KV_padded);
        batch_idx = offset_dst / (num_heads * num_KV_padded);
        head_idx  = (offset_dst % (num_heads * num_KV_padded)) / num_KV_padded;
        row_idx   = offset_dst % num_KV_padded;
        offset_src = (row_idx < num_KV_real) ? (batch_idx * (num_heads * num_KV_real) + head_idx * num_KV_real + row_idx) : -1;
    } else {
        return;
    }

    FP16x8 data = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
    if (offset_src != -1) {
        data = (*(reinterpret_cast<const FP16x8*>(src_ptr + (offset_src * 64) + (idx_h * 8))));
    }
    (*(reinterpret_cast<FP16x8*>(dst_ptr + (offset_dst * 64) + (idx_h * 8)))) = data;
}


void launch_QKV_move_from_unpadded_to_padded_fp16(
    const FP16* Q_ptr_fp16, 
    const FP16* K_ptr_fp16, 
    const FP16* V_ptr_fp16, 
    FP16* Q_ptr_fp16_padded, 
    FP16* K_ptr_fp16_padded, 
    FP16* V_ptr_fp16_padded, 
    int batch, 
    int num_heads, 
    int num_Q_real, 
    int num_KV_real, 
    int num_Q_padded, 
    int num_KV_padded, 
    cudaStream_t stream
) {
    int total_num_threads = (batch * num_heads * (num_Q_padded + num_KV_padded + num_KV_padded)) * (64 / 8);    // 8 means : each thread move 8 elements
    dim3 grid((total_num_threads + 256 - 1) / 256);
    dim3 block(256);

    QKV_move_from_unpadded_to_padded_fp16<<<grid, block, 0, stream>>>(
        Q_ptr_fp16, 
        K_ptr_fp16, 
        V_ptr_fp16, 
        Q_ptr_fp16_padded, 
        K_ptr_fp16_padded, 
        V_ptr_fp16_padded, 
        batch, 
        num_heads, 
        num_Q_real, 
        num_KV_real, 
        num_Q_padded, 
        num_KV_padded
    );
}


__global__ void O_move_from_padded_to_unpadded_fp16(
    const FP16* O_ptr_src, 
    FP16* O_ptr_dst, 
    int batch, 
    int num_heads, 
    int num_O_real, 
    int num_O_padded
) {
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    int idx_v = thread_idx / (64 / 8);  // 8 means : each thread move 8 elements
    int idx_h = thread_idx % (64 / 8);

    const FP16* src_ptr = O_ptr_src;
    FP16* dst_ptr = O_ptr_dst;
    int offset_dst = idx_v - 0;
    int batch_idx = offset_dst / (num_heads * num_O_real);
    int head_idx  = (offset_dst % (num_heads * num_O_real)) / num_O_real;
    int row_idx   = offset_dst % num_O_real;
    int offset_src = batch_idx * (num_heads * num_O_padded) + head_idx * num_O_padded + row_idx;

    (*(reinterpret_cast<      FP16x8*>(dst_ptr + (offset_dst * 64) + (idx_h * 8)))) = \
    (*(reinterpret_cast<const FP16x8*>(src_ptr + (offset_src * 64) + (idx_h * 8))));
}


void launch_O_move_from_padded_to_unpadded_fp16(
    const FP16* O_ptr_src, 
    FP16* O_ptr_dst, 
    int batch, 
    int num_heads, 
    int num_O_real, 
    int num_O_padded, 
    cudaStream_t stream
) {
    int total_num_threads = (batch * num_heads * num_O_real) * (64 / 8);    // 8 means : each thread move 8 elements
    dim3 grid((total_num_threads + 256 - 1) / 256);
    dim3 block(256);

    O_move_from_padded_to_unpadded_fp16<<<grid, block, 0, stream>>>(
        O_ptr_src, 
        O_ptr_dst, 
        batch, 
        num_heads, 
        num_O_real, 
        num_O_padded
    );
}


void launch_attention_kernel_headdim_64_no_remainder_fp16in_fp16out(
    const FP16* Q_ptr, 
    const FP16* K_ptr, 
    const FP16* V_ptr, 
    FP16* O_ptr, 
    int batch, 
    int num_heads, 
    int num_Q_padded, 
    int num_KV_padded, 
    int num_Q_real, 
    int num_KV_real, 
    cudaStream_t stream
) {
    dim3 block(KernelConfigHeadDim64_FP16IN_FP16OUT::get_num_threads_per_block());
    dim3 grid(KernelConfigHeadDim64_FP16IN_FP16OUT::get_num_blocks_per_grid(batch, num_heads, num_Q_padded));
    attention_kernel_headdim_64_no_remainder_fp16in_fp16out<<<grid, block, KernelConfigHeadDim64_FP16IN_FP16OUT::shm_byte_size, stream>>>(
        Q_ptr, 
        K_ptr, 
        V_ptr, 
        O_ptr, 
        batch, 
        num_heads, 
        num_Q_padded, 
        num_KV_padded, 
        num_Q_real, 
        num_KV_real
    );
}

}   // namespace AttentionHeadDim64


