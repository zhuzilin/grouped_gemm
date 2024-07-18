#include <torch/extension.h>

namespace grouped_gemm {

void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b);

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12500
void GroupedGemmDev(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b);
#endif

}  // namespace grouped_gemm
