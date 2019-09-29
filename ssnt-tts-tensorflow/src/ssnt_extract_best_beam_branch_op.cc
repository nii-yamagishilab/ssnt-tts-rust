#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/allocator.h"


extern "C" bool ssnt_extract_best_beam_branch(int best_final_branch, const int *beam_branch, const int *t_history,
                                              int beam_width, int max_u,
                                              int *best_beam_branch, int *best_t_history);


REGISTER_OP("SSNTExtractBestBeamBranch")
        .Input("best_final_branch: int32")
        .Input("beam_branch: int32")
        .Input("t_history: int32")
        .Attr("beam_width: int")
        .Output("best_beam_branch: int32")
        .Output("best_t_history: int32");

namespace tf = tensorflow;

namespace ssnt {

    class SSNTExtractBestBeamBranchOpCPU : public tf::OpKernel {
    public:
        explicit SSNTExtractBestBeamBranchOpCPU(tf::OpKernelConstruction *ctx) : tf::OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
        }

        void Compute(tf::OpKernelContext *ctx) override {
            // Grab the input tensors
            const tf::Tensor *best_final_branch;
            const tf::Tensor *beam_branch;
            const tf::Tensor *t_history;
            OP_REQUIRES_OK(ctx, ctx->input("best_final_branch", &best_final_branch));
            OP_REQUIRES_OK(ctx, ctx->input("beam_branch", &beam_branch));
            OP_REQUIRES_OK(ctx, ctx->input("t_history", &t_history));

            OP_REQUIRES(ctx, best_final_branch->shape().dims() == 0,
                        tf::errors::InvalidArgument("best_final_branch is not a 0D-Tensor"));
            OP_REQUIRES(ctx, beam_branch->shape().dims() == 2,
                        tf::errors::InvalidArgument("beam_branch is not a 2D-Tensor"));
            OP_REQUIRES(ctx, t_history->shape().dims() == 2,
                        tf::errors::InvalidArgument("t_history is not 2D-Tensor"));
            OP_REQUIRES(ctx, beam_branch->shape().dim_size(1) == beam_width_ &&
                        beam_width_ == t_history->shape().dim_size(1),
                        tf::errors::InvalidArgument("Incompatible beam widths"));

            const auto &beam_branch_shape = beam_branch->shape();
            const auto max_u = beam_branch_shape.dim_size(0);

            OP_REQUIRES(ctx, beam_branch->shape().num_elements() == max_u * beam_width_,
                        tf::errors::InvalidArgument("beam_branch has invalid size"));
            OP_REQUIRES(ctx, t_history->shape().num_elements() == max_u * beam_width_,
                        tf::errors::InvalidArgument("t_history has invalid size"));

            auto best_final_branch_t = best_final_branch->scalar<int32_t>();
            auto beam_branch_t = beam_branch->tensor<int32_t, 2>();
            auto t_history_t = t_history->tensor<int32_t, 2>();

            tf::Tensor *best_beam_branch = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("best_beam_branch", tf::TensorShape({max_u}), &best_beam_branch));
            auto best_beam_branch_t = best_beam_branch->vec<int32_t>();

            tf::Tensor *best_t_history = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("best_t_history", tf::TensorShape({max_u}), &best_t_history));
            auto best_t_history_t = best_t_history->vec<int32_t>();

            ssnt_extract_best_beam_branch(best_final_branch_t(),
                    beam_branch_t.data(),
                    t_history_t.data(),
                    beam_width_,
                    max_u,
                    best_beam_branch_t.data(),
                    best_t_history_t.data());
        }

    private:
        int beam_width_;
    };

    REGISTER_KERNEL_BUILDER(Name("SSNTExtractBestBeamBranch").Device(::tensorflow::DEVICE_CPU), SSNTExtractBestBeamBranchOpCPU);

}
