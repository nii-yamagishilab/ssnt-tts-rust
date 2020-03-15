#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/allocator.h"


extern "C" bool ssnt_order_beam_branch(const int *final_branch,
                                       const int *beam_branch,
                                       int batch_size,
                                       int beam_width,
                                       int max_t,
                                       int *ordered_beam_branch);

REGISTER_OP("SSNTOrderBeamBranch")
        .Input("final_branch: int32")
        .Input("beam_branch: int32")
        .Attr("beam_width: int")
        .Output("ordered_beam_branch: int32");

namespace tf = tensorflow;

namespace ssnt {

    class SSNTSSNTOrderBeamBranchOpCPU : public tf::OpKernel {
    public:
        explicit SSNTSSNTOrderBeamBranchOpCPU(tf::OpKernelConstruction *ctx) : tf::OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
        }

        void Compute(tf::OpKernelContext *ctx) override {
            // Grab the input tensors
            const tf::Tensor *final_branch;
            const tf::Tensor *beam_branch;
            OP_REQUIRES_OK(ctx, ctx->input("final_branch", &final_branch));
            OP_REQUIRES_OK(ctx, ctx->input("beam_branch", &beam_branch));

            OP_REQUIRES(ctx, final_branch->shape().dims() == 2,
                        tf::errors::InvalidArgument("final_branch is not a 2D-Tensor"));
            OP_REQUIRES(ctx, beam_branch->shape().dims() == 2,
                        tf::errors::InvalidArgument("beam_branch is not a 3D-Tensor"));
            OP_REQUIRES(ctx, beam_branch->shape().dim_size(0) == final_branch->shape().dim_size(0),
                        tf::errors::InvalidArgument("Incompatible batch sizes"));
            OP_REQUIRES(ctx, beam_branch->shape().dim_size(2) == beam_width_ &&
                             beam_width_ == final_branch->shape().dim_size(1),
                        tf::errors::InvalidArgument("Incompatible beam widths"));

            // (B, T, W)
            const auto &beam_branch_shape = beam_branch->shape();
            const auto batch_size = beam_branch_shape.dim_size(0);
            const auto max_t = beam_branch_shape.dim_size(1);


            auto final_branch_t = final_branch->tensor<int32_t, 2>();
            auto beam_branch_t = beam_branch->tensor<int32_t, 3>();

            tf::Tensor *ordered_beam_branch = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("ordered_beam_branch",
                                                     tf::TensorShape({batch_size, beam_width_, max_t}),
                                                     &ordered_beam_branch));
            auto ordered_beam_branch_t = ordered_beam_branch->tensor<int32_t, 3>();

            ssnt_order_beam_branch(final_branch_t.data(),
                                   beam_branch_t.data(),
                                   batch_size,
                                   beam_width_,
                                   max_t,
                                   ordered_beam_branch_t.data());
        }

    private:
        int beam_width_;
    };

    REGISTER_KERNEL_BUILDER(Name("SSNTOrderBeamBranch").Device(::tensorflow::DEVICE_CPU),
                            SSNTSSNTOrderBeamBranchOpCPU);

}
