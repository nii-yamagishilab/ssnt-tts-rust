#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"


extern "C" bool ssnt_upsample_source_indexes(const int *duration,
                                             const int *output_length,
                                             int batch_size,
                                             int beam_width,
                                             int max_t,
                                             int max_u,
                                             int *upsampled_source_indexes);


REGISTER_OP("SSNTUpsampleSourceIndexes")
        .Input("duration: int32")
        .Input("output_length: int32")
        .Input("max_u: int32")
        .Input("out_of_range_source_index: int32")
        .Attr("beam_width: int")
        .Output("upsampled_source_indexes: int32");

namespace tf = tensorflow;

namespace ssnt {

    class SSNTUpsampleSourceIndexesOpCPU : public tf::OpKernel {
    public:
        explicit SSNTUpsampleSourceIndexesOpCPU(tf::OpKernelConstruction *ctx) : tf::OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
        }

        void Compute(tf::OpKernelContext *ctx) override {
            // Grab the input tensors
            const tf::Tensor *duration;
            const tf::Tensor *output_length;
            const tf::Tensor *max_u;
            const tf::Tensor *out_of_range_source_index;
            OP_REQUIRES_OK(ctx, ctx->input("duration", &duration));
            OP_REQUIRES_OK(ctx, ctx->input("output_length", &output_length));
            OP_REQUIRES_OK(ctx, ctx->input("max_u", &max_u));
            OP_REQUIRES_OK(ctx, ctx->input("out_of_range_source_index", &out_of_range_source_index));

            OP_REQUIRES(ctx, duration->shape().dims() == 3,
                        tf::errors::InvalidArgument("duration is not a 3D-Tensor"));
            OP_REQUIRES(ctx, output_length->shape().dims() == 2,
                        tf::errors::InvalidArgument("output_length is not a 2D-Tensor"));
            OP_REQUIRES(ctx, max_u->shape().dims() == 0,
                        tf::errors::InvalidArgument("max_u is not a 0D-Tensor"));
            OP_REQUIRES(ctx, out_of_range_source_index->shape().dims() == 0,
                        tf::errors::InvalidArgument("out_of_range_source_index is not a 0D-Tensor"));
            OP_REQUIRES(ctx, output_length->shape().dim_size(0) == duration->shape().dim_size(0),
                        tf::errors::InvalidArgument("Incompatible batch sizes"));
            OP_REQUIRES(ctx, duration->shape().dim_size(1) == beam_width_,
                        tf::errors::InvalidArgument("Incompatible beam width of duration"));
            OP_REQUIRES(ctx, output_length->shape().dim_size(1) == beam_width_,
                        tf::errors::InvalidArgument("Incompatible beam width of output_length"));

            // (B, W, T)
            const auto &duration_shape = duration->shape();
            const auto batch_size = duration_shape.dim_size(0);
            const int32_t max_t = duration_shape.dim_size(2);


            auto duration_t = duration->tensor<int32_t, 3>();
            auto output_length_t = output_length->tensor<int32_t, 2>();
            const auto max_u_t = max_u->scalar<int32_t>();
            const auto out_of_range_source_index_t = out_of_range_source_index->scalar<int32_t>();

            tf::Tensor *upsampled_source_indexes = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("upsampled_source_indexes",
                                                     tf::TensorShape({batch_size, beam_width_, max_u_t()}),
                                                     &upsampled_source_indexes));

            FillOutOfTargetRange(upsampled_source_indexes, out_of_range_source_index_t());
            auto upsampled_source_indexes_t = upsampled_source_indexes->tensor<int32_t, 3>();

            ssnt_upsample_source_indexes(duration_t.data(),
                                         output_length_t.data(),
                                         batch_size,
                                         beam_width_,
                                         max_t,
                                         max_u_t(),
                                         upsampled_source_indexes_t.data());
        }

    private:
        int beam_width_;

        void FillOutOfTargetRange(tf::Tensor *t, int32_t out_of_range_source_index) {
            t->flat<int32_t>().setConstant(out_of_range_source_index);
        };
    };

    REGISTER_KERNEL_BUILDER(Name("SSNTUpsampleSourceIndexes").Device(::tensorflow::DEVICE_CPU),
                            SSNTUpsampleSourceIndexesOpCPU);

}
