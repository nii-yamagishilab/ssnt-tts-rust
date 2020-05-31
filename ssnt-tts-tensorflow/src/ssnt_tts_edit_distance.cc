#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"


extern "C" void tone_latent_levenshtein_edit_distance(const int *a, const int *b,
                                                      const int *a_lengths, const int *b_lengths,
                                                      int batch_size, int max_length,
                                                      int *distance);


REGISTER_OP("ToneLatentLevenshteinEditDistance")
        .Input("a: int32")
        .Input("b: int32")
        .Input("a_lengths: int32")
        .Input("b_lengths: int32")
        .Output("distance: int32");

namespace tf = tensorflow;

namespace ssnt {

    class ToneLatentLevenshteinEditDistanceOpCPU : public tf::OpKernel {
    public:
        explicit ToneLatentLevenshteinEditDistanceOpCPU(tf::OpKernelConstruction *ctx) : tf::OpKernel(ctx) {

        }

        void Compute(tf::OpKernelContext *ctx) override {
            // Grab the input tensors
            const tf::Tensor *a;
            const tf::Tensor *b;
            const tf::Tensor *a_lengths;
            const tf::Tensor *b_lengths;
            OP_REQUIRES_OK(ctx, ctx->input("a", &a));
            OP_REQUIRES_OK(ctx, ctx->input("b", &b));
            OP_REQUIRES_OK(ctx, ctx->input("a_lengths", &a_lengths));
            OP_REQUIRES_OK(ctx, ctx->input("b_lengths", &b_lengths));

            OP_REQUIRES(ctx, a->shape().dims() == 2,
                        tf::errors::InvalidArgument("a is not a 2D-Tensor"));
            OP_REQUIRES(ctx, b->shape().dims() == 2,
                        tf::errors::InvalidArgument("b is not a 2D-Tensor"));
            OP_REQUIRES(ctx, a_lengths->shape().dims() == 1,
                        tf::errors::InvalidArgument("a_lengths is not a 1D-Tensor"));
            OP_REQUIRES(ctx, b_lengths->shape().dims() == 1,
                        tf::errors::InvalidArgument("b_lengths is not a 1D-Tensor"));
            OP_REQUIRES(ctx, a->shape().dim_size(0) == b->shape().dim_size(0) &&
                             a->shape().dim_size(1) == b->shape().dim_size(1),
                        tf::errors::InvalidArgument("Incompatible shape between a and b"));
            OP_REQUIRES(ctx, a_lengths->shape().dim_size(0) == b_lengths->shape().dim_size(0) &&
                             a->shape().dim_size(0) == a_lengths->shape().dim_size(0),
                        tf::errors::InvalidArgument("Incompatible shape between inputs and their lengths"));

            const auto &input_shape = a->shape();
            const int batch_size = input_shape.dim_size(0);
            const int max_length = input_shape.dim_size(1);

            auto a_t = a->tensor<int32_t, 2>();
            auto b_t = b->tensor<int32_t, 2>();
            auto a_lengths_t = a_lengths->vec<int32_t>();
            auto b_lengths_t = b_lengths->vec<int32_t>();

            tf::Tensor *distance = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("distance", tf::TensorShape({batch_size}),
                                                     &distance));
            auto distance_t = distance->vec<int32_t>();

            tone_latent_levenshtein_edit_distance(a_t.data(), b_t.data(),
                                                  a_lengths_t.data(), b_lengths_t.data(),
                                                  batch_size, max_length,
                                                  distance_t.data());

        }
    };

    REGISTER_KERNEL_BUILDER(Name("ToneLatentLevenshteinEditDistance").Device(::tensorflow::DEVICE_CPU),
                            ToneLatentLevenshteinEditDistanceOpCPU);

}
