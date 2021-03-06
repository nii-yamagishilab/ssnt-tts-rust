#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"

extern "C" void tone_latent_beam_search_decode(const float *h,
                                               const float *log_prob_history,
                                               const bool *is_finished,
                                               const int *t,
                                               const int *u,
                                               const int *input_length,
                                               int batch_size,
                                               int beam_width,
                                               int tone_class_size,
                                               int empty_tone_id,
                                               int *prediction,
                                               float *log_prob,
                                               int *next_t,
                                               int *next_u,
                                               bool *next_is_finished,
                                               int *beam_branch);


REGISTER_OP("ToneLatentBeamSearchDecode")
        .Input("h: float32")
        .Input("log_prob_history: float32")
        .Input("is_finished: bool")
        .Input("t: int32")
        .Input("u: int32")
        .Input("input_length: int32")
        .Attr("beam_width: int")
        .Attr("tone_class_size: int")
        .Attr("empty_tone_id: int")
        .Output("prediction: int32")
        .Output("log_prob: float32")
        .Output("next_t: int32")
        .Output("next_u: int32")
        .Output("next_is_finished: bool")
        .Output("beam_branch: int32");

namespace tf = tensorflow;

namespace ssnt {

    class ToneLatentBeamSearchDecodeOpCPU : public tf::OpKernel {
    public:
        explicit ToneLatentBeamSearchDecodeOpCPU(tf::OpKernelConstruction *ctx) : tf::OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("tone_class_size", &tone_class_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("empty_tone_id", &empty_tone_id_));
        }

        void Compute(tf::OpKernelContext *ctx) override {
            // Grab the input tensors
            const tf::Tensor *h;
            const tf::Tensor *log_prob_history;
            const tf::Tensor *is_finished;
            const tf::Tensor *t;
            const tf::Tensor *u;
            const tf::Tensor *input_length;
            OP_REQUIRES_OK(ctx, ctx->input("h", &h));
            OP_REQUIRES_OK(ctx, ctx->input("log_prob_history", &log_prob_history));
            OP_REQUIRES_OK(ctx, ctx->input("is_finished", &is_finished));
            OP_REQUIRES_OK(ctx, ctx->input("t", &t));
            OP_REQUIRES_OK(ctx, ctx->input("u", &u));
            OP_REQUIRES_OK(ctx, ctx->input("input_length", &input_length));

            OP_REQUIRES(ctx, h->shape().dims() == 3,
                        tf::errors::InvalidArgument("h is not a 3D-Tensor"));
            OP_REQUIRES(ctx, log_prob_history->shape().dims() == 2,
                        tf::errors::InvalidArgument("log_prob_history is not a 2D-Tensor"));
            OP_REQUIRES(ctx, is_finished->shape().dims() == 2,
                        tf::errors::InvalidArgument("is_finished is not a 2D-Tensor"));
            OP_REQUIRES(ctx, t->shape().dims() == 2,
                        tf::errors::InvalidArgument("t is not 2D-Tensor"));
            OP_REQUIRES(ctx, u->shape().dims() == 2,
                        tf::errors::InvalidArgument("u is not 2D-Tensor"));
            OP_REQUIRES(ctx, input_length->shape().dims() == 1,
                        tf::errors::InvalidArgument("input_length is not 1D-Tensor"));

            // h: (B, W, D)
            OP_REQUIRES(ctx, h->shape().dim_size(1) == beam_width_,
                        tf::errors::InvalidArgument("h does not have beam width: ", beam_width_));
            OP_REQUIRES(ctx, h->shape().dim_size(2) == tone_class_size_,
                        tf::errors::InvalidArgument("h does not have duration class size: ", tone_class_size_));
            // log_prob_history: (B, W)
            OP_REQUIRES(ctx, log_prob_history->shape().dim_size(1) == beam_width_,
                        tf::errors::InvalidArgument("log_prob_history does not have beam width: ", beam_width_));
            // is_finished: (B, W)
            OP_REQUIRES(ctx, is_finished->shape().dim_size(1) == beam_width_,
                        tf::errors::InvalidArgument("is_finished does not have beam width: ", beam_width_));

            OP_REQUIRES(ctx, h->shape().dim_size(0) == log_prob_history->shape().dim_size(0) &&
                             log_prob_history->shape().dim_size(0) == t->shape().dim_size(0) &&
                             t->shape().dim_size(0) == u->shape().dim_size(0),
                        tf::errors::InvalidArgument("Incompatible batch sizes"));
            OP_REQUIRES(ctx, beam_width_ == h->shape().dim_size(1) &&
                             h->shape().dim_size(1) == log_prob_history->shape().dim_size(1) &&
                             log_prob_history->shape().dim_size(1) == t->shape().dim_size(1) &&
                             t->shape().dim_size(1) == u->shape().dim_size(1),
                        tf::errors::InvalidArgument("Incompatible beam widths"));

            const auto &h_shape = h->shape();
            const int batch_size = h_shape.dim_size(0);
            const int beam_width = h_shape.dim_size(1);
            const int num_classes = h_shape.dim_size(2);

            auto h_t = h->tensor<float, 3>();
            auto log_prob_history_t = log_prob_history->tensor<float, 2>();
            auto is_finished_t = is_finished->tensor<bool, 2>();
            auto t_t = t->tensor<int32_t, 2>();
            auto u_t = u->tensor<int32_t, 2>();
            auto input_length_t = input_length->vec<int32_t>();


            tf::Tensor *prediction = nullptr;
            OP_REQUIRES_OK(ctx,
                           ctx->allocate_output("prediction", tf::TensorShape({batch_size, beam_width}), &prediction));
            SetZeroDuration(prediction);
            auto prediction_t = prediction->tensor<int32_t, 2>();

            tf::Tensor *log_prob = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("log_prob", tf::TensorShape({batch_size, beam_width}), &log_prob));
            auto log_prob_t = log_prob->tensor<float, 2>();

            tf::Tensor *next_t = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("next_t", tf::TensorShape({batch_size, beam_width}), &next_t));
            auto next_t_t = next_t->tensor<int32_t, 2>();

            tf::Tensor *next_u = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("next_u", tf::TensorShape({batch_size, beam_width}), &next_u));
            auto next_u_t = next_u->tensor<int32_t, 2>();

            tf::Tensor *next_is_finished = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("next_is_finished", tf::TensorShape({batch_size, beam_width}),
                                                     &next_is_finished));
            auto next_is_finished_t = next_is_finished->tensor<bool, 2>();

            tf::Tensor *beam_branch = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("beam_branch", tf::TensorShape({batch_size, beam_width}),
                                                     &beam_branch));
            auto beam_branch_t = beam_branch->tensor<int32_t, 2>();

            tone_latent_beam_search_decode(h_t.data(),
                                           log_prob_history_t.data(),
                                           is_finished_t.data(),
                                           t_t.data(),
                                           u_t.data(),
                                           input_length_t.data(),
                                           batch_size,
                                           beam_width_,
                                           tone_class_size_,
                                           empty_tone_id_,
                                           prediction_t.data(),
                                           log_prob_t.data(),
                                           next_t_t.data(),
                                           next_u_t.data(),
                                           next_is_finished_t.data(),
                                           beam_branch_t.data());

        }

    private:
        int beam_width_;
        int tone_class_size_;
        int empty_tone_id_;

        void SetZeroDuration(tf::Tensor *t) {
            t->flat<int32_t>().setConstant(empty_tone_id_);
        };
    };

    REGISTER_KERNEL_BUILDER(Name("ToneLatentBeamSearchDecode").Device(::tensorflow::DEVICE_CPU), ToneLatentBeamSearchDecodeOpCPU);

}
