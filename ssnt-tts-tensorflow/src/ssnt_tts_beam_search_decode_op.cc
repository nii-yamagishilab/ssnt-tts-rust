#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"

extern "C" void ssnt_tts_beam_search_decode(const float *h, const float *log_prob_history, const bool *is_finished,
                                            const int *t, const int *u,
                                            int max_t, int beam_width, int *prediction, float *log_prob, int *next_t,
                                            int *next_u, bool *next_is_finished, int *beam_branch);


REGISTER_OP("SSNTBeamSearchDecode")
    .Input("h: float32")
    .Input("log_prob_history: float32")
    .Input("is_finished: bool")
    .Input("t: int32")
    .Input("u: int32")
    .Input("max_t: int32")
    .Attr("beam_width: int")
    .Output("prediction: int32")
    .Output("log_prob: float32")
    .Output("next_t: int32")
    .Output("next_u: int32")
    .Output("next_is_finished: bool")
    .Output("beam_branch: int32");

namespace tf = tensorflow;

namespace ssnt {

    class SSNTBeamSearchDecodeOpCPU : public tf::OpKernel {
    public:
        explicit SSNTBeamSearchDecodeOpCPU(tf::OpKernelConstruction *ctx) : tf::OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
        }

        void Compute(tf::OpKernelContext *ctx) override {
            // Grab the input tensors
            const tf::Tensor *h;
            const tf::Tensor *log_prob_history;
            const tf::Tensor *is_finished;
            const tf::Tensor *t;
            const tf::Tensor *u;
            const tf::Tensor *max_t;
            OP_REQUIRES_OK(ctx, ctx->input("h", &h));
            OP_REQUIRES_OK(ctx, ctx->input("log_prob_history", &log_prob_history));
            OP_REQUIRES_OK(ctx, ctx->input("is_finished", &is_finished));
            OP_REQUIRES_OK(ctx, ctx->input("t", &t));
            OP_REQUIRES_OK(ctx, ctx->input("u", &u));
            OP_REQUIRES_OK(ctx, ctx->input("max_t", &max_t));

            OP_REQUIRES(ctx, h->shape().dims() == 2,
                        tf::errors::InvalidArgument("h is not a 2D-Tensor"));
            OP_REQUIRES(ctx, log_prob_history->shape().dims() == 1,
                        tf::errors::InvalidArgument("log_prob_history is not a 1D-Tensor"));
            OP_REQUIRES(ctx, is_finished->shape().dims() == 1,
                        tf::errors::InvalidArgument("is_finished is not a 1D-Tensor"));
            OP_REQUIRES(ctx, t->shape().dims() == 1,
                        tf::errors::InvalidArgument("t is not 1D-Tensor"));
            OP_REQUIRES(ctx, u->shape().dims() == 1,
                        tf::errors::InvalidArgument("u is not 1D-Tensor"));
            OP_REQUIRES(ctx, max_t->shape().dims() == 0,
                        tf::errors::InvalidArgument("max_t is not 0D-Tensor"));
            OP_REQUIRES(ctx, h->shape().dim_size(0) == beam_width_,
                        tf::errors::InvalidArgument("h does not have beam width: ", beam_width_));
            OP_REQUIRES(ctx, log_prob_history->shape().dim_size(0) == beam_width_,
                        tf::errors::InvalidArgument("log_prob_history does not have beam width: ", beam_width_));
            OP_REQUIRES(ctx, is_finished->shape().dim_size(0) == beam_width_,
                        tf::errors::InvalidArgument("is_finished does not have beam width: ", beam_width_));
            OP_REQUIRES(ctx, h->shape().dim_size(0) == t->shape().dim_size(0) &&
                             t->shape().dim_size(0) == u->shape().dim_size(0),
                        tf::errors::InvalidArgument("Incompatible beam widths"));

            const auto &h_shape = h->shape();
            const int beam_width = h_shape.dim_size(0);
            const int num_classes = h_shape.dim_size(1);

            auto h_t = h->tensor<float, 2>();
            auto log_prob_history_t = log_prob_history->vec<float>();
            auto is_finished_t = is_finished->vec<bool>();
            auto t_t = t->vec<int32_t>();
            auto u_t = u->vec<int32_t>();
            auto max_t_t = max_t->scalar<int32_t>();

            OP_REQUIRES(ctx, num_classes == 2,
                        tf::errors::InvalidArgument("The size of transition class should be 2."));


            tf::Tensor *prediction = nullptr;
            OP_REQUIRES_OK(ctx,
                           ctx->allocate_output("prediction", tf::TensorShape({beam_width}), &prediction));
            prediction->flat<int32_t>().setConstant(-1);
            auto prediction_t = prediction->vec<int32_t>();

            tf::Tensor *log_prob = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("log_prob", tf::TensorShape({beam_width}), &log_prob));
            auto log_prob_t = log_prob->vec<float>();

            tf::Tensor *next_t = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("next_t", tf::TensorShape({beam_width}), &next_t));
            auto next_t_t = next_t->vec<int32_t>();

            tf::Tensor *next_u = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("next_u", tf::TensorShape({beam_width}), &next_u));
            auto next_u_t = next_u->vec<int32_t>();

            tf::Tensor *next_is_finished = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("next_is_finished", tf::TensorShape({beam_width}),
                                                     &next_is_finished));
            auto next_is_finished_t = next_is_finished->vec<bool>();

            tf::Tensor *beam_branch = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("beam_branch", tf::TensorShape({beam_width}),
                                                     &beam_branch));
            auto beam_branch_t = beam_branch->vec<int32_t>();

            ssnt_tts_beam_search_decode(h_t.data(),
                                        log_prob_history_t.data(),
                                        is_finished_t.data(),
                                        t_t.data(),
                                        u_t.data(),
                                        max_t_t(),
                                        beam_width_,
                                        prediction_t.data(),
                                        log_prob_t.data(),
                                        next_t_t.data(),
                                        next_u_t.data(),
                                        next_is_finished_t.data(),
                                        beam_branch_t.data());
        }

    private:
        int beam_width_;

        void set_zero(tf::Tensor *t) {
            t->flat<float>().setZero();
        };
    };

    REGISTER_KERNEL_BUILDER(Name("SSNTBeamSearchDecode").Device(::tensorflow::DEVICE_CPU), SSNTBeamSearchDecodeOpCPU);

}
