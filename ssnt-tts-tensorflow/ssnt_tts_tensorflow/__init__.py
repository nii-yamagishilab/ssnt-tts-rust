import imp
import tensorflow as tf

lib_file = imp.find_module('kernels', __path__)[1]
_ssnt = tf.load_op_library(lib_file)


def beam_search_decode(h, log_prob_history, is_finished, t, u, max_t, beam_width):
    prediction, log_prob, next_t, next_u, is_finished, beam_branch = _ssnt.ssnt_beam_search_decode(h,
                                                                                                   log_prob_history,
                                                                                                   is_finished,
                                                                                                   t, u,
                                                                                                   max_t,
                                                                                                   beam_width)
    prediction.set_shape(tf.TensorShape([beam_width]))
    log_prob.set_shape(tf.TensorShape([beam_width]))
    next_t.set_shape(tf.TensorShape([beam_width]))
    next_u.set_shape(tf.TensorShape([beam_width]))
    is_finished.set_shape(tf.TensorShape([beam_width]))
    beam_branch.set_shape(tf.TensorShape([beam_width]))
    return prediction, log_prob, next_t, next_u, is_finished, beam_branch


def extract_best_beam_branch(best_final_branch, beam_branch, t_history, beam_width):
    best_beam_branch, best_t_history = _ssnt.ssnt_extract_best_beam_branch(best_final_branch, beam_branch, t_history,
                                                                           beam_width)
    beam_branch_shape = beam_branch.get_shape()
    best_beam_branch.set_shape([beam_branch_shape[0].value])
    best_t_history.set_shape([beam_branch_shape[0].value])
    return best_beam_branch, best_t_history


def ssnt_tts_v2_beam_search_decode(h,
                                   log_prob_history,
                                   is_finished,
                                   total_duration,
                                   duration_table,
                                   t,
                                   u,
                                   input_length,
                                   output_length,
                                   beam_width,
                                   duration_class_size,
                                   zero_duration_id,
                                   allow_skip,
                                   test_mode):
    output_length = tf.zeros_like(input_length) if test_mode else output_length
    prediction, log_prob, next_t, next_u, next_is_finished, next_total_duration, beam_branch = _ssnt.ssntv2_beam_search_decode(
        h,
        log_prob_history,
        is_finished,
        total_duration,
        duration_table,
        t,
        u,
        tf.cast(input_length, dtype=tf.int32),
        tf.cast(output_length, dtype=tf.int32),
        beam_width,
        duration_class_size,
        zero_duration_id,
        allow_skip,
        test_mode)

    batch_size = h.shape[0].value
    prediction.set_shape(tf.TensorShape([batch_size, beam_width]))
    log_prob.set_shape(tf.TensorShape([batch_size, beam_width]))
    next_t.set_shape(tf.TensorShape([batch_size, beam_width]))
    next_u.set_shape(tf.TensorShape([batch_size, beam_width]))
    next_is_finished.set_shape(tf.TensorShape([batch_size, beam_width]))
    next_total_duration.set_shape(tf.TensorShape([batch_size, beam_width]))
    beam_branch.set_shape(tf.TensorShape([batch_size, beam_width]))

    return prediction, log_prob, next_t, next_u, next_is_finished, next_total_duration, beam_branch


def order_beam_branch(final_branch, beam_branch, beam_width):
    ordered_beam_branch = _ssnt.ssnt_order_beam_branch(final_branch, beam_branch, beam_width)
    beam_branch_shape = beam_branch.get_shape()
    batch_size = beam_branch_shape[0].value
    max_t = beam_branch_shape[1].value
    ordered_beam_branch.set_shape([batch_size, beam_width, max_t])
    return ordered_beam_branch


def upsample_source_indexes(duration, output_length, out_of_range_source_index, beam_width):
    max_u = tf.reduce_max(output_length)
    upsampled_source_indexes = _ssnt.ssnt_upsample_source_indexes(duration,
                                                                  output_length,
                                                                  max_u,
                                                                  out_of_range_source_index,
                                                                  beam_width)
    duration_shape = duration.get_shape()
    batch_size = duration_shape[0].value
    max_t = duration_shape[2].value
    upsampled_source_indexes.set_shape([batch_size, beam_width, max_t])
    return upsampled_source_indexes


def tone_latent_beam_search_decode(h,
                                   log_prob_history,
                                   is_finished,
                                   t,
                                   u,
                                   input_length,
                                   beam_width,
                                   tone_class_size,
                                   empty_tone_id):
    prediction, log_prob, next_t, next_u, next_is_finished, beam_branch = _ssnt.tone_latent_beam_search_decode(
        h,
        log_prob_history,
        is_finished,
        t,
        u,
        tf.cast(input_length, dtype=tf.int32),
        beam_width,
        tone_class_size,
        empty_tone_id)

    batch_size = h.shape[0].value
    prediction.set_shape(tf.TensorShape([batch_size, beam_width]))
    log_prob.set_shape(tf.TensorShape([batch_size, beam_width]))
    next_t.set_shape(tf.TensorShape([batch_size, beam_width]))
    next_u.set_shape(tf.TensorShape([batch_size, beam_width]))
    next_is_finished.set_shape(tf.TensorShape([batch_size, beam_width]))
    beam_branch.set_shape(tf.TensorShape([batch_size, beam_width]))

    return prediction, log_prob, next_t, next_u, next_is_finished, beam_branch
