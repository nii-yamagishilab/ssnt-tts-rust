import imp
import tensorflow as tf

lib_file = imp.find_module('kernels', __path__)[1]
_ssnt = tf.load_op_library(lib_file)


def beam_search_decode(h, log_prob_history, t, u, beam_width, max_t):
    prediction, log_prob, next_t, next_u, is_finished, beam_branch = _ssnt.ssnt_beam_search_decode(h,
                                                                                                   log_prob_history,
                                                                                                   t, u,
                                                                                                   beam_width,
                                                                                                   max_t)
    prediction.set_shape(tf.TensorShape([beam_width]))
    log_prob.set_shape(tf.TensorShape([beam_width]))
    next_t.set_shape(tf.TensorShape([beam_width]))
    next_u.set_shape(tf.TensorShape([beam_width]))
    is_finished.set_shape(tf.TensorShape([beam_width]))
    beam_branch.set_shape(tf.TensorShape([beam_width]))
    return prediction, log_prob, next_t, next_u, is_finished, beam_branch

