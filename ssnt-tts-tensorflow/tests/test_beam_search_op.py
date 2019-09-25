import tensorflow as tf
import numpy as np
from ssnt_tts_tensorflow import beam_search_decode


class SsntTtsTest(tf.test.TestCase):

    def test_beam_search(self):
        beam_width = 3
        max_t = 4
        acts1 = [[0.2, 0.8],
                 [0.2, 0.8],
                 [0.2, 0.8]]
        acts2 = [[0.7, 0.3],
                 [0.4, 0.6],
                 [0.5, 0.5]]
        acts3 = [[0.1, 0.9],
                 [0.6, 0.4],
                 [0.4, 0.6]]
        acts4 = [[0.7, 0.3],
                 [0.5, 0.5],
                 [0.1, 0.9]]
        acts5 = [[0.6, 0.4],
                 [0.3, 0.7],
                 [0.4, 0.6]]
        acts6 = [[0.1, 0.9],
                 [0.6, 0.4],
                 [0.4, 0.6]]
        acts7 = [[0.3, 0.7],
                 [0.4, 0.6],
                 [0.6, 0.4]]

        acts = list(map(lambda a: np.log(np.array(a, dtype=np.float32)),
                        [acts1, acts2, acts3, acts4, acts5, acts6, acts7]))

        log_prob_history = np.array([0, 0, 0], dtype=np.float32)
        next_t = np.array([0, 0, 0], dtype=np.int32)
        next_u = np.array([0, 0, 0], dtype=np.int32)

        with self.cached_session() as sess:
            for a in acts:
                prediction, log_prob_history, next_t, next_u, is_finished, beam_branch = beam_search_decode(a,
                                                                                                            log_prob_history,
                                                                                                            next_t,
                                                                                                            next_u,
                                                                                                            max_t,
                                                                                                            beam_width)
                prediction, log_prob_history, next_t, next_u, beam_branch, is_finished = sess.run(
                    [prediction, log_prob_history, next_t, next_u, beam_branch, is_finished])
                print(prediction, log_prob_history, next_t, next_u, beam_branch, is_finished)
