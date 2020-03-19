import tensorflow as tf
import numpy as np
from ssnt_tts_tensorflow import upsample_source_indexes


class UpsamplingTest(tf.test.TestCase):

    def test_beam_search(self):
        batch_size = 3
        beam_width = 2
        max_t = 6

        duration = tf.constant([
            [
                [0, 3, 2, 1, 0, 0],
                [0, 3, 2, 1, 0, 0],
            ],
            [
                [2, 4, 1, 2, 1, 0],
                [2, 4, 1, 2, 1, 0],
            ],
            [
                [1, 3, 2, 2, 1, 2],
                [1, 3, 2, 2, 1, 2],
            ]
        ], dtype=tf.int32)
        assert duration.shape.as_list() == [batch_size, beam_width, max_t]

        output_length = tf.constant([6, 10, 11], dtype=tf.int32)

        out_of_range_source_index = -1

        upsampled_source_indexes = upsample_source_indexes(duration,
                                                           output_length,
                                                           beam_width,
                                                           out_of_range_source_index)

        with self.cached_session() as sess:
            upsampled_source_indexes = sess.run(upsampled_source_indexes)
            print(upsampled_source_indexes)
