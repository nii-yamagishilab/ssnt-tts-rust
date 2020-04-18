extern crate rayon;

use std::cmp::Ordering;
use rayon::prelude::*;


struct BatchView<'a, T> {
    batch_offset: usize,
    data: &'a [T],
}

impl<'a, T> BatchView<'a, T> {
    fn new(batch_size: usize, data: &'a [T]) -> BatchView<'a, T> {
        BatchView {
            batch_offset: data.len() / batch_size,
            data,
        }
    }

    fn batch(&self, index: usize) -> &[T] {
        self.data[index * self.batch_offset..(index + 1) * self.batch_offset].as_ref()
    }
}

struct DecodingTable {
    log_prob: f32,
    duration_class: i32,
    duration: i32,
    total_duration: i32,
    is_finished: bool,
}

pub struct BeamSearchDecodingTable<'a> {
    // (W, D)
    input: &'a [f32],
    // (W)
    log_prob_history: &'a [f32],
    // (W)
    is_finished: &'a [bool],
    // (W)
    total_duration: &'a [i32],
    // (D)
    duration_table: &'a [i32],
    duration_class_size: usize,
    input_length: usize,
    output_length: usize,
    beam_width: usize,
    max_beam_width: usize,
}

impl<'a> BeamSearchDecodingTable<'a> {
    pub fn new(input: &'a [f32], log_prob_history: &'a [f32], is_finished: &'a [bool], total_duration: &'a [i32], duration_table: &'a [i32], duration_class_size: usize, input_length: usize, output_length: usize, beam_width: usize, max_beam_width: usize) -> BeamSearchDecodingTable<'a> {
        assert_eq!(input.len(), beam_width * duration_class_size, "input: {}, beam_width: {}, duration_class_size: {}", input.len(), beam_width, duration_class_size);
        assert_eq!(log_prob_history.len(), beam_width);
        assert_eq!(is_finished.len(), beam_width);
        assert_eq!(total_duration.len(), beam_width);
        assert_eq!(duration_table.len(), duration_class_size);
        BeamSearchDecodingTable {
            input,
            log_prob_history,
            is_finished,
            total_duration,
            duration_table,
            duration_class_size,
            input_length,
            output_length,
            beam_width,
            max_beam_width,
        }
    }

    fn beam_branch(&self, w: usize) -> &'a [f32] {
        let size = self.duration_class_size;
        let start = w * size;
        &self.input[start..start + size]
    }

    fn is_defined_at(&self, t: usize) -> bool {
        t < self.input_length
    }

    fn total_duration_bounds(&self, t: usize) -> (i32, i32) {
        let diagonal: f32 = self.output_length as f32 / self.input_length as f32 * (t + 1) as f32;
        // ToDo: configure the range by arguments
        // + 10% of total frames
        let upper_range = self.output_length as f32 * 0.1;
        // - 5% of total frames
        let lower_range = self.output_length as f32 * 0.05;
        let lower_bound: i32 = (diagonal - lower_range).max(0.0) as i32;
        let upper_bound: i32 = (diagonal + upper_range).min(self.output_length as f32) as i32;
        (lower_bound, upper_bound)
    }

    fn on_diagonal(&self, result: &DecodeResult) -> bool {
        let diagonal: f32 = self.output_length as f32 / self.input_length as f32 * result.next_t as f32;
        let diff: f32 = result.total_duration as f32 - diagonal;
        diff.abs() <= 5.0
    }

    fn decode_beam_at(&self, w: usize, t: usize, test_mode: bool) -> Option<Vec<DecodingTable>> {
        if !self.is_defined_at(t) {
            return None;
        }
        if self.is_finished[w] {
            return None;
        }
        let branch = self.beam_branch(w);
        let input_copy: Vec<DecodingTable> = branch.iter().enumerate().filter_map(|(i, v)| {
            let duration = self.duration_table[i];
            let total_duration = self.total_duration[w] + duration;
            let (lower_bound, upper_bound) = self.total_duration_bounds(t);
            if !test_mode && (total_duration < lower_bound || total_duration > upper_bound as i32) {
                None
            } else if t == self.input_length - 1 {
                if !test_mode && total_duration != self.output_length as i32 {
                    None
                } else {
                    Some(DecodingTable {
                        log_prob: *v,
                        duration_class: i as i32,
                        duration,
                        total_duration,
                        is_finished: true,
                    })
                }
            } else {
                Some(DecodingTable {
                    log_prob: *v,
                    duration_class: i as i32,
                    duration,
                    total_duration,
                    is_finished: false,
                })
            }
        }).collect();
        Some(input_copy)
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct DecodeResult {
    prediction: i32,
    log_prob: f32,
    next_t: usize,
    next_u: usize,
    is_finished: bool,
    parent_branch: usize,
    total_duration: i32,
}

impl DecodeResult {
    fn eq_ignore_parent(&self, other: &DecodeResult) -> bool {
        self.prediction == other.prediction &&
            self.log_prob == other.log_prob &&
            self.next_t == other.next_t &&
            self.next_u == other.next_u &&
            self.is_finished == other.is_finished &&
            self.total_duration == other.total_duration
    }
}

pub struct SsntTtsV2Cpu {
    batch_size: i32,
    duration_class_size: usize,
    zero_duration_id: i32,
    test_mode: bool,
}

impl SsntTtsV2Cpu {
    pub fn new(batch_size: i32, duration_class_size: usize, zero_duration_id: i32, test_mode: bool) -> SsntTtsV2Cpu {
        SsntTtsV2Cpu {
            batch_size,
            duration_class_size,
            zero_duration_id,
            test_mode,
        }
    }
}

pub trait SsntTtsV2 {
    fn beam_search_decode(&self, h: &[f32], log_prob_history: &[f32], is_finished: &[bool], total_duration: &[i32], duration_table: &[i32], t: &[i32], u: &[i32], max_t: &[i32], max_u: &[i32], batch_size: i32, beam_width: i32, max_beam_width: i32, prediction: &mut [i32], log_probs: &mut [f32], next_t: &mut [i32], next_u: &mut [i32], next_is_finished: &mut [bool], next_total_duration: &mut [i32], beam_branch: &mut [i32]) -> ();

    fn beam_search_kernel<'a>(&self, h: &BeamSearchDecodingTable<'a>, start_t: &[usize], u: &[usize]) -> Vec<DecodeResult>;

    fn beam_search_kernel_internal<'a>(&self, h: &BeamSearchDecodingTable<'a>, w: usize, t: usize, u: usize, log_prob_history: f32) -> Vec<DecodeResult>;
}


impl SsntTtsV2 for SsntTtsV2Cpu {
    fn beam_search_decode(&self, h: &[f32], log_prob_history: &[f32], is_finished: &[bool], total_duration: &[i32], duration_table: &[i32], t: &[i32], u: &[i32], input_length: &[i32], output_length: &[i32], batch_size: i32, beam_width: i32, max_beam_width: i32, prediction: &mut [i32], log_probs: &mut [f32], next_t: &mut [i32], next_u: &mut [i32], next_is_finished: &mut [bool], next_total_duration: &mut [i32], beam_branch: &mut [i32]) -> () {
        assert_eq!(prediction.len(), (batch_size * max_beam_width) as usize);
        assert_eq!(log_probs.len(), (batch_size * beam_width) as usize);
        assert_eq!(next_is_finished.len(), (batch_size * beam_width) as usize);
        assert_eq!(next_total_duration.len(), (batch_size * beam_width) as usize);
        assert_eq!(beam_branch.len(), (batch_size * beam_width) as usize);
        h.par_chunks(beam_width as usize * self.duration_class_size)
            .zip(log_prob_history.par_chunks(beam_width as usize))
            .zip(is_finished.par_chunks(beam_width as usize))
            .zip(total_duration.par_chunks(beam_width as usize))
            .zip(t.par_chunks(beam_width as usize))
            .zip(u.par_chunks(beam_width as usize))
            .zip(input_length.par_chunks(1))
            .zip(output_length.par_chunks(1))
            .zip(prediction.par_chunks_mut(max_beam_width as usize))
            .zip(log_probs.par_chunks_mut(max_beam_width as usize))
            .zip(next_t.par_chunks_mut(max_beam_width as usize))
            .zip(next_u.par_chunks_mut(max_beam_width as usize))
            .zip(beam_branch.par_chunks_mut(max_beam_width as usize))
            .zip(next_is_finished.par_chunks_mut(max_beam_width as usize))
            .zip(next_total_duration.par_chunks_mut(max_beam_width as usize))
            .for_each(|((((((((((((((h, log_prob_history), is_finished), total_duration), t), u), input_length), output_length), prediction), log_probs), next_t), next_u), beam_branch), next_is_finished), next_total_duration)| {
                let table = BeamSearchDecodingTable::new(h,
                                                         log_prob_history,
                                                         is_finished,
                                                         total_duration,
                                                         duration_table,
                                                         self.duration_class_size,
                                                         input_length[0] as usize,
                                                         output_length[0] as usize,
                                                         beam_width as usize,
                                                         max_beam_width as usize);
                let t: Vec<usize> = t.iter().map(|v| *v as usize).collect();
                let u: Vec<usize> = u.iter().map(|v| *v as usize).collect();
                let results = self.beam_search_kernel(&table, t.as_slice(), u.as_slice());
                results.iter().enumerate().for_each(|(i, result)| {
                    prediction[i] = result.prediction;
                    log_probs[i] = result.log_prob;
                    next_t[i] = result.next_t as i32;
                    next_u[i] = result.next_u as i32;
                    beam_branch[i] = result.parent_branch as i32;
                    next_is_finished[i] = result.is_finished;
                    next_total_duration[i] = result.total_duration;
                });
            });
    }

    fn beam_search_kernel<'a>(&self, h: &BeamSearchDecodingTable<'a>, start_t: &[usize], u: &[usize]) -> Vec<DecodeResult> {
        let mut results: Vec<DecodeResult> = (0..h.beam_width)
            .into_par_iter()
            .flat_map(|w| {
                let t = start_t[w];
                let u = u[w];
                let log_prob_history = h.log_prob_history[w];
                self.beam_search_kernel_internal(h, w, t, u, log_prob_history)
            }).collect();

        // Here the sorting does not consider prefixes. This is because we are interested in intermediate features which is path dependent.
        results.sort_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap_or(Ordering::Equal).reverse());
        results.dedup_by(|a, b| a.eq_ignore_parent(b));
        // Add a diagonal duration candidate to avoid empty search
        let diagonal_result: Option<DecodeResult> = if !self.test_mode {
            results.iter().find(|result| {
                h.on_diagonal(result)
            }).map(|result| result.clone())
        } else {
            None
        };

        let n_results: usize = results.len();
        assert_ne!(n_results, 0, "Beam search could not find a duration sequence with compatible output length: {} for input with length: {}. Please increase duration class size and beam width.", h.output_length, h.input_length);
        if n_results < h.max_beam_width {
            for i in 0..(h.max_beam_width - n_results) {
                results.push(results[i % n_results].clone());
            }
        }
        match diagonal_result {
            Some(result) => {
                results.truncate(h.max_beam_width - 1);
                results.push(result);
                results
            }
            None => {
                results.truncate(h.max_beam_width);
                results
            }
        }
    }

    fn beam_search_kernel_internal<'a>(&self, h: &BeamSearchDecodingTable<'a>, w: usize, t: usize, u: usize, log_prob_history: f32) -> Vec<DecodeResult> {
        match h.decode_beam_at(w, t, self.test_mode) {
            // End of input. Return values to fill padding region.
            None => {
                vec![DecodeResult {
                    prediction: self.zero_duration_id,
                    log_prob: log_prob_history,
                    next_t: t,
                    next_u: u,
                    is_finished: true,
                    parent_branch: w,
                    total_duration: h.total_duration[w],
                }]
            }
            Some(results) => {
                results.into_iter().map(|v| {
                    DecodeResult {
                        prediction: v.duration_class,
                        log_prob: log_prob_history + v.log_prob,
                        next_t: if v.is_finished { t } else { t + 1 },
                        next_u: if v.is_finished { u } else { u + 1 },
                        is_finished: v.is_finished,
                        parent_branch: w,
                        total_duration: v.total_duration,
                    }
                }).collect()
            }
        }
    }
}