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
    tone_class: i32,
    is_finished: bool,
}

pub struct BeamSearchDecodingTable<'a> {
    // (W, D)
    input: &'a [f32],
    // (W)
    log_prob_history: &'a [f32],
    // (W)
    is_finished: &'a [bool],
    tone_class_size: usize,
    input_length: usize,
    beam_width: usize,
    max_beam_width: usize,
    empty_tone_id: i32,
}

impl<'a> BeamSearchDecodingTable<'a> {
    pub fn new(input: &'a [f32],
               log_prob_history: &'a [f32],
               is_finished: &'a [bool],
               tone_class_size: usize,
               input_length: usize,
               beam_width: usize,
               max_beam_width: usize,
               empty_tone_id: i32) -> BeamSearchDecodingTable<'a> {
        assert_eq!(input.len(), beam_width * tone_class_size, "input: {}, beam_width: {}, tone_class_size: {}", input.len(), beam_width, tone_class_size);
        assert_eq!(log_prob_history.len(), beam_width);
        assert_eq!(is_finished.len(), beam_width);
        BeamSearchDecodingTable {
            input,
            log_prob_history,
            is_finished,
            tone_class_size,
            input_length,
            beam_width,
            max_beam_width,
            empty_tone_id,
        }
    }

    fn beam_branch(&self, w: usize) -> &'a [f32] {
        let size = self.tone_class_size;
        let start = w * size;
        &self.input[start..start + size]
    }

    fn is_defined_at(&self, t: usize) -> bool {
        t < self.input_length
    }

    fn decode_beam_at(&self, w: usize, t: usize) -> Option<Vec<DecodingTable>> {
        if !self.is_defined_at(t) {
            return None;
        }
        if self.is_finished[w] {
            return None;
        }
        let branch: &[f32] = self.beam_branch(w);
        let input_copy: Vec<DecodingTable> = branch.iter().enumerate().filter_map(|(i, v)| {
            Some(DecodingTable {
                log_prob: *v,
                tone_class: i as i32,
                is_finished: false,
            })
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
}

impl DecodeResult {
    fn eq_ignore_parent(&self, other: &DecodeResult) -> bool {
        self.prediction == other.prediction &&
            self.log_prob == other.log_prob &&
            self.next_t == other.next_t &&
            self.next_u == other.next_u &&
            self.is_finished == other.is_finished
    }
}

pub struct ToneLatentCpu {
    batch_size: i32,
    tone_class_size: usize,
    empty_tone_id: i32,
}

impl ToneLatentCpu {
    pub fn new(batch_size: i32, tone_class_size: usize, empty_tone_id: i32) -> ToneLatentCpu {
        ToneLatentCpu {
            batch_size,
            tone_class_size,
            empty_tone_id,
        }
    }
}

pub trait ToneLatent {
    fn beam_search_decode(&self, h: &[f32], log_prob_history: &[f32], is_finished: &[bool], t: &[i32], u: &[i32], max_t: &[i32], batch_size: i32, beam_width: i32, max_beam_width: i32, prediction: &mut [i32], log_probs: &mut [f32], next_t: &mut [i32], next_u: &mut [i32], next_is_finished: &mut [bool], beam_branch: &mut [i32]) -> ();

    fn beam_search_kernel<'a>(&self, h: &BeamSearchDecodingTable<'a>, start_t: &[usize], u: &[usize]) -> Vec<DecodeResult>;

    fn beam_search_kernel_internal<'a>(&self, h: &BeamSearchDecodingTable<'a>, w: usize, t: usize, u: usize, log_prob_history: f32) -> Vec<DecodeResult>;
}


impl ToneLatent for ToneLatentCpu {
    fn beam_search_decode(&self, h: &[f32], log_prob_history: &[f32], is_finished: &[bool], t: &[i32], u: &[i32], input_length: &[i32], batch_size: i32, beam_width: i32, max_beam_width: i32, prediction: &mut [i32], log_probs: &mut [f32], next_t: &mut [i32], next_u: &mut [i32], next_is_finished: &mut [bool], beam_branch: &mut [i32]) -> () {
        assert_eq!(prediction.len(), (batch_size * max_beam_width) as usize);
        assert_eq!(log_probs.len(), (batch_size * beam_width) as usize);
        assert_eq!(next_is_finished.len(), (batch_size * beam_width) as usize);
        assert_eq!(beam_branch.len(), (batch_size * beam_width) as usize);
        h.par_chunks(beam_width as usize * self.tone_class_size)
            .zip(log_prob_history.par_chunks(beam_width as usize))
            .zip(is_finished.par_chunks(beam_width as usize))
            .zip(t.par_chunks(beam_width as usize))
            .zip(u.par_chunks(beam_width as usize))
            .zip(input_length.par_chunks(1))
            .zip(prediction.par_chunks_mut(max_beam_width as usize))
            .zip(log_probs.par_chunks_mut(max_beam_width as usize))
            .zip(next_t.par_chunks_mut(max_beam_width as usize))
            .zip(next_u.par_chunks_mut(max_beam_width as usize))
            .zip(beam_branch.par_chunks_mut(max_beam_width as usize))
            .zip(next_is_finished.par_chunks_mut(max_beam_width as usize))
            .for_each(|(((((((((((h, log_prob_history), is_finished), t), u), input_length), prediction), log_probs), next_t), next_u), beam_branch), next_is_finished)| {
                let table = BeamSearchDecodingTable::new(h,
                                                         log_prob_history,
                                                         is_finished,
                                                         self.tone_class_size,
                                                         input_length[0] as usize,
                                                         beam_width as usize,
                                                         max_beam_width as usize,
                                                         self.empty_tone_id);
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

        let n_results: usize = results.len();
        if n_results < h.max_beam_width {
            for i in 0..(h.max_beam_width - n_results) {
                results.push(results[i % n_results].clone());
            }
        }
        results.truncate(h.max_beam_width);
        results
    }

    fn beam_search_kernel_internal<'a>(&self, h: &BeamSearchDecodingTable<'a>, w: usize, t: usize, u: usize, log_prob_history: f32) -> Vec<DecodeResult> {
        match h.decode_beam_at(w, t) {
            // End of input. Return values to fill padding region.
            None => {
                vec![DecodeResult {
                    prediction: self.empty_tone_id,
                    log_prob: log_prob_history,
                    next_t: t,
                    next_u: u,
                    is_finished: true,
                    parent_branch: w,
                }]
            }
            Some(results) => {
                results.into_iter().map(|v| {
                    DecodeResult {
                        prediction: v.tone_class,
                        log_prob: log_prob_history + v.log_prob,
                        next_t: if v.is_finished { t } else { t + 1 },
                        next_u: if v.is_finished { u } else { u + 1 },
                        is_finished: v.is_finished,
                        parent_branch: w,
                    }
                }).collect()
            }
        }
    }
}