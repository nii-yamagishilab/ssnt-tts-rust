extern crate rayon;

pub mod util;
pub mod v2;

use std::cmp::Ordering;
use rayon::prelude::*;

#[derive(PartialEq)]
enum Transition {
    Emit = 0,
    Shift = 1,
}


pub struct BeamSearchDecodingTable<'a> {
    // (W, T, V)
    input: &'a [f32],
    // (W)
    log_prob_history: &'a [f32],
    is_finished: &'a [bool],
    transition_size: usize,
    input_length: usize,
    beam_width: usize,
    max_beam_width: usize,
}

impl<'a> BeamSearchDecodingTable<'a> {
    pub fn new(input: &'a [f32], log_prob_history: &'a [f32], is_finished: &'a [bool], input_length: usize, beam_width: usize, max_beam_width: usize) -> BeamSearchDecodingTable<'a> {
        let transition_size = 2;
        assert_eq!(input.len(), beam_width * transition_size, "input.len(): {}, beam_width: {}, transition_size: {}", input.len(), beam_width, transition_size);
        assert_eq!(log_prob_history.len(), beam_width);
        BeamSearchDecodingTable {
            input,
            log_prob_history,
            is_finished,
            transition_size,
            input_length,
            beam_width,
            max_beam_width,
        }
    }

    fn beam_branch(&self, w: usize) -> &'a [f32] {
        let size = self.transition_size;
        let start = w * size;
        &self.input[start..start + size]
    }

    fn is_defined_at(&self, t: usize) -> bool {
        t >= 0 && t < self.input_length
    }

    fn decode_beam_at(&self, w: usize, t: usize, _u: usize) -> Option<Vec<(Transition, f32)>> {
        if !self.is_defined_at(t) {
            return None;
        }
        if self.is_finished[w] {
            return None;
        }
        let branch = self.beam_branch(w);
        let input = vec![(Transition::Emit, branch[0]), (Transition::Shift, branch[1])];
        Some(input)
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct DecodeResult {
    prediction: i32,
    pub log_prob: f32,
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


pub struct SsntTtsCpu {
    batch_size: i32,
    input_length: usize,
    max_u: usize,
    transition_size: usize,
}

impl SsntTtsCpu {
    pub fn new(batch_size: i32, input_length: usize, max_u: usize) -> SsntTtsCpu {
        let transition_size = 2;
        SsntTtsCpu {
            batch_size,
            input_length,
            max_u,
            transition_size,
        }
    }
}

pub trait SsntTts {
    fn beam_search_decode(&self, h: &[f32], log_prob_history: &[f32], is_finished: &[bool], t: &[i32], u: &[i32], beam_width: i32, max_beam_width: i32, prediction: &mut [i32], log_probs: &mut [f32], next_t: &mut [i32], next_u: &mut [i32], next_is_finished: &mut [bool], beam_branch: &mut [i32]) -> ();

    fn beam_search_kernel<'a>(&self, h: &BeamSearchDecodingTable<'a>, start_t: &[usize], u: &[usize]) -> Vec<DecodeResult>;

    fn beam_search_kernel_internal<'a>(&self, h: &BeamSearchDecodingTable<'a>, w: usize, t: usize, u: usize, log_prob_history: f32, is_finished: bool) -> Vec<DecodeResult>;
}


impl SsntTts for SsntTtsCpu {

    fn beam_search_decode(&self, h: &[f32], log_prob_history: &[f32], is_finished: &[bool], t: &[i32], u: &[i32], beam_width: i32, max_beam_width: i32, prediction: &mut [i32], log_probs: &mut [f32], next_t: &mut [i32], next_u: &mut [i32], next_is_finished: &mut [bool], beam_branch: &mut [i32]) -> () {
        h.par_chunks(beam_width as usize * self.transition_size)
            .zip(log_prob_history.par_chunks(beam_width as usize))
            .zip(is_finished.par_chunks(beam_width as usize))
            .zip(t.par_chunks(beam_width as usize))
            .zip(u.par_chunks(beam_width as usize))
            .zip(prediction.par_chunks_mut(max_beam_width as usize))
            .zip(log_probs.par_chunks_mut(max_beam_width as usize))
            .zip(next_t.par_chunks_mut(max_beam_width as usize))
            .zip(next_u.par_chunks_mut(max_beam_width as usize))
            .zip(beam_branch.par_chunks_mut(max_beam_width as usize))
            .zip(next_is_finished.par_chunks_mut(max_beam_width as usize))
            .for_each(|((((((((((h, log_prob_history), is_finished), t), u), prediction), log_probs), next_t), next_u), w), next_is_finished)| {
                let table = BeamSearchDecodingTable::new(h, log_prob_history, is_finished, self.input_length, beam_width as usize, max_beam_width as usize);
                let t: Vec<usize> = t.iter().map(|v| *v as usize).collect();
                let u: Vec<usize> = u.iter().map(|v| *v as usize).collect();
                let results = self.beam_search_kernel(&table, t.as_slice(), u.as_slice());
                results.iter().enumerate().for_each(|(i, result)| {
                    prediction[i] = result.prediction;
                    log_probs[i] = result.log_prob;
                    next_t[i] = result.next_t as i32;
                    next_u[i] = result.next_u as i32;
                    w[i] = result.parent_branch as i32;
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
                let is_finished = h.is_finished[w];
                self.beam_search_kernel_internal(h, w, t, u, log_prob_history, is_finished)
            }).collect();

        // Here the sorting does not consider prefixes. This is because we are interested in intermediate features which is path dependent.
        results.sort_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap_or(Ordering::Equal).reverse());
        results.dedup_by(|a, b| a.eq_ignore_parent(b));
        if results.len() < h.max_beam_width {
            for i in 0..(h.max_beam_width - results.len()) {
                results.push(results[i].clone());
            }
        }
        results.truncate(h.max_beam_width);
        results
    }

    fn beam_search_kernel_internal<'a>(&self, h: &BeamSearchDecodingTable<'a>, w: usize, t: usize, u: usize, log_prob_history: f32, is_finished: bool) -> Vec<DecodeResult> {
        match h.decode_beam_at(w, t, u) {
            // End of input. Return values to fill padding region.
            None => {
                vec![DecodeResult {
                    prediction: Transition::Emit as i32,
                    log_prob: log_prob_history,
                    next_t: t,
                    next_u: u,
                    is_finished: true,
                    parent_branch: w,
                }]
            }
            Some(results) => {
                results.into_iter().map(|(prediction, log_prob)| {
                    if prediction == Transition::Emit && t == self.input_length - 1 {
                        DecodeResult {
                            prediction: prediction as i32,
                            log_prob: log_prob_history + log_prob,
                            next_t: t,
                            next_u: u,
                            is_finished: true,
                            parent_branch: w,
                        }
                    } else if prediction == Transition::Shift && t == self.input_length - 1 {
                        // Shift transition is prohibited.
                        DecodeResult {
                            prediction: Transition::Emit as i32,
                            log_prob: log_prob_history,
                            next_t: t,
                            next_u: u,
                            is_finished: true,
                            parent_branch: w,
                        }
                    } else if prediction == Transition::Shift {
                        // Shift transition. Proceed to t + 1.
                        DecodeResult {
                            prediction: prediction as i32,
                            log_prob: log_prob_history + log_prob,
                            next_t: t + 1,
                            next_u: u + 1,
                            is_finished: false,
                            parent_branch: w,
                        }
                    } else {
                        // Emit transition. Keep the same t for next step.
                        DecodeResult {
                            prediction: prediction as i32,
                            log_prob: log_prob_history + log_prob,
                            next_t: t,
                            next_u: u + 1,
                            is_finished: false,
                            parent_branch: w,
                        }
                    }
                }).collect()
            }
        }
    }
}