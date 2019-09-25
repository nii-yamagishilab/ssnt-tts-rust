extern crate ssnt_tts;
extern crate libc;

use ssnt_tts::{SsntTts, SsntTtsCpu};
use libc::c_float;


#[no_mangle]
pub extern fn ssnt_tts_beam_search_decode(h: *const c_float, log_prob_history: *const c_float, t: *const i32, u: *const i32, max_t: i32, beam_width: i32, prediction: *mut i32, log_probs: *mut c_float, next_t: *mut i32, next_u: *mut i32, is_finished: *mut bool, beam_branch: *mut i32) -> () {
    // Restricted to single batch.
    let batch_size = 1;
    let n_transition_classes = 2;
    let h = unsafe {
        assert!(!h.is_null());
        let h_len = batch_size * beam_width * n_transition_classes;
        std::slice::from_raw_parts(h, h_len as usize)
    };

    let log_prob_history = unsafe {
        assert!(!log_prob_history.is_null());
        let log_prob_history_len = batch_size * beam_width;
        std::slice::from_raw_parts(log_prob_history, log_prob_history_len as usize)
    };

    let t = unsafe {
        assert!(!t.is_null());
        let t_len = batch_size * beam_width;
        std::slice::from_raw_parts(t, t_len as usize)
    };

    let u = unsafe {
        assert!(!u.is_null());
        let u_len = batch_size * beam_width;
        std::slice::from_raw_parts(u, u_len as usize)
    };

    let prediction = unsafe {
        assert!(!prediction.is_null());
        let prediction_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(prediction, prediction_len as usize)
    };

    let log_probs = unsafe {
        assert!(!log_probs.is_null());
        let log_probs_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(log_probs, log_probs_len as usize)
    };

    let next_t = unsafe {
        assert!(!next_t.is_null());
        let next_t_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(next_t, next_t_len as usize)
    };

    let next_u = unsafe {
        assert!(!next_u.is_null());
        let next_u_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(next_u, next_u_len as usize)
    };

    let is_finished = unsafe {
        assert!(!is_finished.is_null());
        let is_finished_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(is_finished, is_finished_len as usize)
    };

    let beam_branch = unsafe {
        assert!(!beam_branch.is_null());
        let beam_branch_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(beam_branch, beam_branch_len as usize)
    };

    let ssnt_tts = SsntTtsCpu::new(batch_size, max_t as usize, 0 as usize);
    ssnt_tts.beam_search_decode(h, log_prob_history, t, u, beam_width, beam_width, prediction, log_probs, next_t, next_u, is_finished, beam_branch);
}