extern crate ssnt_tts;
extern crate libc;

use ssnt_tts::{SsntTts, SsntTtsCpu, util, v2, v2_util, tone_latent, edit_distance};
use libc::c_float;
use ssnt_tts::v2::SsntTtsV2;
use ssnt_tts::tone_latent::{ToneLatent, ToneLatentCpu};


#[no_mangle]
pub extern fn ssnt_tts_beam_search_decode(h: *const c_float, log_prob_history: *const c_float, is_finished: *const bool, t: *const i32, u: *const i32, max_t: i32, beam_width: i32, prediction: *mut i32, log_probs: *mut c_float, next_t: *mut i32, next_u: *mut i32, next_is_finished: *mut bool, beam_branch: *mut i32) -> () {
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

    let is_finished = unsafe {
        assert!(!is_finished.is_null());
        let is_finished_len = batch_size * beam_width;
        std::slice::from_raw_parts(is_finished, is_finished_len as usize)
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

    let next_is_finished = unsafe {
        assert!(!next_is_finished.is_null());
        let next_is_finished_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(next_is_finished, next_is_finished_len as usize)
    };

    let beam_branch = unsafe {
        assert!(!beam_branch.is_null());
        let beam_branch_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(beam_branch, beam_branch_len as usize)
    };

    let ssnt_tts = SsntTtsCpu::new(batch_size, max_t as usize, 0 as usize);
    ssnt_tts.beam_search_decode(h, log_prob_history, is_finished, t, u, beam_width, beam_width, prediction, log_probs, next_t, next_u, next_is_finished, beam_branch);
}


#[no_mangle]
pub extern fn ssnt_extract_best_beam_branch(best_final_branch: i32, beam_branch: *const i32, t_history: *const i32, beam_width: i32, max_u: i32, best_beam_branch: *mut i32, best_t_history: *mut i32) -> () {
    let beam_branch = unsafe {
        assert!(!beam_branch.is_null());
        let beam_branch_len = max_u * beam_width;
        std::slice::from_raw_parts(beam_branch, beam_branch_len as usize)
    };

    let t_history = unsafe {
        assert!(!t_history.is_null());
        let t_history_len = max_u * beam_width;
        std::slice::from_raw_parts(t_history, t_history_len as usize)
    };

    let best_beam_branch = unsafe {
        assert!(!best_beam_branch.is_null());
        let best_beam_branch_len = max_u;
        std::slice::from_raw_parts_mut(best_beam_branch, best_beam_branch_len as usize)
    };

    let best_t_history = unsafe {
        assert!(!best_t_history.is_null());
        let best_t_history_len = max_u;
        std::slice::from_raw_parts_mut(best_t_history, best_t_history_len as usize)
    };

    let (_best_beam_branch, _best_t_history) = util::extract_best_beam_branch_kernel(best_final_branch, beam_branch, t_history, beam_width, max_u);

    best_beam_branch.copy_from_slice(_best_beam_branch.as_slice());
    best_t_history.copy_from_slice(_best_t_history.as_slice());
}

#[no_mangle]
pub extern fn ssnt_tts_v2_beam_search_decode(h: *const c_float, log_prob_history: *const c_float, is_finished: *const bool, total_duration: *const i32, duration_table: *const i32, t: *const i32, u: *const i32, input_length: *const i32, output_length: *const i32, batch_size: i32, beam_width: i32, duration_class_size: i32, zero_duration_id: i32, allow_skip: bool, test_mode: bool, prediction: *mut i32, log_probs: *mut c_float, next_t: *mut i32, next_u: *mut i32, next_is_finished: *mut bool, next_total_duration: *mut i32, beam_branch: *mut i32) -> () {
    let h = unsafe {
        assert!(!h.is_null());
        let h_len = batch_size * beam_width * duration_class_size;
        std::slice::from_raw_parts(h, h_len as usize)
    };

    let log_prob_history = unsafe {
        assert!(!log_prob_history.is_null());
        let log_prob_history_len = batch_size * beam_width;
        std::slice::from_raw_parts(log_prob_history, log_prob_history_len as usize)
    };

    let is_finished = unsafe {
        assert!(!is_finished.is_null());
        let is_finished_len = batch_size * beam_width;
        std::slice::from_raw_parts(is_finished, is_finished_len as usize)
    };

    let total_duration = unsafe {
        assert!(!total_duration.is_null());
        let total_duration_len = batch_size * beam_width;
        std::slice::from_raw_parts(total_duration, total_duration_len as usize)
    };

    let duration_table = unsafe {
        assert!(!duration_table.is_null());
        let duration_table_len = duration_class_size;
        std::slice::from_raw_parts(duration_table, duration_table_len as usize)
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

    let input_length = unsafe {
        assert!(!input_length.is_null());
        let input_length_len = batch_size;
        std::slice::from_raw_parts(input_length, input_length_len as usize)
    };

    let output_length = unsafe {
        assert!(!output_length.is_null());
        let output_length_len = batch_size;
        std::slice::from_raw_parts(output_length, output_length_len as usize)
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

    let next_is_finished = unsafe {
        assert!(!next_is_finished.is_null());
        let next_is_finished_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(next_is_finished, next_is_finished_len as usize)
    };

    let next_total_duration = unsafe {
        assert!(!next_total_duration.is_null());
        let next_total_duration_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(next_total_duration, next_total_duration_len as usize)
    };

    let beam_branch = unsafe {
        assert!(!beam_branch.is_null());
        let beam_branch_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(beam_branch, beam_branch_len as usize)
    };

    let ssnt_tts = v2::SsntTtsV2Cpu::new(batch_size, duration_class_size as usize, zero_duration_id, allow_skip, test_mode);
    ssnt_tts.beam_search_decode(h, log_prob_history, is_finished, total_duration, duration_table, t, u, input_length, output_length, batch_size, beam_width, beam_width, prediction, log_probs, next_t, next_u, next_is_finished, next_total_duration, beam_branch);
}

#[no_mangle]
pub extern fn ssnt_order_beam_branch(final_branch: *const i32, beam_branch: *const i32, batch_size: i32, beam_width: i32, max_t: i32, ordered_beam_branch: *mut i32) -> () {
    let final_branch: &[i32] = unsafe {
        assert!(!final_branch.is_null());
        let final_branch_len: i32 = batch_size * beam_width;
        std::slice::from_raw_parts(final_branch, final_branch_len as usize)
    };

    let beam_branch: &[i32] = unsafe {
        assert!(!beam_branch.is_null());
        let beam_branch_len: i32 = batch_size * max_t * beam_width;
        std::slice::from_raw_parts(beam_branch, beam_branch_len as usize)
    };

    let ordered_beam_branch: &mut [i32] = unsafe {
        assert!(!ordered_beam_branch.is_null());
        let ordered_beam_branch_len = batch_size * beam_width * max_t;
        std::slice::from_raw_parts_mut(ordered_beam_branch, ordered_beam_branch_len as usize)
    };

    v2_util::order_beam_branch(final_branch, beam_branch, beam_width, max_t, ordered_beam_branch);
}


#[no_mangle]
pub extern fn ssnt_upsample_source_indexes(duration: *const i32, output_length: *const i32, batch_size: i32, beam_width: i32, max_t: i32, max_u: i32, upsampled_source_indexes: *mut i32) -> () {
    let duration: &[i32] = unsafe {
        assert!(!duration.is_null());
        let duration_len: i32 = batch_size * beam_width * max_t;
        std::slice::from_raw_parts(duration, duration_len as usize)
    };

    let output_length: &[i32] = unsafe {
        assert!(!output_length.is_null());
        let output_length_len: i32 = batch_size * beam_width;
        std::slice::from_raw_parts(output_length, output_length_len as usize)
    };

    let upsampled_source_indexes: &mut [i32] = unsafe {
        assert!(!upsampled_source_indexes.is_null());
        let upsampled_source_indexes_len = batch_size * beam_width * max_u;
        std::slice::from_raw_parts_mut(upsampled_source_indexes, upsampled_source_indexes_len as usize)
    };

    v2_util::upsample_source_indexes(duration, output_length, beam_width, max_t, max_u, upsampled_source_indexes);
}

#[no_mangle]
pub extern fn tone_latent_beam_search_decode(h: *const c_float, log_prob_history: *const c_float, is_finished: *const bool, t: *const i32, u: *const i32, input_length: *const i32, batch_size: i32, beam_width: i32, tone_class_size: i32, empty_tone_id: i32, prediction: *mut i32, log_probs: *mut c_float, next_t: *mut i32, next_u: *mut i32, next_is_finished: *mut bool, beam_branch: *mut i32) -> () {
    let h = unsafe {
        assert!(!h.is_null());
        let h_len = batch_size * beam_width * tone_class_size;
        std::slice::from_raw_parts(h, h_len as usize)
    };

    let log_prob_history = unsafe {
        assert!(!log_prob_history.is_null());
        let log_prob_history_len = batch_size * beam_width;
        std::slice::from_raw_parts(log_prob_history, log_prob_history_len as usize)
    };

    let is_finished = unsafe {
        assert!(!is_finished.is_null());
        let is_finished_len = batch_size * beam_width;
        std::slice::from_raw_parts(is_finished, is_finished_len as usize)
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

    let input_length = unsafe {
        assert!(!input_length.is_null());
        let input_length_len = batch_size;
        std::slice::from_raw_parts(input_length, input_length_len as usize)
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

    let next_is_finished = unsafe {
        assert!(!next_is_finished.is_null());
        let next_is_finished_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(next_is_finished, next_is_finished_len as usize)
    };

    let beam_branch = unsafe {
        assert!(!beam_branch.is_null());
        let beam_branch_len = batch_size * beam_width;
        std::slice::from_raw_parts_mut(beam_branch, beam_branch_len as usize)
    };

    let tone_latent: ToneLatentCpu = tone_latent::ToneLatentCpu::new(batch_size, tone_class_size as usize, empty_tone_id);
    tone_latent.beam_search_decode(h, log_prob_history, is_finished, t, u, input_length, batch_size, beam_width, beam_width, prediction, log_probs, next_t, next_u, next_is_finished, beam_branch);
}


#[no_mangle]
pub extern fn tone_latent_levenshtein_edit_distance(a: *const i32, b: *const i32, a_lengths: *const i32, b_lengths: *const i32, batch_size: i32, max_length: i32, distance: *mut i32) -> () {
    let a: &[i32] = unsafe {
        assert!(!a.is_null());
        let a_len = batch_size * max_length;
        std::slice::from_raw_parts(a, a_len as usize)
    };

    let b: &[i32] = unsafe {
        assert!(!b.is_null());
        let b_len = batch_size * max_length;
        std::slice::from_raw_parts(b, b_len as usize)
    };

    let a_lengths: &[i32] = unsafe {
        assert!(!a_lengths.is_null());
        let a_lengths_len = batch_size;
        std::slice::from_raw_parts(a_lengths, a_lengths_len as usize)
    };

    let b_lengths: &[i32] = unsafe {
        assert!(!b_lengths.is_null());
        let b_lengths_len = batch_size;
        std::slice::from_raw_parts(b_lengths, b_lengths_len as usize)
    };

    let distance: &mut [i32] = unsafe {
        assert!(!distance.is_null());
        let distance_len = batch_size;
        std::slice::from_raw_parts_mut(distance, distance_len as usize)
    };

    let led: Vec<i32> = edit_distance::levenshtein_edit_distance(a, b, a_lengths, b_lengths,
                                                                 batch_size as usize, max_length as usize);
    distance.copy_from_slice(&led);
}