extern crate rayon;

use rayon::prelude::*;
use std::collections::VecDeque;

pub fn order_beam_branch(final_branch: &[i32], beam_branch: &[i32], beam_width: i32, max_t: i32, ordered_beam_branch: &mut [i32]) {
    // (B, W)
    final_branch.par_chunks(beam_width as usize)
        // (B, T, W)
        .zip(beam_branch.par_chunks((max_t * beam_width) as usize))
        // (B, W, T)
        .zip(ordered_beam_branch.par_chunks_mut((beam_width * max_t) as usize))
        .for_each(|((final_branch, beam_branch), ordered_beam_branch)| {
            // (W)
            final_branch.par_chunks(1)
                // (W, T)
                .zip(ordered_beam_branch.par_chunks_mut(max_t as usize))
                .for_each(|(final_branch, ordered_beam_branch)| {
                let single_final_branch: i32 = final_branch[0];
                let beam_branch: Vec<i32> = extract_beam_branch_kernel(single_final_branch, beam_branch, beam_width, max_t);
                ordered_beam_branch.copy_from_slice(beam_branch.as_slice());
            })
        });
}

fn extract_beam_branch_kernel(best_final_branch: i32, beam_branch: &[i32], beam_width: i32, max_t: i32) -> Vec<i32> {
    let mut branch_buf: VecDeque<i32> = VecDeque::with_capacity(max_t as usize);
    // (T, W)
    beam_branch.chunks(beam_width as usize)
        .rfold(best_final_branch, |current_branch, branch| {
            let prev_branch: i32 = branch[current_branch as usize];
            branch_buf.push_front(current_branch);
            prev_branch
        });
    Vec::from(branch_buf)
}