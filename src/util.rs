extern crate rayon;

use rayon::prelude::*;
use std::collections::VecDeque;

pub fn extract_best_beam_branch_kernel(best_final_branch: i32, beam_branch: &[i32], t_history: &[i32], beam_width: i32, max_u: i32) -> (Vec<i32>, Vec<i32>) {
    let mut branch_buf: VecDeque<i32> = VecDeque::with_capacity(max_u as usize);
    let mut t_buf: VecDeque<i32> = VecDeque::with_capacity(max_u as usize);
    beam_branch.chunks(beam_width as usize)
        .zip(t_history.chunks(beam_width as usize))
        .rfold(best_final_branch as usize, |current_branch, (branch, ts)| {
            let prev_branch = branch[current_branch];
            let prev_t = ts[prev_branch as usize];
            branch_buf.push_front(prev_branch);
            t_buf.push_front(prev_t);
            prev_branch as usize
        });
    (Vec::from(branch_buf), Vec::from(t_buf))
}