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


pub fn upsample_source_indexes(duration: &[i32], output_length: &[i32], beam_width: i32, max_t: i32, max_u: i32, upsampled_source_indexes: &mut [i32]) {
    // (B, W, T)
    duration.par_chunks((beam_width * max_t) as usize)
        // (B)
        .zip(output_length.par_chunks(1))
        // (B, W, U)
        .zip(upsampled_source_indexes.par_chunks_mut((beam_width * max_u) as usize))
        .for_each(|((duration, output_length), upsampled_source_indexes)| {
            duration.par_chunks(max_t as usize)
                .zip(upsampled_source_indexes.par_chunks_mut(max_u as usize))
                .for_each(|(duration, upsampled_source_indexes)| {
                    let upsampled: Vec<i32> = duration.into_iter().enumerate().flat_map(|(t, d)| {
                        if *d == 0 {
                            vec![]
                        } else {
                            vec![t as i32; *d as usize]
                        }
                    }).collect();
                    println!("{:?}", upsampled);
                    assert_eq!(upsampled.len(), output_length[0] as usize);
                    upsampled.into_iter()
                        .zip(upsampled_source_indexes.into_iter().take(output_length[0] as usize))
                        .for_each(|(s, t)| {
                            *t = s;
                        });
                })
        });
}