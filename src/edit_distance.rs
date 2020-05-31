use std::cmp;


use rayon::prelude::*;

pub fn levenshtein_edit_distance(a: &[i32], b: &[i32], a_lengths: &[i32], b_lengths: &[i32],
                                 batch_size: usize, max_length: usize) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), batch_size * max_length);
    assert_eq!(a_lengths.len(), batch_size);
    assert_eq!(a_lengths.len(), b_lengths.len());

    let led: Vec<i32> = a.par_chunks(max_length)
        .zip(b.par_chunks(max_length))
        .zip(a_lengths.par_chunks(1)
            .zip(b_lengths.par_chunks(1)))
        .map(|ab| {
            let ((a, b), (a_length, b_length)) = ab;
            let a = &a[..a_length[0] as usize];
            let b = &b[..b_length[0] as usize];
            levenshtein_edit_distance_kernel(a, b)
        }).collect();
    return led;
}


// http://kaldi-asr.org/doc/edit-distance-inl_8h_source.html
pub fn levenshtein_edit_distance_kernel(a: &[i32], b: &[i32]) -> i32 {
    let M = a.len();
    let N = b.len();
    let mut e: Vec<i32> = (0..=N as i32).collect();
    let mut e_tmp: Vec<i32> = vec![-1; N + 1];

    for m in 1..=M {
        e_tmp[0] = e[0] + 1;

        for n in 1..=N {
            // E(m-1, n-1) + delta(a_{m-1}, b_{n-1})
            let term1 = e[n - 1] + delta(a[m - 1], b[n - 1]);
            // E(m-1, n) + 1
            let term2 = e[n] + 1;
            // E(m, n-1) + 1
            let term3 = e_tmp[n - 1] + 1;
            e_tmp[n] = cmp::min(term1, cmp::min(term2, term3));
        }
        e = e_tmp.clone();
    }

    return e[N];
}


#[inline]
fn delta(a: i32, b: i32) -> i32 {
    if a == b {
        return 0;
    } else {
        return 1;
    }
}
