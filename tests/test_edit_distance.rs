extern crate ssnt_tts;


use ssnt_tts::edit_distance::{levenshtein_edit_distance_kernel, levenshtein_edit_distance};


// http://kaldi-asr.org/doc/edit-distance-test_8cc_source.html#l00026

#[test]
fn test_edit_distance0() {
    let a0: Vec<i32> = vec![];
    let b0: Vec<i32> = vec![];
    let error0 = levenshtein_edit_distance_kernel(&a0, &b0);
    assert_eq!(error0, 0);

    let a1: Vec<i32> = vec![1];
    let b1: Vec<i32> = vec![1];
    let error1 = levenshtein_edit_distance_kernel(&a1, &b1);
    assert_eq!(error1, 0);

    let a2: Vec<i32> = vec![1, 2];
    let b2: Vec<i32> = vec![1, 2];
    let error2 = levenshtein_edit_distance_kernel(&a2, &b2);
    assert_eq!(error2, 0);
}

#[test]
fn test_edit_distance1() {
    let a0: Vec<i32> = vec![1];
    let b0: Vec<i32> = vec![];
    let error0 = levenshtein_edit_distance_kernel(&a0, &b0);
    assert_eq!(error0, 1);

    let a1: Vec<i32> = vec![1];
    let b1: Vec<i32> = vec![1, 2];
    let error1 = levenshtein_edit_distance_kernel(&a1, &b1);
    assert_eq!(error1, 1);

    let a2: Vec<i32> = vec![1, 2, 3, 4];
    let b2: Vec<i32> = vec![1, 2, 4];
    let error2 = levenshtein_edit_distance_kernel(&a2, &b2);
    assert_eq!(error2, 1);
}

#[test]
fn test_edit_distance2() {
    let a0: Vec<i32> = vec![1, 2, 3, 4, 5];
    let b0: Vec<i32> = vec![1, 2, 4];
    let error0 = levenshtein_edit_distance_kernel(&a0, &b0);
    assert_eq!(error0, 2);

    let a1: Vec<i32> = vec![1, 2, 3, 4, 5];
    let b1: Vec<i32> = vec![1, 2, 4, 6];
    let error1 = levenshtein_edit_distance_kernel(&a1, &b1);
    assert_eq!(error1, 2);

    let a2: Vec<i32> = vec![1, 2, 3, 4, 5, 1];
    let b2: Vec<i32> = vec![1, 2, 4, 6, 1];
    let error2 = levenshtein_edit_distance_kernel(&a2, &b2);
    assert_eq!(error2, 2);
}

#[test]
fn test_edit_distance3() {
    let a0: Vec<i32> = vec![1, 2, 3, 4, 5, 1];
    let b0: Vec<i32> = vec![1, 2, 4, 6, 1, 10];
    let error0 = levenshtein_edit_distance_kernel(&a0, &b0);
    assert_eq!(error0, 3);
}

#[test]
fn test_edit_distance_batched() {
    let batch_size: usize = 10;
    let max_length: usize = 6;
    let a: Vec<i32> = vec![
        vec![-1, -2, -3, -4, -5, -6],
        vec![1, -1, -2, -3, -4, -5],
        vec![1, 2, -1, -2, -3, -4],
        vec![1, -1, -2, -3, -4, -5],
        vec![1, -1, -2, -3, -4, -5],
        vec![1, 2, 3, 4, -1, -2],
        vec![1, 2, 3, 4, 5, -1],
        vec![1, 2, 3, 4, 5, -1],
        vec![1, 2, 3, 4, 5, 1],
        vec![1, 2, 3, 4, 5, 1]].into_iter().flatten().collect();
    let a_length = vec![0, 1, 2, 1, 1, 4, 5, 5, 6, 6];
    let b: Vec<i32> = vec![
        vec![-1, -1, -1, -1, -1, -1],
        vec![1, -1, -1, -1, -1, -1],
        vec![1, 2, -1, -1, -1, -1],
        vec![-6, -5, -4, -3, -2, -1],
        vec![1, 2, -1, -1, -1, -1],
        vec![1, 2, 4, -3, -2, -1],
        vec![1, 2, 4, -3, -2, -1],
        vec![1, 2, 4, 6, -2, -1],
        vec![1, 2, 4, 6, 1, -1],
        vec![1, 2, 4, 6, 1, 10]].into_iter().flatten().collect();
    let b_length = vec![0, 1, 2, 0, 2, 3, 3, 4, 5, 6];
    let error = levenshtein_edit_distance(a.as_slice(),
                                          b.as_slice(),
                                          a_length.as_slice(),
                                          b_length.as_slice(),
                                          batch_size,
                                          max_length);
    let error_answer = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3];
    assert_eq!(error, error_answer);
}