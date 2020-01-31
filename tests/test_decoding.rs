extern crate ssnt_tts;

use ssnt_tts::{SsntTts, SsntTtsCpu, BeamSearchDecodingTable, util};


fn log(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    input.iter().map(|row| {
        row.iter().map(|item| item.ln()).collect()
    }).collect()
}


#[test]
fn beam_search_decode_test() {
    let T: usize = 4;
    let U: usize = 0;
    let batch_size: i32 = 1;
    let beam_width = 3;
    let max_beam_width = 3;
    let is_finished = vec![false, false, false];
    let ssnt_tts_cpu = SsntTtsCpu::new(batch_size, T, U);

    let log_prob_history: Vec<f32> = vec![0.0, 0.0, 0.0];

    let input1: Vec<f32> = log(&vec![  // beam
                                       vec![0.8, 0.2],  // 0
                                       vec![0.8, 0.2],  // 0
                                       vec![0.8, 0.2],  // 0
    ]).into_iter().flatten().collect();

    let table1 = BeamSearchDecodingTable::new(input1.as_slice(), log_prob_history.as_slice(), is_finished.as_slice(), T, beam_width, max_beam_width);
    let start_t: Vec<usize> = vec![0, 0, 0];
    let u = vec![0, 0, 0];

    let result1 = ssnt_tts_cpu.beam_search_kernel(&table1, start_t.as_slice(), u.as_slice());
    println!("{:?}", result1);

    let log_prob2: Vec<f32> = result1.iter().map(|dr| dr.log_prob).collect();
    println!("{:?}", log_prob2);

    let input2: Vec<f32> = log(&vec![  // beam
                                       vec![0.8, 0.2],  // 0
                                       vec![0.8, 0.2],  // 0
                                       vec![0.8, 0.2],  // 0
    ]).into_iter().flatten().collect();
    let table2 = BeamSearchDecodingTable::new(input2.as_slice(), log_prob2.as_slice(), is_finished.as_slice(), T, beam_width, max_beam_width);


    let result2 = ssnt_tts_cpu.beam_search_kernel(&table2, start_t.as_slice(), u.as_slice());
    println!("{:?}", result2);
}

#[test]
fn extract_best_beam_branch_test() {
    let beam_width = 10;
    let max_u = 60;
    let beam_branch: Vec<i32> = vec![
        vec![0,3,0,5,2,3,4,1,1,9],
        vec![0,5,0,1,1,3,2,2,3,4],
        vec![0,5,0,1,2,3,4,2,1,3],
        vec![8,3,0,0,7,1,2,1,3,4],
        vec![0,0,1,1,2,3,4,5,6,7],
        vec![1,0,1,2,3,4,5,0,3,6],
        vec![0,0,7,1,8,3,4,5,6,2],
        vec![0,0,1,1,4,2,3,5,2,6],
        vec![0,1,0,2,2,3,4,6,4,5],
        vec![0,4,0,1,3,2,4,2,5,6],
        vec![0,7,0,1,2,1,3,4,6,8],
        vec![0,0,2,1,4,1,3,5,3,6],
        vec![3,1,0,5,0,6,2,4,3,5],
        vec![0,4,5,0,1,2,3,4,3,6],
        vec![0,0,1,2,1,2,3,4,5,7],
        vec![0,1,1,3,2,2,3,4,5,6],
        vec![2,3,0,1,2,3,4,5,5,6],
        vec![7,0,0,2,1,3,4,5,6,1],
        vec![1,9,0,2,1,0,3,4,5,6],
        vec![0,0,1,2,3,1,4,5,6,7],
        vec![1,0,1,3,4,5,2,7,6,2],
        vec![0,0,1,2,7,3,4,5,6,8],
        vec![0,0,1,2,3,4,4,5,6,7],
        vec![0,1,0,2,3,4,5,6,7,8],
        vec![2,0,1,2,3,4,5,6,7,8],
        vec![0,0,1,2,3,4,5,6,7,8],
        vec![0,0,1,2,3,4,5,6,7,8],
        vec![0,1,1,2,3,4,5,6,7,8],
        vec![0,1,2,1,3,4,5,6,7,8],
        vec![3,0,1,2,3,4,5,6,7,8],
        vec![0,0,1,2,3,4,5,6,7,8],
        vec![1,2,0,3,0,4,5,6,7,8],
        vec![4,0,1,2,3,5,4,6,7,8],
        vec![0,0,1,2,3,4,5,6,7,8],
        vec![1,0,1,2,3,4,5,6,7,8],
        vec![0,0,1,2,3,4,5,6,7,8],
        vec![1,0,1,2,3,4,5,6,7,8],
        vec![0,0,1,2,3,4,5,6,7,8],
        vec![0,0,1,2,3,4,5,6,7,8],
        vec![0,1,0,2,3,4,5,6,7,8],
        vec![0,1,2,2,3,4,5,6,7,8],
        vec![0,1,2,3,4,3,5,6,7,8],
        vec![0,1,2,3,4,5,6,7,5,8],
        vec![0,1,2,8,3,4,5,6,7,8],
        vec![0,1,2,3,4,3,5,6,7,8],
        vec![0,1,2,3,4,5,5,6,7,8],
        vec![0,1,2,3,5,4,5,6,7,8],
        vec![0,1,2,4,3,4,5,6,7,8],
        vec![0,1,2,3,3,4,5,6,7,8],
        vec![0,1,2,3,4,4,5,6,7,8],
        vec![0,1,2,3,5,4,5,6,7,8],
        vec![0,1,2,3,4,5,6,4,7,8],
        vec![0,1,2,3,4,5,6,7,7,8],
        vec![0,1,2,3,7,4,5,6,7,8],
        vec![0,1,2,3,4,5,4,6,7,8],
        vec![0,1,2,3,4,5,6,7,6,8],
        vec![0,8,1,2,3,4,5,6,7,8],
        vec![0,1,2,1,3,4,5,6,7,8],
        vec![0,1,2,3,4,5,6,3,7,8],
        vec![0,1,2,3,4,5,6,7,8,9],
    ].into_iter().flatten().collect();

    let (best_beam_branch, best_t) = util::extract_best_beam_branch_kernel(9,
                                                                           beam_branch.as_slice(),
                                                                           beam_branch.as_slice(),
                                                                           beam_width, max_u);

    assert_eq!(best_beam_branch, vec![5, 1, 8, 0, 1, 0, 0, 0, 2, 7,
                                      1, 3, 0, 0, 1, 2, 0, 1, 0, 1,
                                      0, 0, 0, 2, 0, 0, 1, 1, 3, 0,
                                      0, 4, 0, 1, 0, 1, 0, 0, 0, 2,
                                      3, 5, 8, 3, 5, 5, 4, 3, 4, 5,
                                      4, 7, 7, 4, 6, 6, 7, 8, 9, 9]);
}