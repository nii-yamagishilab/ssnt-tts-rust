extern crate ssnt_tts;

use ssnt_tts::{SsntTts, SsntTtsCpu, BeamSearchDecodingTable};


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
    let ssnt_tts_cpu = SsntTtsCpu::new(batch_size, T, U);

    let log_prob_history: Vec<f32> = vec![0.0, 0.0, 0.0];

    let input1: Vec<f32> = log(&vec![  // beam
                                       vec![0.8, 0.2],  // 0
                                       vec![0.8, 0.2],  // 0
                                       vec![0.8, 0.2],  // 0
    ]).into_iter().flatten().collect();

    let table1 = BeamSearchDecodingTable::new(input1.as_slice(), log_prob_history.as_slice(), T, beam_width, max_beam_width);
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
    let table2 = BeamSearchDecodingTable::new(input2.as_slice(), log_prob2.as_slice(), T, beam_width, max_beam_width);


    let result2 = ssnt_tts_cpu.beam_search_kernel(&table2, start_t.as_slice(), u.as_slice());
    println!("{:?}", result2);
}