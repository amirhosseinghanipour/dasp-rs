use ndarray::Array2;

pub fn amplitude_to_db(S: &Array2<f32>, ref_val: Option<f32>, amin: Option<f32>, top_db: Option<f32>) -> Array2<f32> {
    let ref_val = ref_val.unwrap_or(1.0);
    let amin_val = amin.unwrap_or(1e-5);
    let top_db_val = top_db.unwrap_or(80.0);
    S.mapv(|x| {
        let db = 20.0 * (x.max(amin_val) / ref_val).log10();
        db.max(-top_db_val)
    })
}

pub fn db_to_amplitude(S_db: &Array2<f32>, ref_val: Option<f32>) -> Array2<f32> {
    let ref_val = ref_val.unwrap_or(1.0);
    S_db.mapv(|x| ref_val * 10.0f32.powf(x / 20.0))
}

pub fn power_to_db(S: &Array2<f32>, ref_val: Option<f32>, amin: Option<f32>, top_db: Option<f32>) -> Array2<f32> {
    let ref_val = ref_val.unwrap_or(1.0);
    let amin_val = amin.unwrap_or(1e-10);
    let top_db_val = top_db.unwrap_or(80.0);
    S.mapv(|x| {
        let db = 10.0 * (x.max(amin_val) / ref_val).log10();
        db.max(-top_db_val)
    })
}

pub fn db_to_power(S_db: &Array2<f32>, ref_val: Option<f32>) -> Array2<f32> {
    let ref_val = ref_val.unwrap_or(1.0);
    S_db.mapv(|x| ref_val * 10.0f32.powf(x / 10.0))
}

pub fn perceptual_weighting(_S: &Array2<f32>, _frequencies: &[f32], _kind: Option<&str>) -> Array2<f32> { unimplemented!() }
pub fn frequency_weighting(frequencies: &[f32], kind: Option<&str>) -> Vec<f32> {
    match kind.unwrap_or("A") {
        "A" => A_weighting(frequencies, None),
        "B" => B_weighting(frequencies, None),
        "C" => C_weighting(frequencies, None),
        "D" => D_weighting(frequencies, None),
        _ => unimplemented!(),
    }
}

pub fn multi_frequency_weighting(_frequencies: &[f32], _kinds: &[&str]) -> Vec<Vec<f32>> { unimplemented!() }

pub fn A_weighting(frequencies: &[f32], min_db: Option<f32>) -> Vec<f32> {
    let min_db_val = min_db.unwrap_or(-80.0);
    frequencies.iter().map(|&f| {
        let f2 = f * f;
        let num = 12194.0f32.powi(2) * f2.powi(2);
        let den = (f2 + 20.6f32.powi(2)) * (f2 + 12194.0f32.powi(2)) * ((f2 + 107.7f32.powi(2)) * (f2 + 737.9f32.powi(2))).sqrt();
        let db = 20.0 * (num / den).log10() + 2.0;
        db.max(min_db_val)
    }).collect()
}

pub fn B_weighting(_frequencies: &[f32], _min_db: Option<f32>) -> Vec<f32> { unimplemented!() }
pub fn C_weighting(_frequencies: &[f32], _min_db: Option<f32>) -> Vec<f32> { unimplemented!() }
pub fn D_weighting(_frequencies: &[f32], _min_db: Option<f32>) -> Vec<f32> { unimplemented!() }
pub fn pcen(_S: &Array2<f32>, _sr: Option<u32>, _hop_length: Option<usize>, _gain: Option<f32>, _bias: Option<f32>) -> Array2<f32> { unimplemented!() }