use crate::distance::closest_2_points;
use num_traits::Float;
use std::{fmt::Debug, iter::Sum};

pub fn silhouette_score_km<F: Float + Send + Sync + Sum + Debug>(
    data: &[Vec<F>],
    centers: &[Vec<F>],
) -> F {
    let n_samples = data.len();
    let mut scores = vec![F::zero(); n_samples];
    for (idx, point) in data.iter().enumerate() {
        let ((_, d1), (_, d2)) = closest_2_points(point, centers);
        scores[idx] = (d2 - d1) / d1.max(d2);
    }
    scores.into_iter().sum::<F>() / F::from(n_samples).unwrap()
}
