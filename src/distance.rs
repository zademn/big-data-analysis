use num_traits::Float;
use rayon::prelude::*;

/// Distance between 2 points
pub fn l2_dist<F: Float>(a: &[F], b: &[F]) -> F {
    let mut res = F::zero();
    for (aa, bb) in a.iter().zip(b) {
        let d = aa.clone() - bb.clone();
        res = res + d.clone() * d
    }
    F::from(res.to_f64().unwrap().sqrt()).unwrap()
}

/// Distances between 1 point and a set of points
pub fn l2_dists<F: Float>(point: &[F], b: &[Vec<F>]) -> Vec<F> {
    let n_samples = b.len();
    let mut res = vec![F::zero(); n_samples];
    for (row, resi) in b.iter().zip(res.iter_mut()) {
        *resi = l2_dist(point, row)
    }
    // b.iter()
    //     .zip(res.iter_mut())
    //     .map(|(row, resi)| *resi = l2_dist(point, row));
    res
}

/// Closest point index and distance to it.
pub fn closest_point<F: Float + Send + Sync>(point: &[F], b: &[Vec<F>]) -> (usize, F) {
    let mut d_min = F::infinity();
    let mut idx_min = 0;

    (idx_min, d_min) = if b.len() > 16 {
        b.par_iter()
            .enumerate()
            .fold(
                || (0, F::infinity()),
                |(mut idx_min, mut d_min), (idx, bb)| {
                    let d = l2_dist(point, bb);
                    if d < d_min {
                        d_min = d;
                        idx_min = idx;
                    }
                    (idx_min, d_min)
                },
            )
            .find_first(|_| true)
            .unwrap()
    } else {
        for (idx, bb) in b.into_iter().enumerate() {
            let d = l2_dist(point, bb);
            if d < d_min {
                d_min = d;
                idx_min = idx;
            }
        }
        (idx_min, d_min)
    };
    (idx_min, d_min)
}

/// Closest point index and distance to it.
pub fn closest_2_points<F: Float + Send + Sync>(
    point: &[F],
    b: &[Vec<F>],
) -> ((usize, F), (usize, F)) {
    let mut d1 = F::infinity();
    let mut d2 = F::infinity();
    let mut i1 = 0;
    let mut i2 = 0;

    ((i1, d1), (i2, d2)) = if b.len() > 16 {
        b.par_iter()
            .enumerate()
            .fold(
                || ((0, F::infinity()), (0, F::infinity())),
                |((mut i1, mut d1), (mut i2, mut d2)), (idx, bb)| {
                    let d = l2_dist(point, bb);
                    if d < d1 {
                        d2 = d1;
                        i2 = i1;
                        d1 = d;
                        i1 = idx;
                    } else if d < d2 {
                        d2 = d;
                        i2 = idx;
                    }
                    ((i1, d1), (i2, d2))
                },
            )
            .find_first(|_| true)
            .unwrap()
    } else {
        for (idx, bb) in b.into_iter().enumerate() {
            let d = l2_dist(point, bb);
            if d < d1 {
                d2 = d1;
                i2 = i1;
                d1 = d;
                i1 = idx;
            } else if d < d2 {
                d2 = d;
                i2 = idx;
            }
        }
        ((i1, d1), (i2, d2))
    };
    ((i1, d1), (i2, d2))
}
