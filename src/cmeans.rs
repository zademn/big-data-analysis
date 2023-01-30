use std::cmp::Ordering;
use std::fmt;
use std::time::{Duration, Instant};

use itertools::izip;
use num_traits::Float;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::distance::{l2_dist, l2_dists};

// ----------------KMeans----------------
#[derive(Clone, Debug, PartialEq)]
pub struct CMeansParams<F: Float> {
    // Number of clusters
    pub n_clusters: usize,
    // Number of iterations. The algorithm stops after `max_iter` iterations
    pub max_iter: usize,
    // Tolerance. The algorithm stops after the distances between the old and new points
    // become less than `tol`
    pub tol: F,
    // Fuzzyness parameter
    pub fuzzyness: i32,
    // Flag to use rayon to pparallelize the computation
    pub parallelized: bool,
    // Flag for verbosity
    pub verbose: bool,
}

/// Compute the membership array for a `point` and the given centroids.
pub fn compute_memberships<F: Float + Send + Sync>(
    point: &[F],
    centroids: &[Vec<F>],
    fuzzyness: i32,
) -> Vec<F> {
    let dists = l2_dists(point, centroids);
    // let total_dist = dists.iter().fold(F::zero(), |acc, &a| acc + a);
    let denom = dists.iter().fold(F::zero(), |acc, &a| {
        acc + F::one() / a.powi(2 / (fuzzyness - 1))
    });
    let mut memberships = vec![F::zero(); centroids.len()];

    for (idx, centroid) in centroids.into_iter().enumerate() {
        let d = l2_dist(point, centroid);
        let num = F::one() / d.powi(2 / (fuzzyness - 1));
        // memberships[idx] = F::one() / (d / total_dist).powi(2 / (fuzzyness - 1))
        memberships[idx] = num / denom;
    }
    return memberships;
}

impl<F: Float> CMeansParams<F> {
    pub fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: F,
        parallelized: bool,
        verbose: bool,
        fuzzyness: i32,
    ) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol,
            parallelized,
            verbose,
            fuzzyness,
        }
    }
    // Builder setters
    // Builder setters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }
    pub fn parallelized(mut self, parallelized: bool) -> Self {
        self.parallelized = parallelized;
        self
    }
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    pub fn fuzzyness(mut self, fuzzyness: i32) -> Self {
        self.fuzzyness = fuzzyness;
        self
    }
    // Getters
}

impl<F: Float> Default for CMeansParams<F> {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 100,
            tol: F::from(0.001).unwrap(),
            parallelized: false,
            verbose: false,
            fuzzyness: 2,
        }
    }
}

// State
#[derive(Default, Clone)]
struct CMResults<F: Float> {
    pub(crate) centroids: Vec<Vec<F>>,
    pub(crate) memberships: Vec<Vec<F>>,
    pub(crate) labels: Vec<usize>,
}
// State types
#[derive(Default, Clone)]
pub struct Init;
#[derive(Default, Clone)]
pub struct Built<F: Float>(CMResults<F>);

pub struct CMeans<F: Float, S> {
    params: CMeansParams<F>,
    state: S,
}

impl<F: Float + Send + Sync + fmt::Debug> CMeans<F, Init> {
    pub fn new(params: CMeansParams<F>) -> Self {
        Self {
            params,
            state: Init::default(),
        }
    }

    pub fn fit(&self, data: &[Vec<F>]) -> CMeans<F, Built<F>> {
        // Get params
        let fuzzyness = self.params.fuzzyness;
        let n_clusters = self.params.n_clusters;
        let max_iter = self.params.max_iter;
        let tol = self.params.tol;
        let parallelized = self.params.parallelized;
        // Default rng
        let mut rng = thread_rng();

        // Data info
        let (n_samples, n_dim) = (data.len(), data[0].len());
        // initial centroids are chosen randomly

        // Results
        // Total inertia
        let mut inertia_ = F::infinity();
        // Distance from each point to its corresponding centroid
        // Loop
        let mut n_iter = 0;
        let mut current_tol = F::infinity();

        // Init centroids as random points
        let mut centroids = vec![vec![F::zero(); n_dim]; n_clusters];
        centroids = (0..n_clusters)
            .map(|_| {
                (0..n_dim)
                    .map(|_| F::from(rng.gen::<f64>()).unwrap())
                    .collect()
            })
            .collect();

        let mut time_assign = Duration::ZERO;
        let mut time_centroids = Duration::ZERO;
        let mut total_time = Duration::ZERO;
        let mut best_memberships;
        loop {
            let total_start = Instant::now();
            let mut memberships = vec![vec![F::zero(); n_clusters]; n_samples];
            // 1. Assign labels to points
            // For each data point, update its closest center and the distance to it.
            // This function can be parallelized.
            let assign_start = Instant::now();
            if parallelized {
                data.par_iter()
                    .zip(&mut memberships)
                    .for_each(|(point, memb)| {
                        *memb = compute_memberships(point, &centroids, fuzzyness);
                    });
            } else {
                for (point, memb) in izip!(data, memberships.iter_mut()) {
                    *memb = compute_memberships(point, &centroids, fuzzyness);
                }
            }
            // println!("centroids {centroids:?}");
            // println!("memberships {memberships:?}");
            let duration = assign_start.elapsed();
            time_assign += duration;

            // 2. Compute centroids
            // For each data point and labels compute the mean. Add to the centroids the samples labeled
            // then divide with the counts
            let centroids_start = Instant::now();
            let mut new_centroids = vec![vec![F::zero(); n_dim]; n_clusters];
            // Demominator
            let mut memb_sum = vec![F::zero(); n_clusters];
            if parallelized {
                let (nc, ms): (Vec<_>, Vec<_>) = data
                    .par_iter()
                    .zip(&memberships)
                    .fold(
                        || {
                            (
                                vec![vec![F::zero(); n_dim]; n_clusters],
                                vec![F::zero(); n_clusters],
                            )
                        },
                        |(mut new_centroids, mut memb_sum), (point, memb)| {
                            for (c_idx, centroid) in new_centroids.iter_mut().enumerate() {
                                let u = memb[c_idx].powi(fuzzyness);
                                *centroid = centroid
                                    .iter()
                                    .zip(point)
                                    .map(|(&ci, &xi)| ci + xi * u)
                                    .collect();
                                memb_sum[c_idx] = memb_sum[c_idx] + u;
                            }
                            (new_centroids, memb_sum)
                        },
                    )
                    .reduce(
                        || {
                            (
                                vec![vec![F::zero(); n_dim]; n_clusters],
                                vec![F::zero(); n_clusters],
                            )
                        },
                        |(mut a0, mut a1), (b0, b1)| {
                            a0 = a0
                                .iter()
                                .zip(b0)
                                .map(|(aa0, bb0)| {
                                    aa0.iter().zip(bb0).map(|(&ai, bi)| ai + bi).collect()
                                })
                                .collect();
                            a1 = a1.iter().zip(b1).map(|(&ai, bi)| ai + bi).collect();
                            (a0, a1)
                        },
                    );
                // .find_first(|_| true)
                // .unwrap();
                new_centroids = nc;
                memb_sum = ms;
            } else {
                for (point, memb) in data.iter().zip(&memberships) {
                    for (c_idx, centroid) in new_centroids.iter_mut().enumerate() {
                        let u = memb[c_idx].powi(fuzzyness);
                        *centroid = centroid
                            .iter()
                            .zip(point)
                            .map(|(&ci, &xi)| ci + xi * u)
                            .collect();
                        memb_sum[c_idx] = memb_sum[c_idx] + u;
                    }
                }
            }

            // Divide centroids
            for (centroid, s) in new_centroids.iter_mut().zip(memb_sum) {
                *centroid = centroid.iter().map(|&e| e / s).collect();
            }

            let duration = centroids_start.elapsed();
            time_centroids += duration;

            // Check stopping conditions
            current_tol = F::zero();
            for (c1, c2) in centroids.iter().zip(new_centroids.iter()) {
                current_tol = current_tol + l2_dist(c1, c2)
            }
            centroids = new_centroids.clone();
            n_iter += 1;
            let duration = total_start.elapsed();
            total_time += duration;

            best_memberships = memberships.clone();
            if n_iter >= max_iter || current_tol < tol {
                break;
            }
        }
        let labels: Vec<usize> = best_memberships
            .iter()
            .map(|memb| {
                memb.iter()
                    .enumerate()
                    .max_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .collect();
        if self.params.verbose {
            println!("Total time {}ms", total_time.as_millis());
            println!(
                "Total time to assign centroids {}ms",
                time_assign.as_millis()
            );
            println!(
                "Total time to compute new centroids: {}ms",
                time_centroids.as_millis()
            );
        }
        CMeans {
            params: self.params.clone(),
            state: Built(CMResults {
                centroids,
                memberships: best_memberships,
                labels,
            }),
        }
    }
}

impl<F: Float> CMeans<F, Built<F>> {
    pub fn centroids(&self) -> Vec<Vec<F>> {
        self.state.0.centroids.clone()
    }
    pub fn memberships(&self) -> Vec<Vec<F>> {
        self.state.0.memberships.clone()
    }
    pub fn labels(&self) -> Vec<usize> {
        self.state.0.labels.clone()
    }
    // pub fn dists(&self) -> Vec<F> {
    //     self.state.0.dists.clone()
    // }
}
