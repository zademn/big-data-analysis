use std::collections::HashMap;
use std::fmt;
use std::iter::Sum;
use std::time::{Duration, Instant};

use itertools::izip;
use num_traits::Float;
use rand::thread_rng;
use rayon::prelude::*;

use crate::distance::{closest_point, l2_dist, l2_dists};

// ----------------KMeans----------------
#[derive(Clone, Debug, PartialEq)]
pub struct KMeansParams<F: Float> {
    // Number of clusters
    pub n_clusters: usize,
    // Number of iterations. The algorithm stops after `max_iter` iterations
    pub max_iter: usize,
    // Number of initializations (how many times to run kmeans from scratch)
    pub n_init: usize,
    // Tolerance. The algorithm stops after the distances between the old and new points
    // become less than `tol`
    pub tol: F,
    // Flag to use rayon to pparallelize the computation
    pub parallelized: bool,
    pub verbose: bool,
}
impl<F: Float> KMeansParams<F> {
    pub fn new(
        n_clusters: usize,
        max_iter: usize,
        n_init: usize,
        tol: F,
        parallelized: bool,
        verbose: bool,
    ) -> Self {
        Self {
            n_clusters,
            max_iter,
            n_init,
            tol,
            parallelized,
            verbose,
        }
    }
    // Builder setters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
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
}

impl<F: Float> Default for KMeansParams<F> {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            n_init: 10,
            max_iter: 100,
            tol: F::from(0.001).unwrap(),
            parallelized: false,
            verbose: false,
        }
    }
}

#[derive(Default, Clone)]
struct KMResults<F: Float> {
    pub(crate) centroids: Vec<Vec<F>>,
    pub(crate) inertia: F,
    pub(crate) labels: Vec<usize>,
    pub(crate) dists: Vec<F>,
}
// State types
#[derive(Default, Clone)]
pub struct Init;
#[derive(Default, Clone)]
pub struct Built<F: Float>(KMResults<F>);

#[derive(Default)]
pub struct KMeans<F: Float, S> {
    params: KMeansParams<F>,
    state: S,
}

impl<F: Float + Send + Sync + fmt::Debug + Sum> KMeans<F, Init> {
    pub fn new(params: KMeansParams<F>) -> Self {
        Self {
            params,
            state: Init::default(),
        }
    }

    pub fn fit(self, data: &[Vec<F>]) -> KMeans<F, Built<F>> {
        // Get params
        let n_clusters = self.params.n_clusters;
        let max_iter = self.params.max_iter;
        let tol = self.params.tol;
        let parallelized = self.params.parallelized;
        let n_init = self.params.n_init;
        // Times for info
        let mut time_assign = Duration::ZERO;
        let mut time_centroids = Duration::ZERO;
        let mut total_time = Duration::ZERO;

        // Default rng
        let mut rng = thread_rng();

        // Data info
        let (n_samples, n_dim) = (data.len(), data[0].len());

        // Results
        let mut best_inertia = F::infinity();
        let mut best_centroids = vec![vec![F::zero(); n_dim]; n_clusters];
        let mut best_labels = vec![0; n_samples];
        let mut best_dists = vec![F::zero(); n_samples];

        // Distance from each point to its corresponding centroid
        // Loop
        let mut n_iter = 0;
        let mut current_tol;

        for _ in 0..n_init {
            let mut final_labels;
            let mut final_dists;

            // initial centroids are chosen randomly
            let init_idxs = rand::seq::index::sample(&mut rng, n_samples, n_clusters).into_vec();
            let mut centroids: Vec<Vec<F>> =
                init_idxs.into_iter().map(|i| data[i].clone()).collect();
            loop {
                let total_start = Instant::now();
                let mut dists = vec![F::zero(); n_samples];
                let mut labels = vec![0; n_samples];
                // 1. Assign labels to points
                // For each data point, update its closest center and the distance to it.
                // This function can be parallelized.
                let assign_start = Instant::now();
                if parallelized {
                    data.par_iter().zip(&mut labels).zip(&mut dists).for_each(
                        |((point, label), dist)| {
                            let (l, d) = closest_point(point, &centroids);
                            *label = l;
                            *dist = d;
                        },
                    );
                } else {
                    // assign labels and dists
                    for (point, label, dist) in izip!(data, labels.iter_mut(), dists.iter_mut()) {
                        let (l, d) = closest_point(point, &centroids);
                        *label = l;
                        *dist = d;
                    }
                }
                let duration = assign_start.elapsed();
                time_assign += duration;

                // 2. Compute centroids
                // For each data point and labels compute the mean. Add to the centroids the samples labeled
                // then divide with the counts
                let centroids_start = Instant::now();
                let mut new_centroids = vec![vec![F::zero(); n_dim]; n_clusters];
                let mut counts = vec![F::zero(); n_clusters];
                if parallelized {
                    let (nc, c): (Vec<_>, Vec<_>) = data
                        .par_iter()
                        .zip(&labels)
                        .fold(
                            || {
                                (
                                    vec![vec![F::zero(); n_dim]; n_clusters],
                                    vec![F::zero(); n_clusters],
                                )
                            },
                            |(mut new_centroids, mut counts), (point, label)| {
                                new_centroids[*label] = new_centroids[*label]
                                    .iter()
                                    .zip(point)
                                    .map(|(&a, &b)| a + b)
                                    .collect();
                                counts[*label] = counts[*label] + F::one();
                                (new_centroids, counts)
                            },
                        )
                        .reduce(
                            || {
                                (
                                    centroids.clone(),
                                    // vec![vec![F::zero(); n_dim]; n_clusters],
                                    vec![F::one(); n_clusters],
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

                    // println!("{}", nc.len());
                    // println!("{c:?}");
                    new_centroids = nc;
                    counts = c;
                } else {
                    for (point, label) in data.iter().zip(labels.iter()) {
                        new_centroids[*label] = new_centroids[*label]
                            .iter()
                            .zip(point)
                            .map(|(&a, &b)| a + b)
                            .collect();
                        counts[*label] = counts[*label] + F::one();
                    }
                }
                for (label, centroid) in new_centroids.iter_mut().enumerate() {
                    let count = counts[label];
                    *centroid = centroid.iter().map(|&c| c / count).collect();
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

                final_labels = labels;
                final_dists = dists;

                if n_iter >= max_iter || current_tol < tol {
                    break;
                }
            }

            let inertia = final_dists.clone().into_iter().sum();
            if inertia < best_inertia {
                best_inertia = inertia;
                best_centroids = centroids;
                best_labels = final_labels;
                best_dists = final_dists;
            }
        }

        // Print times if verbose
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

        KMeans {
            params: self.params.clone(),
            state: Built(KMResults {
                centroids: best_centroids,
                inertia: best_inertia,
                labels: best_labels,
                dists: best_dists,
            }),
        }
    }
}

impl<F: Float> KMeans<F, Built<F>> {
    pub fn centroids(&self) -> Vec<Vec<F>> {
        self.state.0.centroids.clone()
    }
    pub fn inertia(&self) -> F {
        self.state.0.inertia.clone()
    }
    pub fn labels(&self) -> Vec<usize> {
        self.state.0.labels.clone()
    }
    pub fn dists(&self) -> Vec<F> {
        self.state.0.dists.clone()
    }
}
