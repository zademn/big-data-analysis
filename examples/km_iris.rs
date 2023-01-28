use std::{time::Instant, vec};

use cluster_rs::{
    kmeans::{KMeans, KMeansParams},
    metrics::silhouette_score_km,
};
use serde::Deserialize;

use plotters::prelude::*;
#[derive(Debug, Deserialize)]
struct IrisRecord {
    col1: f64,
    col2: f64,
    col3: f64,
    col4: f64,
    label: String,
}
fn main() {
    let mut data: Vec<Vec<_>> = vec![];
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("data/BDA/Dataset 1/Iris-150.txt")
        .expect("Failed to open csv");

    for record in reader.deserialize() {
        let record: IrisRecord = record.unwrap();
        let mut row = vec![vec![record.col1, record.col2, record.col3, record.col4]];
        data.append(&mut row);
    }
    println!("Data length {}", data.len());
    let params = KMeansParams::default()
        .n_clusters(3)
        .parallelized(false)
        .verbose(true);

    println!("KM parameters {params:?}");
    let km = KMeans::new(params).fit(&data);
    let t = Instant::now();
    let centroids = km.centroids();
    let inertia = km.inertia();
    let labels = km.labels();
    let duration = t.elapsed();
    println!("Centroids: {centroids:?}");
    println!("Inertia: {inertia:?}");
    println!(
        "Silhouette Score: {}",
        silhouette_score_km(&data, &centroids)
    );
    println!("Labels: {:?}", labels);
    println!("Duration: {}ms", duration.as_millis());
}
