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
        .from_path("data/BDA/Dataset 1/Iris-1500.txt")
        .expect("Failed to open csv");

    for record in reader.deserialize() {
        let record: IrisRecord = record.unwrap();
        let mut row = vec![vec![record.col1, record.col2, record.col3, record.col4]];
        data.append(&mut row);
    }
    println!("Data length {}", data.len());

    let ks = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut inertia_diffs = vec![];
    let mut silhouette_scores = vec![];
    let mut prev_inertia = 0.;
    for k in ks {
        let params = KMeansParams::default().n_clusters(k);
        let km = KMeans::new(params).fit(&data);
        let centroids = km.centroids();
        let inertia = km.inertia();
        let s = silhouette_score_km(&data, &centroids);
        inertia_diffs.push(prev_inertia - inertia);
        prev_inertia = inertia;
        silhouette_scores.push(s);
    }
    println!("n_cluster tried: {ks:?}");
    println!("Inertia scores: {inertia_diffs:?}");
    println!("Silhouette scores: {silhouette_scores:?}");
}
