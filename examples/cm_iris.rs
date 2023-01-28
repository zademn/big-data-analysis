use std::{time::Instant, vec};

use cluster_rs::{
    cmeans::{CMeans, CMeansParams},
};
use serde::Deserialize;

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
    let params = CMeansParams::default()
        .n_clusters(3)
        .parallelized(false)
        .verbose(true)
        .max_iter(100);

    println!("{params:?}");
    let t = Instant::now();
    let cm = CMeans::new(params).fit(&data);
    let duration = t.elapsed();
    println!("{:?}", cm.centroids());
    println!("{}", duration.as_millis());
}
