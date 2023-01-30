use std::{time::Instant, vec};

use cluster_rs::cmeans::{CMeans, CMeansParams};
use serde::Deserialize;

fn main() {
    let mut data: Vec<Vec<_>> = vec![];
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("data/BDA/Dataset 2/Dataset2.csv")
        .expect("Failed to open csv");

    for record in reader.records() {
        let mut row = record
            .unwrap()
            .iter()
            .map(|e| e.parse::<f64>().unwrap())
            .collect();
        data.push(row);
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
    println!("{:?}", cm.labels());
    println!("{}", duration.as_millis());
}
