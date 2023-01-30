use cluster_rs::cmeans::{CMeans, CMeansParams};
use criterion::{criterion_group, criterion_main, Criterion};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct IrisRecord {
    col1: f64,
    col2: f64,
    col3: f64,
    col4: f64,
    label: String,
}

pub fn cmeans_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmeans");

    for size in [150, 1_500, 15_000, 150_000, 1_500_000, 15_000_000] {
        let filename = format!("Iris-{}.txt", size);
        let mut data: Vec<Vec<_>> = vec![];
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(format!("data/BDA/Dataset 1/{filename}"))
            .expect("Failed to open csv");

        for record in reader.deserialize() {
            let record: IrisRecord = record.unwrap();
            let mut row = vec![vec![record.col1, record.col2, record.col3, record.col4]];
            data.append(&mut row);
        }

        group
            .sample_size(10)
            .bench_function(format!("iris-{}, parallelized=false", size), |b| {
                b.iter(|| {
                    let params = CMeansParams::default()
                        .n_clusters(15)
                        .parallelized(false)
                        .max_iter(100)
                        .tol(0.001);
                    let cm = CMeans::new(params);
                    cm.fit(&data);
                })
            });

        group
            .sample_size(10)
            .bench_function(format!("iris-{}, parallelized=true", size), |b| {
                b.iter(|| {
                    let params = CMeansParams::default()
                        .n_clusters(15)
                        .parallelized(true)
                        .max_iter(100)
                        .tol(0.001);
                    let cm = CMeans::new(params);
                    cm.fit(&data);
                })
            });
    }
}

criterion_group!(cmeans, cmeans_benchmark);
criterion_main!(cmeans);
