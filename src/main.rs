mod distance;
mod kmeans;
use std::{time::Instant, vec};

use cluster_rs::{
    cmeans::{CMeans, CMeansParams},
    metrics::silhouette_score_km,
};
use kmeans::{KMeans, KMeansParams};
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
    // let mut data: Vec<Vec<_>> = vec![];
    // let mut reader = csv::ReaderBuilder::new()
    //     .has_headers(false)
    //     .from_path("data/BDA/Dataset 1/Iris-150.txt")
    //     .expect("Failed to open csv");

    // for record in reader.deserialize() {
    //     let record: IrisRecord = record.unwrap();
    //     let mut row = vec![vec![record.col1, record.col2, record.col3, record.col4]];
    //     data.append(&mut row);
    // }
    // println!("{}", data.len());
    // let params = KMeansParams::default()
    //     .n_clusters(3)
    //     .parallelized(true)
    //     .verbose(true);

    // println!("{params:?}");
    // let km = KMeans::new(params).fit(&data);
    // let t = Instant::now();
    // let centroids = km.centroids();
    // let inertia = km.inertia();
    // let labels = km.labels();
    // let duration = t.elapsed();
    // println!("{centroids:?}");
    // println!("{inertia:?}");
    // println!("{:?}", labels);
    // // println!("{:?}");

    // println!("{}", duration.as_millis());

    // let params = CMeansParams::default()
    //     .n_clusters(3)
    //     .parallelized(true)
    //     .verbose(true)
    //     .max_iter(100);

    // println!("{params:?}");
    // let t = Instant::now();
    // let cm = CMeans::new(params).fit(&data);
    // let duration = t.elapsed();
    // println!("{:?}", cm.centroids());
    // println!("{}", duration.as_millis());

    // // Plotting
    // let x_lim = 0.0..10.0f32;
    // let y_lim = 0.0..10.0f32;
    // let root = BitMapBackend::new("kmeans.png", (600, 400)).into_drawing_area();

    // root.fill(&WHITE).unwrap();

    // let mut ctx = ChartBuilder::on(&root)
    //     .set_label_area_size(LabelAreaPosition::Left, 40) // Put in some margins
    //     .set_label_area_size(LabelAreaPosition::Right, 40)
    //     .set_label_area_size(LabelAreaPosition::Bottom, 40)
    //     .caption("Iris KMeans", ("sans-serif", 25)) // Set a caption and font
    //     .build_cartesian_2d(x_lim, y_lim)
    //     .expect("Couldn't build our ChartBuilder");

    // ctx.configure_mesh().draw().unwrap();
    // let root_area = ctx.plotting_area();

    // let colors = [RED, BLUE, GREEN];
    // for (record, label) in data.iter().zip(labels) {
    //     let coords = (record[0] as f32, record[1] as f32);
    //     let point = Circle::new(coords, 3, ShapeStyle::from(&colors[label]).filled());
    //     root_area
    //         .draw(&point)
    //         .expect("An error occurred while drawing the point!");
    // }

    let x = vec![
        vec![0.1, 0.1],
        vec![0., 0.],
        vec![-0.1, -0.1],
        vec![1.1, 1.1],
        vec![1., 1.],
        vec![0.9, 0.9],
        vec![1.9, 1.9],
        vec![2., 2.],
        vec![2.1, 2.1],
    ];
    let params = KMeansParams::default().n_clusters(3);
    let km = KMeans::new(params).fit(&x);
    let centers = km.centroids();
    println!("{centers:?}");
}
