use cluster_rs::kmeans::{KMeansParams, KMeans};

fn main() {
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
