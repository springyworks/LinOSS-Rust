// examples/burn_iris_loader.rs
// Minimal example: Load iris_features.npy and iris_labels.npy, wrap as Burn dataset, use DataLoader

#[cfg(feature = "npy_files")]
use burn::data::dataloader::Dataset;
#[cfg(feature = "npy_files")]
use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "npy_files")]
use burn::tensor::Tensor;
#[cfg(feature = "npy_files")]
use ndarray_npy::read_npy;
#[cfg(feature = "npy_files")]
use std::sync::Arc;

// Custom dataset struct for Burn
#[cfg(feature = "npy_files")]
struct IrisDataset {
    features: Arc<Vec<[f32; 4]>>,
    labels: Arc<Vec<u8>>,
    device: NdArrayDevice,
}

#[cfg(feature = "npy_files")]
impl Dataset<(Tensor<NdArray, 2>, Tensor<NdArray, 1>)> for IrisDataset {
    fn get(&self, index: usize) -> Option<(Tensor<NdArray, 2>, Tensor<NdArray, 1>)> {
        let x = self.features.get(index)?;
        let y = self.labels.get(index)?;
        let x_tensor = Tensor::<NdArray, 2>::from_data([[x[0], x[1], x[2], x[3]]], &self.device);
        let y_tensor = Tensor::<NdArray, 1>::from_data([*y], &self.device);
        Some((x_tensor, y_tensor))
    }
    fn len(&self) -> usize {
        self.features.len()
    }
}

#[cfg(not(feature = "npy_files"))]
fn main() {
    println!("⚠️  This example requires the 'npy_files' feature to be enabled.");
    println!("   Run with: cargo run --example burn_iris_loader --features npy_files");
}

#[cfg(feature = "npy_files")]
fn main() -> anyhow::Result<()> {
    // Load features and labels from .npy
    let features_path = "datastore/processed-by-python/iris_features.npy";
    let labels_path = "datastore/processed-by-python/iris_labels.npy";
    let features: ndarray::Array2<f32> = read_npy(features_path)?;
    let labels: ndarray::Array1<u8> = read_npy(labels_path)?;

    // Ensure features are in standard (row-major) layout
    let features = features.to_owned();
    let features_vec: Vec<[f32; 4]> = features
        .outer_iter()
        .enumerate()
        .map(|(i, row)| {
            let v: Vec<f32> = row.iter().copied().collect();
            if v.len() != 4 {
                println!("Row {} has wrong shape: {:?}", i, v);
                panic!("Feature row shape error");
            }
            [v[0], v[1], v[2], v[3]]
        })
        .collect();
    let labels_vec: Vec<u8> = labels.iter().copied().collect();
    println!("features_vec len: {} labels_vec len: {}", features_vec.len(), labels_vec.len());
    for (i, row) in features.outer_iter().enumerate().take(5) {
        println!("features[{}]: {:?}", i, row);
    }
    for (i, label) in labels.iter().enumerate().take(5) {
        println!("labels[{}]: {:?}", i, label);
    }
    if features_vec.len() != labels_vec.len() {
        println!("ERROR: features and labels have different lengths!");
        return Err(anyhow::anyhow!("features/labels length mismatch"));
    }

    let device = NdArrayDevice::default();
    let dataset = IrisDataset {
        features: Arc::new(features_vec),
        labels: Arc::new(labels_vec),
        device,
    };

    let batch_size = 16;
    let num_batches = dataset.len().div_ceil(batch_size);
    for batch_idx in 0..num_batches.min(1) { // Only print first batch
        let start = batch_idx * batch_size;
        let end = ((batch_idx + 1) * batch_size).min(dataset.len());
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in start..end {
            let (x, y) = dataset.get(i).ok_or_else(|| {
                anyhow::anyhow!("Failed to get dataset item at index {}", i)
            })?;
            xs.push(x);
            ys.push(y);
        }
        // Stack tensors along batch dimension
        let x_batch = Tensor::cat(xs, 0);
        let y_batch = Tensor::cat(ys, 0);
        let x0_data = x_batch.clone().slice([0]).to_data();
        let y0_data = y_batch.clone().slice([0]).to_data();
        let x0_vec: Vec<f32> = x0_data.convert::<f32>().into_vec().map_err(|e| {
            anyhow::anyhow!("Failed to convert x0_data to Vec<f32>: {:?}", e)
        })?;
        let y0_vec: Vec<u8> = y0_data.convert::<u8>().into_vec().map_err(|e| {
            anyhow::anyhow!("Failed to convert y0_data to Vec<u8>: {:?}", e)
        })?;
        println!("Batch {batch_idx}: x shape = {:?}, y shape = {:?}", x_batch.shape(), y_batch.shape());
        println!("x[0] = {:?}, y[0] = {:?}", x0_vec, y0_vec);
    }
    Ok(())
}
