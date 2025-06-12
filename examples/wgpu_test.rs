#![recursion_limit = "256"]

use burn::tensor::{Tensor, TensorData};

// Conditional imports and type aliases for WGPU
#[cfg(feature = "wgpu_backend")]
mod wgpu_specific {
    pub use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub use burn::backend::Autodiff;

    pub type MyWgpuBackend = Wgpu<f32, i32>;
    pub type MyAutodiffWgpuBackend = Autodiff<MyWgpuBackend>;
    pub type MyWgpuDevice = WgpuDevice;
}

// Conditional imports and type aliases for NdArray (fallback)
#[cfg(not(feature = "wgpu_backend"))]
mod ndarray_specific {
    pub use burn::backend::NdArray;
    pub use burn::backend::ndarray::NdArrayDevice;
    pub use burn::backend::Autodiff;

    pub type MyWgpuDevice = NdArrayDevice;
    pub type MyWgpuBackend = NdArray<f32>;
    pub type MyAutodiffWgpuBackend = Autodiff<MyWgpuBackend>;
}

#[cfg(feature = "wgpu_backend")]
fn print_feature_status() {
    println!("wgpu_backend feature is enabled.");
}

#[cfg(not(feature = "wgpu_backend"))]
fn print_feature_status() {
    println!("wgpu_backend feature is NOT enabled. WGPU tests will be skipped (NdArray fallback will run).");
}

// Removed unused check_backend_implemented function

fn main() {
    print_feature_status();

    #[cfg(feature = "wgpu_backend")]
    {
        use wgpu_specific::*;
        println!("Initializing WGPU backend...");
        let device = MyWgpuDevice::default();
        println!("WGPU device: {:?}", device);

        // The check_backend_implemented function is illustrative.
        // If MyWgpuBackend compiles, it means the basic setup is okay.
        // check_backend_implemented::<MyWgpuBackend>();
        // check_backend_implemented::<MyAutodiffWgpuBackend>();

        println!("Creating tensors on the WGPU backend...");
        let tensor_1_data = TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let tensor_2_data = TensorData::from([[5.0f32, 6.0], [7.0, 8.0]]);

        let tensor_1: Tensor<MyWgpuBackend, 2> = Tensor::from_data(tensor_1_data.clone(), &device);
        let tensor_2: Tensor<MyWgpuBackend, 2> = Tensor::from_data(tensor_2_data.clone(), &device);

        println!("Tensor 1 (WGPU): {:?}", tensor_1.to_data());
        println!("Tensor 2 (WGPU): {:?}", tensor_2.to_data());

        let sum_tensor = tensor_1.clone() + tensor_2.clone();
        println!("Sum of tensors (WGPU): {:?}", sum_tensor.to_data());

        // Autodiff example
        let tensor_ad_1: Tensor<MyAutodiffWgpuBackend, 2> = Tensor::from_data(tensor_1_data, &device);
        let tensor_ad_2: Tensor<MyAutodiffWgpuBackend, 2> = Tensor::from_data(tensor_2_data, &device);
        
        let sum_ad_tensor = tensor_ad_1.clone() + tensor_ad_2.clone();
        println!("Sum of tensors (Autodiff WGPU): {:?}", sum_ad_tensor.to_data());

        println!("WGPU backend test completed successfully!");
    }

    #[cfg(not(feature = "wgpu_backend"))]
    {
        use ndarray_specific::*;
        println!("Initializing NdArray backend as fallback...");
        let device = MyWgpuDevice::default();
        println!("NdArray device: {:?}", device);

        println!("Creating tensors on the NdArray backend...");
        let tensor_1_data = TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let tensor_2_data = TensorData::from([[5.0f32, 6.0], [7.0, 8.0]]);

        let tensor_1: Tensor<MyWgpuBackend, 2> = Tensor::from_data(tensor_1_data.clone(), &device);
        let tensor_2: Tensor<MyWgpuBackend, 2> = Tensor::from_data(tensor_2_data.clone(), &device);

        println!("Tensor 1 (NdArray): {:?}", tensor_1.to_data());
        println!("Tensor 2 (NdArray): {:?}", tensor_2.to_data());

        let sum_tensor = tensor_1.clone() + tensor_2.clone();
        println!("Sum of tensors (NdArray): {:?}", sum_tensor.to_data());
        
        // Autodiff example for NdArray
        let tensor_ad_1: Tensor<MyAutodiffWgpuBackend, 2> = Tensor::from_data(tensor_1_data, &device);
        let tensor_ad_2: Tensor<MyAutodiffWgpuBackend, 2> = Tensor::from_data(tensor_2_data, &device);

        let sum_ad_tensor = tensor_ad_1.clone() + tensor_ad_2.clone();
        println!("Sum of tensors (Autodiff NdArray): {:?}", sum_ad_tensor.to_data());

        println!("NdArray backend fallback test completed successfully!");
    }
}
