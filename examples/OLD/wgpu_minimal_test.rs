#![recursion_limit = "256"] // Suggested for WGPU backend

// Common imports
use burn::tensor::{Tensor, TensorData};

// Conditional imports and type aliases for WGPU
#[cfg(feature = "wgpu_backend")]
mod wgpu_specific {
    pub use burn::backend::wgpu::{Wgpu, WgpuDevice}; // WGPU types from burn's wgpu backend module
    pub use burn::backend::Autodiff;         // Autodiff backend wrapper from burn's backend module
    // Backend trait is usually brought in via prelude if needed, or directly if specific.
    // For this minimal example, direct Tensor usage might not need it explicitly if types are well-defined.

    // Define the WGPU backend and its autodiff version
    // Wgpu<FloatElement, IntElement>, GraphicsApi defaults to AutoGraphicsApi
    pub type MyWgpuBackend = Wgpu<f32, i32>;
    pub type MyAutodiffWgpuBackend = Autodiff<MyWgpuBackend>;
    pub type MyWgpuDevice = WgpuDevice; 
}

// Conditional imports and type aliases for NdArray (fallback for LSP or non-WGPU builds)
#[cfg(not(feature = "wgpu_backend"))]
mod ndarray_specific {
    pub use burn::backend::NdArray;  // The NdArray backend type
    pub use burn::backend::ndarray::NdArrayDevice; // NdArray device
    pub use burn::backend::Autodiff; // Autodiff from burn crate for the type alias

    // Fallback types
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

fn main() {
    print_feature_status();

    #[cfg(feature = "wgpu_backend")]
    {
        use wgpu_specific::*; 

        println!("Initializing WGPU backend...");
        let device = MyWgpuDevice::default(); 
        println!("WGPU device: {:?}", device);

        println!("Creating a tensor on the WGPU backend...");
        let tensor_data = TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]);
        
        let tensor_direct: Tensor<MyWgpuBackend, 2> = Tensor::from_data(tensor_data.clone(), &device);
        println!("Tensor (direct WGPU) created successfully: {:?}", tensor_direct.to_data());

        let tensor_add_direct = tensor_direct.clone() + tensor_direct.clone();
        println!("Tensor (direct WGPU) after addition: {:?}", tensor_add_direct.to_data());

        // Autodiff example
        let tensor_autodiff: Tensor<MyAutodiffWgpuBackend, 2> = Tensor::from_data(tensor_data.clone(), &device);
        println!("Tensor (Autodiff WGPU) created successfully: {:?}", tensor_autodiff.to_data());

        let tensor_add_autodiff = tensor_autodiff.clone() + tensor_autodiff.clone();
        println!("Tensor (Autodiff WGPU) after addition: {:?}", tensor_add_autodiff.to_data());
        
        println!("WGPU backend test completed successfully!");
    }

    #[cfg(not(feature = "wgpu_backend"))]
    {
        use ndarray_specific::*;
        println!("Initializing NdArray backend as fallback...");
        let device = MyWgpuDevice::default(); 
        println!("NdArray device: {:?}", device);

        println!("Creating a tensor on the NdArray backend...");
        let tensor_data = TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]);
        
        let tensor_direct: Tensor<MyWgpuBackend, 2> = Tensor::from_data(tensor_data.clone(), &device);
        println!("Tensor (direct NdArray) created successfully: {:?}", tensor_direct.to_data());
        let tensor_add_direct = tensor_direct.clone() + tensor_direct.clone();
        println!("Tensor (direct NdArray) after addition: {:?}", tensor_add_direct.to_data());

        // Autodiff example for NdArray
        let tensor_autodiff: Tensor<MyAutodiffWgpuBackend, 2> = Tensor::from_data(tensor_data.clone(), &device);
        println!("Tensor (Autodiff NdArray) created successfully: {:?}", tensor_autodiff.to_data());
        let tensor_add_autodiff = tensor_autodiff.clone() + tensor_autodiff.clone();
        println!("Tensor (Autodiff NdArray) after addition: {:?}", tensor_add_autodiff.to_data());

        println!("NdArray backend fallback test completed successfully!");
    }
}
