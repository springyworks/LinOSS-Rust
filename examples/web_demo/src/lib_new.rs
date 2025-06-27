#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::{string::String, vec::Vec, boxed::Box};

use wasm_bindgen::prelude::*;
use console_error_panic_hook;

// Use Burn with NdArray backend for WASM compatibility
use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    prelude::*,
};

// Import the real D-LinOSS layer using Burn backend
use linoss_rust::dlinoss::dlinoss_layer::DLinOSSLayer;

// Type alias for our WASM-compatible backend
type Backend = NdArray<f32>;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
}

#[wasm_bindgen]
pub struct DLinOSSDemo {
    layer: DLinOSSLayer<Backend>,
    device: NdArrayDevice,
}

#[wasm_bindgen]
impl DLinOSSDemo {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let device = NdArrayDevice::Cpu;
        
        // Create a simple D-LinOSS layer for demonstration
        let layer = DLinOSSLayer::new(
            10,  // input_size
            5,   // output_size
            &device,
        );
        
        Self { layer, device }
    }
    
    #[wasm_bindgen]
    pub fn forward(&self, input_data: &[f32]) -> Vec<f32> {
        // Convert input to Burn tensor
        let input_tensor = Tensor::<Backend, 2>::from_data(
            TensorData::new(input_data.to_vec(), [1, input_data.len()]),
            &self.device,
        );
        
        // Run forward pass through D-LinOSS layer
        let output = self.layer.forward(input_tensor);
        
        // Convert back to Vec<f32> for JavaScript
        output.to_data().to_vec().unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_info(&self) -> String {
        String::from("D-LinOSS Neural Dynamics Layer running in WebAssembly with Burn NdArray backend")
    }
}
