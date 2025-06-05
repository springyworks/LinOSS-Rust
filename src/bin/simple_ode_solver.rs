use burn::backend::{Autodiff, NdArray}; // Import Autodiff and NdArray backends
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

// Define the backend type globally
//type B = Autodiff<NdArrayBackend<f64>>;
// type B = burn::backend::Autodiff<burn::backend::ndarray::NdArray<f32>>; // Unused type alias, commented out

fn euler_method_autodiff<B: Backend + AutodiffBackend>(
    x0: Tensor<B, 1>,
    f: impl Fn(&Tensor<B, 1>) -> Tensor<B, 1>,
    t_span: (f64, f64),
    h: f64,
) -> Vec<Tensor<B, 1>> {
    let mut results = Vec::new();
    let mut x = x0.clone();
    let mut t = t_span.0;
    results.push(x.clone());
    while t < t_span.1 {
        let dx = f(&x); // Call f directly with the outer tensor
        x = x + dx.mul_scalar(h); // Update x using the Euler method
        t += h;
        results.push(x.clone());
    }
    results
}

// fn test_solver(device: &<B as Backend>::Device) { // Unused function, commented out
//     // Example test case for the solver
//     let x0 = Tensor::<B, 1>::from_data([1.0], device);
//     let t_span = (0.0, 1.0);
//     let h = 0.1;
//     let f = |x: &Tensor<B, 1>| -x.clone();
//     let results = euler_method_autodiff(x0, f, t_span, h);
//     for (i, result) in results.iter().enumerate() {
//         println!("Test Step {}: {:?}", i, result.to_data());
//     }
// }

fn main() {
    // Create the device
    let device = <Autodiff<NdArray<f64>> as Backend>::Device::default();

    // Define the initial condition and time span
    let x0 = Tensor::<Autodiff<NdArray<f64>>, 1>::from_data([1.0], &device);
    let t_span = (0.0, 1.0);
    let h = 0.1;

    // Define the ODE function: dx/dt = -x
    let f = |x: &Tensor<Autodiff<NdArray<f64>>, 1>| {
        -x.clone()
    };

    // Solve the ODE using the Euler method
    let results = euler_method_autodiff(x0, f, t_span, h);

    // Print the results
    for (i, result) in results.iter().enumerate() {
        println!("Step {}: {:?}", i, result.to_data());
    }
}