use burn::backend::{Autodiff, NdArray}; // Correct import paths for Autodiff and NdArray
use burn::tensor::TensorData; // Use TensorData instead of Data
use burn::prelude::*; // Import general Burn prelude
use std::convert::TryInto; // Import for byte conversion

type MyBackend = NdArray<f32>; // Or some other element type like f64
type MyAutodiffBackend = Autodiff<MyBackend>;

// Function to print a tensor's data in a 2D grid
fn print_tensor_2d<T: Backend>(name: &str, tensor: &Tensor<T, 2>) {
    let data = tensor.to_data();
    let values: Vec<f32> = data
        .bytes
        .chunks_exact(4) // Each f32 is 4 bytes
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let shape = data.shape;
    println!("{}:", name);
    for row in values.chunks(shape[1]) {
        println!("{:?}", row);
    }
}

// Function to solve a simple ODE using the Euler method with autodiff
fn solve_ode_euler_autodiff<T: Backend>() {
    // Define the ODE: dy/dt = -y (implicitly via autodiff)
    let device = T::Device::default();
    let step_size = 0.1;
    let num_steps = 100;

    // Initial condition: y(0) = 1
    let mut y = Tensor::<T, 1>::from_data(TensorData::from([1.0]), &device).require_grad();

    println!("Time, y");
    for step in 0..num_steps {
        let t = step as f32 * step_size;
        let y_value: f32 = f32::from_le_bytes(y.to_data().bytes[0..4].try_into().unwrap());
        println!("{:.2}, {:.4}", t, y_value);

        // Compute dy/dt using autodiff
        let dy_dt = y.clone().neg();

        // Update y using Euler's method: y_next = y + step_size * dy/dt
        y = y.add(dy_dt.mul_scalar(step_size));

        // Reset gradients for the next step
        y = y.detach().require_grad();
    }
}

fn main() {
    // Create a device for the backend
    let device = <MyAutodiffBackend as Backend>::Device::default();

    // Create a 3x3 tensor with requires_grad enabled
    let x = Tensor::<MyAutodiffBackend, 2>::from_data(TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), &device).require_grad();

    // Print tensor x
    print_tensor_2d("Tensor x", &x);

    // Perform a simple operation: y = x^2
    let y = x.clone().mul(x.clone());

    // Print tensor y
    print_tensor_2d("Tensor y", &y);

    // Compute gradients
    let grads = y.backward(); // `backward()` returns the gradients

    // Access the gradient of `x`
    let grad_x = x.grad(&grads);

    // Print the gradient of `x`
    if let Some(grad_x) = grad_x {
        print_tensor_2d("Gradient of x", &grad_x);
    } else {
        println!("No gradient available for x.");
    }

    // Run the ODE solver demo with autodiff
    solve_ode_euler_autodiff::<MyAutodiffBackend>();
}
