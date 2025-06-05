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

// Function to print a tensor's data in a 1D format
fn print_tensor_1d<T: Backend>(name: &str, tensor: &Tensor<T, 1>) {
    let data = tensor.to_data();
    let values: Vec<f32> = data
        .bytes
        .chunks_exact(4) // Each f32 is 4 bytes
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    println!("{}: {:?}", name, values);
}

// Function to solve a simple ODE using the Euler method with autodiff for a 1D tensor
fn solve_ode_euler_autodiff_vector<T: Backend>() {
    // Define the ODE: dy/dt = -y (implicitly via autodiff)
    let device = T::Device::default();
    let step_size = 0.1;
    let num_steps = 100;

    // Initial condition: y(0) = [1.0, 2.0, 3.0, 4.0, 5.0]
    let mut y = Tensor::<T, 1>::from_data(TensorData::from([1.0, 2.0, 3.0, 4.0, 5.0]), &device).require_grad();

    for _step in 0..num_steps {
        // Compute dy/dt using autodiff
        let dy_dt = y.clone().neg();

        // Update y using Euler's method: y_next = y + step_size * dy/dt
        y = y.add(dy_dt.mul_scalar(step_size));

        // Reset gradients for the next step
        y = y.detach().require_grad();
    }

    // Print the final result using the specialized print function
    print_tensor_1d("Final result of y", &y);
}

// Simplified state-space model function
fn simplified_ssm<T: Backend>() {
    let device = T::Device::default();

    // Define dimensions
    let n = 20; // State vector size

    // Initialize state vector x(k) and input scalar u(k)
    let mut x = Tensor::<T, 2>::from_data(TensorData::zeros::<f32, _>([n, 1]), &device).require_grad();
    let u = Tensor::<T, 1>::from_data(TensorData::from([1.0]), &device);

    // Learnable B matrix (vector)
    let b_values: Vec<f32> = vec![0.1; n];
    let b = Tensor::<T, 2>::from_data(TensorData::new::<f32, _>(b_values, [n, 1]), &device).require_grad();

    // Constant A matrix (identity for simplicity)
    let a_values: Vec<f32> = (0..n)
        .flat_map(|i| (0..n).map(move |j| if i == j { 1.0 } else { 0.0 }))
        .collect();
    let a = Tensor::<T, 2>::from_data(TensorData::new::<f32, _>(a_values, [n, n]), &device);

    println!("Initial state x: {:?}", x.to_data().bytes);

    for step in 0..10 {
        // Clone tensors to avoid moving them
        let a_clone = a.clone();
        let b_clone = b.clone();

        // Compute A * x(k)
        let ax = a_clone.matmul(x.clone());

        // Compute B * u(k)
        let bu = b_clone.mul_scalar(u.to_data().bytes[0]);

        // Update state: x(k+1) = A * x(k) + B * u(k)
        x = ax.add(bu);

        // Reset gradients for the next step
        x = x.detach().require_grad();

        println!("State at step {}: {:?}", step + 1, x.to_data().bytes);
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

    // Run the ODE solver demo with autodiff for a vector
    solve_ode_euler_autodiff_vector::<MyAutodiffBackend>();

    // Run the simplified state-space model
    simplified_ssm::<MyAutodiffBackend>();
}
