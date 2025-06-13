#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    /// Create a synthetic dataset for testing
    fn create_test_dataset(n_samples: usize, n_features: usize, n_classes: usize) -> (ndarray::Array2<f32>, ndarray::Array1<u8>) {
        let mut features = ndarray::Array2::<f32>::zeros((n_samples, n_features));
        let mut labels = ndarray::Array1::<u8>::zeros(n_samples);
        
        for i in 0..n_samples {
            let class = i % n_classes;
            labels[i] = class as u8;
            
            // Generate features with class-dependent means
            for j in 0..n_features {
                let class_mean = (class as f32) * 2.0;
                let feature_offset = (j as f32) * 0.3;
                let noise = ((i * 100 + j * 10 + class) as f32).sin() * 0.2;
                features[[i, j]] = class_mean + feature_offset + noise;
            }
        }
        
        (features, labels)
    }

    /// Test basic model parameter save and load functionality
    #[test]
    fn test_modelparams_save_load() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let test_path = temp_dir.path().join("test_model.bin");
        
        let device = <MyBackend as Backend>::Device::default();
        let (input_dim, hidden_dim, output_dim) = (4, 8, 3);
        
        // Create and train a model briefly
        let mut model = TinyMLP::<MyBackend>::new(input_dim, hidden_dim, output_dim, &device);
        
        // Create test data
        let (features, labels) = create_test_dataset(60, input_dim, output_dim);
        let (x_vec, _) = features.into_raw_vec_and_offset();
        let (y_vec, _) = labels.into_raw_vec_and_offset();
        
        let x_data = TensorData::new(x_vec, [60, input_dim]);
        let y_data = TensorData::new(y_vec, [60]);
        
        let x = Tensor::<MyBackend, 2>::from_data(x_data, &device);
        let y = Tensor::<MyBackend, 1>::from_data(y_data, &device);
        
        // Train for a few steps to get non-random weights
        let mut optimizer = AdamConfig::new().init();
        for _ in 0..3 {
            let logits = model.forward(x.clone());
            let yb2 = y.clone().unsqueeze_dim(1);
            let loss = cross_entropy_with_logits(logits, yb2);
            
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(0.01, model, grads);
        }
        
        // Get output before saving
        let test_input = x.clone().slice([0..1, 0..input_dim]);
        let original_output = model.forward(test_input.clone());
        
        // Test save functionality
        save_modelparams(&model, test_path.to_str().unwrap())
            .expect("Failed to save model parameters");
        
        assert!(test_path.exists(), "Model parameters file should exist after saving");
        
        // Test load functionality
        let loaded_model = load_modelparams::<MyBackend>(
            input_dim, hidden_dim, output_dim, &device, test_path.to_str().unwrap()
        ).expect("Failed to load model parameters");
        
        // Verify outputs are identical
        let loaded_output = loaded_model.forward(test_input);
        let diff = (original_output - loaded_output).abs().max();
        let max_diff = diff.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
        
        assert!(max_diff < 1e-6, "Loaded model should produce identical outputs (diff: {:.2e})", max_diff);
    }

    /// Test checkpoint functionality during training interruption and resume
    #[test]
    fn test_checkpoint_resume_functionality() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let checkpoint_path = temp_dir.path().join("checkpoint.bin");
        
        let device = <MyBackend as Backend>::Device::default();
        let (input_dim, hidden_dim, output_dim) = (4, 8, 3);
        let (n_samples, batch_size) = (60, 8);
        
        // Create synthetic dataset
        let (features, labels) = create_test_dataset(n_samples, input_dim, output_dim);
        let (x_vec, _) = features.into_raw_vec_and_offset();
        let (y_vec, _) = labels.into_raw_vec_and_offset();
        
        let x_data = TensorData::new(x_vec, [n_samples, input_dim]);
        let y_data = TensorData::new(y_vec, [n_samples]);
        
        let x = Tensor::<MyBackend, 2>::from_data(x_data, &device);
        let y = Tensor::<MyBackend, 1>::from_data(y_data, &device);
        
        // Phase 1: Train model and save checkpoint
        let mut model = TinyMLP::<MyBackend>::new(input_dim, hidden_dim, output_dim, &device);
        let mut optimizer = AdamConfig::new().init();
        let interruption_epochs = 5;
        
        let mut checkpoint_loss = 0.0;
        for epoch in 0..interruption_epochs {
            let mut total_loss = 0.0;
            let mut n_batches = 0;
            
            for i in (0..n_samples).step_by(batch_size) {
                let end = (i + batch_size).min(n_samples);
                let xb = x.clone().slice([i..end, 0..input_dim]);
                let yb = y.clone().slice([i..end]);
                let yb2 = yb.unsqueeze_dim(1);
                
                let logits = model.forward(xb);
                let loss = cross_entropy_with_logits(logits, yb2);
                
                total_loss += loss.clone().mean().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
                n_batches += 1;
                
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(0.01, model, grads);
            }
            
            checkpoint_loss = total_loss / n_batches as f32;
        }
        
        // Save checkpoint
        save_modelparams(&model, checkpoint_path.to_str().unwrap())
            .expect("Failed to save checkpoint");
        
        assert!(checkpoint_path.exists(), "Checkpoint file should exist");
        
        // Get model state at checkpoint
        let test_input = x.clone().slice([0..1, 0..input_dim]);
        let checkpoint_output = model.forward(test_input.clone());
        
        // Phase 2: Load checkpoint and continue training
        let mut resumed_model = load_modelparams::<MyBackend>(
            input_dim, hidden_dim, output_dim, &device, checkpoint_path.to_str().unwrap()
        ).expect("Failed to load checkpoint");
        
        // Verify checkpoint loaded correctly
        let resumed_output = resumed_model.forward(test_input.clone());
        let diff = (checkpoint_output - resumed_output).abs().max();
        let max_diff = diff.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
        
        assert!(max_diff < 1e-6, "Resumed model should match checkpoint state (diff: {:.2e})", max_diff);
        
        // Continue training from checkpoint
        let mut resumed_optimizer = AdamConfig::new().init();
        let additional_epochs = 3;
        
        for _ in 0..additional_epochs {
            for i in (0..n_samples).step_by(batch_size) {
                let end = (i + batch_size).min(n_samples);
                let xb = x.clone().slice([i..end, 0..input_dim]);
                let yb = y.clone().slice([i..end]);
                let yb2 = yb.unsqueeze_dim(1);
                
                let logits = resumed_model.forward(xb);
                let loss = cross_entropy_with_logits(logits, yb2);
                
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &resumed_model);
                resumed_model = resumed_optimizer.step(0.01, resumed_model, grads);
            }
        }
        
        // Verify model has changed from checkpoint (training continued)
        let final_output = resumed_model.forward(test_input);
        let training_diff = (checkpoint_output - final_output).abs().max();
        let training_max_diff = training_diff.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
        
        assert!(training_max_diff > 1e-8, "Model should have changed after additional training (diff: {:.2e})", training_max_diff);
    }

    /// Test multiple save/load cycles for parameter stability
    #[test]
    fn test_multiple_save_load_cycles() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let device = <MyBackend as Backend>::Device::default();
        let (input_dim, hidden_dim, output_dim) = (3, 6, 2);
        
        // Create and train a model
        let mut model = TinyMLP::<MyBackend>::new(input_dim, hidden_dim, output_dim, &device);
        let (features, labels) = create_test_dataset(30, input_dim, output_dim);
        let (x_vec, _) = features.into_raw_vec_and_offset();
        let (y_vec, _) = labels.into_raw_vec_and_offset();
        
        let x_data = TensorData::new(x_vec, [30, input_dim]);
        let y_data = TensorData::new(y_vec, [30]);
        
        let x = Tensor::<MyBackend, 2>::from_data(x_data, &device);
        let y = Tensor::<MyBackend, 1>::from_data(y_data, &device);
        
        // Brief training
        let mut optimizer = AdamConfig::new().init();
        let logits = model.forward(x.clone());
        let yb2 = y.unsqueeze_dim(1);
        let loss = cross_entropy_with_logits(logits, yb2);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads);
        
        // Get original output
        let test_input = x.slice([0..1, 0..input_dim]);
        let original_output = model.forward(test_input.clone());
        
        // Perform multiple save/load cycles
        let cycles = 5;
        let mut current_model = model;
        
        for cycle in 0..cycles {
            let cycle_path = temp_dir.path().join(format!("cycle_{}.bin", cycle));
            
            // Save current model
            save_modelparams(&current_model, cycle_path.to_str().unwrap())
                .expect("Failed to save model in cycle");
            
            // Load model
            current_model = load_modelparams::<MyBackend>(
                input_dim, hidden_dim, output_dim, &device, cycle_path.to_str().unwrap()
            ).expect("Failed to load model in cycle");
            
            // Verify output hasn't changed
            let cycle_output = current_model.forward(test_input.clone());
            let diff = (original_output.clone() - cycle_output).abs().max();
            let max_diff = diff.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
            
            assert!(max_diff < 1e-6, "Model output should remain identical after cycle {} (diff: {:.2e})", cycle, max_diff);
        }
    }

    /// Test error handling for invalid checkpoint files
    #[test]
    fn test_checkpoint_error_handling() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let device = <MyBackend as Backend>::Device::default();
        let (input_dim, hidden_dim, output_dim) = (2, 4, 2);
        
        // Test loading non-existent file
        let nonexistent_path = temp_dir.path().join("nonexistent.bin");
        let result = load_modelparams::<MyBackend>(
            input_dim, hidden_dim, output_dim, &device, nonexistent_path.to_str().unwrap()
        );
        
        assert!(result.is_err(), "Loading non-existent file should fail");
        
        // Test loading corrupted file
        let corrupted_path = temp_dir.path().join("corrupted.bin");
        fs::write(&corrupted_path, b"invalid model data").expect("Failed to write corrupted file");
        
        let result = load_modelparams::<MyBackend>(
            input_dim, hidden_dim, output_dim, &device, corrupted_path.to_str().unwrap()
        );
        
        assert!(result.is_err(), "Loading corrupted file should fail");
    }

    /// Integration test: Full training workflow with checkpoints
    #[test]
    fn test_full_training_workflow_with_checkpoints() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let checkpoint_path = temp_dir.path().join("workflow_checkpoint.bin");
        let final_path = temp_dir.path().join("workflow_final.bin");
        
        let device = <MyBackend as Backend>::Device::default();
        let (input_dim, hidden_dim, output_dim) = (4, 8, 3);
        let (n_samples, batch_size) = (90, 15);
        
        // Create synthetic dataset
        let (features, labels) = create_test_dataset(n_samples, input_dim, output_dim);
        let (x_vec, _) = features.into_raw_vec_and_offset();
        let (y_vec, _) = labels.into_raw_vec_and_offset();
        
        let x_data = TensorData::new(x_vec, [n_samples, input_dim]);
        let y_data = TensorData::new(y_vec, [n_samples]);
        
        let x = Tensor::<MyBackend, 2>::from_data(x_data, &device);
        let y = Tensor::<MyBackend, 1>::from_data(y_data, &device);
        
        // Training configuration
        let total_epochs = 10;
        let checkpoint_interval = 3;
        
        // Simulate full training with checkpointing
        let mut model = TinyMLP::<MyBackend>::new(input_dim, hidden_dim, output_dim, &device);
        let mut optimizer = AdamConfig::new().init();
        let mut losses = Vec::new();
        
        for epoch in 0..total_epochs {
            let mut total_loss = 0.0;
            let mut n_batches = 0;
            
            for i in (0..n_samples).step_by(batch_size) {
                let end = (i + batch_size).min(n_samples);
                let xb = x.clone().slice([i..end, 0..input_dim]);
                let yb = y.clone().slice([i..end]);
                let yb2 = yb.unsqueeze_dim(1);
                
                let logits = model.forward(xb);
                let loss = cross_entropy_with_logits(logits, yb2);
                
                total_loss += loss.clone().mean().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
                n_batches += 1;
                
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(0.01, model, grads);
            }
            
            let avg_loss = total_loss / n_batches as f32;
            losses.push(avg_loss);
            
            // Save checkpoint at intervals
            if (epoch + 1) % checkpoint_interval == 0 {
                save_modelparams(&model, checkpoint_path.to_str().unwrap())
                    .expect("Failed to save checkpoint during training");
                
                assert!(checkpoint_path.exists(), "Checkpoint should exist after saving");
            }
        }
        
        // Save final model
        save_modelparams(&model, final_path.to_str().unwrap())
            .expect("Failed to save final model");
        
        // Verify training progressed (loss should generally decrease)
        let initial_loss = losses[0];
        let final_loss = losses[losses.len() - 1];
        
        // Allow for some fluctuation, but expect overall improvement or stability
        assert!(final_loss <= initial_loss * 1.1, 
               "Training should show progress or stability (initial: {:.4}, final: {:.4})", 
               initial_loss, final_loss);
        
        // Test loading final model
        let final_model = load_modelparams::<MyBackend>(
            input_dim, hidden_dim, output_dim, &device, final_path.to_str().unwrap()
        ).expect("Failed to load final model");
        
        // Verify final model works for inference
        let test_input = x.slice([0..1, 0..input_dim]);
        let output = final_model.forward(test_input);
        
        assert_eq!(output.dims(), [1, output_dim], "Final model should produce correct output shape");
        
        // Cleanup is automatic with TempDir
    }
}
