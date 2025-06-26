pub mod eeg_decoder;

pub use eeg_decoder::{
    EEGDecoder, 
    EEGDecoderConfig,
    TrainingData,
    train_eeg_decoder,
};
