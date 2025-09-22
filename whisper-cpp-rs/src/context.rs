use crate::error::{Result, WhisperError};
use crate::quantize::{ModelQuantizer, QuantizationType};
use std::path::Path;
use std::sync::Arc;
use whisper_sys as ffi;

pub struct WhisperContext {
    pub(crate) ptr: Arc<ContextPtr>,
}

pub(crate) struct ContextPtr(pub(crate) *mut ffi::whisper_context);

unsafe impl Send for ContextPtr {}
unsafe impl Sync for ContextPtr {}

impl Drop for ContextPtr {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::whisper_free(self.0);
            }
        }
    }
}

impl WhisperContext {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| WhisperError::ModelLoadError("Invalid path".into()))?;

        let c_path = std::ffi::CString::new(path_str)?;

        let ptr = unsafe {
            ffi::whisper_init_from_file_with_params(
                c_path.as_ptr(),
                ffi::whisper_context_default_params(),
            )
        };

        if ptr.is_null() {
            return Err(WhisperError::ModelLoadError(
                "Failed to load model".into(),
            ));
        }

        Ok(Self {
            ptr: Arc::new(ContextPtr(ptr)),
        })
    }

    pub fn new_from_buffer(buffer: &[u8]) -> Result<Self> {
        let ptr = unsafe {
            ffi::whisper_init_from_buffer_with_params(
                buffer.as_ptr() as *mut std::os::raw::c_void,
                buffer.len(),
                ffi::whisper_context_default_params(),
            )
        };

        if ptr.is_null() {
            return Err(WhisperError::ModelLoadError(
                "Failed to load model from buffer".into(),
            ));
        }

        Ok(Self {
            ptr: Arc::new(ContextPtr(ptr)),
        })
    }

    pub fn is_multilingual(&self) -> bool {
        unsafe { ffi::whisper_is_multilingual(self.ptr.0) != 0 }
    }

    pub fn n_vocab(&self) -> i32 {
        unsafe { ffi::whisper_n_vocab(self.ptr.0) }
    }

    pub fn n_audio_ctx(&self) -> i32 {
        unsafe { ffi::whisper_n_audio_ctx(self.ptr.0) }
    }

    pub fn n_text_ctx(&self) -> i32 {
        unsafe { ffi::whisper_n_text_ctx(self.ptr.0) }
    }

    pub fn n_len(&self) -> i32 {
        unsafe { ffi::whisper_n_len(self.ptr.0) }
    }

    /// Quantize a Whisper model file (static method)
    ///
    /// This is a convenience method for quantizing models without needing to
    /// create a WhisperContext instance.
    ///
    /// # Arguments
    /// * `input_path` - Path to the input model file
    /// * `output_path` - Path where the quantized model will be saved
    /// * `qtype` - The quantization type to use
    ///
    /// # Example
    /// ```no_run
    /// use whisper_cpp_rs::{WhisperContext, QuantizationType};
    ///
    /// WhisperContext::quantize_model(
    ///     "models/ggml-base.bin",
    ///     "models/ggml-base-q5_0.bin",
    ///     QuantizationType::Q5_0
    /// ).expect("Failed to quantize model");
    /// ```
    pub fn quantize_model<P: AsRef<Path>>(
        input_path: P,
        output_path: P,
        qtype: QuantizationType,
    ) -> Result<()> {
        ModelQuantizer::quantize_model_file(input_path, output_path, qtype)
    }
}

impl Clone for WhisperContext {
    fn clone(&self) -> Self {
        Self {
            ptr: Arc::clone(&self.ptr),
        }
    }
}