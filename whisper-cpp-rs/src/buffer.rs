//! Audio buffer utilities for streaming support

use std::collections::VecDeque;

/// A circular audio buffer for streaming transcription
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    buffer: VecDeque<f32>,
    max_capacity: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer with specified maximum capacity
    pub fn new(max_capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_capacity),
            max_capacity,
        }
    }

    /// Add audio samples to the buffer
    pub fn push_samples(&mut self, samples: &[f32]) {
        // If adding these samples would exceed capacity, remove old samples
        let space_needed = samples.len().saturating_sub(self.available_space());
        if space_needed > 0 {
            self.buffer.drain(..space_needed.min(self.buffer.len()));
        }

        self.buffer.extend(samples);
    }

    /// Get the number of samples currently in the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get available space in the buffer
    pub fn available_space(&self) -> usize {
        self.max_capacity.saturating_sub(self.buffer.len())
    }

    /// Extract a chunk of audio from the buffer
    /// Returns None if not enough samples are available
    pub fn extract_chunk(&mut self, size: usize, keep_overlap: usize) -> Option<Vec<f32>> {
        if self.buffer.len() < size {
            return None;
        }

        // Collect the chunk
        let chunk: Vec<f32> = self.buffer.iter().take(size).copied().collect();

        // Remove processed samples, keeping overlap
        let to_remove = size.saturating_sub(keep_overlap);
        self.buffer.drain(..to_remove);

        Some(chunk)
    }

    /// Get a view of the buffer without removing samples
    pub fn peek(&self, size: usize) -> Vec<f32> {
        self.buffer.iter().take(size).copied().collect()
    }

    /// Clear all samples from the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Drain all samples from the buffer
    pub fn drain_all(&mut self) -> Vec<f32> {
        self.buffer.drain(..).collect()
    }
}

/// A pool of reusable audio buffers for zero-allocation streaming
pub struct BufferPool {
    buffers: Vec<Vec<f32>>,
    available: VecDeque<usize>,
    buffer_size: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(count: usize, buffer_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(count);
        let mut available = VecDeque::with_capacity(count);

        for i in 0..count {
            let mut buffer = Vec::with_capacity(buffer_size);
            buffer.resize(buffer_size, 0.0);
            buffers.push(buffer);
            available.push_back(i);
        }

        Self {
            buffers,
            available,
            buffer_size,
        }
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self) -> Option<BufferHandle> {
        self.available.pop_front().map(|index| BufferHandle {
            pool: self as *mut BufferPool,
            index,
        })
    }

    /// Return a buffer to the pool (internal use)
    fn release(&mut self, index: usize) {
        // Clear the buffer before returning it to the pool
        self.buffers[index].fill(0.0);
        self.available.push_back(index);
    }

    /// Get a reference to a buffer by index
    fn get_buffer(&self, index: usize) -> &[f32] {
        &self.buffers[index]
    }

    /// Get a mutable reference to a buffer by index
    fn get_buffer_mut(&mut self, index: usize) -> &mut Vec<f32> {
        &mut self.buffers[index]
    }
}

/// A handle to a buffer from the pool
pub struct BufferHandle {
    pool: *mut BufferPool,
    index: usize,
}

impl BufferHandle {
    /// Get the buffer as a slice
    pub fn as_slice(&self) -> &[f32] {
        unsafe { (*self.pool).get_buffer(self.index) }
    }

    /// Get the buffer as a mutable vector
    pub fn as_mut_vec(&mut self) -> &mut Vec<f32> {
        unsafe { (*self.pool).get_buffer_mut(self.index) }
    }
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        unsafe {
            (*self.pool).release(self.index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_push_and_extract() {
        let mut buffer = AudioBuffer::new(100);

        // Push samples
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.push_samples(&samples);
        assert_eq!(buffer.len(), 5);

        // Extract chunk with overlap
        let chunk = buffer.extract_chunk(4, 2);
        assert!(chunk.is_some());
        assert_eq!(chunk.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buffer.len(), 3); // 2 samples kept as overlap + 1 remaining
    }

    #[test]
    fn test_audio_buffer_overflow() {
        let mut buffer = AudioBuffer::new(5);

        // Fill buffer to capacity
        buffer.push_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(buffer.len(), 5);

        // Push more samples, should remove oldest
        buffer.push_samples(&[6.0, 7.0]);
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.peek(5), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(2, 10);

        // Acquire buffers
        let handle1 = pool.acquire();
        assert!(handle1.is_some());

        let handle2 = pool.acquire();
        assert!(handle2.is_some());

        // Pool exhausted
        let handle3 = pool.acquire();
        assert!(handle3.is_none());

        // Return a buffer
        drop(handle1);

        // Can acquire again
        let handle4 = pool.acquire();
        assert!(handle4.is_some());
    }
}