#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]

// Include the bindgen-generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Manual error code constants not documented in whisper.h
// These are based on common error patterns seen in whisper.cpp
pub const WHISPER_ERR_INVALID_MODEL: i32 = -1;
pub const WHISPER_ERR_NOT_ENOUGH_MEMORY: i32 = -2;
pub const WHISPER_ERR_FAILED_TO_PROCESS: i32 = -3;
pub const WHISPER_ERR_INVALID_CONTEXT: i32 = -4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_defined() {
        // Just verify that our custom error constants are accessible
        assert_eq!(WHISPER_ERR_INVALID_MODEL, -1);
        assert_eq!(WHISPER_ERR_NOT_ENOUGH_MEMORY, -2);
        assert_eq!(WHISPER_ERR_FAILED_TO_PROCESS, -3);
        assert_eq!(WHISPER_ERR_INVALID_CONTEXT, -4);
    }
}