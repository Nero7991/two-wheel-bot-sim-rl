/**
 * WebGPU Polyfill for Safe Constants Access
 * 
 * This module ensures WebGPU constants are available even when WebGPU is not.
 * It should be imported before any other WebGPU-related modules.
 */

// Check if WebGPU constants are already defined
if (typeof globalThis.GPUBufferUsage === 'undefined') {
    // Define WebGPU constants as fallbacks based on the WebGPU specification
    globalThis.GPUBufferUsage = {
        MAP_READ: 0x0001,
        MAP_WRITE: 0x0002,
        COPY_SRC: 0x0004,
        COPY_DST: 0x0008,
        INDEX: 0x0010,
        VERTEX: 0x0020,
        UNIFORM: 0x0040,
        STORAGE: 0x0080,
        INDIRECT: 0x0100,
        QUERY_RESOLVE: 0x0200
    };
}

if (typeof globalThis.GPUMapMode === 'undefined') {
    globalThis.GPUMapMode = {
        READ: 0x0001,
        WRITE: 0x0002
    };
}

if (typeof globalThis.GPUShaderStage === 'undefined') {
    globalThis.GPUShaderStage = {
        VERTEX: 0x1,
        FRAGMENT: 0x2,
        COMPUTE: 0x4
    };
}

// Export a flag to indicate whether we're using polyfilled values
export const isPolyfilled = !navigator.gpu;

// Export the constants for convenience
export const BufferUsage = globalThis.GPUBufferUsage;
export const MapMode = globalThis.GPUMapMode;
export const ShaderStage = globalThis.GPUShaderStage;

console.log('WebGPU polyfill loaded:', isPolyfilled ? 'Using polyfilled constants' : 'Using native WebGPU constants');