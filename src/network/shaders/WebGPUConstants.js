/**
 * WebGPU Constants Helper
 * Provides safe access to WebGPU constants with fallback values
 */

/**
 * Get WebGPU buffer usage flags safely
 * @returns {Object} Buffer usage flags
 */
export function getBufferUsage() {
    if (typeof GPUBufferUsage !== 'undefined') {
        return GPUBufferUsage;
    }
    
    // Fallback values based on WebGPU spec
    return {
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

/**
 * Get WebGPU map mode flags safely
 * @returns {Object} Map mode flags
 */
export function getMapMode() {
    if (typeof GPUMapMode !== 'undefined') {
        return GPUMapMode;
    }
    
    // Fallback values based on WebGPU spec
    return {
        READ: 0x0001,
        WRITE: 0x0002
    };
}

/**
 * Get WebGPU shader stage flags safely
 * @returns {Object} Shader stage flags
 */
export function getShaderStage() {
    if (typeof GPUShaderStage !== 'undefined') {
        return GPUShaderStage;
    }
    
    // Fallback values based on WebGPU spec
    return {
        VERTEX: 0x1,
        FRAGMENT: 0x2,
        COMPUTE: 0x4
    };
}

/**
 * Check if WebGPU is available
 * @returns {boolean} True if WebGPU is available
 */
export function isWebGPUAvailable() {
    return typeof navigator !== 'undefined' && 
           'gpu' in navigator &&
           typeof GPUBufferUsage !== 'undefined';
}

/**
 * Safe WebGPU constants wrapper
 */
export const WebGPUConstants = {
    get BufferUsage() {
        return getBufferUsage();
    },
    
    get MapMode() {
        return getMapMode();
    },
    
    get ShaderStage() {
        return getShaderStage();
    },
    
    isAvailable: isWebGPUAvailable
};

export default WebGPUConstants;