/**
 * Matrix Utilities for Neural Network Computations
 * 
 * Efficient matrix operations using Float32Array for optimal performance
 * in both CPU and preparation for WebGPU implementations.
 */

/**
 * Perform matrix multiplication: C = A * B + bias
 * @param {Float32Array} A - Input matrix (flattened, row-major)
 * @param {Float32Array} B - Weight matrix (flattened, row-major)
 * @param {Float32Array} bias - Bias vector
 * @param {number} aRows - Number of rows in A
 * @param {number} aCols - Number of columns in A (must equal bRows)
 * @param {number} bCols - Number of columns in B
 * @returns {Float32Array} Result matrix C
 */
export function matrixMultiply(A, B, bias, aRows, aCols, bCols) {
    if (A.length !== aRows * aCols) {
        throw new Error(`Matrix A size mismatch. Expected ${aRows * aCols}, got ${A.length}`);
    }
    if (B.length !== aCols * bCols) {
        throw new Error(`Matrix B size mismatch. Expected ${aCols * bCols}, got ${B.length}`);
    }
    if (bias && bias.length !== bCols) {
        throw new Error(`Bias size mismatch. Expected ${bCols}, got ${bias.length}`);
    }
    
    const result = new Float32Array(aRows * bCols);
    
    // Optimized matrix multiplication with optional bias
    for (let i = 0; i < aRows; i++) {
        for (let j = 0; j < bCols; j++) {
            let sum = 0;
            for (let k = 0; k < aCols; k++) {
                sum += A[i * aCols + k] * B[k * bCols + j];
            }
            // Add bias if provided
            if (bias) {
                sum += bias[j];
            }
            result[i * bCols + j] = sum;
        }
    }
    
    return result;
}

/**
 * Perform vector-matrix multiplication for single input: y = x * W + b
 * Optimized version for single sample forward pass
 * @param {Float32Array} input - Input vector
 * @param {Float32Array} weights - Weight matrix (flattened, row-major)
 * @param {Float32Array} bias - Bias vector
 * @param {number} inputSize - Size of input vector
 * @param {number} outputSize - Size of output vector
 * @returns {Float32Array} Result vector
 */
export function vectorMatrixMultiply(input, weights, bias, inputSize, outputSize) {
    if (input.length !== inputSize) {
        throw new Error(`Input size mismatch. Expected ${inputSize}, got ${input.length}`);
    }
    if (weights.length !== inputSize * outputSize) {
        throw new Error(`Weight matrix size mismatch. Expected ${inputSize * outputSize}, got ${weights.length}`);
    }
    if (bias.length !== outputSize) {
        throw new Error(`Bias size mismatch. Expected ${outputSize}, got ${bias.length}`);
    }
    
    const result = new Float32Array(outputSize);
    
    // Optimized vector-matrix multiplication
    for (let j = 0; j < outputSize; j++) {
        let sum = bias[j]; // Start with bias
        for (let i = 0; i < inputSize; i++) {
            sum += input[i] * weights[i * outputSize + j];
        }
        result[j] = sum;
    }
    
    return result;
}

/**
 * Apply ReLU activation function element-wise
 * @param {Float32Array} input - Input array
 * @returns {Float32Array} Output array with ReLU applied
 */
export function relu(input) {
    const result = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
        result[i] = Math.max(0, input[i]);
    }
    return result;
}

/**
 * Apply ReLU activation function in-place
 * @param {Float32Array} input - Input array to modify
 */
export function reluInPlace(input) {
    for (let i = 0; i < input.length; i++) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}

/**
 * Copy array (useful for avoiding in-place modifications)
 * @param {Float32Array} input - Input array
 * @returns {Float32Array} Copy of input array
 */
export function copyArray(input) {
    return new Float32Array(input);
}

/**
 * Initialize weights using Xavier/Glorot initialization
 * Optimal for sigmoid and tanh activations
 * @param {number} inputSize - Number of input neurons
 * @param {number} outputSize - Number of output neurons
 * @returns {Float32Array} Initialized weight matrix
 */
export function xavierInit(inputSize, outputSize) {
    const limit = Math.sqrt(6.0 / (inputSize + outputSize));
    const weights = new Float32Array(inputSize * outputSize);
    
    for (let i = 0; i < weights.length; i++) {
        weights[i] = (Math.random() * 2 - 1) * limit;
    }
    
    return weights;
}

/**
 * Initialize weights using He initialization
 * Optimal for ReLU activations
 * @param {number} inputSize - Number of input neurons
 * @param {number} outputSize - Number of output neurons
 * @returns {Float32Array} Initialized weight matrix
 */
export function heInit(inputSize, outputSize) {
    const stddev = Math.sqrt(2.0 / inputSize);
    const weights = new Float32Array(inputSize * outputSize);
    
    for (let i = 0; i < weights.length; i++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        weights[i] = normal * stddev;
    }
    
    return weights;
}

/**
 * Initialize bias vector with zeros
 * @param {number} size - Size of bias vector
 * @returns {Float32Array} Zero-initialized bias vector
 */
export function zeroInit(size) {
    return new Float32Array(size);
}

/**
 * Initialize bias vector with small random values
 * @param {number} size - Size of bias vector
 * @param {number} scale - Scale factor for random values (default: 0.01)
 * @returns {Float32Array} Small random initialized bias vector
 */
export function smallRandomInit(size, scale = 0.01) {
    const bias = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        bias[i] = (Math.random() * 2 - 1) * scale;
    }
    return bias;
}

/**
 * Validate matrix dimensions for multiplication
 * @param {number} aRows - Rows in matrix A
 * @param {number} aCols - Columns in matrix A
 * @param {number} bRows - Rows in matrix B
 * @param {number} bCols - Columns in matrix B
 * @throws {Error} If dimensions are incompatible
 */
export function validateMatrixDimensions(aRows, aCols, bRows, bCols) {
    if (aCols !== bRows) {
        throw new Error(`Matrix dimensions incompatible for multiplication: (${aRows}x${aCols}) * (${bRows}x${bCols})`);
    }
}

/**
 * Performance utilities for benchmarking
 */
export const Performance = {
    /**
     * Time a function execution
     * @param {Function} fn - Function to time
     * @param {...any} args - Arguments to pass to function
     * @returns {Object} Result and execution time
     */
    time(fn, ...args) {
        const start = performance.now();
        const result = fn(...args);
        const end = performance.now();
        return {
            result,
            time: end - start
        };
    },
    
    /**
     * Run multiple iterations and return average time
     * @param {Function} fn - Function to benchmark
     * @param {number} iterations - Number of iterations
     * @param {...any} args - Arguments to pass to function
     * @returns {Object} Average time and total iterations
     */
    benchmark(fn, iterations, ...args) {
        let totalTime = 0;
        let lastResult = null;
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            lastResult = fn(...args);
            const end = performance.now();
            totalTime += (end - start);
        }
        
        return {
            averageTime: totalTime / iterations,
            totalTime,
            iterations,
            lastResult
        };
    }
};