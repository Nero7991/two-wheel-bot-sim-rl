// Matrix Multiplication Compute Shader for Neural Network
// Performs C = A * B + bias operation
// Optimized for small neural networks (2->4-16->3 architecture)

// Workgroup size optimized for GPU efficiency (8x8 = 64 threads)
@group(0) @binding(0) var<storage, read> matrixA: array<f32>;    // Input matrix [M x K]
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;    // Weight matrix [K x N]
@group(0) @binding(2) var<storage, read> bias: array<f32>;       // Bias vector [N]
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>; // Output matrix [M x N]
@group(0) @binding(4) var<uniform> params: MatMulParams;

struct MatMulParams {
    M: u32,    // Number of rows in A (batch size)
    K: u32,    // Number of columns in A / rows in B (input size)
    N: u32,    // Number of columns in B (output size)
    padding: u32, // Padding for alignment
}

// Local workgroup memory for cache optimization
var<workgroup> tileA: array<array<f32, 8>, 8>;
var<workgroup> tileB: array<array<f32, 8>, 8>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let row = global_id.x;
    let col = global_id.y;
    
    // Bounds checking for matrix dimensions
    if (row >= params.M || col >= params.N) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Process matrix multiplication in tiles for better cache performance
    let num_tiles = (params.K + 7u) / 8u; // Ceiling division
    
    for (var tile: u32 = 0u; tile < num_tiles; tile++) {
        // Load tile into workgroup memory
        let tileRowA = tile * 8u + local_id.x;
        let tileColB = tile * 8u + local_id.y;
        
        // Load A tile (with bounds checking)
        if (row < params.M && tileRowA < params.K) {
            tileA[local_id.y][local_id.x] = matrixA[row * params.K + tileRowA];
        } else {
            tileA[local_id.y][local_id.x] = 0.0;
        }
        
        // Load B tile (with bounds checking)
        if (tileColB < params.K && col < params.N) {
            tileB[local_id.y][local_id.x] = matrixB[tileColB * params.N + col];
        } else {
            tileB[local_id.y][local_id.x] = 0.0;
        }
        
        // Synchronize workgroup
        workgroupBarrier();
        
        // Compute partial dot product using tile data
        for (var k: u32 = 0u; k < 8u; k++) {
            sum += tileA[local_id.y][k] * tileB[k][local_id.x];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Add bias and store result
    let output_index = row * params.N + col;
    matrixC[output_index] = sum + bias[col];
}

// Alternative simpler implementation for very small matrices (fallback)
@compute @workgroup_size(64, 1, 1)
fn matmul_simple(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = params.M * params.N;
    
    if (index >= total_elements) {
        return;
    }
    
    let row = index / params.N;
    let col = index % params.N;
    
    var sum: f32 = 0.0;
    
    // Simple dot product computation
    for (var k: u32 = 0u; k < params.K; k++) {
        sum += matrixA[row * params.K + k] * matrixB[k * params.N + col];
    }
    
    // Add bias and store result
    matrixC[index] = sum + bias[col];
}

// Batch matrix multiplication for training efficiency
@compute @workgroup_size(8, 8, 1)
fn matmul_batch(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let batch_idx = global_id.z;
    let row = global_id.x;
    let col = global_id.y;
    
    // Bounds checking
    if (row >= params.M || col >= params.N) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Calculate offsets for this batch
    let batch_offset_A = batch_idx * params.M * params.K;
    let batch_offset_C = batch_idx * params.M * params.N;
    
    // Matrix multiplication for this batch element
    for (var k: u32 = 0u; k < params.K; k++) {
        let a_val = matrixA[batch_offset_A + row * params.K + k];
        let b_val = matrixB[k * params.N + col]; // Weights are shared across batch
        sum += a_val * b_val;
    }
    
    // Add bias and store result
    let output_index = batch_offset_C + row * params.N + col;
    matrixC[output_index] = sum + bias[col];
}