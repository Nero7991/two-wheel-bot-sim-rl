// ReLU Activation Function Compute Shader
// Applies ReLU(x) = max(0, x) activation element-wise
// Optimized for neural network hidden layer activation

// Workgroup size optimized for element-wise operations (64 threads)
@group(0) @binding(0) var<storage, read> input_data: array<f32>;     // Input values
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>; // Output values
@group(0) @binding(2) var<uniform> params: ActivationParams;

struct ActivationParams {
    size: u32,        // Total number of elements
    padding1: u32,    // Padding for alignment
    padding2: u32,    // Padding for alignment  
    padding3: u32,    // Padding for alignment
}

// Standard ReLU activation: f(x) = max(0, x)
@compute @workgroup_size(64, 1, 1)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds checking
    if (index >= params.size) {
        return;
    }
    
    let input_val = input_data[index];
    output_data[index] = max(0.0, input_val);
}

// Leaky ReLU activation: f(x) = x if x > 0, alpha * x if x <= 0
// Alpha is hardcoded to 0.01 for simplicity
@compute @workgroup_size(64, 1, 1)
fn leaky_relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= params.size) {
        return;
    }
    
    let input_val = input_data[index];
    let alpha = 0.01; // Leaky ReLU slope for negative values
    
    if (input_val > 0.0) {
        output_data[index] = input_val;
    } else {
        output_data[index] = alpha * input_val;
    }
}

// ReLU derivative for backpropagation: f'(x) = 1 if x > 0, 0 if x <= 0
@compute @workgroup_size(64, 1, 1)
fn relu_derivative(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= params.size) {
        return;
    }
    
    let input_val = input_data[index];
    
    if (input_val > 0.0) {
        output_data[index] = 1.0;
    } else {
        output_data[index] = 0.0;
    }
}

// Leaky ReLU derivative for backpropagation
@compute @workgroup_size(64, 1, 1)
fn leaky_relu_derivative(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= params.size) {
        return;
    }
    
    let input_val = input_data[index];
    let alpha = 0.01;
    
    if (input_val > 0.0) {
        output_data[index] = 1.0;
    } else {
        output_data[index] = alpha;
    }
}

// In-place ReLU for memory efficiency (input and output are the same buffer)
@compute @workgroup_size(64, 1, 1)
fn relu_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= params.size) {
        return;
    }
    
    let input_val = input_data[index];
    input_data[index] = max(0.0, input_val);
}

// Vectorized ReLU for processing multiple elements per thread
@compute @workgroup_size(16, 1, 1)
fn relu_vectorized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * 4u;
    
    // Process 4 elements per thread for better memory throughput
    for (var i: u32 = 0u; i < 4u; i++) {
        let index = base_index + i;
        
        if (index >= params.size) {
            return;
        }
        
        let input_val = input_data[index];
        output_data[index] = max(0.0, input_val);
    }
}

// Batch ReLU processing for training
@compute @workgroup_size(64, 1, 1)
fn relu_batch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= params.size) {
        return;
    }
    
    // Apply ReLU activation
    let input_val = input_data[index];
    output_data[index] = max(0.0, input_val);
}

// Softmax activation for output layer (alternative to raw outputs)
@compute @workgroup_size(64, 1, 1)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(workgroup_id) workgroup_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let index = global_id.x;
    
    if (index >= params.size) {
        return;
    }
    
    // Find maximum value for numerical stability
    var max_val = input_data[0];
    for (var i: u32 = 1u; i < params.size; i++) {
        max_val = max(max_val, input_data[i]);
    }
    
    // Compute exponentials and sum
    var sum_exp = 0.0;
    for (var i: u32 = 0u; i < params.size; i++) {
        sum_exp += exp(input_data[i] - max_val);
    }
    
    // Compute softmax probability
    let exp_val = exp(input_data[index] - max_val);
    output_data[index] = exp_val / sum_exp;
}

// Clipped ReLU (ReLU6): f(x) = min(max(0, x), 6)
// Useful for preventing activation explosion in some networks
@compute @workgroup_size(64, 1, 1)
fn relu6(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= params.size) {
        return;
    }
    
    let input_val = input_data[index];
    output_data[index] = min(max(0.0, input_val), 6.0);
}