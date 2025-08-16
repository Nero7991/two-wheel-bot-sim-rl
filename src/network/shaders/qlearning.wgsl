// Q-Learning Weight Update Compute Shader
// Performs Q-learning updates with TD error calculation and gradient-based weight updates
// Optimized for reinforcement learning training of neural networks

// Workgroup size optimized for parameter updates (32 threads)
@group(0) @binding(0) var<storage, read> q_values: array<f32>;           // Current Q-values [batch_size * output_size]
@group(0) @binding(1) var<storage, read> target_q_values: array<f32>;     // Target Q-values [batch_size * output_size]
@group(0) @binding(2) var<storage, read> actions: array<u32>;            // Taken actions [batch_size]
@group(0) @binding(3) var<storage, read> rewards: array<f32>;            // Rewards [batch_size]
@group(0) @binding(4) var<storage, read> dones: array<u32>;              // Episode done flags [batch_size]
@group(0) @binding(5) var<storage, read> hidden_activations: array<f32>; // Hidden layer activations [batch_size * hidden_size]
@group(0) @binding(6) var<storage, read> input_states: array<f32>;       // Input states [batch_size * input_size]
@group(0) @binding(7) var<storage, read_write> weights_hidden: array<f32>; // Hidden layer weights [input_size * hidden_size]
@group(0) @binding(8) var<storage, read_write> bias_hidden: array<f32>;   // Hidden layer bias [hidden_size]
@group(0) @binding(9) var<storage, read_write> weights_output: array<f32>; // Output layer weights [hidden_size * output_size]
@group(0) @binding(10) var<storage, read_write> bias_output: array<f32>;  // Output layer bias [output_size]
@group(0) @binding(11) var<storage, read_write> td_errors: array<f32>;    // TD errors [batch_size]
@group(0) @binding(12) var<uniform> params: QLearningParams;

struct QLearningParams {
    batch_size: u32,
    input_size: u32,
    hidden_size: u32,
    output_size: u32,
    learning_rate: f32,
    gamma: f32,          // Discount factor
    epsilon: f32,        // Small value for numerical stability
    clip_grad_norm: f32, // Gradient clipping threshold
}

// Compute TD errors for Q-learning
@compute @workgroup_size(32, 1, 1)
fn compute_td_errors(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= params.batch_size) {
        return;
    }
    
    let action = actions[batch_idx];
    let reward = rewards[batch_idx];
    let done = dones[batch_idx];
    
    // Get current Q-value for taken action
    let q_idx = batch_idx * params.output_size + action;
    let current_q = q_values[q_idx];
    
    // Compute target Q-value
    var target_q = reward;
    
    if (done == 0u) { // Episode not done
        // Find maximum target Q-value for next state
        var max_target_q = target_q_values[batch_idx * params.output_size];
        for (var i: u32 = 1u; i < params.output_size; i++) {
            let target_idx = batch_idx * params.output_size + i;
            max_target_q = max(max_target_q, target_q_values[target_idx]);
        }
        target_q += params.gamma * max_target_q;
    }
    
    // Compute TD error
    td_errors[batch_idx] = target_q - current_q;
}

// Update output layer weights using gradients
@compute @workgroup_size(32, 1, 1)
fn update_output_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let weight_idx = global_id.x;
    let total_output_weights = params.hidden_size * params.output_size;
    
    if (weight_idx >= total_output_weights) {
        return;
    }
    
    let hidden_idx = weight_idx / params.output_size;
    let output_idx = weight_idx % params.output_size;
    
    var gradient = 0.0;
    
    // Accumulate gradients across batch
    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {
        let action = actions[batch_idx];
        
        // Only update weights for the action that was taken
        if (action == output_idx) {
            let td_error = td_errors[batch_idx];
            let hidden_activation = hidden_activations[batch_idx * params.hidden_size + hidden_idx];
            
            // Gradient for Q-learning: td_error * hidden_activation
            gradient += td_error * hidden_activation;
        }
    }
    
    // Average gradient over batch
    gradient /= f32(params.batch_size);
    
    // Gradient clipping
    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);
    
    // Update weight using gradient ascent (positive TD error increases weight)
    weights_output[weight_idx] += params.learning_rate * gradient;
}

// Update output layer biases
@compute @workgroup_size(32, 1, 1)
fn update_output_bias(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    
    if (output_idx >= params.output_size) {
        return;
    }
    
    var gradient = 0.0;
    
    // Accumulate gradients across batch
    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {
        let action = actions[batch_idx];
        
        // Only update bias for the action that was taken
        if (action == output_idx) {
            let td_error = td_errors[batch_idx];
            gradient += td_error;
        }
    }
    
    // Average gradient over batch
    gradient /= f32(params.batch_size);
    
    // Gradient clipping
    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);
    
    // Update bias
    bias_output[output_idx] += params.learning_rate * gradient;
}

// Update hidden layer weights (backpropagated gradients)
@compute @workgroup_size(32, 1, 1)
fn update_hidden_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let weight_idx = global_id.x;
    let total_hidden_weights = params.input_size * params.hidden_size;
    
    if (weight_idx >= total_hidden_weights) {
        return;
    }
    
    let input_idx = weight_idx / params.hidden_size;
    let hidden_idx = weight_idx % params.hidden_size;
    
    var gradient = 0.0;
    
    // Accumulate gradients across batch
    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {
        let action = actions[batch_idx];
        let td_error = td_errors[batch_idx];
        let input_value = input_states[batch_idx * params.input_size + input_idx];
        let hidden_activation = hidden_activations[batch_idx * params.hidden_size + hidden_idx];
        
        // Backpropagate error through output layer
        let output_weight = weights_output[hidden_idx * params.output_size + action];
        
        // ReLU derivative: 1 if hidden_activation > 0, 0 otherwise
        let relu_derivative = select(0.0, 1.0, hidden_activation > 0.0);
        
        // Gradient computation
        let backprop_error = td_error * output_weight * relu_derivative;
        gradient += backprop_error * input_value;
    }
    
    // Average gradient over batch
    gradient /= f32(params.batch_size);
    
    // Gradient clipping
    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);
    
    // Update weight
    weights_hidden[weight_idx] += params.learning_rate * gradient;
}

// Update hidden layer biases
@compute @workgroup_size(32, 1, 1)
fn update_hidden_bias(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let hidden_idx = global_id.x;
    
    if (hidden_idx >= params.hidden_size) {
        return;
    }
    
    var gradient = 0.0;
    
    // Accumulate gradients across batch
    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {
        let action = actions[batch_idx];
        let td_error = td_errors[batch_idx];
        let hidden_activation = hidden_activations[batch_idx * params.hidden_size + hidden_idx];
        
        // Backpropagate error through output layer
        let output_weight = weights_output[hidden_idx * params.output_size + action];
        
        // ReLU derivative
        let relu_derivative = select(0.0, 1.0, hidden_activation > 0.0);
        
        // Gradient computation
        let backprop_error = td_error * output_weight * relu_derivative;
        gradient += backprop_error;
    }
    
    // Average gradient over batch
    gradient /= f32(params.batch_size);
    
    // Gradient clipping
    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);
    
    // Update bias
    bias_hidden[hidden_idx] += params.learning_rate * gradient;
}

// Compute gradient norms for monitoring training stability
@compute @workgroup_size(32, 1, 1)
fn compute_gradient_norms(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    
    if (thread_id != 0u) {
        return; // Only use first thread
    }
    
    // This is a simple implementation - in practice you'd want more sophisticated
    // gradient norm computation and statistics
    
    // For now, we'll just use the existing gradient clipping in the update functions
    // A more complete implementation would compute actual L2 norms of gradients
}

// Experience replay buffer update (for more advanced Q-learning)
@compute @workgroup_size(32, 1, 1)
fn update_experience_replay(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= params.batch_size) {
        return;
    }
    
    // This would implement experience replay buffer management
    // For now, we'll leave this as a placeholder for future implementation
    // when we add more advanced RL algorithms
}

// Combined weight update function (calls all update functions)
@compute @workgroup_size(1, 1, 1)
fn update_all_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // This function would coordinate all weight updates
    // In practice, you'd call the individual update functions separately
    // This is just a placeholder to show the overall structure
}

// Utility function to decay learning rate
@compute @workgroup_size(1, 1, 1)
fn decay_learning_rate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Learning rate decay would be handled on the CPU side
    // This is just a placeholder for potential GPU-side learning rate scheduling
}