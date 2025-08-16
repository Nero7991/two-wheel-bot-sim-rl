# Two-Wheel Balancing Robot RL Training - Focused Web App Requirements

## Project Overview
Build a web application for training a neural network to balance a two-wheeled robot using Q-learning reinforcement learning, with WebGPU acceleration for fast training and C++ export for embedded deployment.

## Technology Stack
- **Frontend**: Vanilla JavaScript or React
- **Build Tool**: Vite
- **ML**: WebGPU-accelerated custom implementation with CPU fallback
- **Visualization**: HTML5 Canvas (2D) or simple Three.js (3D)

---

## Core Features Checklist

### 1. Neural Network Implementation
- [ ] **Configurable architecture**
  - `function createNetwork(inputSize: 2, hiddenSize: number, outputSize: 3): NeuralNetwork`
  - Hidden layer size: 4-16 neurons
  - ReLU activation for hidden layer
  - Linear output layer
  - Total parameters under 200 for embedded constraints

- [ ] **Forward pass**
  - `function forward(input: Float32Array): Float32Array`
  - Efficient matrix multiplication
  - Batch processing support

- [ ] **Q-learning training**
  - `function train(state, action, reward, nextState, done, lr, gamma): void`
  - Simple gradient descent
  - Momentum optimizer (optional)

### 2. WebGPU Acceleration (Priority Feature)
- [ ] **WebGPU initialization**
  ```javascript
  async function initWebGPU() {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    return { device, queue: device.queue };
  }
  ```

- [ ] **Matrix multiplication kernel**
  ```wgsl
  @compute @workgroup_size(8, 8)
  fn matmul(@builtin(global_invocation_id) id: vec3<u32>) {
    // Compute C = A * B + bias
    // For layers: hidden = input * W1 + b1
    //            output = hidden * W2 + b2
  }
  ```

- [ ] **Activation function kernel**
  ```wgsl
  @compute @workgroup_size(64)
  fn relu(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    output[idx] = max(0.0, input[idx]);
  }
  ```

- [ ] **Batch training kernel**
  ```wgsl
  @compute @workgroup_size(32)
  fn qlearning_update(@builtin(global_invocation_id) id: vec3<u32>) {
    // Parallel Q-value updates for experience batch
    // Compute TD error and gradient updates
  }
  ```

- [ ] **CPU fallback**
  - `class CPUNetwork implements NeuralNetwork`
  - Automatic detection and switching
  - Same API as GPU version
### 3. Physics Simulation
- [ ] **Robot dynamics**
  - `class BalancingRobot`
  - Inverted pendulum physics
  - Configurable parameters:
    - Mass: 0.5 - 3.0 kg
    - Center of mass height: 0.2 - 1.0 m
    - Motor strength: 2 - 10 N⋅m
  - Fixed timestep: 20ms (50 Hz)

- [ ] **State representation**
  ```javascript
  class RobotState {
    angle: number;        // radians
    angularVelocity: number;  // rad/s
    position: number;     // meters
    velocity: number;     // m/s
  }
  ```

- [ ] **Reward function**
  ```javascript
  function calculateReward(state: RobotState): number {
    let reward = 1.0;  // Alive bonus
    reward -= Math.abs(state.angle) * 10;
    reward -= Math.abs(state.angularVelocity) * 0.5;
    reward -= Math.abs(motorTorque) * 0.1;
    if (Math.abs(state.angle) > Math.PI/3) reward = -100;
    return reward;
  }
  ```
### 4. Training System
- [ ] **Experience replay buffer**
  ```javascript
  class ReplayBuffer {
    constructor(maxSize: 10000) {}
    add(state, action, reward, nextState, done): void
    sample(batchSize: 32): Experience[]
    clear(): void
  }
  ```

- [ ] **Training loop**
  ```javascript
  async function trainEpisode() {
    // Collect experience
    // Sample from replay buffer
    // Update network with WebGPU
    // Update epsilon
  }
  ```

- [ ] **Hyperparameters**
  - Learning rate: 0.0001 - 0.01
  - Epsilon (exploration): 0.0 - 0.5
  - Discount factor (gamma): 0.8 - 0.99
  - Batch size: 32 - 128

### 5. Visualization
- [ ] **2D Canvas rendering**
  - Robot visualization (pendulum + wheels)
  - Real-time angle display
  - Motor torque indicator
  - Ground/environment
- [ ] **Performance graphs**
  - Episode reward over time
  - Average reward (last 100 episodes)
  - Loss curve
  - Simple line charts using Canvas

- [ ] **Status display**
  - Current episode
  - Steps in episode
  - Current angle/velocity
  - FPS counter
  - GPU/CPU mode indicator

### 6. User Interface
- [ ] **Training controls**
  - Start/Pause/Reset buttons
  - Training speed slider (1x - 100x)
  - Save/Load model buttons

- [ ] **Configuration panel**
  - Network size slider
  - Hyperparameter sliders
  - Physics parameter sliders
  - GPU/CPU toggle

- [ ] **Info panel**
  - Network architecture display
  - Parameter count
  - Memory usage estimate
  - Training mode (GPU/CPU)
### 7. Model Export
- [ ] **C++ code generation**
  ```javascript
  function exportToCpp(network: NeuralNetwork): string {
    // Generate fixed-point C++ code
    // Include weight arrays
    // Simple forward pass implementation
    return cppCode;
  }
  ```

- [ ] **Weight quantization**
  ```javascript
  function quantizeWeights(weights: Float32Array): Int8Array {
    // Convert float32 to int8 for embedded systems
    // Scale and clip values
    return quantized;
  }
  ```

- [ ] **Arduino template**
  ```javascript
  function generateArduinoSketch(network, config): string {
    // Complete Arduino sketch with:
    // - IMU reading
    // - Neural network inference
    // - Motor control
    return sketch;
  }
  ```
---

## WebGPU Implementation Details

### Buffer Management
```javascript
class GPUBufferManager {
  createBuffer(device, data, usage): GPUBuffer
  updateBuffer(queue, buffer, data): void
  readBuffer(device, buffer): Promise<Float32Array>
}
```

### Training Pipeline
```javascript
class GPUTrainingPipeline {
  constructor(device, network) {
    this.createComputePipelines();
    this.allocateBuffers();
  }
  
  async forward(input): Float32Array
  async backward(gradient): void
  async updateWeights(lr): void
}
```

### Performance Optimizations
- Batch size optimization for GPU occupancy
- Async queue management
- Buffer reuse
- Minimal CPU-GPU data transfer
---

## Performance Targets

### Training Speed
- **CPU Mode**: 100+ episodes/minute
- **WebGPU Mode**: 1000+ episodes/minute
- **Batch processing**: 32-128 samples parallel

### Rendering
- **Canvas 2D**: 60 FPS
- **Simulation**: 50 Hz update rate
- **UI responsiveness**: < 16ms

### Memory Usage
- **Network weights**: < 1KB (int8 quantized)
- **Replay buffer**: < 10MB (10K experiences)
- **GPU buffers**: < 50MB total

---

## File Structure
```
project/
├── index.html
├── src/
│   ├── main.js
│   ├── network/
│   │   ├── NeuralNetwork.js
│   │   ├── WebGPUBackend.js
│   │   ├── CPUBackend.js│   │   └── shaders/
│   │       ├── matmul.wgsl
│   │       ├── relu.wgsl
│   │       └── qlearning.wgsl
│   ├── physics/
│   │   └── BalancingRobot.js
│   ├── training/
│   │   ├── QLearning.js
│   │   └── ReplayBuffer.js
│   ├── visualization/
│   │   ├── Renderer.js
│   │   └── Charts.js
│   └── export/
│       ├── CppExporter.js
│       └── templates/
│           └── arduino.template
├── package.json
└── vite.config.js
```

---

## Implementation Priority

### Phase 1: Core CPU Implementation (Week 1)
1. Basic neural network (CPU)
2. Physics simulation
3. Q-learning algorithm
4. Simple 2D visualization
5. Basic UI controls
### Phase 2: WebGPU Acceleration (Week 2)
1. WebGPU device initialization
2. Matrix multiplication shader
3. Forward pass on GPU
4. Batch training on GPU
5. CPU fallback system

### Phase 3: Polish & Export (Week 3)
1. Experience replay buffer
2. Performance graphs
3. C++ code export
4. Arduino template generation
5. Testing & optimization

---

## Success Criteria

1. **Training converges** in under 100 episodes
2. **WebGPU provides 5-10x speedup** over CPU
3. **Exported C++ code runs** on 48MHz MCU
4. **Memory usage** stays under 384KB on embedded
5. **Real-time visualization** at 60 FPS