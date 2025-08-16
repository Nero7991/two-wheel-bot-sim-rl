# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a web application for training a neural network to balance a two-wheeled robot using Q-learning reinforcement learning. The project features WebGPU acceleration for fast training and C++ export capabilities for embedded deployment on microcontrollers.

## Technology Stack

- **Frontend**: Vanilla JavaScript or React
- **Build Tool**: Vite  
- **ML**: WebGPU-accelerated custom implementation with CPU fallback
- **Visualization**: HTML5 Canvas (2D) or simple Three.js (3D)

## Development Commands

Since this is a new project without existing build configuration, you will likely need to set up:

- `npm init` - Initialize package.json
- `npm install vite` - Install build tool
- `npm run dev` - Start development server (after Vite setup)
- `npm run build` - Build for production (after Vite setup)

## Architecture Overview

### Core Components

**Neural Network Layer** (`src/network/`)
- `NeuralNetwork.js` - Main network interface with configurable architecture (2 inputs, 4-16 hidden neurons, 3 outputs)
- `WebGPUBackend.js` - GPU-accelerated implementation for fast training
- `CPUBackend.js` - Fallback implementation for compatibility
- `shaders/` - WGSL compute shaders for matrix operations, activation functions, and Q-learning updates

**Physics Simulation** (`src/physics/`)
- `BalancingRobot.js` - Inverted pendulum physics with configurable mass, height, and motor strength
- Fixed 20ms timestep (50 Hz) simulation
- State representation: angle, angular velocity, position, velocity

**Training System** (`src/training/`)
- `QLearning.js` - Q-learning algorithm implementation with experience replay
- `ReplayBuffer.js` - Circular buffer for storing training experiences (max 10K samples)
- Hyperparameters: learning rate (0.0001-0.01), epsilon (0.0-0.5), gamma (0.8-0.99)

**Visualization** (`src/visualization/`)
- `Renderer.js` - 2D Canvas rendering of robot, environment, and real-time metrics
- `Charts.js` - Performance graphs for episode rewards, loss curves, and training progress

**Export System** (`src/export/`)
- `CppExporter.js` - Generates fixed-point C++ code for embedded deployment
- `templates/arduino.template` - Complete Arduino sketch template with IMU reading and motor control

### Key Constraints

- **Network Size**: Total parameters under 200 for embedded deployment
- **Memory**: Network weights < 1KB when quantized to int8
- **Performance**: Target 100+ episodes/minute on CPU, 1000+ on WebGPU
- **Embedded Target**: 48MHz MCU with 384KB memory limit

### Training Pipeline

1. **Initialization**: WebGPU device setup with CPU fallback detection
2. **Experience Collection**: Robot simulation with exploration (epsilon-greedy)
3. **Batch Training**: Parallel Q-value updates on GPU using experience replay
4. **Model Export**: Weight quantization and C++ code generation for embedded deployment

### WebGPU Implementation

- **Buffer Management**: Efficient GPU buffer allocation and reuse
- **Compute Pipelines**: Separate shaders for matrix multiplication, ReLU activation, and Q-learning updates
- **Async Operations**: Non-blocking training with minimal CPU-GPU data transfer
- **Batch Processing**: 32-128 samples processed in parallel for optimal GPU utilization

## File Structure

```
project/
├── index.html
├── src/
│   ├── main.js
│   ├── network/
│   │   ├── NeuralNetwork.js
│   │   ├── WebGPUBackend.js
│   │   ├── CPUBackend.js
│   │   └── shaders/
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

## Implementation Phases

**Phase 1: Core CPU Implementation**
- Basic neural network with forward pass and Q-learning
- Physics simulation and reward function
- Simple 2D visualization and UI controls

**Phase 2: WebGPU Acceleration**  
- WebGPU device initialization and compute shaders
- GPU-accelerated forward pass and batch training
- Performance optimization and fallback handling

**Phase 3: Export & Polish**
- Experience replay buffer and performance metrics
- C++ code generation and Arduino template
- Testing and optimization for embedded deployment

## Success Criteria

- Training converges within 100 episodes
- WebGPU provides 5-10x speedup over CPU
- Exported C++ code runs on 48MHz MCU
- Memory usage under 384KB on embedded systems
- Real-time visualization at 60 FPS

Use Port 3005 for this app. If something exists, you are free to kill it. 3005 is reserved for this.

## Testing Requirements

Before committing changes, test the changes using Puppeteer with the following resolutions to ensure proper layout and alignment:

- **1280x800** - Standard desktop resolution
- **1920x1080** - Full HD resolution  

Make sure everything is nice and aligned across both test resolutions.
