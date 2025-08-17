# Two-Wheel Balancing Robot RL Simulator

**Live Demo**: [https://botsim.orenslab.com](https://botsim.orenslab.com)

Train a neural network to balance a two-wheeled robot directly in your browser. This simulator uses Deep Q-Network (DQN) reinforcement learning with optional WebGPU acceleration for fast training. Export trained models as C++ code for real hardware deployment.

![Two-Wheel Bot Simulator](https://img.shields.io/badge/Status-Live-brightgreen) ![WebGPU](https://img.shields.io/badge/WebGPU-Accelerated-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

### Core Capabilities
- DQN training with target networks and experience replay
- WebGPU acceleration providing 5-10x training speedup
- Real-time 2D physics visualization
- Manual control mode for debugging and testing
- Multiple network architectures from 4 to 256 neurons

### Machine Learning
- Deep Q-Network with separate target network (updated every 100 steps)
- Huber loss for stable Q-value updates
- Linear epsilon decay from 0.9 to 0.01 over 2500 steps
- Gradient clipping and weight clamping for stability
- Configurable learning rate, batch size, and network architecture

### Performance
- Training speed control from 0.1x to 100x real-time
- 100+ episodes per minute on CPU, 1000+ with WebGPU
- C++ export for Arduino and embedded systems
- Memory footprint under 1KB for quantized models

### User Interface
- Real-time parameter adjustment during training
- Debug mode with manual control and adjustable speed
- Live performance charts (rewards, loss, Q-values)
- Pre-configured network architectures for different hardware targets

## Quick Start

### Online Demo
Visit [https://botsim.orenslab.com](https://botsim.orenslab.com) to try the simulator immediately in your browser.

### Local Development

```bash
# Clone the repository
git clone git@github.com:Nero7991/two-wheel-bot-sim-rl.git
cd two-wheel-bot-sim-rl

# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:3005
```

## Usage Guide

### 1. Quick Training
1. Open the simulator
2. Select "DQN Standard" network preset
3. Click "Start Training"
4. Use speed controls (1x, 10x, 100x) to accelerate
5. Robot typically learns to balance within 50-100 episodes

### 2. Manual Testing
1. Expand "Debug Controls"
2. Enable "User Control"
3. Use arrow keys to control the robot
4. Adjust speed slider for slow-motion testing
5. Monitor real-time reward values

### 3. Advanced Configuration
- Network Architecture: Choose from Micro (4 neurons) to DQN Standard (128 neurons)
- Hyperparameters: Adjust learning rate, epsilon, batch size
- Physics Parameters: Modify robot mass, height, motor strength
- Training Speed: Fine-tune with slider or use quick presets

## Architecture

### Network Presets
| Preset | Hidden Neurons | Parameters | Use Case |
|--------|---------------|------------|----------|
| Micro Bot | 4 | <50 | Ultra-lightweight embedded |
| Nano Bot | 6 | <80 | Small microcontrollers |
| Classic Bot | 8 | <150 | Standard embedded (ESP32) |
| DQN Standard | 128 | ~771 | Web training (recommended) |
| Enhanced Bot | 12→8 | <400 | Multi-layer embedded |
| Advanced Bot | 16→12→8 | <800 | Complex control tasks |

### DQN Implementation
- Target Networks: Separate target network updated every 100 steps
- Experience Replay: 10K sample buffer for decorrelated training
- Epsilon-Greedy: Linear decay from 0.9 to 0.01 over 2500 steps
- Batch Training: 128 samples per batch (PyTorch standard)
- Huber Loss: More robust than MSE for Q-value regression

### Physics Model
- Inverted Pendulum: Realistic two-wheel balancing dynamics
- State Space: Robot angle and angular velocity (2D input)
- Action Space: Motor torque [-1.0, 0.0, 1.0] (3 discrete actions)
- Reward Function: Simple upright reward (1.0 when vertical, 0.0 at failure)
- Fixed Timestep: 20ms simulation steps (50Hz)

## Training Performance

### Training Results
The DQN Standard configuration typically achieves:

- Baseline (Classic): 12.83 average reward
- DQN Standard: 78.51 average reward (6x improvement)
- Convergence within 50-100 episodes
- Q-values stabilize between 10-15 for trained models

### Speed Comparison
| Configuration | CPU | WebGPU |
|--------------|-----|--------|
| Training Episodes/min | 100+ | 1000+ |
| Speedup Factor | 1x | 5-10x |
| Batch Processing | Sequential | Parallel |

## Export & Deployment

### Arduino C++ Export
The simulator can export trained networks as optimized C++ code:

```cpp
// Generated Arduino code example
const float weights_input_hidden[256] = { /* optimized weights */ };
const float weights_hidden_output[384] = { /* optimized weights */ };

int predict_action(float angle, float angular_velocity) {
    // Fixed-point neural network inference
    // Optimized for 48MHz microcontrollers
    // Memory usage: <1KB
}
```

### Embedded Constraints
- Memory: Under 384KB total (weights + code)
- Compute: Runs on 48MHz ARM Cortex-M processors
- Real-time: 50Hz control loop capability
- Quantization: Int8 weights for minimal memory

## Development

### Project Structure
```
src/
├── main.js                 # Application entry point
├── network/               # Neural network implementations
│   ├── NeuralNetwork.js   # Network architecture definitions
│   ├── WebGPUBackend.js   # GPU-accelerated backend
│   ├── CPUBackend.js      # CPU fallback backend
│   └── NetworkPresets.js  # Pre-configured architectures
├── training/              # RL algorithms
│   ├── QLearning.js       # DQN implementation
│   └── ReplayBuffer.js    # Experience replay
├── physics/               # Physics simulation
│   └── BalancingRobot.js  # Inverted pendulum model
├── visualization/         # Rendering and UI
│   └── Renderer.js        # 2D canvas visualization
└── export/                # Code generation
    └── CppExporter.js     # Arduino C++ export
```

### Key Technologies
- Frontend: Vanilla JavaScript with Vite build system
- ML: Custom DQN implementation (WebGPU + CPU backends)
- Physics: Inverted pendulum simulation
- Visualization: HTML5 Canvas 2D rendering
- Export: Template-based C++ code generation

### Build Commands
```bash
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview production build
npm run lint         # Code linting
npm run test         # Run tests
```

## Contributing

Contributions are welcome. Key areas for improvement:

- Performance: WebGPU shader optimizations
- Algorithms: Additional RL algorithms (PPO, A3C, SAC)
- Physics: More complex robot models
- Export: Support for other embedded platforms
- UI: Enhanced visualization and debugging tools

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Research & References

This implementation follows standard DQN practices:

- Deep Q-Networks: Mnih et al. (2015) - Human-level control through deep reinforcement learning
- PyTorch DQN Tutorial: [Official PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- Inverted Pendulum Control: Classical control theory applied to RL
- Embedded ML: Optimization for resource-constrained deployment

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- Live Demo: [https://botsim.orenslab.com](https://botsim.orenslab.com)
- Repository: [https://github.com/Nero7991/two-wheel-bot-sim-rl](https://github.com/Nero7991/two-wheel-bot-sim-rl)
- Issues: [GitHub Issues](https://github.com/Nero7991/two-wheel-bot-sim-rl/issues)

---

Train neural networks in your browser, deploy to real robots.