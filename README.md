# Two-Wheel Balancing Robot RL Simulator

🤖 **Live Demo**: [https://botsim.orenslab.com](https://botsim.orenslab.com)

A web-based reinforcement learning environment for training neural networks to balance a two-wheeled robot using Deep Q-Network (DQN) algorithms. Features WebGPU acceleration, real-time visualization, and C++ export for embedded deployment.

![Two-Wheel Bot Simulator](https://img.shields.io/badge/Status-Live-brightgreen) ![WebGPU](https://img.shields.io/badge/WebGPU-Accelerated-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

### 🚀 **Core Capabilities**
- **DQN Training**: PyTorch-standard Deep Q-Network implementation with proven hyperparameters
- **WebGPU Acceleration**: GPU-accelerated training with 5-10x speedup over CPU
- **Real-time Visualization**: Interactive 2D physics simulation with performance metrics
- **Manual Control**: Debug interface with speed control for testing reward functions
- **Multiple Network Architectures**: From micro (4 neurons) to DQN standard (128 neurons)

### 🧠 **Machine Learning**
- **Deep Q-Network (DQN)** with target networks and experience replay
- **Huber Loss** for robust Q-value regression
- **Linear Epsilon Decay** (PyTorch standard: 0.9 → 0.01 over 2500 steps)
- **Gradient Clipping** to prevent exploding gradients
- **Configurable Hyperparameters**: Learning rate, batch size, network architecture

### ⚡ **Performance**
- **Training Speed Control**: 0.1x to 100x with quick preset buttons (1x, 2x, 10x, 100x)
- **Real-time Training**: 100+ episodes/minute on CPU, 1000+ on WebGPU
- **Embedded Export**: Generate optimized C++ code for Arduino/microcontrollers
- **Memory Efficient**: Networks under 1KB when quantized for embedded deployment

### 🎮 **User Interface**
- **Interactive Controls**: Real-time parameter adjustment during training
- **Debug Interface**: Manual control with speed slider (0.1x-2x) for reward function testing
- **Performance Metrics**: Live charts for rewards, loss, Q-values, and epsilon decay
- **Network Presets**: Pre-configured architectures for different use cases

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

### 1. **Quick Training**
1. Open the simulator
2. Select "DQN Standard" network preset (recommended)
3. Click "Start Training"
4. Use speed controls (1x, 10x, 100x) to accelerate training
5. Watch the robot learn to balance in real-time!

### 2. **Manual Testing**
1. Expand "Debug Controls" section
2. Enable "User Control"
3. Use arrow keys to manually control the robot
4. Adjust debug speed slider for slow-motion analysis
5. Observe real-time reward values during manual control

### 3. **Advanced Configuration**
- **Network Architecture**: Choose from Micro (4 neurons) to DQN Standard (128 neurons)
- **Hyperparameters**: Adjust learning rate, epsilon, batch size
- **Physics Parameters**: Modify robot mass, height, motor strength
- **Training Speed**: Fine-tune with slider or use quick presets

## Architecture

### Network Presets
| Preset | Hidden Neurons | Parameters | Use Case |
|--------|---------------|------------|----------|
| Micro Bot | 4 | <50 | Ultra-lightweight embedded |
| Nano Bot | 6 | <80 | Small microcontrollers |
| Classic Bot | 8 | <150 | Standard embedded (ESP32) |
| **DQN Standard** | **128** | **~771** | **Web training (recommended)** |
| Enhanced Bot | 12→8 | <400 | Multi-layer embedded |
| Advanced Bot | 16→12→8 | <800 | Complex control tasks |

### DQN Implementation
- **Target Networks**: Separate target network updated every 100 steps
- **Experience Replay**: 10K sample buffer for decorrelated training
- **Epsilon-Greedy**: Linear decay from 0.9 to 0.01 over 2500 steps
- **Batch Training**: 128 samples per batch (PyTorch standard)
- **Huber Loss**: More robust than MSE for Q-value regression

### Physics Model
- **Inverted Pendulum**: Realistic two-wheel balancing dynamics
- **State Space**: Robot angle and angular velocity (2D input)
- **Action Space**: Motor torque [-1.0, 0.0, 1.0] (3 discrete actions)
- **Reward Function**: Simple upright reward (1.0 when vertical, 0.0 at failure)
- **Fixed Timestep**: 20ms simulation steps (50Hz)

## Training Performance

### Proven Results
The DQN Standard configuration has been validated with significant performance improvements:

- **Baseline (Classic)**: 12.83 average reward
- **DQN Standard**: 78.51 average reward (**+65.68 improvement!**)
- **Convergence**: Clear learning progression within 50 episodes
- **Q-values**: Stable growth from 1.8 to 12.8

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
- **Memory**: <384KB total (weights + code)
- **Compute**: Optimized for 48MHz ARM Cortex-M
- **Real-time**: 50Hz control loop capability
- **Quantization**: Int8 weights for minimal memory usage

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
- **Frontend**: Vanilla JavaScript with Vite build system
- **ML**: Custom DQN implementation (WebGPU + CPU backends)
- **Physics**: Custom inverted pendulum simulation
- **Visualization**: HTML5 Canvas 2D rendering
- **Export**: Template-based C++ code generation

### Build Commands
```bash
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview production build
npm run lint         # Code linting
npm run test         # Run tests
```

## Contributing

We welcome contributions! Areas of interest:

- **Performance**: WebGPU shader optimizations
- **Algorithms**: Additional RL algorithms (PPO, A3C, SAC)
- **Physics**: More complex robot models
- **Export**: Support for other embedded platforms
- **UI**: Enhanced visualization and debugging tools

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Research & References

This implementation is based on proven DQN research and follows PyTorch tutorial standards:

- **Deep Q-Networks**: Mnih et al. (2015) - Human-level control through deep reinforcement learning
- **PyTorch DQN Tutorial**: [Official PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- **Inverted Pendulum Control**: Classical control theory and modern RL approaches
- **Embedded ML**: Optimization techniques for resource-constrained deployment

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Live Demo**: [https://botsim.orenslab.com](https://botsim.orenslab.com)
- **Repository**: [https://github.com/Nero7991/two-wheel-bot-sim-rl](https://github.com/Nero7991/two-wheel-bot-sim-rl)
- **Issues**: [GitHub Issues](https://github.com/Nero7991/two-wheel-bot-sim-rl/issues)

---

**Built with ❤️ for the robotics and ML community**

*Train neural networks in your browser, deploy to real robots!*