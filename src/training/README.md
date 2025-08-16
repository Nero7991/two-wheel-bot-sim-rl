# Training Module

This module contains the reinforcement learning training logic for the two-wheel balancing robot.

## Core Components

### QLearning.js - Deep Q-Network Implementation
Complete DQN (Deep Q-Network) implementation with:
- **Neural Network Function Approximation**: Uses CPUBackend for Q-value estimation
- **Epsilon-Greedy Exploration**: Configurable exploration with decay
- **Experience Replay**: Replay buffer for stable training
- **Target Network**: Separate target network for stable Q-learning
- **Hyperparameter Management**: Comprehensive parameter validation and defaults
- **Training Episode Management**: Complete episode loop with metrics
- **Convergence Detection**: Automatic detection of training convergence
- **Model Save/Load**: Serialization for model persistence

### Key Classes

#### QLearning
Main Q-learning algorithm implementation:
```javascript
const qlearning = new QLearning({
    learningRate: 0.001,
    epsilon: 0.1,
    gamma: 0.95,
    batchSize: 32,
    hiddenSize: 8
});

await qlearning.initialize();
const metrics = await qlearning.runTraining(robot);
```

#### Hyperparameters
Manages and validates all training hyperparameters:
- Learning rate: 0.001 (default)
- Epsilon: 0.1 (default, with decay)
- Discount factor (gamma): 0.95 (default)
- Batch size: 32 (default)
- Target update frequency: 100 steps (default)

#### ReplayBuffer
Experience replay buffer for storing and sampling training experiences:
- Configurable buffer size (10,000 default)
- Circular buffer implementation
- Random sampling for training batches
- Efficient memory management

#### TrainingMetrics
Comprehensive training statistics and monitoring:
- Episode rewards and lengths
- Training losses over time
- Convergence detection
- Performance metrics
- Training time tracking

## Architecture

### Network Structure
- **Input**: 2 neurons (robot angle, angular velocity)
- **Hidden**: 4-16 neurons (configurable, ReLU activation)
- **Output**: 3 neurons (left motor, brake, right motor)
- **Total Parameters**: <200 for embedded deployment

### Action Space
Discrete action space with 3 actions:
- **Action 0**: Left motor (-1.0 torque)
- **Action 1**: Brake (0.0 torque)
- **Action 2**: Right motor (+1.0 torque)

### State Space
Continuous state space (normalized):
- **Angle**: Robot tilt angle in radians (normalized to [-1, 1])
- **Angular Velocity**: Angular velocity in rad/s (normalized to [-1, 1])

## Training Process

### Episode Loop
1. Reset environment to initial state
2. For each timestep:
   - Select action using epsilon-greedy policy
   - Execute action in physics simulation
   - Observe reward and next state
   - Store experience in replay buffer
   - Train network on batch of experiences
3. Update target network periodically
4. Decay epsilon for exploration reduction
5. Check convergence criteria

### Q-Learning Update
Target Q-value calculation:
```
Q_target = reward + gamma * max(Q_next) (if not terminal)
Q_target = reward (if terminal)
```

Loss function: Mean Squared Error between current Q-value and target

## Usage Examples

### Basic Training
```javascript
import { createDefaultQLearning } from './QLearning.js';
import { createTrainingRobot } from '../physics/BalancingRobot.js';

const qlearning = createDefaultQLearning();
await qlearning.initialize();

const robot = createTrainingRobot();
const metrics = await qlearning.runTraining(robot, {
    verbose: true,
    onEpisodeEnd: (result, stats) => {
        console.log(`Episode ${result.episode}: Reward=${result.reward}`);
    }
});
```

### Custom Hyperparameters
```javascript
const qlearning = new QLearning({
    learningRate: 0.01,
    epsilon: 0.3,
    epsilonDecay: 0.99,
    hiddenSize: 12,
    maxEpisodes: 500
});
```

### Agent Evaluation
```javascript
// Evaluate trained agent (no exploration)
const results = qlearning.evaluate(robot, 10);
console.log(`Average reward: ${results.averageReward}`);
```

### Model Persistence
```javascript
// Save trained model
const modelData = qlearning.save();

// Load model into new agent
const newAgent = new QLearning();
await newAgent.load(modelData);
```

## Testing

### Unit Tests
Comprehensive test suite in `tests/QLearningTests.js`:
- Hyperparameter validation
- Replay buffer functionality
- Q-learning algorithm correctness
- Integration with physics simulation
- Model save/load functionality
- Error handling and edge cases

Run tests:
```bash
# Open in browser
open src/training/tests/testRunner.html

# Or run individual test components
node --experimental-modules tests/QLearningTests.js
```

### Example Demonstrations
Complete examples in `example.js`:
- Basic training workflow
- Custom hyperparameter configuration
- Real-time training monitoring
- Agent evaluation and analysis
- Model saving and loading
- Configuration comparison
- Q-value inspection

## Performance Characteristics

### Training Speed
- **Fast Configuration**: ~10-50 episodes for basic balancing
- **Optimal Configuration**: ~100-500 episodes for robust performance
- **Real-time**: Capable of real-time training at 50Hz physics simulation

### Memory Usage
- Network parameters: <200 total
- Replay buffer: ~10MB for 10,000 experiences
- Training overhead: <50MB total memory usage

### Convergence
- **Simple environments**: 50-200 episodes
- **Complex environments**: 200-1000 episodes
- **Convergence detection**: Automatic based on reward thresholds

## Integration

### Physics Simulation
Seamless integration with `BalancingRobot.js`:
- Standardized state representation
- Reward function integration
- Episode management
- Stability checking

### Neural Network Backend
Uses `CPUBackend.js` for neural network operations:
- Optimized CPU-based inference
- Efficient weight updates
- Memory-optimized operations
- Export capabilities for embedded deployment

## Future Enhancements

### Planned Improvements
- [ ] Double DQN implementation
- [ ] Dueling network architecture
- [ ] Prioritized experience replay
- [ ] Multi-step returns
- [ ] Distributional Q-learning
- [ ] Continuous action space support

### Advanced Features
- [ ] Curriculum learning
- [ ] Transfer learning between robot configurations
- [ ] Real-time hyperparameter tuning
- [ ] Advanced exploration strategies
- [ ] Multi-agent training support

## Implementation Status:
- [x] ✅ Deep Q-Network implementation
- [x] ✅ Experience replay system
- [x] ✅ Reward function integration
- [x] ✅ Training loop with episodes
- [x] ✅ Hyperparameter management
- [x] ✅ Training metrics and logging
- [x] ✅ Convergence detection
- [x] ✅ Model save/load functionality
- [x] ✅ Comprehensive unit tests
- [x] ✅ Integration with physics simulation
- [x] ✅ Performance optimization
- [x] ✅ Documentation and examples