# Visualization Module

High-performance 2D visualization system for the two-wheel balancing robot RL application.

## Implemented Components:

### Renderer.js - Core 2D Visualization System
- **CoordinateTransform**: Physics-to-screen coordinate conversion with automatic scaling
- **PerformanceMonitor**: Real-time FPS and frame time tracking
- **Renderer**: Main rendering class with 60 FPS rendering loop

#### Features:
- Real-time robot visualization (pendulum + wheels)
- Motor torque indicators with color-coded direction
- Physics state display (angle, velocity, position)
- Training metrics overlay (episode, step, reward, epsilon)
- Responsive canvas management
- Grid and coordinate system reference
- Performance monitoring and optimization

### example.js - Integration Example
- **VisualizationExample**: Complete demo showing renderer integration
- Physics demo with PD controller
- Training visualization with Q-learning
- Evaluation mode for trained agents
- Mode switching and episode management

### Tests
- **RendererTests.js**: Comprehensive test suite
- **testRenderer.html**: Interactive test page
- Performance benchmarking
- Integration testing

## Implementation Status:
- [x] 2D robot rendering system
- [x] Environment visualization (grid, ground, axes)
- [x] Real-time training metrics display
- [x] Motor torque visualization
- [x] Performance monitoring
- [x] Coordinate system transformation
- [x] Responsive canvas sizing
- [x] Integration with physics simulation
- [x] Integration with Q-learning training
- [ ] Real-time neural network visualization
- [ ] Training metrics charts/graphs
- [ ] Advanced UI components

## Usage

### Basic Renderer Setup
```javascript
import { createRenderer } from './Renderer.js';

const canvas = document.getElementById('canvas');
const renderer = createRenderer(canvas, {
    targetFPS: 60,
    showGrid: true,
    showDebugInfo: true
});

// Start rendering
renderer.start();

// Update with robot state
renderer.updateRobot(robotState, robotConfig, motorTorque);
renderer.updateTraining(trainingMetrics);
```

### Complete Integration
```javascript
import { VisualizationExample } from './example.js';

const demo = new VisualizationExample('canvas-id');
await demo.start('physics'); // or 'training', 'evaluation'
```

## Architecture

### Coordinate System
- Physics: Meters, with (0,0) at robot base, Y-up
- Screen: Pixels, with (0,0) at top-left, Y-down
- Automatic transformation with configurable scaling (default: 100px/meter)

### Rendering Pipeline
1. **Clear**: Background fill
2. **Environment**: Grid, ground line, coordinate axes
3. **Robot**: Pendulum body, wheels, torque indicators, angle arc
4. **UI Overlays**: State info, training metrics, performance data

### Performance Optimization
- Frame rate limiting to target FPS
- Efficient drawing operations
- Minimal canvas state changes
- Performance monitoring and warnings

## Configuration Options

```javascript
const config = {
    // Rendering
    targetFPS: 60,
    backgroundColor: '#0a0a0a',
    
    // Visual elements
    showGrid: true,
    showDebugInfo: true,
    showPerformance: true,
    
    // Colors
    robotColor: '#00d4ff',
    wheelColor: '#ffffff',
    torqueColor: '#ff6b35',
    textColor: '#ffffff',
    gridColor: '#1a1a1a'
};
```

## Testing

### Run Tests
```bash
# Open test page in browser
open src/visualization/tests/testRenderer.html

# Or run programmatically
import { runRendererTests } from './tests/RendererTests.js';
const results = await runRendererTests();
```

### Performance Testing
- Target: 60 FPS at 800x600 resolution
- Typical frame time: 5-15ms
- Performance warnings for >25ms frames
- Automatic FPS monitoring and display

## Future Enhancements
- WebGL rendering for higher performance
- Neural network visualization overlay
- Training progress charts and graphs
- Interactive debugging tools
- Export functionality for animations/videos