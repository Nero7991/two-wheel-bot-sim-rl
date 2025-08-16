/**
 * Neural Network Usage Example
 * 
 * This example demonstrates how to use the CPU-based neural network
 * for the two-wheel balancing robot RL application.
 */

import { CPUBackend } from './CPUBackend.js';
import { NetworkConfig } from './NeuralNetwork.js';

/**
 * Basic usage example
 */
async function basicExample() {
    console.log('=== Basic Neural Network Example ===\n');
    
    // Create and initialize network
    const network = new CPUBackend();
    await network.createNetwork(2, 8, 3);  // 2 inputs, 8 hidden, 3 outputs
    
    console.log('Network created successfully!');
    console.log(`Architecture: ${network.getArchitecture().inputSize}-${network.getArchitecture().hiddenSize}-${network.getArchitecture().outputSize}`);
    console.log(`Parameters: ${network.getParameterCount()}`);
    console.log();
    
    // Test with different robot states
    const robotStates = [
        { angle: 0.0, angularVelocity: 0.0, description: "Perfectly balanced" },
        { angle: 0.1, angularVelocity: 0.05, description: "Small forward tilt, tilting more" },
        { angle: 0.1, angularVelocity: -0.05, description: "Small forward tilt, correcting" },
        { angle: -0.15, angularVelocity: 0.1, description: "Backward tilt, correcting fast" },
        { angle: 0.3, angularVelocity: 0.2, description: "Large forward tilt, falling" }
    ];
    
    console.log('Testing different robot states:\n');
    
    for (const state of robotStates) {
        const input = new Float32Array([state.angle, state.angularVelocity]);
        const output = network.forward(input);
        
        console.log(`State: ${state.description}`);
        console.log(`  Input: [angle=${state.angle.toFixed(3)}, angVel=${state.angularVelocity.toFixed(3)}]`);
        console.log(`  Output: [left=${output[0].toFixed(3)}, right=${output[1].toFixed(3)}, brake=${output[2].toFixed(3)}]`);
        console.log();
    }
}

/**
 * Performance benchmark example
 */
async function performanceExample() {
    console.log('=== Performance Benchmark ===\n');
    
    const network = new CPUBackend();
    await network.createNetwork(2, 8, 3);
    
    // Benchmark different iteration counts
    const iterationCounts = [100, 1000, 10000];
    
    for (const iterations of iterationCounts) {
        console.log(`Benchmarking ${iterations} forward passes...`);
        const benchmark = network.benchmark(iterations);
        
        console.log(`  Average time: ${benchmark.averageTime.toFixed(3)} ms`);
        console.log(`  Total time: ${benchmark.totalTime.toFixed(1)} ms`);
        console.log(`  Operations/sec: ${(1000 / benchmark.averageTime).toFixed(0)}`);
        console.log();
    }
    
    // Memory usage
    const memory = network.getMemoryUsage();
    console.log('Memory Usage:');
    console.log(`  Total: ${memory.totalKB.toFixed(2)} KB`);
    console.log(`  Parameters: ${memory.parameterKB.toFixed(2)} KB`);
    console.log(`  Breakdown:`);
    for (const [key, bytes] of Object.entries(memory.breakdown)) {
        console.log(`    ${key}: ${(bytes / 1024).toFixed(3)} KB`);
    }
    console.log();
}

/**
 * Network size comparison example
 */
async function sizeComparisonExample() {
    console.log('=== Network Size Comparison ===\n');
    
    const hiddenSizes = [4, 8, 12, 16];
    
    console.log('Testing different network sizes:');
    console.log('Size\tParams\tMemory(KB)\tAvg Time(ms)\tOps/sec');
    console.log('-'.repeat(60));
    
    for (const hiddenSize of hiddenSizes) {
        const network = new CPUBackend();
        await network.createNetwork(2, hiddenSize, 3);
        
        const params = network.getParameterCount();
        const memory = network.getMemoryUsage();
        const benchmark = network.benchmark(1000);
        const opsPerSec = Math.round(1000 / benchmark.averageTime);
        
        console.log(`${hiddenSize}\t${params}\t${memory.totalKB.toFixed(2)}\t\t${benchmark.averageTime.toFixed(3)}\t\t${opsPerSec}`);
    }
    console.log();
}

/**
 * Weight serialization example
 */
async function serializationExample() {
    console.log('=== Weight Serialization Example ===\n');
    
    // Create original network
    const network1 = new CPUBackend();
    await network1.createNetwork(2, 8, 3);
    
    console.log('Original network created');
    
    // Test with a specific input
    const testInput = new Float32Array([0.1, -0.05]);
    const output1 = network1.forward(testInput);
    
    console.log(`Original output: [${output1.map(x => x.toFixed(3)).join(', ')}]`);
    
    // Serialize weights
    const weights = network1.getWeights();
    console.log('Weights serialized');
    console.log(`  Serialized data size: ${JSON.stringify(weights).length} characters`);
    
    // Create new network and load weights
    const network2 = new CPUBackend();
    network2.setWeights(weights);
    
    console.log('New network created from serialized weights');
    
    // Test with same input
    const output2 = network2.forward(testInput);
    console.log(`Restored output: [${output2.map(x => x.toFixed(3)).join(', ')}]`);
    
    // Verify they're identical
    const identical = output1.every((val, idx) => Math.abs(val - output2[idx]) < 1e-6);
    console.log(`Networks identical: ${identical ? '✓' : '✗'}`);
    console.log();
    
    // Clone example
    const network3 = network1.clone();
    const output3 = network3.forward(testInput);
    const cloneIdentical = output1.every((val, idx) => Math.abs(val - output3[idx]) < 1e-6);
    console.log(`Clone identical: ${cloneIdentical ? '✓' : '✗'}`);
    console.log();
}

/**
 * Real-time simulation example
 */
async function simulationExample() {
    console.log('=== Real-time Simulation Example ===\n');
    
    const network = new CPUBackend();
    await network.createNetwork(2, 8, 3);
    
    console.log('Simulating robot control loop...\n');
    
    // Simulate a falling robot trying to balance
    let angle = 0.2;  // Start with forward tilt
    let angularVelocity = 0.1;  // Falling forward
    
    const dt = 0.02;  // 20ms timestep (50 Hz control)
    const steps = 20;
    
    console.log('Step\tAngle\t\tAngVel\t\tLeft\t\tRight\t\tBrake');
    console.log('-'.repeat(80));
    
    for (let step = 0; step < steps; step++) {
        // Get action from neural network
        const input = new Float32Array([angle, angularVelocity]);
        const output = network.forward(input);
        
        const [leftMotor, rightMotor, brake] = output;
        
        console.log(`${step}\t${angle.toFixed(3)}\t\t${angularVelocity.toFixed(3)}\t\t${leftMotor.toFixed(3)}\t\t${rightMotor.toFixed(3)}\t\t${brake.toFixed(3)}`);
        
        // Simple physics simulation (very basic)
        // In real application, this would be much more sophisticated
        const motorTorque = leftMotor - rightMotor;  // Differential drive
        const gravity = 9.81;
        const length = 0.1;  // Robot height
        
        // Simple inverted pendulum dynamics
        const angularAccel = (gravity / length) * Math.sin(angle) - 0.1 * motorTorque;
        
        // Update state
        angularVelocity += angularAccel * dt;
        angle += angularVelocity * dt;
        
        // Add some damping
        angularVelocity *= 0.99;
        
        // Check if robot fell
        if (Math.abs(angle) > Math.PI / 3) {
            console.log(`\nRobot fell at step ${step}! |angle| = ${Math.abs(angle).toFixed(3)} > π/3`);
            break;
        }
    }
    
    console.log('\nSimulation complete.');
    console.log('Note: This is a random network, so balancing is not expected.');
    console.log('In a trained network, the robot should learn to balance.\n');
}

/**
 * Main function to run all examples
 */
async function runExamples() {
    try {
        await basicExample();
        await performanceExample();
        await sizeComparisonExample();
        await serializationExample();
        await simulationExample();
        
        console.log('=== All Examples Complete ===');
        
    } catch (error) {
        console.error('Error running examples:', error);
    }
}

// Export for use in other modules
export { 
    basicExample, 
    performanceExample, 
    sizeComparisonExample, 
    serializationExample, 
    simulationExample,
    runExamples 
};

// Run examples if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runExamples();
}