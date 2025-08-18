/**
 * Test script to debug Q-value calculation issues
 */

import { CPUBackend } from './src/network/CPUBackend.js';
import { NetworkConfig } from './src/network/NeuralNetwork.js';

async function testQValues() {
    console.log('=== Q-Value Debug Test ===\n');
    
    // Create a simple CPU network
    const network = new CPUBackend();
    await network.createNetwork(
        NetworkConfig.INPUT_SIZE,  // 2 inputs
        8,                         // 8 hidden neurons
        3,                         // 3 outputs (actions)
        { initMethod: NetworkConfig.INITIALIZATION.HE }
    );
    
    // Get initial weights
    const initialWeights = network.getWeights();
    console.log('Initial network architecture:', initialWeights.architecture);
    
    // Test with various inputs
    const testInputs = [
        new Float32Array([0.0, 0.0]),      // Balanced
        new Float32Array([0.1, 0.0]),      // Slight tilt
        new Float32Array([-0.1, 0.0]),     // Opposite tilt
        new Float32Array([0.0, 0.1]),      // Angular velocity
        new Float32Array([0.5, -0.5]),     // Complex state
    ];
    
    console.log('\n--- Initial Q-Values ---');
    for (let i = 0; i < testInputs.length; i++) {
        const qValues = network.forward(testInputs[i]);
        console.log(`Input [${testInputs[i][0].toFixed(2)}, ${testInputs[i][1].toFixed(2)}]:`, 
                    `Q-values: [${Array.from(qValues).map(v => v.toFixed(4)).join(', ')}]`);
    }
    
    // Check weight magnitudes
    console.log('\n--- Weight Statistics ---');
    const weightsIH = initialWeights.weightsInputHidden;
    const weightsHO = initialWeights.weightsHiddenOutput;
    
    const ihStats = getArrayStats(weightsIH);
    const hoStats = getArrayStats(weightsHO);
    
    console.log('Input->Hidden weights:', ihStats);
    console.log('Hidden->Output weights:', hoStats);
    
    // Manually set some weights to test
    console.log('\n--- Testing with Manual Weights ---');
    const testWeights = {
        ...initialWeights,
        weightsInputHidden: new Float32Array(16).fill(0.5),  // 2*8 = 16
        biasHidden: new Float32Array(8).fill(0.1),
        weightsHiddenOutput: new Float32Array(24).fill(0.3), // 8*3 = 24
        biasOutput: new Float32Array(3).fill(0.2)
    };
    
    network.setWeights(testWeights);
    
    for (let i = 0; i < testInputs.length; i++) {
        const qValues = network.forward(testInputs[i]);
        console.log(`Input [${testInputs[i][0].toFixed(2)}, ${testInputs[i][1].toFixed(2)}]:`, 
                    `Q-values: [${Array.from(qValues).map(v => v.toFixed(4)).join(', ')}]`);
    }
    
    // Test gradient updates
    console.log('\n--- Testing Gradient Updates ---');
    
    // Simulate a simple TD error update
    const state = new Float32Array([0.1, -0.05]);
    const actionIndex = 1; // Middle action
    const tdError = 0.5; // Positive TD error
    const learningRate = 0.01;
    
    // Get current Q-value
    const beforeQValues = network.forward(state);
    console.log('Before update - Q-values:', Array.from(beforeQValues).map(v => v.toFixed(4)));
    
    // Manually update weights (simplified version of what Q-learning should do)
    const currentWeights = network.getWeights();
    
    // Get hidden activations
    const hiddenLinear = new Float32Array(8);
    for (let h = 0; h < 8; h++) {
        let sum = currentWeights.biasHidden[h];
        for (let i = 0; i < 2; i++) {
            sum += state[i] * currentWeights.weightsInputHidden[i * 8 + h];
        }
        hiddenLinear[h] = Math.max(0, sum); // ReLU
    }
    
    // Update output weights for the selected action
    const newWeightsHO = Array.from(currentWeights.weightsHiddenOutput);
    for (let h = 0; h < 8; h++) {
        const idx = h * 3 + actionIndex;
        newWeightsHO[idx] += learningRate * tdError * hiddenLinear[h];
    }
    
    // Update output bias
    const newBiasOutput = Array.from(currentWeights.biasOutput);
    newBiasOutput[actionIndex] += learningRate * tdError;
    
    // Set updated weights
    network.setWeights({
        ...currentWeights,
        weightsHiddenOutput: newWeightsHO,
        biasOutput: newBiasOutput
    });
    
    // Check Q-values after update
    const afterQValues = network.forward(state);
    console.log('After update - Q-values:', Array.from(afterQValues).map(v => v.toFixed(4)));
    console.log('Change in Q-values:', Array.from(afterQValues).map((v, i) => (v - beforeQValues[i]).toFixed(4)));
}

function getArrayStats(arr) {
    const values = Array.from(arr);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const absMax = Math.max(...values.map(Math.abs));
    const zeros = values.filter(v => v === 0).length;
    
    return {
        min: min.toFixed(4),
        max: max.toFixed(4),
        mean: mean.toFixed(4),
        absMax: absMax.toFixed(4),
        zeros: zeros,
        total: values.length
    };
}

// Run the test
testQValues().catch(console.error);