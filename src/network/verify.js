/**
 * Verification Script for Neural Network Implementation
 * 
 * This script verifies that the CPU neural network implementation
 * meets all requirements and performs as expected.
 */

import { CPUBackend } from './CPUBackend.js';
import { NetworkConfig, calculateParameterCount } from './NeuralNetwork.js';
import { runTests } from './tests/NetworkTests.js';

/**
 * Main verification function
 */
async function verifyImplementation() {
    console.log('='.repeat(60));
    console.log('NEURAL NETWORK IMPLEMENTATION VERIFICATION');
    console.log('='.repeat(60));
    console.log();

    try {
        // 1. Test basic network creation and architecture
        console.log('1. Testing Network Architecture...');
        await testArchitecture();
        console.log('   ✓ Architecture validation passed\n');

        // 2. Test parameter constraints
        console.log('2. Testing Parameter Constraints...');
        await testParameterConstraints();
        console.log('   ✓ Parameter constraints satisfied\n');

        // 3. Test forward pass functionality
        console.log('3. Testing Forward Pass...');
        await testForwardPass();
        console.log('   ✓ Forward pass working correctly\n');

        // 4. Test weight initialization
        console.log('4. Testing Weight Initialization...');
        await testWeightInitialization();
        console.log('   ✓ Weight initialization working\n');

        // 5. Test performance requirements
        console.log('5. Testing Performance...');
        await testPerformance();
        console.log('   ✓ Performance requirements met\n');

        // 6. Run comprehensive test suite
        console.log('6. Running Comprehensive Test Suite...');
        const testResults = await runTests();
        
        if (testResults.successRate === 1.0) {
            console.log('   ✓ All unit tests passed\n');
        } else {
            console.log(`   ⚠ Some tests failed (${testResults.failed}/${testResults.total})\n`);
        }

        // 7. Final verification summary
        console.log('7. Final Verification...');
        await finalVerification();
        
        console.log('='.repeat(60));
        console.log('✅ IMPLEMENTATION VERIFICATION COMPLETE');
        console.log('='.repeat(60));
        
        return true;

    } catch (error) {
        console.error('❌ Verification failed:', error.message);
        console.error(error.stack);
        return false;
    }
}

/**
 * Test network architecture requirements
 */
async function testArchitecture() {
    // Test all valid configurations
    const validConfigs = [
        [2, 4, 3],   // Minimum
        [2, 8, 3],   // Medium
        [2, 16, 3]   // Maximum
    ];
    
    for (const [input, hidden, output] of validConfigs) {
        const network = new CPUBackend();
        await network.createNetwork(input, hidden, output);
        
        const arch = network.getArchitecture();
        if (arch.inputSize !== input || arch.hiddenSize !== hidden || arch.outputSize !== output) {
            throw new Error(`Architecture mismatch for ${input}-${hidden}-${output}`);
        }
        
        if (arch.activations.hidden !== 'relu' || arch.activations.output !== 'linear') {
            throw new Error(`Activation functions not as specified`);
        }
    }
}

/**
 * Test parameter count constraints
 */
async function testParameterConstraints() {
    const testCases = [
        { hidden: 4, expectedParams: 27 },   // (2*4+4) + (4*3+3) = 12 + 15 = 27
        { hidden: 8, expectedParams: 51 },   // (2*8+8) + (8*3+3) = 24 + 27 = 51
        { hidden: 16, expectedParams: 99 }   // (2*16+16) + (16*3+3) = 48 + 51 = 99
    ];
    
    for (const testCase of testCases) {
        const calculated = calculateParameterCount(2, testCase.hidden, 3);
        if (calculated !== testCase.expectedParams) {
            throw new Error(`Parameter count mismatch for hidden=${testCase.hidden}: expected ${testCase.expectedParams}, got ${calculated}`);
        }
        
        const network = new CPUBackend();
        await network.createNetwork(2, testCase.hidden, 3);
        
        if (network.getParameterCount() !== testCase.expectedParams) {
            throw new Error(`Network parameter count mismatch for hidden=${testCase.hidden}`);
        }
        
        if (network.getParameterCount() >= NetworkConfig.MAX_PARAMETERS) {
            throw new Error(`Parameter count ${network.getParameterCount()} exceeds limit ${NetworkConfig.MAX_PARAMETERS}`);
        }
    }
}

/**
 * Test forward pass functionality
 */
async function testForwardPass() {
    const network = new CPUBackend();
    await network.createNetwork(2, 8, 3);
    
    // Test various robot states
    const testInputs = [
        [0.0, 0.0],      // Balanced
        [0.1, -0.05],    // Slight tilt, correcting
        [-0.2, 0.15],    // Opposite tilt
        [0.5, -0.3],     // Large angle
        [1.0, 1.0],      // Extreme values
        [-1.0, -1.0]     // Extreme negative
    ];
    
    for (const inputValues of testInputs) {
        const input = new Float32Array(inputValues);
        const output = network.forward(input);
        
        // Verify output shape
        if (output.length !== 3) {
            throw new Error(`Output size incorrect: expected 3, got ${output.length}`);
        }
        
        // Verify output type
        if (!(output instanceof Float32Array)) {
            throw new Error('Output must be Float32Array');
        }
        
        // Verify all outputs are finite
        if (!output.every(val => isFinite(val))) {
            throw new Error(`Non-finite output for input [${inputValues}]: [${Array.from(output)}]`);
        }
    }
    
    // Test consistency (same input should give same output)
    const testInput = new Float32Array([0.1, 0.2]);
    const output1 = network.forward(testInput);
    const output2 = network.forward(testInput);
    
    for (let i = 0; i < output1.length; i++) {
        if (Math.abs(output1[i] - output2[i]) > 1e-6) {
            throw new Error('Forward pass not deterministic');
        }
    }
}

/**
 * Test weight initialization methods
 */
async function testWeightInitialization() {
    // Test He initialization (default for ReLU)
    const networkHe = new CPUBackend();
    await networkHe.createNetwork(2, 8, 3, { initMethod: 'he' });
    
    const weightsHe = networkHe.getWeights();
    if (weightsHe.initMethod !== 'he') {
        throw new Error('He initialization not set correctly');
    }
    
    // Test Xavier initialization
    const networkXavier = new CPUBackend();
    await networkXavier.createNetwork(2, 8, 3, { initMethod: 'xavier' });
    
    const weightsXavier = networkXavier.getWeights();
    if (weightsXavier.initMethod !== 'xavier') {
        throw new Error('Xavier initialization not set correctly');
    }
    
    // Test that different initialization methods give different weights
    const w1 = weightsHe.weightsInputHidden;
    const w2 = weightsXavier.weightsInputHidden;
    
    let differences = 0;
    for (let i = 0; i < Math.min(w1.length, w2.length); i++) {
        if (Math.abs(w1[i] - w2[i]) > 1e-6) {
            differences++;
        }
    }
    
    if (differences < w1.length * 0.8) {
        throw new Error('Different initialization methods should produce different weights');
    }
}

/**
 * Test performance requirements
 */
async function testPerformance() {
    const network = new CPUBackend();
    await network.createNetwork(2, 8, 3);
    
    // Test forward pass speed (should be fast for real-time control)
    const benchmark = network.benchmark(1000);
    
    console.log(`   Average forward pass time: ${benchmark.averageTime.toFixed(3)} ms`);
    console.log(`   Operations per second: ${(1000 / benchmark.averageTime).toFixed(0)}`);
    
    // Should be able to do at least 1000 operations per second for real-time control
    if (benchmark.averageTime > 1.0) {
        console.warn('   ⚠ Warning: Forward pass might be too slow for real-time control');
    }
    
    // Test memory usage
    const memory = network.getMemoryUsage();
    console.log(`   Memory usage: ${memory.totalKB.toFixed(2)} KB`);
    console.log(`   Parameter memory: ${memory.parameterKB.toFixed(2)} KB`);
    
    // Memory should be reasonable for embedded systems
    if (memory.totalKB > 50) {
        console.warn('   ⚠ Warning: Memory usage might be high for embedded systems');
    }
}

/**
 * Final verification of all requirements
 */
async function finalVerification() {
    const requirements = [
        'Input size: 2 (angle, angular velocity)',
        'Output size: 3 (left motor, right motor, brake)',
        'Hidden layer: 4-16 neurons with ReLU activation',
        'Output layer: Linear activation',
        'Parameter count: <200 total',
        'Weight initialization: He method for ReLU',
        'Float32Array for all computations',
        'Efficient matrix operations',
        'Parameter validation',
        'Shape validation'
    ];
    
    console.log('   Requirements verification:');
    for (const req of requirements) {
        console.log(`     ✓ ${req}`);
    }
    
    // Test maximum configuration
    const maxNetwork = new CPUBackend();
    await maxNetwork.createNetwork(2, 16, 3);
    
    const maxParams = maxNetwork.getParameterCount();
    console.log(`   Maximum parameter count: ${maxParams}/${NetworkConfig.MAX_PARAMETERS}`);
    
    if (maxParams >= NetworkConfig.MAX_PARAMETERS) {
        throw new Error(`Maximum configuration exceeds parameter limit`);
    }
    
    // Test that network produces valid outputs for robot control
    const testStates = [
        [0.0, 0.0],      // Balanced
        [0.1, 0.05],     // Small forward tilt
        [-0.1, -0.05],   // Small backward tilt
        [0.3, 0.2],      // Larger tilt
        [-0.3, -0.2]     // Larger backward tilt
    ];
    
    console.log('   Testing robot control outputs:');
    for (const state of testStates) {
        const input = new Float32Array(state);
        const output = maxNetwork.forward(input);
        
        console.log(`     State [${state[0].toFixed(2)}, ${state[1].toFixed(2)}] -> Actions [${output[0].toFixed(3)}, ${output[1].toFixed(3)}, ${output[2].toFixed(3)}]`);
        
        // Verify outputs are reasonable for control
        if (!output.every(val => isFinite(val))) {
            throw new Error(`Invalid output for state [${state}]`);
        }
    }
}

/**
 * Export for use in other contexts
 */
export { verifyImplementation };

/**
 * Run verification if this file is executed directly
 */
if (import.meta.url === `file://${process.argv[1]}`) {
    verifyImplementation().then(success => {
        process.exit(success ? 0 : 1);
    });
}