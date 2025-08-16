/**
 * Unit Tests for Neural Network Implementation
 * 
 * Comprehensive tests for CPU backend neural network implementation
 * covering forward pass, shape validation, parameter counting, and performance.
 */

import { CPUBackend } from '../CPUBackend.js';
import { NetworkConfig, calculateParameterCount, validateArchitecture } from '../NeuralNetwork.js';
import { heInit, xavierInit, vectorMatrixMultiply, relu } from '../MatrixUtils.js';

/**
 * Test suite for neural network implementation
 */
export class NetworkTests {
    constructor() {
        this.testResults = [];
        this.totalTests = 0;
        this.passedTests = 0;
    }

    /**
     * Run all tests
     * @returns {Object} Test results summary
     */
    async runAllTests() {
        console.log('Starting Neural Network Tests...\n');
        
        // Reset counters
        this.testResults = [];
        this.totalTests = 0;
        this.passedTests = 0;
        
        // Run test suites
        await this.testArchitectureValidation();
        await this.testParameterCounting();
        await this.testWeightInitialization();
        await this.testNetworkCreation();
        await this.testForwardPass();
        await this.testShapeValidation();
        await this.testWeightSerialization();
        await this.testPerformance();
        await this.testMemoryUsage();
        await this.testEdgeCases();
        
        // Print summary
        this.printTestSummary();
        
        return {
            total: this.totalTests,
            passed: this.passedTests,
            failed: this.totalTests - this.passedTests,
            successRate: this.passedTests / this.totalTests,
            results: this.testResults
        };
    }

    /**
     * Test architecture validation
     */
    async testArchitectureValidation() {
        console.log('Testing Architecture Validation...');
        
        // Valid architectures
        await this.runTest('Valid architecture 2-4-3', () => {
            validateArchitecture(2, 4, 3);
            return true;
        });
        
        await this.runTest('Valid architecture 2-16-3', () => {
            validateArchitecture(2, 16, 3);
            return true;
        });
        
        // Invalid input size
        await this.runTest('Invalid input size', () => {
            try {
                validateArchitecture(3, 8, 3);
                return false; // Should throw
            } catch (e) {
                return e.message.includes('Input size must be 2');
            }
        });
        
        // Invalid output size
        await this.runTest('Invalid output size', () => {
            try {
                validateArchitecture(2, 8, 4);
                return false; // Should throw
            } catch (e) {
                return e.message.includes('Output size must be 3');
            }
        });
        
        // Invalid hidden size (too small)
        await this.runTest('Hidden size too small', () => {
            try {
                validateArchitecture(2, 3, 3);
                return false; // Should throw
            } catch (e) {
                return e.message.includes('Hidden size must be between');
            }
        });
        
        // Invalid hidden size (too large)
        await this.runTest('Hidden size too large', () => {
            try {
                validateArchitecture(2, 17, 3);
                return false; // Should throw
            } catch (e) {
                return e.message.includes('Hidden size must be between');
            }
        });
    }

    /**
     * Test parameter counting
     */
    async testParameterCounting() {
        console.log('Testing Parameter Counting...');
        
        await this.runTest('Parameter count 2-4-3', () => {
            const count = calculateParameterCount(2, 4, 3);
            // Input->Hidden: (2*4) + 4 = 12
            // Hidden->Output: (4*3) + 3 = 15
            // Total: 27
            return count === 27;
        });
        
        await this.runTest('Parameter count 2-8-3', () => {
            const count = calculateParameterCount(2, 8, 3);
            // Input->Hidden: (2*8) + 8 = 24
            // Hidden->Output: (8*3) + 3 = 27
            // Total: 51
            return count === 51;
        });
        
        await this.runTest('Parameter count 2-16-3', () => {
            const count = calculateParameterCount(2, 16, 3);
            // Input->Hidden: (2*16) + 16 = 48
            // Hidden->Output: (16*3) + 3 = 51
            // Total: 99
            return count === 99;
        });
        
        await this.runTest('Parameter limit validation', () => {
            // Test that maximum hidden size (16) stays under 200 parameters
            const maxCount = calculateParameterCount(2, 16, 3);
            return maxCount < NetworkConfig.MAX_PARAMETERS;
        });
    }

    /**
     * Test weight initialization methods
     */
    async testWeightInitialization() {
        console.log('Testing Weight Initialization...');
        
        await this.runTest('He initialization shape', () => {
            const weights = heInit(2, 8);
            return weights.length === 16 && weights instanceof Float32Array;
        });
        
        await this.runTest('Xavier initialization shape', () => {
            const weights = xavierInit(2, 8);
            return weights.length === 16 && weights instanceof Float32Array;
        });
        
        await this.runTest('He initialization statistics', () => {
            const weights = heInit(100, 50);
            const mean = weights.reduce((sum, w) => sum + w, 0) / weights.length;
            const variance = weights.reduce((sum, w) => sum + (w - mean) ** 2, 0) / weights.length;
            
            // Check if mean is close to 0 and variance is reasonable for He init
            return Math.abs(mean) < 0.1 && variance > 0.01 && variance < 0.1;
        });
    }

    /**
     * Test network creation
     */
    async testNetworkCreation() {
        console.log('Testing Network Creation...');
        
        await this.runTest('Create basic network', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            return network.isInitialized && network.getParameterCount() === 51;
        });
        
        await this.runTest('Create minimum size network', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 4, 3);
            return network.isInitialized && network.getParameterCount() === 27;
        });
        
        await this.runTest('Create maximum size network', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 16, 3);
            return network.isInitialized && network.getParameterCount() === 99;
        });
        
        await this.runTest('Network architecture info', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            const arch = network.getArchitecture();
            
            return arch.inputSize === 2 && 
                   arch.hiddenSize === 8 && 
                   arch.outputSize === 3 &&
                   arch.backend === 'cpu' &&
                   arch.activations.hidden === 'relu' &&
                   arch.activations.output === 'linear';
        });
    }

    /**
     * Test forward pass functionality
     */
    async testForwardPass() {
        console.log('Testing Forward Pass...');
        
        await this.runTest('Basic forward pass', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const input = new Float32Array([0.1, -0.05]);
            const output = network.forward(input);
            
            return output instanceof Float32Array && output.length === 3;
        });
        
        await this.runTest('Forward pass output range', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const input = new Float32Array([0.5, -0.3]);
            const output = network.forward(input);
            
            // Check that outputs are finite numbers
            return output.every(val => isFinite(val));
        });
        
        await this.runTest('Multiple forward passes consistency', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const input = new Float32Array([0.2, -0.1]);
            const output1 = network.forward(input);
            const output2 = network.forward(input);
            
            // Same input should give same output
            return output1.every((val, idx) => Math.abs(val - output2[idx]) < 1e-6);
        });
        
        await this.runTest('Different inputs give different outputs', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const input1 = new Float32Array([0.1, 0.2]);
            const input2 = new Float32Array([0.3, -0.1]);
            
            const output1 = network.forward(input1);
            const output2 = network.forward(input2);
            
            // Different inputs should give different outputs
            return !output1.every((val, idx) => Math.abs(val - output2[idx]) < 1e-6);
        });
    }

    /**
     * Test input/output shape validation
     */
    async testShapeValidation() {
        console.log('Testing Shape Validation...');
        
        await this.runTest('Valid input shape', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const input = new Float32Array([0.1, 0.2]);
            const output = network.forward(input);
            
            return output.length === 3;
        });
        
        await this.runTest('Invalid input size', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            try {
                const input = new Float32Array([0.1]); // Wrong size
                network.forward(input);
                return false; // Should throw
            } catch (e) {
                return e.message.includes('Input size mismatch');
            }
        });
        
        await this.runTest('Invalid input type', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            try {
                const input = [0.1, 0.2]; // Wrong type
                network.forward(input);
                return false; // Should throw
            } catch (e) {
                return e.message.includes('Input must be a Float32Array');
            }
        });
        
        await this.runTest('Forward pass before initialization', async () => {
            const network = new CPUBackend();
            
            try {
                const input = new Float32Array([0.1, 0.2]);
                network.forward(input);
                return false; // Should throw
            } catch (e) {
                return e.message.includes('Network not initialized');
            }
        });
    }

    /**
     * Test weight serialization and loading
     */
    async testWeightSerialization() {
        console.log('Testing Weight Serialization...');
        
        await this.runTest('Weight serialization', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const weights = network.getWeights();
            
            return weights.architecture && 
                   weights.weightsInputHidden && 
                   weights.biasHidden &&
                   weights.weightsHiddenOutput &&
                   weights.biasOutput;
        });
        
        await this.runTest('Weight loading', async () => {
            const network1 = new CPUBackend();
            await network1.createNetwork(2, 8, 3);
            
            const weights = network1.getWeights();
            
            const network2 = new CPUBackend();
            network2.setWeights(weights);
            
            // Both networks should give same output
            const input = new Float32Array([0.1, 0.2]);
            const output1 = network1.forward(input);
            const output2 = network2.forward(input);
            
            return output1.every((val, idx) => Math.abs(val - output2[idx]) < 1e-6);
        });
        
        await this.runTest('Clone network', async () => {
            const network1 = new CPUBackend();
            await network1.createNetwork(2, 8, 3);
            
            const network2 = network1.clone();
            
            const input = new Float32Array([0.3, -0.1]);
            const output1 = network1.forward(input);
            const output2 = network2.forward(input);
            
            return output1.every((val, idx) => Math.abs(val - output2[idx]) < 1e-6);
        });
    }

    /**
     * Test performance benchmarking
     */
    async testPerformance() {
        console.log('Testing Performance...');
        
        await this.runTest('Performance benchmark', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const benchmark = network.benchmark(100);
            
            return benchmark.averageTime > 0 && 
                   benchmark.totalTime > 0 && 
                   benchmark.iterations === 100 &&
                   benchmark.lastResult instanceof Float32Array;
        });
        
        await this.runTest('Forward pass speed', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const benchmark = network.benchmark(1000);
            
            // Should be able to do forward pass in less than 1ms on average
            return benchmark.averageTime < 1.0;
        });
    }

    /**
     * Test memory usage tracking
     */
    async testMemoryUsage() {
        console.log('Testing Memory Usage...');
        
        await this.runTest('Memory usage calculation', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const memory = network.getMemoryUsage();
            
            return memory.totalBytes > 0 && 
                   memory.totalKB > 0 &&
                   memory.parameterBytes > 0 &&
                   memory.breakdown;
        });
        
        await this.runTest('Parameter memory matches count', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const memory = network.getMemoryUsage();
            const expectedBytes = network.getParameterCount() * 4; // 4 bytes per Float32
            
            return memory.parameterBytes === expectedBytes;
        });
    }

    /**
     * Test edge cases and error handling
     */
    async testEdgeCases() {
        console.log('Testing Edge Cases...');
        
        await this.runTest('Zero input values', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const input = new Float32Array([0.0, 0.0]);
            const output = network.forward(input);
            
            return output.every(val => isFinite(val));
        });
        
        await this.runTest('Large input values', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const input = new Float32Array([100.0, -100.0]);
            const output = network.forward(input);
            
            return output.every(val => isFinite(val));
        });
        
        await this.runTest('Reset weights', async () => {
            const network = new CPUBackend();
            await network.createNetwork(2, 8, 3);
            
            const weights1 = network.getWeights();
            network.resetWeights();
            const weights2 = network.getWeights();
            
            // Weights should be different after reset
            const w1 = weights1.weightsInputHidden;
            const w2 = weights2.weightsInputHidden;
            
            return !w1.every((val, idx) => Math.abs(val - w2[idx]) < 1e-6);
        });
    }

    /**
     * Run a single test
     * @param {string} name - Test name
     * @param {Function} testFn - Test function
     */
    async runTest(name, testFn) {
        this.totalTests++;
        
        try {
            const result = await testFn();
            
            if (result) {
                this.passedTests++;
                console.log(`  ✓ ${name}`);
                this.testResults.push({ name, passed: true, error: null });
            } else {
                console.log(`  ✗ ${name} - Test failed`);
                this.testResults.push({ name, passed: false, error: null });
            }
        } catch (error) {
            console.log(`  ✗ ${name} - Error: ${error.message}`);
            this.testResults.push({ name, passed: false, error: error.message });
        }
    }

    /**
     * Print test summary
     */
    printTestSummary() {
        console.log(`\n=== Test Summary ===`);
        console.log(`Total Tests: ${this.totalTests}`);
        console.log(`Passed: ${this.passedTests}`);
        console.log(`Failed: ${this.totalTests - this.passedTests}`);
        console.log(`Success Rate: ${((this.passedTests / this.totalTests) * 100).toFixed(1)}%`);
        
        if (this.passedTests === this.totalTests) {
            console.log(`🎉 All tests passed!`);
        } else {
            console.log(`❌ Some tests failed. Check output above for details.`);
        }
    }
}

/**
 * Utility function to run tests from browser console or other contexts
 */
export async function runTests() {
    const tester = new NetworkTests();
    return await tester.runAllTests();
}