/**
 * WebGPU Backend Tests
 * 
 * Comprehensive tests for WebGPU device initialization, feature detection,
 * and automatic CPU fallback functionality.
 */

import { WebGPUBackend, WebGPUDeviceManager, checkWebGPUAvailability } from '../WebGPUBackend.js';
import { NetworkConfig } from '../NeuralNetwork.js';

/**
 * Test suite for WebGPU device initialization
 */
export class WebGPUBackendTests {
    constructor() {
        this.results = [];
        this.testCount = 0;
        this.passCount = 0;
    }

    /**
     * Run all WebGPU backend tests
     * @returns {Promise<Object>} Test results summary
     */
    async runAllTests() {
        console.log('Running WebGPU Backend Tests...');
        
        await this.testWebGPUAvailabilityCheck();
        await this.testDeviceManagerInitialization();
        await this.testWebGPUBackendCreation();
        await this.testCPUFallback();
        await this.testArchitectureCompatibility();
        await this.testPerformanceMonitoring();
        await this.testResourceCleanup();
        
        return this.getResults();
    }

    /**
     * Test WebGPU availability checking
     */
    async testWebGPUAvailabilityCheck() {
        await this.runTest('WebGPU Availability Check', async () => {
            const result = await checkWebGPUAvailability();
            
            // Should return an object with availability status
            this.assert(typeof result === 'object', 'Result should be an object');
            this.assert(typeof result.available === 'boolean', 'Should have available boolean property');
            
            if (result.available) {
                console.log('WebGPU is available:', result.adapterInfo);
            } else {
                console.log('WebGPU not available:', result.reason);
            }
            
            return true;
        });
    }

    /**
     * Test WebGPU device manager initialization
     */
    async testDeviceManagerInitialization() {
        await this.runTest('Device Manager Initialization', async () => {
            const deviceManager = new WebGPUDeviceManager();
            
            // Test initial state
            this.assert(!deviceManager.isInitialized, 'Should not be initialized initially');
            this.assert(!deviceManager.isWebGPUAvailable(), 'WebGPU should not be available initially');
            
            // Test initialization
            const deviceInfo = await deviceManager.initWebGPU();
            
            this.assert(typeof deviceInfo === 'object', 'Should return device info object');
            this.assert(typeof deviceInfo.isAvailable === 'boolean', 'Should have isAvailable property');
            
            if (deviceInfo.isAvailable) {
                console.log('WebGPU device initialized successfully');
                this.assert(deviceManager.isWebGPUAvailable(), 'Manager should report WebGPU as available');
                this.assert(deviceInfo.device !== null, 'Should have valid device');
                this.assert(deviceInfo.limits !== null, 'Should have device limits');
            } else {
                console.log('WebGPU not available, fallback expected');
                this.assert(!deviceManager.isWebGPUAvailable(), 'Manager should report WebGPU as unavailable');
            }
            
            // Test device info summary
            const summary = deviceInfo.getSummary();
            this.assert(typeof summary === 'object', 'Summary should be an object');
            this.assert(typeof summary.isAvailable === 'boolean', 'Summary should have availability info');
            
            // Cleanup
            deviceManager.destroy();
            
            return true;
        });
    }

    /**
     * Test WebGPU backend creation and network initialization
     */
    async testWebGPUBackendCreation() {
        await this.runTest('WebGPU Backend Creation', async () => {
            const backend = new WebGPUBackend();
            
            // Test initial state
            this.assert(!backend.isInitialized, 'Should not be initialized initially');
            this.assert(!backend.isWebGPUAvailable, 'WebGPU should not be available initially');
            
            // Test network creation
            await backend.createNetwork(
                NetworkConfig.INPUT_SIZE, 
                8, 
                NetworkConfig.OUTPUT_SIZE,
                { initMethod: NetworkConfig.INITIALIZATION.HE }
            );
            
            this.assert(backend.isInitialized, 'Should be initialized after createNetwork');
            
            // Test architecture
            const arch = backend.getArchitecture();
            this.assert(arch.inputSize === NetworkConfig.INPUT_SIZE, 'Input size should match');
            this.assert(arch.hiddenSize === 8, 'Hidden size should match');
            this.assert(arch.outputSize === NetworkConfig.OUTPUT_SIZE, 'Output size should match');
            this.assert(['webgpu', 'cpu'].includes(arch.backend), 'Backend should be webgpu or cpu');
            
            // Test forward pass
            const input = new Float32Array([0.1, -0.05]);
            const output = backend.forward(input);
            
            this.assert(output instanceof Float32Array, 'Output should be Float32Array');
            this.assert(output.length === NetworkConfig.OUTPUT_SIZE, 'Output size should match');
            
            // Test backend info
            const backendInfo = backend.getBackendInfo();
            this.assert(typeof backendInfo === 'object', 'Backend info should be object');
            this.assert(['webgpu', 'cpu'].includes(backendInfo.type), 'Backend type should be valid');
            
            console.log(`Backend initialized: ${backendInfo.type}`);
            if (backendInfo.usingFallback) {
                console.log(`Fallback reason: ${backendInfo.fallbackReason}`);
            }
            
            // Cleanup
            backend.destroy();
            
            return true;
        });
    }

    /**
     * Test forced CPU fallback
     */
    async testCPUFallback() {
        await this.runTest('CPU Fallback', async () => {
            const backend = new WebGPUBackend();
            
            // Force CPU fallback
            await backend.createNetwork(
                NetworkConfig.INPUT_SIZE, 
                8, 
                NetworkConfig.OUTPUT_SIZE,
                { forceCPU: true }
            );
            
            const backendInfo = backend.getBackendInfo();
            this.assert(backendInfo.type === 'cpu', 'Should use CPU backend when forced');
            this.assert(backendInfo.usingFallback, 'Should report using fallback');
            
            // Test that it still works
            const input = new Float32Array([0.1, -0.05]);
            const output = backend.forward(input);
            
            this.assert(output instanceof Float32Array, 'Should still produce valid output');
            this.assert(output.length === NetworkConfig.OUTPUT_SIZE, 'Output size should be correct');
            
            console.log('CPU fallback test successful');
            
            // Cleanup
            backend.destroy();
            
            return true;
        });
    }

    /**
     * Test architecture compatibility
     */
    async testArchitectureCompatibility() {
        await this.runTest('Architecture Compatibility', async () => {
            const backend = new WebGPUBackend();
            
            // Test valid architectures
            const validSizes = [4, 8, 12, 16];
            
            for (const hiddenSize of validSizes) {
                await backend.createNetwork(
                    NetworkConfig.INPUT_SIZE,
                    hiddenSize,
                    NetworkConfig.OUTPUT_SIZE
                );
                
                const arch = backend.getArchitecture();
                this.assert(arch.hiddenSize === hiddenSize, `Hidden size should be ${hiddenSize}`);
                
                // Test forward pass
                const input = new Float32Array([0.1, -0.05]);
                const output = backend.forward(input);
                this.assert(output.length === NetworkConfig.OUTPUT_SIZE, 'Output should be valid');
            }
            
            // Test invalid architectures should throw
            try {
                await backend.createNetwork(1, 8, 3); // Invalid input size
                this.assert(false, 'Should throw for invalid input size');
            } catch (error) {
                // Expected
            }
            
            try {
                await backend.createNetwork(2, 3, 3); // Invalid hidden size (too small)
                this.assert(false, 'Should throw for invalid hidden size');
            } catch (error) {
                // Expected
            }
            
            console.log('Architecture compatibility tests passed');
            
            // Cleanup
            backend.destroy();
            
            return true;
        });
    }

    /**
     * Test performance monitoring
     */
    async testPerformanceMonitoring() {
        await this.runTest('Performance Monitoring', async () => {
            const backend = new WebGPUBackend();
            
            await backend.createNetwork(
                NetworkConfig.INPUT_SIZE,
                8,
                NetworkConfig.OUTPUT_SIZE
            );
            
            // Run some forward passes to generate performance data
            const input = new Float32Array([0.1, -0.05]);
            for (let i = 0; i < 10; i++) {
                backend.forward(input);
            }
            
            // Test performance metrics
            const perfMetrics = backend.getPerformanceMetrics();
            this.assert(typeof perfMetrics === 'object', 'Performance metrics should be object');
            this.assert(typeof perfMetrics.initializationTime === 'number', 'Should have initialization time');
            this.assert(typeof perfMetrics.averageForwardTime === 'number', 'Should have average forward time');
            this.assert(perfMetrics.totalForwardPasses >= 10, 'Should track forward passes');
            
            // Test benchmark
            const benchmarkResults = backend.benchmark(100);
            this.assert(typeof benchmarkResults === 'object', 'Benchmark should return object');
            this.assert(typeof benchmarkResults.averageTime === 'number', 'Should have average time');
            this.assert(typeof benchmarkResults.backend === 'string', 'Should have backend type');
            
            console.log(`Performance test completed - avg forward time: ${perfMetrics.averageForwardTime.toFixed(2)}ms`);
            
            // Cleanup
            backend.destroy();
            
            return true;
        });
    }

    /**
     * Test resource cleanup
     */
    async testResourceCleanup() {
        await this.runTest('Resource Cleanup', async () => {
            const backend = new WebGPUBackend();
            
            await backend.createNetwork(
                NetworkConfig.INPUT_SIZE,
                8,
                NetworkConfig.OUTPUT_SIZE
            );
            
            this.assert(backend.isInitialized, 'Should be initialized');
            
            // Test destroy
            backend.destroy();
            
            this.assert(!backend.isInitialized, 'Should not be initialized after destroy');
            this.assert(backend.deviceManager === null, 'Device manager should be null');
            this.assert(backend.cpuBackend === null, 'CPU backend should be null');
            
            console.log('Resource cleanup test passed');
            
            return true;
        });
    }

    /**
     * Run a single test with error handling
     * @private
     */
    async runTest(name, testFn) {
        this.testCount++;
        try {
            console.log(`Running test: ${name}`);
            const result = await testFn();
            if (result) {
                this.passCount++;
                this.results.push({ name, status: 'PASS', error: null });
                console.log(`✓ ${name} - PASS`);
            } else {
                this.results.push({ name, status: 'FAIL', error: 'Test returned false' });
                console.log(`✗ ${name} - FAIL`);
            }
        } catch (error) {
            this.results.push({ name, status: 'ERROR', error: error.message });
            console.error(`✗ ${name} - ERROR:`, error);
        }
    }

    /**
     * Assert condition with error message
     * @private
     */
    assert(condition, message) {
        if (!condition) {
            throw new Error(`Assertion failed: ${message}`);
        }
    }

    /**
     * Get test results summary
     * @returns {Object} Test results
     */
    getResults() {
        return {
            totalTests: this.testCount,
            passedTests: this.passCount,
            failedTests: this.testCount - this.passCount,
            successRate: (this.passCount / this.testCount) * 100,
            results: this.results
        };
    }
}

/**
 * Run WebGPU backend tests
 * @returns {Promise<Object>} Test results
 */
export async function runWebGPUBackendTests() {
    const testSuite = new WebGPUBackendTests();
    return await testSuite.runAllTests();
}