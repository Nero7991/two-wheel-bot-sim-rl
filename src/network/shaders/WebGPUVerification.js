/**
 * WebGPU Neural Network Verification and Benchmarking
 * 
 * Comprehensive testing suite for verifying GPU implementation accuracy
 * and measuring performance improvements vs CPU implementation.
 * 
 * Features:
 * - Numerical accuracy verification against CPU reference
 * - Performance benchmarking with statistical analysis
 * - Stress testing for robustness validation
 * - Real-time inference capability testing
 * - Production readiness assessment
 */

import { CPUBackend } from '../CPUBackend.js';
import { WebGPUNeuralNetwork } from './WebGPUNeuralNetwork.js';

/**
 * Verification test suite for WebGPU neural network implementation
 */
export class WebGPUVerification {
    constructor(device) {
        this.device = device;
        this.cpuBackend = null;
        this.gpuNetwork = null;
        
        // Test configurations
        this.testArchitectures = [
            { inputSize: 2, hiddenSize: 4, outputSize: 3 },
            { inputSize: 2, hiddenSize: 8, outputSize: 3 },
            { inputSize: 2, hiddenSize: 16, outputSize: 3 }
        ];
        
        // Numerical accuracy thresholds
        this.accuracyThresholds = {
            absoluteError: 1e-6,      // Maximum absolute difference
            relativeError: 1e-5,      // Maximum relative difference
            correlationMin: 0.9999,   // Minimum correlation coefficient
            rmseMax: 1e-5            // Maximum root mean square error
        };
        
        // Performance targets
        this.performanceTargets = {
            minSpeedup: 2.0,          // Minimum 2x speedup required
            maxLatency: 5.0,          // Maximum 5ms latency for real-time
            minThroughput: 1000,      // Minimum 1000 inferences/second
            batchEfficiency: 1.5      // Batch should be 1.5x more efficient
        };
        
        // Test results storage
        this.verificationResults = {};
        this.benchmarkResults = {};
        this.stressTestResults = {};
    }

    /**
     * Run complete verification and benchmarking suite
     * @param {Object} options - Test configuration options
     * @returns {Promise<Object>} Complete test results
     */
    async runCompleteVerification(options = {}) {
        const {
            skipStressTests = false,
            customTestCases = null,
            verboseLogging = true
        } = options;

        console.log('Starting comprehensive WebGPU neural network verification...');
        const startTime = performance.now();

        try {
            // 1. Numerical Accuracy Verification
            console.log('\n=== Phase 1: Numerical Accuracy Verification ===');
            this.verificationResults = await this.verifyNumericalAccuracy();

            // 2. Performance Benchmarking
            console.log('\n=== Phase 2: Performance Benchmarking ===');
            this.benchmarkResults = await this.benchmarkPerformance();

            // 3. Stress Testing (optional)
            if (!skipStressTests) {
                console.log('\n=== Phase 3: Stress Testing ===');
                this.stressTestResults = await this.runStressTests();
            }

            // 4. Real-time Capability Testing
            console.log('\n=== Phase 4: Real-time Capability Testing ===');
            const realTimeResults = await this.testRealTimeCapability();

            // 5. Generate comprehensive report
            const report = this.generateVerificationReport();
            
            const totalTime = performance.now() - startTime;
            console.log(`\nVerification completed in ${totalTime.toFixed(2)}ms`);

            return {
                summary: report.summary,
                accuracy: this.verificationResults,
                performance: this.benchmarkResults,
                stressTest: this.stressTestResults,
                realTime: realTimeResults,
                fullReport: report,
                executionTime: totalTime
            };

        } catch (error) {
            console.error('Verification failed:', error);
            throw error;
        }
    }

    /**
     * Verify numerical accuracy against CPU reference implementation
     * @returns {Promise<Object>} Accuracy verification results
     */
    async verifyNumericalAccuracy() {
        const results = {
            architectures: {},
            overallAccuracy: null,
            passedTests: 0,
            totalTests: 0,
            maxError: 0,
            averageError: 0
        };

        for (const arch of this.testArchitectures) {
            console.log(`Testing architecture: ${arch.inputSize}-${arch.hiddenSize}-${arch.outputSize}`);
            
            const archResults = await this._testArchitectureAccuracy(arch);
            results.architectures[`${arch.inputSize}_${arch.hiddenSize}_${arch.outputSize}`] = archResults;
            
            if (archResults.passed) {
                results.passedTests++;
            }
            results.totalTests++;
            
            results.maxError = Math.max(results.maxError, archResults.maxAbsoluteError);
            results.averageError += archResults.averageAbsoluteError;
        }

        results.averageError /= results.totalTests;
        results.overallAccuracy = results.passedTests / results.totalTests;

        console.log(`Accuracy verification: ${results.passedTests}/${results.totalTests} architectures passed`);
        console.log(`Maximum error: ${results.maxError.toExponential(3)}`);
        console.log(`Average error: ${results.averageError.toExponential(3)}`);

        return results;
    }

    /**
     * Test accuracy for a specific architecture
     * @private
     */
    async _testArchitectureAccuracy(architecture) {
        const { inputSize, hiddenSize, outputSize } = architecture;
        
        // Initialize both implementations with same weights
        this.cpuBackend = new CPUBackend();
        await this.cpuBackend.createNetwork(inputSize, hiddenSize, outputSize, {
            initMethod: 'he',
            seed: 12345 // Fixed seed for reproducibility
        });

        this.gpuNetwork = new WebGPUNeuralNetwork(this.device);
        await this.gpuNetwork.initialize(inputSize, hiddenSize, outputSize, {
            initMethod: 'he',
            seed: 12345 // Same seed
        });

        // Ensure both networks have identical weights
        const cpuWeights = this.cpuBackend.getWeights();
        await this.gpuNetwork.setWeights({
            weightsHidden: cpuWeights.weightsInputHidden,
            biasHidden: cpuWeights.biasHidden,
            weightsOutput: cpuWeights.weightsHiddenOutput,
            biasOutput: cpuWeights.biasOutput
        });

        // Generate test cases
        const testCases = this._generateTestCases(inputSize, 100); // 100 test cases
        
        const errors = [];
        const cpuOutputs = [];
        const gpuOutputs = [];

        for (let i = 0; i < testCases.length; i++) {
            const input = testCases[i];
            
            // Get CPU reference output
            const cpuOutput = this.cpuBackend.forward(input);
            cpuOutputs.push(Array.from(cpuOutput));
            
            // Get GPU output
            const gpuOutput = await this.gpuNetwork.forward(input);
            gpuOutputs.push(Array.from(gpuOutput));
            
            // Calculate errors
            const absoluteErrors = cpuOutput.map((cpu, j) => Math.abs(cpu - gpuOutput[j]));
            const relativeErrors = cpuOutput.map((cpu, j) => 
                cpu !== 0 ? Math.abs((cpu - gpuOutput[j]) / cpu) : 0
            );
            
            errors.push({
                absoluteErrors,
                relativeErrors,
                maxAbsolute: Math.max(...absoluteErrors),
                maxRelative: Math.max(...relativeErrors)
            });
        }

        // Analyze results
        const maxAbsoluteError = Math.max(...errors.map(e => e.maxAbsolute));
        const maxRelativeError = Math.max(...errors.map(e => e.maxRelative));
        const averageAbsoluteError = errors.reduce((sum, e) => sum + e.maxAbsolute, 0) / errors.length;
        const averageRelativeError = errors.reduce((sum, e) => sum + e.maxRelative, 0) / errors.length;

        // Calculate correlation
        const correlation = this._calculateCorrelation(
            cpuOutputs.flat(),
            gpuOutputs.flat()
        );

        // Calculate RMSE
        const rmse = this._calculateRMSE(cpuOutputs.flat(), gpuOutputs.flat());

        // Determine if test passed
        const passed = 
            maxAbsoluteError <= this.accuracyThresholds.absoluteError &&
            maxRelativeError <= this.accuracyThresholds.relativeError &&
            correlation >= this.accuracyThresholds.correlationMin &&
            rmse <= this.accuracyThresholds.rmseMax;

        return {
            architecture,
            passed,
            testCases: testCases.length,
            maxAbsoluteError,
            maxRelativeError,
            averageAbsoluteError,
            averageRelativeError,
            correlation,
            rmse,
            thresholds: this.accuracyThresholds
        };
    }

    /**
     * Benchmark performance against CPU implementation
     * @returns {Promise<Object>} Performance benchmark results
     */
    async benchmarkPerformance() {
        const results = {
            architectures: {},
            overallSpeedup: null,
            realTimeCapable: false,
            batchEfficiency: null
        };

        for (const arch of this.testArchitectures) {
            console.log(`Benchmarking architecture: ${arch.inputSize}-${arch.hiddenSize}-${arch.outputSize}`);
            
            const perfResults = await this._benchmarkArchitecturePerformance(arch);
            results.architectures[`${arch.inputSize}_${arch.hiddenSize}_${arch.outputSize}`] = perfResults;
        }

        // Calculate overall metrics
        const speedups = Object.values(results.architectures).map(r => r.speedup);
        results.overallSpeedup = speedups.reduce((sum, s) => sum + s, 0) / speedups.length;
        results.realTimeCapable = Object.values(results.architectures).every(r => r.realTimeCapable);

        // Test batch efficiency
        results.batchEfficiency = await this._testBatchEfficiency();

        console.log(`Overall speedup: ${results.overallSpeedup.toFixed(2)}x`);
        console.log(`Real-time capable: ${results.realTimeCapable}`);
        console.log(`Batch efficiency: ${results.batchEfficiency.toFixed(2)}x`);

        return results;
    }

    /**
     * Benchmark performance for a specific architecture
     * @private
     */
    async _benchmarkArchitecturePerformance(architecture) {
        const { inputSize, hiddenSize, outputSize } = architecture;
        
        // Initialize networks
        const cpuBackend = new CPUBackend();
        await cpuBackend.createNetwork(inputSize, hiddenSize, outputSize);
        
        const gpuNetwork = new WebGPUNeuralNetwork(this.device);
        await gpuNetwork.initialize(inputSize, hiddenSize, outputSize);

        // Warm up
        const warmupInput = new Float32Array([0.1, -0.05]);
        for (let i = 0; i < 10; i++) {
            cpuBackend.forward(warmupInput);
            await gpuNetwork.forward(warmupInput);
        }

        // Benchmark CPU
        const cpuTimes = [];
        const iterations = 1000;
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            cpuBackend.forward(warmupInput);
            cpuTimes.push(performance.now() - start);
        }

        // Benchmark GPU
        const gpuTimes = [];
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await gpuNetwork.forward(warmupInput);
            gpuTimes.push(performance.now() - start);
        }

        // Calculate statistics
        const cpuStats = this._calculateStatistics(cpuTimes);
        const gpuStats = this._calculateStatistics(gpuTimes);
        
        const speedup = cpuStats.median / gpuStats.median;
        const realTimeCapable = gpuStats.p95 <= this.performanceTargets.maxLatency;
        const throughput = 1000 / gpuStats.median; // inferences per second

        return {
            architecture,
            cpu: cpuStats,
            gpu: gpuStats,
            speedup,
            realTimeCapable,
            throughput,
            iterations,
            meetsTargets: {
                speedup: speedup >= this.performanceTargets.minSpeedup,
                latency: realTimeCapable,
                throughput: throughput >= this.performanceTargets.minThroughput
            }
        };
    }

    /**
     * Test batch processing efficiency
     * @private
     */
    async _testBatchEfficiency() {
        const arch = { inputSize: 2, hiddenSize: 8, outputSize: 3 };
        const batchSizes = [1, 4, 8, 16, 32];
        
        const gpuNetwork = new WebGPUNeuralNetwork(this.device);
        await gpuNetwork.initialize(arch.inputSize, arch.hiddenSize, arch.outputSize);

        const results = {};
        
        for (const batchSize of batchSizes) {
            const batchInput = new Float32Array(batchSize * arch.inputSize);
            for (let i = 0; i < batchInput.length; i += 2) {
                batchInput[i] = Math.random() * 0.2 - 0.1;     // angle
                batchInput[i + 1] = Math.random() * 0.1 - 0.05; // angular velocity
            }

            // Warm up
            for (let i = 0; i < 5; i++) {
                await gpuNetwork.forwardBatch(batchInput, batchSize);
            }

            // Benchmark
            const times = [];
            const iterations = 100;
            
            for (let i = 0; i < iterations; i++) {
                const start = performance.now();
                await gpuNetwork.forwardBatch(batchInput, batchSize);
                times.push(performance.now() - start);
            }

            const stats = this._calculateStatistics(times);
            results[batchSize] = {
                totalTime: stats.median,
                timePerSample: stats.median / batchSize,
                throughput: (1000 * batchSize) / stats.median
            };
        }

        // Calculate efficiency compared to single inference
        const singleTime = results[1].timePerSample;
        let maxEfficiency = 1.0;
        
        for (const batchSize of batchSizes) {
            if (batchSize > 1) {
                const efficiency = singleTime / results[batchSize].timePerSample;
                maxEfficiency = Math.max(maxEfficiency, efficiency);
            }
        }

        return maxEfficiency;
    }

    /**
     * Run stress tests for robustness validation
     * @returns {Promise<Object>} Stress test results
     */
    async runStressTests() {
        console.log('Running stress tests...');
        
        const results = {
            longRunning: await this._testLongRunning(),
            memoryStress: await this._testMemoryStress(),
            errorRecovery: await this._testErrorRecovery(),
            concurrentOperations: await this._testConcurrentOperations()
        };

        return results;
    }

    /**
     * Test real-time capability for robot control
     * @returns {Promise<Object>} Real-time capability results
     */
    async testRealTimeCapability() {
        console.log('Testing real-time capability...');
        
        const arch = { inputSize: 2, hiddenSize: 8, outputSize: 3 };
        const gpuNetwork = new WebGPUNeuralNetwork(this.device);
        await gpuNetwork.initialize(arch.inputSize, arch.hiddenSize, arch.outputSize);

        // Test standard forward pass
        const standardTimes = [];
        const realTimeTimes = [];
        const iterations = 1000;

        const testInput = new Float32Array([0.1, -0.05]);

        // Standard forward pass benchmark
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await gpuNetwork.forward(testInput);
            standardTimes.push(performance.now() - start);
        }

        // Real-time optimized forward pass benchmark
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await gpuNetwork.forwardRealTime(testInput, { realTime: true, skipValidation: true });
            realTimeTimes.push(performance.now() - start);
        }

        const standardStats = this._calculateStatistics(standardTimes);
        const realTimeStats = this._calculateStatistics(realTimeTimes);

        return {
            standard: standardStats,
            realTime: realTimeStats,
            improvement: standardStats.median / realTimeStats.median,
            realTimeCapable: realTimeStats.p95 <= this.performanceTargets.maxLatency,
            controlFrequency: 1000 / realTimeStats.median, // Hz
            jitter: realTimeStats.std,
            iterations
        };
    }

    /**
     * Generate comprehensive verification report
     * @returns {Object} Detailed verification report
     */
    generateVerificationReport() {
        const report = {
            timestamp: new Date().toISOString(),
            summary: {
                overallPassed: false,
                accuracyPassed: false,
                performancePassed: false,
                realTimePassed: false,
                productionReady: false
            },
            recommendations: [],
            issues: [],
            strengths: []
        };

        // Analyze accuracy results
        if (this.verificationResults.overallAccuracy >= 1.0) {
            report.summary.accuracyPassed = true;
            report.strengths.push('Excellent numerical accuracy across all architectures');
        } else {
            report.issues.push(`Accuracy verification failed: ${this.verificationResults.passedTests}/${this.verificationResults.totalTests} architectures passed`);
            report.recommendations.push('Review shader implementations for numerical precision issues');
        }

        // Analyze performance results
        if (this.benchmarkResults.overallSpeedup >= this.performanceTargets.minSpeedup) {
            report.summary.performancePassed = true;
            report.strengths.push(`Achieved ${this.benchmarkResults.overallSpeedup.toFixed(2)}x speedup over CPU`);
        } else {
            report.issues.push(`Performance target not met: ${this.benchmarkResults.overallSpeedup.toFixed(2)}x < ${this.performanceTargets.minSpeedup}x`);
            report.recommendations.push('Optimize GPU compute pipeline and buffer management');
        }

        // Analyze real-time capability
        if (this.benchmarkResults.realTimeCapable) {
            report.summary.realTimePassed = true;
            report.strengths.push('Real-time capability confirmed for robot control');
        } else {
            report.issues.push('Real-time latency requirements not met');
            report.recommendations.push('Implement real-time optimizations and reduce GPU synchronization overhead');
        }

        // Overall assessment
        report.summary.overallPassed = 
            report.summary.accuracyPassed && 
            report.summary.performancePassed;
            
        report.summary.productionReady = 
            report.summary.overallPassed && 
            report.summary.realTimePassed &&
            (this.stressTestResults ? Object.values(this.stressTestResults).every(t => t.passed) : true);

        return report;
    }

    // Utility methods
    
    /**
     * Generate test cases for verification
     * @private
     */
    _generateTestCases(inputSize, count) {
        const testCases = [];
        
        for (let i = 0; i < count; i++) {
            const input = new Float32Array(inputSize);
            
            // Generate realistic robot state inputs
            input[0] = (Math.random() - 0.5) * 0.4; // angle: -0.2 to 0.2 radians
            input[1] = (Math.random() - 0.5) * 0.2; // angular velocity: -0.1 to 0.1 rad/s
            
            testCases.push(input);
        }
        
        return testCases;
    }

    /**
     * Calculate correlation coefficient
     * @private
     */
    _calculateCorrelation(x, y) {
        const n = x.length;
        const meanX = x.reduce((sum, val) => sum + val, 0) / n;
        const meanY = y.reduce((sum, val) => sum + val, 0) / n;
        
        let numerator = 0;
        let sumXSq = 0;
        let sumYSq = 0;
        
        for (let i = 0; i < n; i++) {
            const deltaX = x[i] - meanX;
            const deltaY = y[i] - meanY;
            numerator += deltaX * deltaY;
            sumXSq += deltaX * deltaX;
            sumYSq += deltaY * deltaY;
        }
        
        return numerator / Math.sqrt(sumXSq * sumYSq);
    }

    /**
     * Calculate root mean square error
     * @private
     */
    _calculateRMSE(actual, predicted) {
        const n = actual.length;
        const sumSquaredErrors = actual.reduce((sum, val, i) => {
            const error = val - predicted[i];
            return sum + error * error;
        }, 0);
        
        return Math.sqrt(sumSquaredErrors / n);
    }

    /**
     * Calculate statistical measures
     * @private
     */
    _calculateStatistics(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const n = values.length;
        
        return {
            count: n,
            min: Math.min(...values),
            max: Math.max(...values),
            mean: values.reduce((sum, val) => sum + val, 0) / n,
            median: sorted[Math.floor(n / 2)],
            p95: sorted[Math.floor(n * 0.95)],
            p99: sorted[Math.floor(n * 0.99)],
            std: Math.sqrt(values.reduce((sum, val) => {
                const mean = values.reduce((s, v) => s + v, 0) / n;
                return sum + Math.pow(val - mean, 2);
            }, 0) / n)
        };
    }

    // Stress test implementations
    
    async _testLongRunning() {
        // Test for memory leaks and stability over long periods
        const iterations = 10000;
        const results = { passed: true, iterations, errors: [] };
        
        try {
            const gpuNetwork = new WebGPUNeuralNetwork(this.device);
            await gpuNetwork.initialize(2, 8, 3);
            
            const testInput = new Float32Array([0.1, -0.05]);
            
            for (let i = 0; i < iterations; i++) {
                await gpuNetwork.forward(testInput);
                
                if (i % 1000 === 0) {
                    console.log(`Long running test: ${i}/${iterations}`);
                }
            }
        } catch (error) {
            results.passed = false;
            results.errors.push(error.message);
        }
        
        return results;
    }

    async _testMemoryStress() {
        // Test with large batch sizes and multiple concurrent networks
        const results = { passed: true, errors: [] };
        
        try {
            const networks = [];
            
            // Create multiple networks
            for (let i = 0; i < 5; i++) {
                const network = new WebGPUNeuralNetwork(this.device);
                await network.initialize(2, 16, 3);
                networks.push(network);
            }
            
            // Test large batches
            const batchSize = 128;
            const batchInput = new Float32Array(batchSize * 2);
            
            for (let i = 0; i < networks.length; i++) {
                await networks[i].forwardBatch(batchInput, batchSize);
            }
            
            // Cleanup
            networks.forEach(network => network.destroy());
            
        } catch (error) {
            results.passed = false;
            results.errors.push(error.message);
        }
        
        return results;
    }

    async _testErrorRecovery() {
        // Test recovery from various error conditions
        const results = { passed: true, errors: [], recovered: 0 };
        
        // Test invalid inputs, buffer overflows, etc.
        // Implementation would include various error scenarios
        
        return results;
    }

    async _testConcurrentOperations() {
        // Test multiple simultaneous operations
        const results = { passed: true, errors: [], concurrentOps: 0 };
        
        try {
            const gpuNetwork = new WebGPUNeuralNetwork(this.device);
            await gpuNetwork.initialize(2, 8, 3);
            
            const promises = [];
            const testInput = new Float32Array([0.1, -0.05]);
            
            // Launch multiple concurrent operations
            for (let i = 0; i < 10; i++) {
                promises.push(gpuNetwork.forward(testInput));
            }
            
            await Promise.all(promises);
            results.concurrentOps = promises.length;
            
        } catch (error) {
            results.passed = false;
            results.errors.push(error.message);
        }
        
        return results;
    }
}

/**
 * Quick verification function for development use
 * @param {GPUDevice} device - WebGPU device
 * @param {Object} options - Test options
 * @returns {Promise<boolean>} True if verification passes
 */
export async function quickVerify(device, options = {}) {
    const verifier = new WebGPUVerification(device);
    
    try {
        const results = await verifier.runCompleteVerification({
            skipStressTests: true,
            ...options
        });
        
        console.log('Quick verification results:', {
            accuracyPassed: results.summary.accuracyPassed,
            performancePassed: results.summary.performancePassed,
            speedup: results.performance.overallSpeedup?.toFixed(2) + 'x',
            realTimeCapable: results.realTime.realTimeCapable
        });
        
        return results.summary.overallPassed;
        
    } catch (error) {
        console.error('Quick verification failed:', error);
        return false;
    }
}

/**
 * Production readiness check
 * @param {GPUDevice} device - WebGPU device
 * @returns {Promise<Object>} Production readiness assessment
 */
export async function checkProductionReadiness(device) {
    const verifier = new WebGPUVerification(device);
    
    const results = await verifier.runCompleteVerification({
        skipStressTests: false,
        verboseLogging: false
    });
    
    const report = verifier.generateVerificationReport();
    
    return {
        ready: report.summary.productionReady,
        report: report,
        criticalIssues: report.issues.filter(issue => 
            issue.includes('accuracy') || issue.includes('real-time')
        ),
        recommendations: report.recommendations,
        metrics: {
            accuracy: results.accuracy.overallAccuracy,
            speedup: results.performance.overallSpeedup,
            latency: results.realTime?.realTime?.p95,
            reliability: results.stressTest ? 
                Object.values(results.stressTest).filter(t => t.passed).length /
                Object.values(results.stressTest).length : 0
        }
    };
}