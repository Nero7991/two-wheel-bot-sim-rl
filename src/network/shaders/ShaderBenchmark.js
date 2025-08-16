/**
 * Performance Benchmarking for WebGPU Neural Network Shaders
 * 
 * Comprehensive benchmarking suite for measuring and comparing performance
 * of GPU vs CPU neural network operations for the two-wheel robot RL application.
 */

import { ShaderManager } from './ShaderManager.js';
import { BufferManager } from './BufferManager.js';
import { CPUBackend } from '../CPUBackend.js';

/**
 * Shader performance benchmarking utility
 */
export class ShaderBenchmark {
    constructor(device) {
        this.device = device;
        this.shaderManager = null;
        this.bufferManager = null;
        this.cpuBackend = null;
        
        // Benchmark configurations
        this.configurations = [
            { name: 'Tiny (2-4-3)', inputSize: 2, hiddenSize: 4, outputSize: 3 },
            { name: 'Small (2-8-3)', inputSize: 2, hiddenSize: 8, outputSize: 3 },
            { name: 'Medium (2-16-3)', inputSize: 2, hiddenSize: 16, outputSize: 3 }
        ];
        
        this.batchSizes = [1, 4, 8, 16, 32];
        this.iterations = 100;
        
        // Results storage
        this.benchmarkResults = new Map();
    }

    /**
     * Initialize benchmarking environment
     */
    async initialize() {
        console.log('Initializing shader benchmarking environment...');
        
        this.shaderManager = new ShaderManager(this.device);
        this.bufferManager = new BufferManager(this.device);
        this.cpuBackend = new CPUBackend();
        
        // Load and compile shaders
        await this.shaderManager.loadShaders();
        
        console.log('Shader benchmarking environment ready');
    }

    /**
     * Run comprehensive benchmark suite
     * @returns {Promise<Object>} Complete benchmark results
     */
    async runBenchmarkSuite() {
        console.log('Starting comprehensive shader benchmark suite...');
        const startTime = performance.now();
        
        const results = {
            overview: {
                timestamp: new Date().toISOString(),
                device: this.device,
                configurations: this.configurations,
                iterations: this.iterations
            },
            matmulBenchmarks: {},
            activationBenchmarks: {},
            endToEndBenchmarks: {},
            memoryBenchmarks: {},
            comparisonResults: {},
            recommendations: {}
        };

        try {
            // Benchmark matrix multiplication operations
            results.matmulBenchmarks = await this.benchmarkMatrixMultiplication();
            
            // Benchmark activation functions
            results.activationBenchmarks = await this.benchmarkActivationFunctions();
            
            // Benchmark end-to-end forward pass
            results.endToEndBenchmarks = await this.benchmarkEndToEnd();
            
            // Benchmark memory transfer operations
            results.memoryBenchmarks = await this.benchmarkMemoryOperations();
            
            // Compare GPU vs CPU performance
            results.comparisonResults = await this.compareGPUvsCPU();
            
            // Generate performance recommendations
            results.recommendations = this.generateRecommendations(results);
            
        } catch (error) {
            console.error('Benchmark suite failed:', error);
            results.error = error.message;
        }

        const totalTime = performance.now() - startTime;
        results.overview.totalBenchmarkTime = totalTime;
        
        console.log(`Benchmark suite completed in ${totalTime.toFixed(2)}ms`);
        return results;
    }

    /**
     * Benchmark matrix multiplication operations
     */
    async benchmarkMatrixMultiplication() {
        console.log('Benchmarking matrix multiplication operations...');
        const results = {};

        for (const config of this.configurations) {
            const configResults = {};
            
            for (const batchSize of this.batchSizes) {
                console.log(`  Testing ${config.name} with batch size ${batchSize}...`);
                
                const timings = await this._benchmarkMatMul(config, batchSize);
                configResults[`batch_${batchSize}`] = timings;
            }
            
            results[config.name] = configResults;
        }

        return results;
    }

    /**
     * Benchmark a specific matrix multiplication configuration
     * @private
     */
    async _benchmarkMatMul(config, batchSize) {
        const { inputSize, hiddenSize, outputSize } = config;
        const architecture = { inputSize, hiddenSize, outputSize, batchSize };
        
        // Create buffers and bind groups
        const buffers = this.bufferManager.createNetworkBuffers(architecture);
        const layouts = { matmul: this.shaderManager.getBindGroupLayout('matmul') };
        const bindGroups = this.bufferManager.createBindGroups(buffers, layouts);

        // Prepare test data
        const inputData = new Float32Array(batchSize * inputSize).map(() => Math.random());
        const weightData = new Float32Array(inputSize * hiddenSize).map(() => Math.random() * 0.1);
        const biasData = new Float32Array(hiddenSize).map(() => Math.random() * 0.01);

        // Upload data
        await this.bufferManager.uploadData('input', inputData);
        await this.bufferManager.uploadData('weightsHidden', weightData);
        await this.bufferManager.uploadData('biasHidden', biasData);
        this.bufferManager.updateUniformBuffer('matmulParams', { 
            M: batchSize, K: inputSize, N: hiddenSize 
        });

        // Warm-up runs
        for (let i = 0; i < 5; i++) {
            await this._executeMatMul(bindGroups.matmul, batchSize, hiddenSize);
        }

        // Benchmark runs
        const timings = [];
        for (let i = 0; i < this.iterations; i++) {
            const startTime = performance.now();
            await this._executeMatMul(bindGroups.matmul, batchSize, hiddenSize);
            const endTime = performance.now();
            timings.push(endTime - startTime);
        }

        return this._calculateStatistics(timings);
    }

    /**
     * Execute matrix multiplication shader
     * @private
     */
    async _executeMatMul(bindGroup, batchSize, hiddenSize) {
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        
        passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(batchSize * hiddenSize / 64));
        
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Benchmark activation function operations
     */
    async benchmarkActivationFunctions() {
        console.log('Benchmarking activation functions...');
        const results = {};

        const activations = ['relu', 'leaky_relu', 'relu_derivative'];
        
        for (const activation of activations) {
            if (!this.shaderManager.computePipelines.has(activation)) {
                console.log(`  Skipping ${activation} - pipeline not available`);
                continue;
            }

            const activationResults = {};
            
            for (const config of this.configurations) {
                const configResults = {};
                
                for (const batchSize of this.batchSizes) {
                    const timings = await this._benchmarkActivation(activation, config, batchSize);
                    configResults[`batch_${batchSize}`] = timings;
                }
                
                activationResults[config.name] = configResults;
            }
            
            results[activation] = activationResults;
        }

        return results;
    }

    /**
     * Benchmark a specific activation function
     * @private
     */
    async _benchmarkActivation(activationName, config, batchSize) {
        const { hiddenSize } = config;
        const dataSize = batchSize * hiddenSize;
        
        const architecture = { inputSize: 2, hiddenSize, outputSize: 3, batchSize };
        const buffers = this.bufferManager.createNetworkBuffers(architecture);
        const layouts = { activation: this.shaderManager.getBindGroupLayout('activation') };
        const bindGroups = this.bufferManager.createBindGroups(buffers, layouts);

        // Prepare test data
        const inputData = new Float32Array(dataSize).map(() => (Math.random() - 0.5) * 4);
        await this.bufferManager.uploadData('hidden', inputData);
        this.bufferManager.updateUniformBuffer('activationParams', { size: dataSize });

        // Warm-up runs
        for (let i = 0; i < 5; i++) {
            await this._executeActivation(activationName, bindGroups.activation, dataSize);
        }

        // Benchmark runs
        const timings = [];
        for (let i = 0; i < this.iterations; i++) {
            const startTime = performance.now();
            await this._executeActivation(activationName, bindGroups.activation, dataSize);
            const endTime = performance.now();
            timings.push(endTime - startTime);
        }

        return this._calculateStatistics(timings);
    }

    /**
     * Execute activation function shader
     * @private
     */
    async _executeActivation(activationName, bindGroup, dataSize) {
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        
        passEncoder.setPipeline(this.shaderManager.getPipeline(activationName));
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(dataSize / 64));
        
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Benchmark end-to-end forward pass
     */
    async benchmarkEndToEnd() {
        console.log('Benchmarking end-to-end forward pass...');
        const results = {};

        for (const config of this.configurations) {
            const configResults = {};
            
            for (const batchSize of this.batchSizes) {
                console.log(`  Testing ${config.name} end-to-end with batch size ${batchSize}...`);
                
                const timings = await this._benchmarkForwardPass(config, batchSize);
                configResults[`batch_${batchSize}`] = timings;
            }
            
            results[config.name] = configResults;
        }

        return results;
    }

    /**
     * Benchmark complete forward pass
     * @private
     */
    async _benchmarkForwardPass(config, batchSize) {
        const { inputSize, hiddenSize, outputSize } = config;
        const architecture = { inputSize, hiddenSize, outputSize, batchSize };
        
        // Create buffers and bind groups
        const buffers = this.bufferManager.createNetworkBuffers(architecture);
        const layouts = {
            matmul: this.shaderManager.getBindGroupLayout('matmul'),
            activation: this.shaderManager.getBindGroupLayout('activation')
        };
        const bindGroups = this.bufferManager.createBindGroups(buffers, layouts);

        // Prepare network data
        const inputData = new Float32Array(batchSize * inputSize).map(() => Math.random());
        const weightsHidden = new Float32Array(inputSize * hiddenSize).map(() => Math.random() * 0.1);
        const biasHidden = new Float32Array(hiddenSize).map(() => Math.random() * 0.01);
        const weightsOutput = new Float32Array(hiddenSize * outputSize).map(() => Math.random() * 0.1);
        const biasOutput = new Float32Array(outputSize).map(() => Math.random() * 0.01);

        // Upload data
        await this.bufferManager.uploadData('input', inputData);
        await this.bufferManager.uploadData('weightsHidden', weightsHidden);
        await this.bufferManager.uploadData('biasHidden', biasHidden);
        await this.bufferManager.uploadData('weightsOutput', weightsOutput);
        await this.bufferManager.uploadData('biasOutput', biasOutput);

        // Warm-up runs
        for (let i = 0; i < 5; i++) {
            await this._executeForwardPass(config, batchSize, bindGroups);
        }

        // Benchmark runs
        const timings = [];
        for (let i = 0; i < this.iterations; i++) {
            const startTime = performance.now();
            await this._executeForwardPass(config, batchSize, bindGroups);
            const endTime = performance.now();
            timings.push(endTime - startTime);
        }

        return this._calculateStatistics(timings);
    }

    /**
     * Execute complete forward pass
     * @private
     */
    async _executeForwardPass(config, batchSize, bindGroups) {
        const { inputSize, hiddenSize, outputSize } = config;
        
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        
        // Input to hidden layer
        this.bufferManager.updateUniformBuffer('matmulParams', { 
            M: batchSize, K: inputSize, N: hiddenSize 
        });
        passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
        passEncoder.setBindGroup(0, bindGroups.matmul);
        passEncoder.dispatchWorkgroups(Math.ceil(batchSize * hiddenSize / 64));
        
        // ReLU activation
        this.bufferManager.updateUniformBuffer('activationParams', { 
            size: batchSize * hiddenSize 
        });
        passEncoder.setPipeline(this.shaderManager.getPipeline('relu'));
        passEncoder.setBindGroup(0, bindGroups.activation);
        passEncoder.dispatchWorkgroups(Math.ceil(batchSize * hiddenSize / 64));
        
        // Hidden to output layer
        this.bufferManager.updateUniformBuffer('matmulParams', { 
            M: batchSize, K: hiddenSize, N: outputSize 
        });
        passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
        passEncoder.setBindGroup(0, bindGroups.output);
        passEncoder.dispatchWorkgroups(Math.ceil(batchSize * outputSize / 64));
        
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Benchmark memory transfer operations
     */
    async benchmarkMemoryOperations() {
        console.log('Benchmarking memory operations...');
        const results = {};

        const dataSizes = [64, 256, 1024, 4096]; // bytes
        
        for (const size of dataSizes) {
            console.log(`  Testing memory transfers for ${size} bytes...`);
            
            const uploadTimings = [];
            const downloadTimings = [];
            
            // Create test buffer
            const architecture = { inputSize: 2, hiddenSize: 8, outputSize: 3, batchSize: 1 };
            const buffers = this.bufferManager.createNetworkBuffers(architecture);
            
            const testData = new Float32Array(size / 4).map(() => Math.random());
            
            // Benchmark uploads
            for (let i = 0; i < 50; i++) {
                const startTime = performance.now();
                await this.bufferManager.uploadData('input', testData.slice(0, Math.min(testData.length, 2)));
                const endTime = performance.now();
                uploadTimings.push(endTime - startTime);
            }
            
            // Benchmark downloads
            for (let i = 0; i < 50; i++) {
                const startTime = performance.now();
                await this.bufferManager.downloadData('input', Math.min(size, 8));
                const endTime = performance.now();
                downloadTimings.push(endTime - startTime);
            }
            
            results[`${size}_bytes`] = {
                upload: this._calculateStatistics(uploadTimings),
                download: this._calculateStatistics(downloadTimings)
            };
        }

        return results;
    }

    /**
     * Compare GPU vs CPU performance
     */
    async compareGPUvsCPU() {
        console.log('Comparing GPU vs CPU performance...');
        const results = {};

        for (const config of this.configurations) {
            console.log(`  Comparing ${config.name}...`);
            
            const cpuTimings = await this._benchmarkCPUForwardPass(config);
            const gpuTimings = await this._benchmarkForwardPass(config, 1); // Single batch for fair comparison
            
            results[config.name] = {
                cpu: cpuTimings,
                gpu: gpuTimings,
                speedup: cpuTimings.mean / gpuTimings.mean,
                recommendation: cpuTimings.mean > gpuTimings.mean ? 'GPU' : 'CPU'
            };
        }

        return results;
    }

    /**
     * Benchmark CPU forward pass for comparison
     * @private
     */
    async _benchmarkCPUForwardPass(config) {
        const { inputSize, hiddenSize, outputSize } = config;
        
        // Initialize CPU backend
        await this.cpuBackend.createNetwork(inputSize, hiddenSize, outputSize);
        
        const input = new Float32Array([Math.random(), Math.random()]);
        
        // Warm-up runs
        for (let i = 0; i < 5; i++) {
            this.cpuBackend.forward(input);
        }

        // Benchmark runs
        const timings = [];
        for (let i = 0; i < this.iterations; i++) {
            const startTime = performance.now();
            this.cpuBackend.forward(input);
            const endTime = performance.now();
            timings.push(endTime - startTime);
        }

        return this._calculateStatistics(timings);
    }

    /**
     * Calculate timing statistics
     * @private
     */
    _calculateStatistics(timings) {
        const sorted = timings.slice().sort((a, b) => a - b);
        const sum = timings.reduce((a, b) => a + b, 0);
        
        return {
            count: timings.length,
            mean: sum / timings.length,
            median: sorted[Math.floor(sorted.length / 2)],
            min: Math.min(...timings),
            max: Math.max(...timings),
            std: Math.sqrt(timings.reduce((sq, x) => sq + Math.pow(x - (sum / timings.length), 2), 0) / timings.length),
            p95: sorted[Math.floor(sorted.length * 0.95)],
            p99: sorted[Math.floor(sorted.length * 0.99)]
        };
    }

    /**
     * Generate performance recommendations
     */
    generateRecommendations(results) {
        const recommendations = {
            optimal_batch_sizes: {},
            best_configurations: {},
            gpu_vs_cpu_guidance: {},
            memory_optimization: {},
            general_guidelines: []
        };

        // Analyze batch size performance
        for (const [configName, configResults] of Object.entries(results.endToEndBenchmarks)) {
            const batchPerformance = [];
            
            for (const [batchKey, stats] of Object.entries(configResults)) {
                const batchSize = parseInt(batchKey.split('_')[1]);
                const throughput = batchSize / stats.mean; // samples per ms
                batchPerformance.push({ batchSize, throughput, latency: stats.mean });
            }
            
            const bestBatch = batchPerformance.reduce((best, current) => 
                current.throughput > best.throughput ? current : best
            );
            
            recommendations.optimal_batch_sizes[configName] = bestBatch;
        }

        // GPU vs CPU recommendations
        if (results.comparisonResults) {
            for (const [configName, comparison] of Object.entries(results.comparisonResults)) {
                recommendations.gpu_vs_cpu_guidance[configName] = {
                    recommended: comparison.recommendation,
                    speedup: comparison.speedup.toFixed(2),
                    reason: comparison.speedup > 1.5 ? 'Significant GPU advantage' : 
                           comparison.speedup < 0.7 ? 'CPU overhead too high' : 'Marginal difference'
                };
            }
        }

        // General guidelines
        recommendations.general_guidelines = [
            'Use GPU for batch sizes >= 4 for better throughput',
            'CPU may be better for single inferences due to lower overhead',
            'Memory transfers are expensive - minimize data movement',
            'Larger networks benefit more from GPU acceleration',
            'Consider WebGPU availability and fallback to CPU when needed'
        ];

        return recommendations;
    }

    /**
     * Export benchmark results to JSON
     */
    exportResults(results) {
        const exportData = {
            ...results,
            metadata: {
                exported: new Date().toISOString(),
                version: '1.0.0',
                platform: navigator.platform,
                userAgent: navigator.userAgent.substring(0, 100)
            }
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `shader_benchmark_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Benchmark results exported');
    }

    /**
     * Clean up resources
     */
    destroy() {
        if (this.bufferManager) {
            this.bufferManager.destroy();
        }
        if (this.shaderManager) {
            this.shaderManager.destroy();
        }
        
        console.log('Shader benchmark destroyed');
    }
}