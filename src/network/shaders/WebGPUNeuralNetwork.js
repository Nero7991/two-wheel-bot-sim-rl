/**
 * WebGPU Neural Network Implementation with Compute Shaders
 * 
 * Complete GPU-accelerated neural network implementation using WGSL compute shaders
 * for the two-wheel balancing robot RL application. This replaces the placeholder
 * implementation in the main WebGPUBackend.js file.
 */

import { ShaderManager } from './ShaderManager.js';
import { BufferManager, BufferUsageType } from './BufferManager.js';
import { validateArchitecture, calculateParameterCount } from '../NeuralNetwork.js';

/**
 * GPU-accelerated neural network using WebGPU compute shaders
 */
export class WebGPUNeuralNetwork {
    constructor(device) {
        this.device = device;
        this.shaderManager = null;
        this.bufferManager = null;
        
        // Network architecture
        this.inputSize = 0;
        this.hiddenSize = 0;
        this.outputSize = 0;
        
        // GPU resources
        this.buffers = null;
        this.bindGroups = null;
        
        // State tracking
        this.isInitialized = false;
        this.weightsInitialized = false;
        
        // Performance tracking
        this.forwardPassTimes = [];
        this.gpuMemoryUsed = 0;
        
        // Async operation management
        this.pendingOperations = new Map();
        this.operationCounter = 0;
        this.maxConcurrentOps = 4;
        this.asyncQueue = [];
        
        // Error handling and recovery
        this.errorCount = 0;
        this.maxErrors = 10;
        this.lastError = null;
        this.errorLog = [];
        
        // Batch buffer caching
        this._batchBufferCache = new Map();
        this._batchBindGroupCache = new Map();
        
        // Operation timeouts
        this.operationTimeout = 30000; // 30 seconds
        this.activeTimeouts = new Map();
    }

    /**
     * Initialize the neural network with GPU resources
     * @param {number} inputSize - Input layer size (2)
     * @param {number} hiddenSize - Hidden layer size (4-16)
     * @param {number} outputSize - Output layer size (3)
     * @param {Object} options - Configuration options
     */
    async initialize(inputSize, hiddenSize, outputSize, options = {}) {
        console.log('Initializing WebGPU neural network...');
        
        // Validate architecture
        validateArchitecture(inputSize, hiddenSize, outputSize);
        
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        // Initialize shader and buffer managers with enhanced configuration
        this.shaderManager = new ShaderManager(this.device);
        this.bufferManager = new BufferManager(this.device, {
            maxBufferSize: 1024 * 1024 * 16, // 16MB max per buffer
            maxTotalMemory: 1024 * 1024 * 256, // 256MB total limit
            poolEnabled: true,
            poolMaxAge: 60000, // 1 minute
            poolMaxSize: 20,
            enableValidation: true,
            enableProfiling: true
        });
        
        try {
            // Load and compile shaders
            await this.shaderManager.loadShaders();
            
            // Create buffers for network with enhanced features
            const architecture = { inputSize, hiddenSize, outputSize, batchSize: 1 };
            this.buffers = await this.bufferManager.createNetworkBuffers(architecture, {
                persistent: true, // Keep weights persistent
                allowReuse: true  // Allow buffer reuse from pool
            });
            
            // Create specialized bind groups for each layer
            await this._createBindGroups();
            
            // Initialize weights
            await this.initializeWeights(options.initMethod || 'xavier', options.seed);
            
            this.isInitialized = true;
            const memoryUsage = this.bufferManager.getMemoryUsage();
            this.gpuMemoryUsed = memoryUsage.memory.totalActive;
            
            console.log(`WebGPU neural network initialized: ${inputSize}-${hiddenSize}-${outputSize}`);
            console.log(`GPU memory used: ${memoryUsage.memory.totalActiveFormatted}`);
            console.log(`Buffer pool efficiency: ${memoryUsage.pool.hitRate} hit rate`);
            
        } catch (error) {
            console.error('Failed to initialize WebGPU neural network:', error);
            throw error;
        }
    }

    /**
     * Initialize network weights using specified method
     * @param {string} method - Initialization method ('xavier', 'he', 'random')
     * @param {number} seed - Random seed for reproducible initialization
     */
    async initializeWeights(method = 'xavier', seed = null) {
        if (seed !== null) {
            // Note: WebGPU doesn't have built-in seeded random, so we use CPU generation
            console.log(`Initializing weights with ${method} method, seed: ${seed}`);
        }

        // Generate weights on CPU (for deterministic initialization)
        const weightsHidden = this._generateWeights(this.inputSize, this.hiddenSize, method, seed);
        const biasHidden = this._generateBias(this.hiddenSize);
        const weightsOutput = this._generateWeights(this.hiddenSize, this.outputSize, method, seed ? seed + 1 : null);
        const biasOutput = this._generateBias(this.outputSize);

        // Upload weights to GPU using enhanced buffer operations
        await this.bufferManager.updateBuffer(
            this.device.queue, 
            this.buffers.weightsHidden, 
            weightsHidden, 
            0, 
            { label: 'init_weights_hidden' }
        );
        await this.bufferManager.updateBuffer(
            this.device.queue, 
            this.buffers.biasHidden, 
            biasHidden, 
            0, 
            { label: 'init_bias_hidden' }
        );
        await this.bufferManager.updateBuffer(
            this.device.queue, 
            this.buffers.weightsOutput, 
            weightsOutput, 
            0, 
            { label: 'init_weights_output' }
        );
        await this.bufferManager.updateBuffer(
            this.device.queue, 
            this.buffers.biasOutput, 
            biasOutput, 
            0, 
            { label: 'init_bias_output' }
        );

        this.weightsInitialized = true;
        console.log('Weights initialized and uploaded to GPU');
    }

    /**
     * Generate weight matrix using specified initialization method
     * @private
     */
    _generateWeights(inputSize, outputSize, method, seed) {
        const size = inputSize * outputSize;
        const weights = new Float32Array(size);
        
        // Simple random number generator for seeded initialization
        let random = seed ? this._seededRandom(seed) : Math.random;
        
        switch (method) {
            case 'xavier':
                const xavierStd = Math.sqrt(2.0 / (inputSize + outputSize));
                for (let i = 0; i < size; i++) {
                    weights[i] = (random() - 0.5) * 2 * xavierStd;
                }
                break;
                
            case 'he':
                const heStd = Math.sqrt(2.0 / inputSize);
                for (let i = 0; i < size; i++) {
                    weights[i] = this._boxMuller() * heStd;
                }
                break;
                
            default: // random
                for (let i = 0; i < size; i++) {
                    weights[i] = (random() - 0.5) * 0.1;
                }
        }
        
        return weights;
    }

    /**
     * Generate bias vector
     * @private
     */
    _generateBias(size) {
        return new Float32Array(size).fill(0.01);
    }

    /**
     * Simple seeded random number generator
     * @private
     */
    _seededRandom(seed) {
        let s = seed;
        return function() {
            s = (s * 1664525 + 1013904223) % 4294967296;
            return s / 4294967296;
        };
    }

    /**
     * Box-Muller transform for normal distribution
     * @private
     */
    _boxMuller() {
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Perform forward pass through the network
     * @param {Float32Array} input - Input values [angle, angular_velocity]
     * @returns {Promise<Float32Array>} Output action probabilities [left, right, brake]
     */
    async forward(input) {
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized');
        }

        if (input.length !== this.inputSize) {
            throw new Error(`Input size mismatch. Expected ${this.inputSize}, got ${input.length}`);
        }

        const startTime = performance.now();

        try {
            // Upload input data to GPU using enhanced buffer operations
            await this.bufferManager.updateBuffer(
                this.device.queue, 
                this.buffers.input, 
                input, 
                0, 
                { label: 'forward_input' }
            );

            // Execute forward pass on GPU
            await this._executeForwardPass();

            // Download output from GPU using enhanced buffer operations
            const outputBuffer = await this.bufferManager.readBuffer(
                this.device, 
                this.buffers.output, 
                this.outputSize * 4, 
                0, 
                { label: 'forward_output' }
            );
            const output = new Float32Array(outputBuffer);

            // Track performance
            const forwardTime = performance.now() - startTime;
            this.forwardPassTimes.push(forwardTime);
            
            // Keep only recent timing history
            if (this.forwardPassTimes.length > 100) {
                this.forwardPassTimes.shift();
            }

            return output;

        } catch (error) {
            console.error('Forward pass failed:', error);
            throw error;
        }
    }

    /**
     * Execute forward pass using GPU compute shaders with proper synchronization
     * @private
     */
    async _executeForwardPass() {
        try {
            // Create command encoder for the entire forward pass sequence
            const commandEncoder = this.device.createCommandEncoder({
                label: 'neural_network_forward_pass'
            });
            
            const passEncoder = commandEncoder.beginComputePass({
                label: 'forward_pass_compute',
                timestampWrites: this.device.features.has('timestamp-query') ? {
                    querySet: null, // Would be set up for performance monitoring
                    beginningOfPassWriteIndex: undefined,
                    endOfPassWriteIndex: undefined
                } : undefined
            });

            // Step 1: Input to hidden layer (matrix multiplication + bias)
            // Update parameters for hidden layer computation
            const hiddenParams = new Uint32Array([1, this.inputSize, this.hiddenSize, 0]);
            this.device.queue.writeBuffer(this.buffers.matmulParams, 0, hiddenParams);
            
            passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
            passEncoder.setBindGroup(0, this.bindGroups.matmulHidden);
            passEncoder.dispatchWorkgroups(Math.ceil((1 * this.hiddenSize) / 64));

            // Step 2: ReLU activation on hidden layer
            // Update parameters for activation function
            const activationParams = new Uint32Array([this.hiddenSize, 0, 0, 0]);
            this.device.queue.writeBuffer(this.buffers.activationParams, 0, activationParams);
            
            passEncoder.setPipeline(this.shaderManager.getPipeline('relu'));
            passEncoder.setBindGroup(0, this.bindGroups.activation);
            passEncoder.dispatchWorkgroups(Math.ceil(this.hiddenSize / 64));

            // Step 3: Hidden to output layer (matrix multiplication + bias)
            // Update parameters for output layer computation
            const outputParams = new Uint32Array([1, this.hiddenSize, this.outputSize, 0]);
            this.device.queue.writeBuffer(this.buffers.matmulParams, 0, outputParams);
            
            passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
            passEncoder.setBindGroup(0, this.bindGroups.matmulOutput);
            passEncoder.dispatchWorkgroups(Math.ceil((1 * this.outputSize) / 64));

            // End compute pass
            passEncoder.end();

            // Submit all commands in a single batch for optimal GPU utilization
            const commandBuffer = commandEncoder.finish();
            this.device.queue.submit([commandBuffer]);
            
            // Wait for all GPU operations to complete
            await this.device.queue.onSubmittedWorkDone();
            
        } catch (error) {
            console.error('Forward pass execution failed:', error);
            throw new Error(`GPU forward pass failed: ${error.message}`);
        }
    }

    /**
     * Perform batch forward pass for training with optimized GPU utilization
     * @param {Float32Array} batchInput - Batch of inputs [batchSize * inputSize]
     * @param {number} batchSize - Number of samples in batch
     * @param {Object} options - Batch processing options
     * @param {boolean} options.reuseBuffers - Reuse existing batch buffers
     * @param {boolean} options.async - Execute asynchronously
     * @returns {Promise<Float32Array>} Batch output [batchSize * outputSize]
     */
    async forwardBatch(batchInput, batchSize, options = {}) {
        const { reuseBuffers = true, async = false } = options;
        
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized');
        }

        if (batchInput.length !== batchSize * this.inputSize) {
            throw new Error(`Batch input size mismatch. Expected ${batchSize * this.inputSize}, got ${batchInput.length}`);
        }

        const startTime = performance.now();
        const batchKey = `batch_${batchSize}`;

        try {
            // Get or create batch buffers
            let batchBuffers = reuseBuffers ? this._batchBufferCache?.get(batchKey) : null;
            let batchBindGroups = reuseBuffers ? this._batchBindGroupCache?.get(batchKey) : null;

            if (!batchBuffers || !batchBindGroups) {
                console.log(`Creating new batch buffers for batch size: ${batchSize}`);
                
                // Create buffers with batch dimension
                const batchArchitecture = { 
                    inputSize: this.inputSize, 
                    hiddenSize: this.hiddenSize, 
                    outputSize: this.outputSize, 
                    batchSize 
                };
                
                batchBuffers = await this.bufferManager.createNetworkBuffers(batchArchitecture, {
                    persistent: reuseBuffers, // Cache if reusing
                    allowReuse: true
                });

                // Create specialized bind groups for batch processing
                batchBindGroups = await this._createBatchBindGroups(batchBuffers, batchSize);

                // Cache for reuse
                if (reuseBuffers) {
                    this._batchBufferCache = this._batchBufferCache || new Map();
                    this._batchBindGroupCache = this._batchBindGroupCache || new Map();
                    this._batchBufferCache.set(batchKey, batchBuffers);
                    this._batchBindGroupCache.set(batchKey, batchBindGroups);
                }
            }

            // Copy current weights to batch buffers if needed
            await this._syncWeightsToBatchBuffers(batchBuffers);

            // Upload batch input with optimized transfer
            await this.bufferManager.updateBuffer(
                this.device.queue, 
                batchBuffers.input, 
                batchInput, 
                0, 
                { 
                    label: 'batch_input', 
                    useStaging: batchInput.length > 64 * 1024 
                }
            );

            // Execute optimized batch forward pass
            if (async) {
                // Fire and forget for high throughput training
                this._executeBatchForwardPassAsync(batchSize, batchBindGroups, batchBuffers);
                return null; // Caller must retrieve results separately
            } else {
                await this._executeBatchForwardPass(batchSize, batchBindGroups, batchBuffers);
            }

            // Download batch output
            const outputBuffer = await this.bufferManager.readBuffer(
                this.device, 
                batchBuffers.output, 
                batchSize * this.outputSize * 4, 
                0, 
                { label: 'batch_output' }
            );
            
            const output = new Float32Array(outputBuffer);
            
            // Track batch performance
            const batchTime = performance.now() - startTime;
            console.log(`Batch forward (${batchSize}): ${batchTime.toFixed(2)}ms (${(batchTime/batchSize).toFixed(2)}ms per sample)`);
            
            return output;

        } catch (error) {
            console.error('Batch forward pass failed:', error);
            throw error;
        }
    }

    /**
     * Synchronize current weights to batch buffers efficiently
     * @private
     */
    async _syncWeightsToBatchBuffers(batchBuffers) {
        try {
            // Copy weights using GPU-to-GPU transfers for efficiency
            const commandEncoder = this.device.createCommandEncoder({
                label: 'weight_sync_to_batch'
            });

            // Copy hidden layer weights
            commandEncoder.copyBufferToBuffer(
                this.buffers.weightsHidden, 0,
                batchBuffers.weightsHidden, 0,
                this.inputSize * this.hiddenSize * 4
            );

            // Copy hidden layer biases
            commandEncoder.copyBufferToBuffer(
                this.buffers.biasHidden, 0,
                batchBuffers.biasHidden, 0,
                this.hiddenSize * 4
            );

            // Copy output layer weights
            commandEncoder.copyBufferToBuffer(
                this.buffers.weightsOutput, 0,
                batchBuffers.weightsOutput, 0,
                this.hiddenSize * this.outputSize * 4
            );

            // Copy output layer biases
            commandEncoder.copyBufferToBuffer(
                this.buffers.biasOutput, 0,
                batchBuffers.biasOutput, 0,
                this.outputSize * 4
            );

            // Submit copy operations
            this.device.queue.submit([commandEncoder.finish()]);
            
            // Don't wait - subsequent operations will implicitly wait
            
        } catch (error) {
            console.error('Weight synchronization failed:', error);
            throw error;
        }
    }

    /**
     * Create specialized bind groups for batch processing
     * @private
     */
    async _createBatchBindGroups(batchBuffers, batchSize) {
        const layouts = {
            matmul: this.shaderManager.getBindGroupLayout('matmul'),
            activation: this.shaderManager.getBindGroupLayout('activation')
        };

        const bindGroups = {};

        // Batch hidden layer matrix multiplication
        bindGroups.matmulHidden = this.device.createBindGroup({
            label: `batch_matmul_hidden_${batchSize}`,
            layout: layouts.matmul,
            entries: [
                { binding: 0, resource: { buffer: batchBuffers.input } },
                { binding: 1, resource: { buffer: batchBuffers.weightsHidden } },
                { binding: 2, resource: { buffer: batchBuffers.biasHidden } },
                { binding: 3, resource: { buffer: batchBuffers.hidden } },
                { binding: 4, resource: { buffer: batchBuffers.matmulParams } }
            ]
        });

        // Batch output layer matrix multiplication
        bindGroups.matmulOutput = this.device.createBindGroup({
            label: `batch_matmul_output_${batchSize}`,
            layout: layouts.matmul,
            entries: [
                { binding: 0, resource: { buffer: batchBuffers.hidden } },
                { binding: 1, resource: { buffer: batchBuffers.weightsOutput } },
                { binding: 2, resource: { buffer: batchBuffers.biasOutput } },
                { binding: 3, resource: { buffer: batchBuffers.output } },
                { binding: 4, resource: { buffer: batchBuffers.matmulParams } }
            ]
        });

        // Batch activation function
        bindGroups.activation = this.device.createBindGroup({
            label: `batch_activation_${batchSize}`,
            layout: layouts.activation,
            entries: [
                { binding: 0, resource: { buffer: batchBuffers.hidden } },
                { binding: 1, resource: { buffer: batchBuffers.hidden } }, // In-place operation
                { binding: 2, resource: { buffer: batchBuffers.activationParams } }
            ]
        });

        console.log(`Created batch bind groups for batch size: ${batchSize}`);
        return bindGroups;
    }

    /**
     * Execute optimized batch forward pass with proper GPU utilization
     * @private
     */
    async _executeBatchForwardPass(batchSize, bindGroups, buffers) {
        try {
            const commandEncoder = this.device.createCommandEncoder({
                label: `batch_forward_pass_${batchSize}`
            });
            
            const passEncoder = commandEncoder.beginComputePass({
                label: 'batch_forward_compute'
            });

            // Step 1: Input to hidden layer matrix multiplication
            const hiddenParams = new Uint32Array([batchSize, this.inputSize, this.hiddenSize, 0]);
            this.device.queue.writeBuffer(buffers.matmulParams, 0, hiddenParams);
            
            // Use batch-optimized pipeline if available
            if (this.shaderManager.computePipelines.has('matmul_batch')) {
                passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_batch'));
                passEncoder.setBindGroup(0, bindGroups.matmulHidden);
                // Optimal workgroup dispatch for batch processing
                passEncoder.dispatchWorkgroups(
                    Math.ceil(this.hiddenSize / 8),
                    Math.ceil(batchSize / 8),
                    1
                );
            } else {
                // Fallback to simple implementation
                passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
                passEncoder.setBindGroup(0, bindGroups.matmulHidden);
                passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.hiddenSize / 64));
            }

            // Step 2: Batch ReLU activation on hidden layer
            const activationParams = new Uint32Array([batchSize * this.hiddenSize, 0, 0, 0]);
            this.device.queue.writeBuffer(buffers.activationParams, 0, activationParams);
            
            passEncoder.setPipeline(this.shaderManager.getPipeline('relu'));
            passEncoder.setBindGroup(0, bindGroups.activation);
            passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.hiddenSize / 64));

            // Step 3: Hidden to output layer matrix multiplication
            const outputParams = new Uint32Array([batchSize, this.hiddenSize, this.outputSize, 0]);
            this.device.queue.writeBuffer(buffers.matmulParams, 0, outputParams);
            
            if (this.shaderManager.computePipelines.has('matmul_batch')) {
                passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_batch'));
                passEncoder.setBindGroup(0, bindGroups.matmulOutput);
                passEncoder.dispatchWorkgroups(
                    Math.ceil(this.outputSize / 8),
                    Math.ceil(batchSize / 8),
                    1
                );
            } else {
                passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
                passEncoder.setBindGroup(0, bindGroups.matmulOutput);
                passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.outputSize / 64));
            }

            passEncoder.end();
            
            // Submit command buffer
            const commandBuffer = commandEncoder.finish();
            this.device.queue.submit([commandBuffer]);
            
            // Wait for completion
            await this.device.queue.onSubmittedWorkDone();
            
        } catch (error) {
            console.error('Batch forward pass execution failed:', error);
            throw error;
        }
    }

    /**
     * Execute asynchronous batch forward pass for high-throughput training
     * @private
     */
    async _executeBatchForwardPassAsync(batchSize, bindGroups, buffers) {
        try {
            const commandEncoder = this.device.createCommandEncoder({
                label: `async_batch_forward_${batchSize}`
            });
            
            const passEncoder = commandEncoder.beginComputePass({
                label: 'async_batch_compute'
            });

            // Same operations as synchronous version but no await
            const hiddenParams = new Uint32Array([batchSize, this.inputSize, this.hiddenSize, 0]);
            this.device.queue.writeBuffer(buffers.matmulParams, 0, hiddenParams);
            
            passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
            passEncoder.setBindGroup(0, bindGroups.matmulHidden);
            passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.hiddenSize / 64));

            const activationParams = new Uint32Array([batchSize * this.hiddenSize, 0, 0, 0]);
            this.device.queue.writeBuffer(buffers.activationParams, 0, activationParams);
            
            passEncoder.setPipeline(this.shaderManager.getPipeline('relu'));
            passEncoder.setBindGroup(0, bindGroups.activation);
            passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.hiddenSize / 64));

            const outputParams = new Uint32Array([batchSize, this.hiddenSize, this.outputSize, 0]);
            this.device.queue.writeBuffer(buffers.matmulParams, 0, outputParams);
            
            passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
            passEncoder.setBindGroup(0, bindGroups.matmulOutput);
            passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.outputSize / 64));

            passEncoder.end();
            
            // Submit without waiting - fire and forget
            this.device.queue.submit([commandEncoder.finish()]);
            
        } catch (error) {
            console.error('Async batch forward pass failed:', error);
            throw error;
        }
    }

    /**
     * Get current network weights
     * @returns {Promise<Object>} Object containing weights and biases
     */
    async getWeights() {
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized');
        }

        // Download weights from GPU using enhanced buffer operations
        const weightsHiddenBuffer = await this.bufferManager.readBuffer(
            this.device, this.buffers.weightsHidden, this.inputSize * this.hiddenSize * 4, 0, 
            { label: 'get_weights_hidden' }
        );
        const biasHiddenBuffer = await this.bufferManager.readBuffer(
            this.device, this.buffers.biasHidden, this.hiddenSize * 4, 0, 
            { label: 'get_bias_hidden' }
        );
        const weightsOutputBuffer = await this.bufferManager.readBuffer(
            this.device, this.buffers.weightsOutput, this.hiddenSize * this.outputSize * 4, 0, 
            { label: 'get_weights_output' }
        );
        const biasOutputBuffer = await this.bufferManager.readBuffer(
            this.device, this.buffers.biasOutput, this.outputSize * 4, 0, 
            { label: 'get_bias_output' }
        );

        return {
            weightsHidden: new Float32Array(weightsHiddenBuffer),
            biasHidden: new Float32Array(biasHiddenBuffer),
            weightsOutput: new Float32Array(weightsOutputBuffer),
            biasOutput: new Float32Array(biasOutputBuffer)
        };
    }

    /**
     * Set network weights
     * @param {Object} weights - Object containing weights and biases
     */
    async setWeights(weights) {
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized');
        }

        // Upload new weights to GPU using enhanced buffer operations
        await this.bufferManager.updateBuffer(
            this.device.queue, this.buffers.weightsHidden, weights.weightsHidden, 0, 
            { label: 'set_weights_hidden' }
        );
        await this.bufferManager.updateBuffer(
            this.device.queue, this.buffers.biasHidden, weights.biasHidden, 0, 
            { label: 'set_bias_hidden' }
        );
        await this.bufferManager.updateBuffer(
            this.device.queue, this.buffers.weightsOutput, weights.weightsOutput, 0, 
            { label: 'set_weights_output' }
        );
        await this.bufferManager.updateBuffer(
            this.device.queue, this.buffers.biasOutput, weights.biasOutput, 0, 
            { label: 'set_bias_output' }
        );

        this.weightsInitialized = true;
        console.log('Weights updated on GPU');
    }

    /**
     * Get network architecture information
     * @returns {Object} Architecture details
     */
    getArchitecture() {
        return {
            inputSize: this.inputSize,
            hiddenSize: this.hiddenSize,
            outputSize: this.outputSize,
            parameterCount: calculateParameterCount(this.inputSize, this.hiddenSize, this.outputSize),
            backend: 'webgpu',
            isInitialized: this.isInitialized,
            weightsInitialized: this.weightsInitialized
        };
    }

    /**
     * Get comprehensive performance metrics including enhanced buffer management
     * @returns {Object} Performance statistics
     */
    getPerformanceMetrics() {
        const forwardTimes = this.forwardPassTimes;
        const avgForwardTime = forwardTimes.length > 0 
            ? forwardTimes.reduce((a, b) => a + b) / forwardTimes.length 
            : 0;

        const bufferMetrics = this.bufferManager ? this.bufferManager.getMemoryUsage() : null;
        const bufferPerformance = this.bufferManager ? this.bufferManager.getPerformanceMetrics() : null;

        return {
            // Neural network metrics
            averageForwardTime: avgForwardTime,
            totalForwardPasses: forwardTimes.length,
            gpuMemoryUsed: this.gpuMemoryUsed,
            
            // Enhanced buffer management metrics
            bufferManager: bufferMetrics ? {
                memory: bufferMetrics.memory,
                buffers: bufferMetrics.buffers,
                pool: bufferMetrics.pool,
                performance: bufferMetrics.performance,
                config: bufferMetrics.config
            } : null,
            
            // Detailed buffer performance
            bufferPerformance: bufferPerformance ? {
                bufferCreation: bufferPerformance.bufferCreation,
                memoryTransfer: bufferPerformance.memoryTransfer,
                mapping: bufferPerformance.mapping,
                asyncOperations: bufferPerformance.asyncOperations,
                errors: bufferPerformance.errors
            } : null,
            
            // Shader manager metrics
            shaderManager: this.shaderManager ? this.shaderManager.getPerformanceMetrics() : null,
            
            // Async operation metrics
            asyncOperations: this.getAsyncStatus(),
            
            // Overall system efficiency
            efficiency: {
                bufferPoolHitRate: bufferMetrics ? bufferMetrics.pool.hitRate : '0%',
                memoryUtilization: bufferMetrics ? bufferMetrics.memory.utilizationPercent + '%' : '0%',
                averageBufferCreationTime: bufferMetrics ? bufferMetrics.performance.avgCreationTime : '0ms',
                averageTransferTime: bufferMetrics ? bufferMetrics.performance.avgTransferTime : '0ms',
                errorRate: this.operationCounter > 0 ? (this.errorCount / this.operationCounter * 100).toFixed(2) + '%' : '0%'
            }
        };
    }

    /**
     * Comprehensive performance benchmark against CPU implementation
     * @param {Object} cpuBackend - CPU backend instance for comparison
     * @param {Object} options - Benchmark configuration
     * @returns {Promise<Object>} Detailed benchmark results
     */
    async benchmarkAgainstCPU(cpuBackend, options = {}) {
        const {
            iterations = 1000,
            warmupIterations = 100,
            testBatchSizes = [1, 4, 8, 16],
            includeMemoryAnalysis = true,
            includeAccuracyCheck = true
        } = options;

        console.log('Starting comprehensive GPU vs CPU benchmark...');
        const startTime = performance.now();

        const results = {
            config: {
                iterations,
                warmupIterations,
                testBatchSizes,
                architecture: this.getArchitecture()
            },
            singleInference: null,
            batchProcessing: {},
            memoryComparison: null,
            accuracyVerification: null,
            summary: null
        };

        try {
            // 1. Single inference benchmark
            console.log('Benchmarking single inference performance...');
            results.singleInference = await this._benchmarkSingleInference(
                cpuBackend, iterations, warmupIterations
            );

            // 2. Batch processing benchmark
            console.log('Benchmarking batch processing performance...');
            for (const batchSize of testBatchSizes) {
                console.log(`  Testing batch size: ${batchSize}`);
                results.batchProcessing[batchSize] = await this._benchmarkBatchProcessing(
                    cpuBackend, batchSize, Math.floor(iterations / batchSize)
                );
            }

            // 3. Memory usage comparison
            if (includeMemoryAnalysis) {
                console.log('Analyzing memory usage...');
                results.memoryComparison = await this._compareMemoryUsage(cpuBackend);
            }

            // 4. Accuracy verification
            if (includeAccuracyCheck) {
                console.log('Verifying numerical accuracy...');
                results.accuracyVerification = await this._verifyAccuracy(cpuBackend);
            }

            // 5. Generate summary
            results.summary = this._generateBenchmarkSummary(results);

            const totalTime = performance.now() - startTime;
            console.log(`Benchmark completed in ${totalTime.toFixed(2)}ms`);
            results.benchmarkTime = totalTime;

            return results;

        } catch (error) {
            console.error('Benchmark failed:', error);
            throw error;
        }
    }

    /**
     * Benchmark single inference performance
     * @private
     */
    async _benchmarkSingleInference(cpuBackend, iterations, warmupIterations) {
        const testInput = new Float32Array([0.1, -0.05]); // Sample robot state

        // Warmup
        for (let i = 0; i < warmupIterations; i++) {
            cpuBackend.forward(testInput);
            await this.forward(testInput);
        }

        // Benchmark CPU
        const cpuTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            cpuBackend.forward(testInput);
            cpuTimes.push(performance.now() - start);
        }

        // Benchmark GPU
        const gpuTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await this.forward(testInput);
            gpuTimes.push(performance.now() - start);
        }

        // Benchmark GPU real-time optimized
        const gpuRealTimeTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await this.forwardRealTime(testInput, { realTime: true, skipValidation: true });
            gpuRealTimeTimes.push(performance.now() - start);
        }

        const cpuStats = this._calculatePerformanceStats(cpuTimes);
        const gpuStats = this._calculatePerformanceStats(gpuTimes);
        const gpuRealTimeStats = this._calculatePerformanceStats(gpuRealTimeTimes);

        return {
            cpu: cpuStats,
            gpu: gpuStats,
            gpuRealTime: gpuRealTimeStats,
            speedup: {
                standard: cpuStats.median / gpuStats.median,
                realTime: cpuStats.median / gpuRealTimeStats.median
            },
            throughput: {
                cpu: 1000 / cpuStats.median,
                gpu: 1000 / gpuStats.median,
                gpuRealTime: 1000 / gpuRealTimeStats.median
            },
            realTimeCapable: gpuRealTimeStats.p95 <= 5.0, // 5ms threshold
            iterations
        };
    }

    /**
     * Benchmark batch processing performance
     * @private
     */
    async _benchmarkBatchProcessing(cpuBackend, batchSize, iterations) {
        const batchInput = new Float32Array(batchSize * this.inputSize);
        
        // Generate batch input
        for (let i = 0; i < batchInput.length; i += 2) {
            batchInput[i] = Math.random() * 0.2 - 0.1;     // angle
            batchInput[i + 1] = Math.random() * 0.1 - 0.05; // angular velocity
        }

        // Warmup
        for (let i = 0; i < 10; i++) {
            // CPU batch simulation (multiple single forwards)
            for (let j = 0; j < batchSize; j++) {
                const singleInput = batchInput.slice(j * 2, (j + 1) * 2);
                cpuBackend.forward(singleInput);
            }
            
            await this.forwardBatch(batchInput, batchSize);
        }

        // Benchmark CPU batch processing (simulated)
        const cpuBatchTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            for (let j = 0; j < batchSize; j++) {
                const singleInput = batchInput.slice(j * 2, (j + 1) * 2);
                cpuBackend.forward(singleInput);
            }
            cpuBatchTimes.push(performance.now() - start);
        }

        // Benchmark GPU batch processing
        const gpuBatchTimes = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await this.forwardBatch(batchInput, batchSize);
            gpuBatchTimes.push(performance.now() - start);
        }

        const cpuStats = this._calculatePerformanceStats(cpuBatchTimes);
        const gpuStats = this._calculatePerformanceStats(gpuBatchTimes);

        return {
            batchSize,
            cpu: {
                ...cpuStats,
                timePerSample: cpuStats.median / batchSize,
                throughput: (1000 * batchSize) / cpuStats.median
            },
            gpu: {
                ...gpuStats,
                timePerSample: gpuStats.median / batchSize,
                throughput: (1000 * batchSize) / gpuStats.median
            },
            speedup: cpuStats.median / gpuStats.median,
            batchEfficiency: (cpuStats.median / batchSize) / (gpuStats.median / batchSize),
            iterations
        };
    }

    /**
     * Compare memory usage between CPU and GPU implementations
     * @private
     */
    async _compareMemoryUsage(cpuBackend) {
        const cpuMemory = cpuBackend.getMemoryUsage();
        const gpuMemory = this.bufferManager.getMemoryUsage();

        return {
            cpu: {
                totalBytes: cpuMemory.totalBytes,
                totalKB: cpuMemory.totalKB,
                parameterBytes: cpuMemory.parameterBytes,
                breakdown: cpuMemory.breakdown
            },
            gpu: {
                totalBytes: gpuMemory.memory.totalActive,
                totalKB: gpuMemory.memory.totalActive / 1024,
                bufferCount: gpuMemory.buffers.count,
                pooledBytes: gpuMemory.memory.totalPooled,
                efficiency: gpuMemory.pool.hitRate
            },
            comparison: {
                memoryRatio: gpuMemory.memory.totalActive / cpuMemory.totalBytes,
                gpuOverhead: gpuMemory.memory.totalActive - cpuMemory.totalBytes,
                bufferPoolSavings: gpuMemory.memory.totalPooled
            }
        };
    }

    /**
     * Verify numerical accuracy against CPU
     * @private
     */
    async _verifyAccuracy(cpuBackend, testCases = 100) {
        const errors = [];
        const maxErrors = { absolute: 0, relative: 0 };
        let totalAbsoluteError = 0;
        let totalRelativeError = 0;

        for (let i = 0; i < testCases; i++) {
            // Generate test input
            const input = new Float32Array([
                (Math.random() - 0.5) * 0.4, // angle: -0.2 to 0.2
                (Math.random() - 0.5) * 0.2  // angular velocity: -0.1 to 0.1
            ]);

            // Get outputs
            const cpuOutput = cpuBackend.forward(input);
            const gpuOutput = await this.forward(input);

            // Calculate errors
            const absoluteErrors = [];
            const relativeErrors = [];

            for (let j = 0; j < cpuOutput.length; j++) {
                const absError = Math.abs(cpuOutput[j] - gpuOutput[j]);
                const relError = cpuOutput[j] !== 0 ? Math.abs((cpuOutput[j] - gpuOutput[j]) / cpuOutput[j]) : 0;

                absoluteErrors.push(absError);
                relativeErrors.push(relError);

                maxErrors.absolute = Math.max(maxErrors.absolute, absError);
                maxErrors.relative = Math.max(maxErrors.relative, relError);
            }

            const maxAbsError = Math.max(...absoluteErrors);
            const maxRelError = Math.max(...relativeErrors);

            totalAbsoluteError += maxAbsError;
            totalRelativeError += maxRelError;

            errors.push({
                input: Array.from(input),
                cpuOutput: Array.from(cpuOutput),
                gpuOutput: Array.from(gpuOutput),
                absoluteErrors,
                relativeErrors,
                maxAbsoluteError: maxAbsError,
                maxRelativeError: maxRelError
            });
        }

        const averageAbsoluteError = totalAbsoluteError / testCases;
        const averageRelativeError = totalRelativeError / testCases;

        // Calculate correlation
        const allCpuOutputs = errors.flatMap(e => e.cpuOutput);
        const allGpuOutputs = errors.flatMap(e => e.gpuOutput);
        const correlation = this._calculateCorrelation(allCpuOutputs, allGpuOutputs);

        return {
            testCases,
            maxErrors,
            averageErrors: {
                absolute: averageAbsoluteError,
                relative: averageRelativeError
            },
            correlation,
            passed: {
                absolute: maxErrors.absolute < 1e-6,
                relative: maxErrors.relative < 1e-5,
                correlation: correlation > 0.9999
            },
            detailedErrors: errors.slice(0, 5) // First 5 for debugging
        };
    }

    /**
     * Generate benchmark summary
     * @private
     */
    _generateBenchmarkSummary(results) {
        const summary = {
            overallSpeedup: results.singleInference.speedup.standard,
            realTimeSpeedup: results.singleInference.speedup.realTime,
            realTimeCapable: results.singleInference.realTimeCapable,
            bestBatchSpeedup: 0,
            optimalBatchSize: 1,
            memoryEfficient: false,
            numericallyAccurate: false,
            productionReady: false,
            recommendations: []
        };

        // Find best batch performance
        for (const [batchSize, batchResult] of Object.entries(results.batchProcessing)) {
            if (batchResult.speedup > summary.bestBatchSpeedup) {
                summary.bestBatchSpeedup = batchResult.speedup;
                summary.optimalBatchSize = parseInt(batchSize);
            }
        }

        // Memory efficiency check
        if (results.memoryComparison) {
            summary.memoryEfficient = results.memoryComparison.comparison.memoryRatio < 2.0;
        }

        // Accuracy check
        if (results.accuracyVerification) {
            summary.numericallyAccurate = 
                results.accuracyVerification.passed.absolute &&
                results.accuracyVerification.passed.relative &&
                results.accuracyVerification.passed.correlation;
        }

        // Production readiness assessment
        summary.productionReady = 
            summary.overallSpeedup >= 2.0 &&
            summary.realTimeCapable &&
            summary.memoryEfficient &&
            summary.numericallyAccurate;

        // Generate recommendations
        if (summary.overallSpeedup < 2.0) {
            summary.recommendations.push('Optimize GPU compute pipeline for better single inference performance');
        }
        if (!summary.realTimeCapable) {
            summary.recommendations.push('Implement additional real-time optimizations to meet latency requirements');
        }
        if (!summary.memoryEfficient) {
            summary.recommendations.push('Optimize buffer management to reduce GPU memory overhead');
        }
        if (!summary.numericallyAccurate) {
            summary.recommendations.push('Review shader implementations for numerical precision issues');
        }
        if (summary.bestBatchSpeedup < 1.5) {
            summary.recommendations.push('Improve batch processing efficiency for training workflows');
        }

        return summary;
    }

    /**
     * Calculate comprehensive performance statistics
     * @private
     */
    _calculatePerformanceStats(times) {
        const sorted = [...times].sort((a, b) => a - b);
        const n = times.length;
        const mean = times.reduce((sum, t) => sum + t, 0) / n;
        const variance = times.reduce((sum, t) => sum + Math.pow(t - mean, 2), 0) / n;

        return {
            count: n,
            min: Math.min(...times),
            max: Math.max(...times),
            mean,
            median: sorted[Math.floor(n / 2)],
            p90: sorted[Math.floor(n * 0.9)],
            p95: sorted[Math.floor(n * 0.95)],
            p99: sorted[Math.floor(n * 0.99)],
            std: Math.sqrt(variance),
            cv: Math.sqrt(variance) / mean // Coefficient of variation
        };
    }

    /**
     * Calculate correlation coefficient between two arrays
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
     * Validate that the network is ready for inference with enhanced buffer validation
     * @returns {Object} Comprehensive validation results
     */
    validate() {
        const validation = {
            isValid: true,
            issues: [],
            warnings: [],
            bufferStatus: {},
            performance: {}
        };

        // Basic validation
        if (!this.isInitialized) {
            validation.isValid = false;
            validation.issues.push('Network not initialized');
        }

        if (!this.weightsInitialized) {
            validation.isValid = false;
            validation.issues.push('Weights not initialized');
        }

        if (!this.device) {
            validation.isValid = false;
            validation.issues.push('WebGPU device not available');
        }

        if (!this.shaderManager || !this.bufferManager) {
            validation.isValid = false;
            validation.issues.push('GPU managers not initialized');
        }

        // Enhanced buffer validation
        if (this.bufferManager) {
            const bufferMetrics = this.bufferManager.getMemoryUsage();
            const bufferPerformance = this.bufferManager.getPerformanceMetrics();
            
            validation.bufferStatus = {
                totalBuffers: bufferMetrics.buffers.count,
                memoryUsage: bufferMetrics.memory.totalActiveFormatted,
                memoryUtilization: bufferMetrics.memory.utilizationPercent + '%',
                poolEfficiency: bufferMetrics.pool.hitRate,
                errorCount: bufferPerformance.errors.count
            };
            
            // Check for buffer-related issues
            if (bufferPerformance.errors.count > 0) {
                validation.warnings.push(`${bufferPerformance.errors.count} buffer errors detected`);
            }
            
            if (parseFloat(bufferMetrics.memory.utilizationPercent) > 90) {
                validation.warnings.push('High memory utilization (>90%)');
            }
            
            if (parseFloat(bufferMetrics.pool.hitRate) < 50 && bufferMetrics.pool.hits + bufferMetrics.pool.misses > 10) {
                validation.warnings.push('Low buffer pool efficiency (<50% hit rate)');
            }
            
            // Performance validation
            const avgCreationTime = parseFloat(bufferMetrics.performance.avgCreationTime);
            const avgTransferTime = parseFloat(bufferMetrics.performance.avgTransferTime);
            
            validation.performance = {
                avgBufferCreation: bufferMetrics.performance.avgCreationTime,
                avgMemoryTransfer: bufferMetrics.performance.avgTransferTime,
                isPerformant: avgCreationTime < 5.0 && avgTransferTime < 10.0 // ms thresholds
            };
            
            if (!validation.performance.isPerformant) {
                validation.warnings.push('Suboptimal buffer performance detected');
            }
        }

        // Validate shader compilation
        if (this.shaderManager) {
            const shaderValidation = this.shaderManager.validateShaders();
            if (!shaderValidation.allShadersCompiled) {
                validation.isValid = false;
                validation.issues.push(`Missing shaders: ${shaderValidation.missingShaders.join(', ')}`);
            }
        }

        // Validate required buffers exist
        if (this.buffers) {
            const requiredBuffers = ['input', 'hidden', 'output', 'weightsHidden', 'weightsOutput', 'biasHidden', 'biasOutput'];
            for (const bufferName of requiredBuffers) {
                if (!this.buffers[bufferName]) {
                    validation.isValid = false;
                    validation.issues.push(`Missing required buffer: ${bufferName}`);
                }
            }
        }

        return validation;
    }

    /**
     * Execute async operation with timeout and error handling
     * @param {Function} operation - Async operation to execute
     * @param {string} operationName - Name for tracking
     * @param {number} timeoutMs - Timeout in milliseconds
     * @returns {Promise} Operation result
     */
    async executeAsyncOperation(operation, operationName, timeoutMs = this.operationTimeout) {
        const operationId = this._generateOperationId();
        
        try {
            // Check if we're at capacity
            if (this.pendingOperations.size >= this.maxConcurrentOps) {
                // Queue the operation
                await this._queueOperation({ operation, operationName, timeoutMs, operationId });
                return;
            }
            
            // Track pending operation
            this.pendingOperations.set(operationId, {
                name: operationName,
                startTime: performance.now(),
                promise: null
            });
            
            // Set up timeout
            const timeoutPromise = new Promise((_, reject) => {
                const timeoutId = setTimeout(() => {
                    reject(new Error(`Operation ${operationName} timed out after ${timeoutMs}ms`));
                }, timeoutMs);
                this.activeTimeouts.set(operationId, timeoutId);
            });
            
            // Execute operation with timeout
            const result = await Promise.race([
                operation(),
                timeoutPromise
            ]);
            
            // Clean up successful operation
            this._cleanupOperation(operationId);
            
            return result;
            
        } catch (error) {
            // Handle error
            this._handleAsyncError(error, operationName, operationId);
            throw error;
        }
    }
    
    /**
     * Queue operation when at capacity
     * @private
     */
    async _queueOperation(operationData) {
        return new Promise((resolve, reject) => {
            this.asyncQueue.push({ ...operationData, resolve, reject });
            this._processQueue();
        });
    }
    
    /**
     * Process queued operations
     * @private
     */
    async _processQueue() {
        if (this.asyncQueue.length === 0 || this.pendingOperations.size >= this.maxConcurrentOps) {
            return;
        }
        
        const { operation, operationName, timeoutMs, operationId, resolve, reject } = this.asyncQueue.shift();
        
        try {
            const result = await this.executeAsyncOperation(operation, operationName, timeoutMs);
            resolve(result);
        } catch (error) {
            reject(error);
        }
        
        // Process next in queue
        setImmediate(() => this._processQueue());
    }
    
    /**
     * Generate unique operation ID
     * @private
     */
    _generateOperationId() {
        return `op_${++this.operationCounter}_${Date.now()}`;
    }
    
    /**
     * Handle async operation error
     * @private
     */
    _handleAsyncError(error, operationName, operationId) {
        this.errorCount++;
        this.lastError = error;
        
        const errorInfo = {
            timestamp: new Date().toISOString(),
            operationName,
            operationId,
            message: error.message,
            stack: error.stack
        };
        
        this.errorLog.push(errorInfo);
        
        // Keep error log manageable
        if (this.errorLog.length > 100) {
            this.errorLog.shift();
        }
        
        // Clean up failed operation
        this._cleanupOperation(operationId);
        
        console.error(`Async operation ${operationName} failed:`, error);
        
        // Check if we've exceeded error threshold
        if (this.errorCount > this.maxErrors) {
            console.error(`Too many errors (${this.errorCount}). Neural network may be unstable.`);
        }
    }
    
    /**
     * Clean up operation tracking
     * @private
     */
    _cleanupOperation(operationId) {
        this.pendingOperations.delete(operationId);
        
        const timeoutId = this.activeTimeouts.get(operationId);
        if (timeoutId) {
            clearTimeout(timeoutId);
            this.activeTimeouts.delete(operationId);
        }
    }
    
    /**
     * Wait for all pending operations to complete
     * @param {number} timeoutMs - Maximum wait time
     * @returns {Promise<void>}
     */
    async waitForOperations(timeoutMs = 60000) {
        const startTime = performance.now();
        
        while (this.pendingOperations.size > 0) {
            if (performance.now() - startTime > timeoutMs) {
                const pendingOps = Array.from(this.pendingOperations.keys());
                throw new Error(`Timeout waiting for operations: ${pendingOps.join(', ')}`);
            }
            
            // Wait a bit before checking again
            await new Promise(resolve => setTimeout(resolve, 10));
        }
    }
    
    /**
     * Get async operation status
     * @returns {Object} Operation status information
     */
    getAsyncStatus() {
        return {
            pendingOperations: this.pendingOperations.size,
            queuedOperations: this.asyncQueue.length,
            totalErrors: this.errorCount,
            lastError: this.lastError ? {
                message: this.lastError.message,
                timestamp: this.errorLog[this.errorLog.length - 1]?.timestamp
            } : null,
            recentErrors: this.errorLog.slice(-5), // Last 5 errors
            maxConcurrentOps: this.maxConcurrentOps,
            operationTimeout: this.operationTimeout
        };
    }
    
    /**
     * Reset error state
     */
    resetErrorState() {
        this.errorCount = 0;
        this.lastError = null;
        this.errorLog = [];
        console.log('Neural network error state reset');
    }

    /**
     * Clean up GPU resources with enhanced async operation handling
     */
    destroy() {
        // Cancel all pending operations
        for (const [operationId, timeout] of this.activeTimeouts) {
            clearTimeout(timeout);
        }
        this.activeTimeouts.clear();
        this.pendingOperations.clear();
        this.asyncQueue = [];
        
        // Clean up batch caches
        this._batchBufferCache?.clear();
        this._batchBindGroupCache?.clear();
        
        if (this.bufferManager) {
            this.bufferManager.destroy();
            this.bufferManager = null;
        }

        if (this.shaderManager) {
            this.shaderManager.destroy();
            this.shaderManager = null;
        }

        this.buffers = null;
        this.bindGroups = null;
        this.isInitialized = false;
        this.weightsInitialized = false;
        this.forwardPassTimes = [];

        // Log final performance and error metrics
        if (this.errorCount > 0) {
            console.log('Final error stats:', {
                totalErrors: this.errorCount,
                errorRate: `${((this.errorCount / (this.operationCounter || 1)) * 100).toFixed(2)}%`
            });
        }
        
        console.log('Enhanced WebGPU neural network destroyed');
    }

    /**
     * Create specialized bind groups for different forward pass stages
     * @private
     */
    async _createBindGroups() {
        const layouts = {
            matmul: this.shaderManager.getBindGroupLayout('matmul'),
            activation: this.shaderManager.getBindGroupLayout('activation')
        };

        this.bindGroups = {};

        // Hidden layer matrix multiplication bind group
        this.bindGroups.matmulHidden = this.device.createBindGroup({
            label: 'matmul_hidden_bind_group',
            layout: layouts.matmul,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.input } },
                { binding: 1, resource: { buffer: this.buffers.weightsHidden } },
                { binding: 2, resource: { buffer: this.buffers.biasHidden } },
                { binding: 3, resource: { buffer: this.buffers.hidden } },
                { binding: 4, resource: { buffer: this.buffers.matmulParams } }
            ]
        });

        // Output layer matrix multiplication bind group
        this.bindGroups.matmulOutput = this.device.createBindGroup({
            label: 'matmul_output_bind_group',
            layout: layouts.matmul,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.hidden } },
                { binding: 1, resource: { buffer: this.buffers.weightsOutput } },
                { binding: 2, resource: { buffer: this.buffers.biasOutput } },
                { binding: 3, resource: { buffer: this.buffers.output } },
                { binding: 4, resource: { buffer: this.buffers.matmulParams } }
            ]
        });

        // Activation function bind group (in-place ReLU on hidden layer)
        this.bindGroups.activation = this.device.createBindGroup({
            label: 'activation_bind_group',
            layout: layouts.activation,
            entries: [
                { binding: 0, resource: { buffer: this.buffers.hidden } },
                { binding: 1, resource: { buffer: this.buffers.hidden } }, // In-place operation
                { binding: 2, resource: { buffer: this.buffers.activationParams } }
            ]
        });

        console.log('Created specialized bind groups for forward pass');
    }

    /**
     * Perform forward pass with real-time optimization for robot control
     * @param {Float32Array} input - Input values [angle, angular_velocity]
     * @param {Object} options - Forward pass options
     * @param {boolean} options.realTime - Optimize for real-time inference
     * @param {boolean} options.skipValidation - Skip input validation for performance
     * @returns {Promise<Float32Array>} Output action probabilities [left, right, brake]
     */
    async forwardRealTime(input, options = {}) {
        const { realTime = true, skipValidation = false } = options;
        
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized');
        }

        if (!skipValidation && input.length !== this.inputSize) {
            throw new Error(`Input size mismatch. Expected ${this.inputSize}, got ${input.length}`);
        }

        const startTime = performance.now();

        try {
            // Fast path upload - use direct writeBuffer for small data
            this.device.queue.writeBuffer(this.buffers.input, 0, input);

            // Execute optimized forward pass
            if (realTime) {
                await this._executeRealTimeForwardPass();
            } else {
                await this._executeForwardPass();
            }

            // Fast path download - use staging buffer
            const outputBuffer = await this.bufferManager.readBuffer(
                this.device, 
                this.buffers.output, 
                this.outputSize * 4, 
                0, 
                { label: 'realtime_forward_output' }
            );
            const output = new Float32Array(outputBuffer);

            // Track performance for real-time optimization
            const forwardTime = performance.now() - startTime;
            this.forwardPassTimes.push(forwardTime);
            
            // Keep only recent timing history for memory efficiency
            if (this.forwardPassTimes.length > 50) {
                this.forwardPassTimes.shift();
            }

            return output;

        } catch (error) {
            console.error('Real-time forward pass failed:', error);
            throw error;
        }
    }

    /**
     * Optimized forward pass for real-time robot control
     * @private
     */
    async _executeRealTimeForwardPass() {
        try {
            // Minimal command encoding for lowest latency
            const encoder = this.device.createCommandEncoder({ label: 'realtime_forward' });
            const pass = encoder.beginComputePass({ label: 'realtime_compute' });

            // Input -> Hidden (pre-computed parameters)
            const hiddenParams = new Uint32Array([1, this.inputSize, this.hiddenSize, 0]);
            this.device.queue.writeBuffer(this.buffers.matmulParams, 0, hiddenParams);
            
            pass.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
            pass.setBindGroup(0, this.bindGroups.matmulHidden);
            pass.dispatchWorkgroups(Math.ceil(this.hiddenSize / 64));

            // ReLU activation (in-place)
            const activationParams = new Uint32Array([this.hiddenSize, 0, 0, 0]);
            this.device.queue.writeBuffer(this.buffers.activationParams, 0, activationParams);
            
            pass.setPipeline(this.shaderManager.getPipeline('relu'));
            pass.setBindGroup(0, this.bindGroups.activation);
            pass.dispatchWorkgroups(Math.ceil(this.hiddenSize / 64));

            // Hidden -> Output
            const outputParams = new Uint32Array([1, this.hiddenSize, this.outputSize, 0]);
            this.device.queue.writeBuffer(this.buffers.matmulParams, 0, outputParams);
            
            pass.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
            pass.setBindGroup(0, this.bindGroups.matmulOutput);
            pass.dispatchWorkgroups(Math.ceil(this.outputSize / 64));

            pass.end();
            this.device.queue.submit([encoder.finish()]);
            
            // Don't await - fire and forget for minimal latency
            // The read operation will implicitly wait
            
        } catch (error) {
            console.error('Real-time forward pass execution failed:', error);
            throw error;
        }
    }
}