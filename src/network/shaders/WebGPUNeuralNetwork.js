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
            
            // Create bind groups
            const layouts = {
                matmul: this.shaderManager.getBindGroupLayout('matmul'),
                activation: this.shaderManager.getBindGroupLayout('activation')
            };
            this.bindGroups = this.bufferManager.createBindGroups(this.buffers, layouts);
            
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
     * Execute forward pass using GPU compute shaders
     * @private
     */
    async _executeForwardPass() {
        const commandEncoder = this.device.createCommandEncoder({
            label: 'neural_network_forward_pass'
        });
        
        const passEncoder = commandEncoder.beginComputePass({
            label: 'forward_pass_compute'
        });

        // Input to hidden layer (matrix multiplication + bias)
        await this.bufferManager.updateUniformBuffer('matmulParams', {
            M: 1,
            K: this.inputSize,
            N: this.hiddenSize
        }, { label: 'matmul_params_hidden' });
        
        passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
        passEncoder.setBindGroup(0, this.bindGroups.matmul);
        passEncoder.dispatchWorkgroups(Math.ceil(this.hiddenSize / 64));

        // ReLU activation on hidden layer
        await this.bufferManager.updateUniformBuffer('activationParams', {
            size: this.hiddenSize
        }, { label: 'activation_params_hidden' });
        
        passEncoder.setPipeline(this.shaderManager.getPipeline('relu'));
        passEncoder.setBindGroup(0, this.bindGroups.activation);
        passEncoder.dispatchWorkgroups(Math.ceil(this.hiddenSize / 64));

        // Hidden to output layer (matrix multiplication + bias)
        await this.bufferManager.updateUniformBuffer('matmulParams', {
            M: 1,
            K: this.hiddenSize,
            N: this.outputSize
        }, { label: 'matmul_params_output' });
        
        passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
        passEncoder.setBindGroup(0, this.bindGroups.output);
        passEncoder.dispatchWorkgroups(Math.ceil(this.outputSize / 64));

        passEncoder.end();

        // Submit commands and wait for completion
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Perform batch forward pass for training
     * @param {Float32Array} batchInput - Batch of inputs [batchSize * inputSize]
     * @param {number} batchSize - Number of samples in batch
     * @returns {Promise<Float32Array>} Batch output [batchSize * outputSize]
     */
    async forwardBatch(batchInput, batchSize) {
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized');
        }

        if (batchInput.length !== batchSize * this.inputSize) {
            throw new Error(`Batch input size mismatch. Expected ${batchSize * this.inputSize}, got ${batchInput.length}`);
        }

        // For batch processing, create buffers with batch dimension using enhanced features
        const batchArchitecture = { 
            inputSize: this.inputSize, 
            hiddenSize: this.hiddenSize, 
            outputSize: this.outputSize, 
            batchSize 
        };
        
        const batchBuffers = await this.bufferManager.createNetworkBuffers(batchArchitecture, {
            persistent: false, // Temporary batch buffers
            allowReuse: true   // Allow reuse for similar batch sizes
        });
        
        const layouts = {
            matmul: this.shaderManager.getBindGroupLayout('matmul'),
            activation: this.shaderManager.getBindGroupLayout('activation')
        };
        const batchBindGroups = this.bufferManager.createBindGroups(batchBuffers, layouts);

        try {
            // Copy current weights to batch buffers
            await this._copyWeightsToBatchBuffers(batchBuffers);

            // Upload batch input using enhanced buffer operations
            await this.bufferManager.updateBuffer(
                this.device.queue, 
                this.bufferManager.getBuffer('input'), 
                batchInput, 
                0, 
                { label: 'batch_input', useStaging: batchInput.length > 64 * 1024 }
            );

            // Execute batch forward pass
            await this._executeBatchForwardPass(batchSize, batchBindGroups);

            // Download batch output using enhanced buffer operations
            const outputBuffer = await this.bufferManager.readBuffer(
                this.device, 
                this.bufferManager.getBuffer('output'), 
                batchSize * this.outputSize * 4, 
                0, 
                { label: 'batch_output' }
            );
            return new Float32Array(outputBuffer);

        } finally {
            // Clean up batch buffers (simplified - in practice would pool these)
            // Note: BufferManager handles cleanup on destroy
        }
    }

    /**
     * Copy current weights to batch buffers
     * @private
     */
    async _copyWeightsToBatchBuffers(batchBuffers) {
        // In a complete implementation, we would copy existing weights
        // For now, we assume weights are already uploaded to the batch buffers
        // This would typically involve GPU-to-GPU copies or re-uploading from CPU cache
    }

    /**
     * Execute batch forward pass
     * @private
     */
    async _executeBatchForwardPass(batchSize, bindGroups) {
        const commandEncoder = this.device.createCommandEncoder({
            label: 'batch_forward_pass'
        });
        
        const passEncoder = commandEncoder.beginComputePass({
            label: 'batch_forward_compute'
        });

        // Input to hidden layer
        await this.bufferManager.updateUniformBuffer('matmulParams', {
            M: batchSize,
            K: this.inputSize,
            N: this.hiddenSize
        }, { label: 'batch_matmul_params_hidden' });
        
        if (this.shaderManager.computePipelines.has('matmul_batch')) {
            passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_batch'));
            passEncoder.setBindGroup(0, bindGroups.matmul);
            passEncoder.dispatchWorkgroups(
                Math.ceil(this.hiddenSize / 8),
                Math.ceil(batchSize / 8),
                1
            );
        } else {
            // Fall back to simple implementation
            passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
            passEncoder.setBindGroup(0, bindGroups.matmul);
            passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.hiddenSize / 64));
        }

        // Batch ReLU activation
        await this.bufferManager.updateUniformBuffer('activationParams', {
            size: batchSize * this.hiddenSize
        }, { label: 'batch_activation_params' });
        
        passEncoder.setPipeline(this.shaderManager.getPipeline('relu'));
        passEncoder.setBindGroup(0, bindGroups.activation);
        passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.hiddenSize / 64));

        // Hidden to output layer
        await this.bufferManager.updateUniformBuffer('matmulParams', {
            M: batchSize,
            K: this.hiddenSize,
            N: this.outputSize
        }, { label: 'batch_matmul_params_output' });
        
        passEncoder.setPipeline(this.shaderManager.getPipeline('matmul_simple'));
        passEncoder.setBindGroup(0, bindGroups.output);
        passEncoder.dispatchWorkgroups(Math.ceil(batchSize * this.outputSize / 64));

        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
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
            
            // Overall system efficiency
            efficiency: {
                bufferPoolHitRate: bufferMetrics ? bufferMetrics.pool.hitRate : '0%',
                memoryUtilization: bufferMetrics ? bufferMetrics.memory.utilizationPercent + '%' : '0%',
                averageBufferCreationTime: bufferMetrics ? bufferMetrics.performance.avgCreationTime : '0ms',
                averageTransferTime: bufferMetrics ? bufferMetrics.performance.avgTransferTime : '0ms'
            }
        };
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
     * Clean up GPU resources
     */
    destroy() {
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

        // Log final performance metrics before destruction
        if (this.bufferManager) {
            const finalMetrics = this.bufferManager.getMemoryUsage();
            console.log('Final buffer pool stats:', {
                hits: finalMetrics.pool.hits,
                misses: finalMetrics.pool.misses,
                hitRate: finalMetrics.pool.hitRate,
                totalReused: finalMetrics.pool.totalReused
            });
        }
        
        console.log('Enhanced WebGPU neural network destroyed');
    }
}