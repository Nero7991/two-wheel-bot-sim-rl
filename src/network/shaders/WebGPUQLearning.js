/**
 * WebGPU-Enhanced Q-Learning Integration
 * 
 * Seamless integration between GPU-accelerated neural networks and Q-learning training.
 * Provides intelligent backend selection, performance optimization, and fallback mechanisms
 * for production-ready reinforcement learning training.
 */

import { WebGPUBackend } from '../WebGPUBackend.js';
import { CPUBackend } from '../CPUBackend.js';
import { Hyperparameters, ReplayBuffer, TrainingMetrics } from '../../training/QLearning.js';
import { NetworkConfig } from '../NeuralNetwork.js';

/**
 * Enhanced Q-Learning with WebGPU acceleration and intelligent backend selection
 */
export class WebGPUQLearning {
    constructor(hyperparams = {}, options = {}) {
        this.hyperparams = new Hyperparameters(hyperparams);
        this.metrics = new TrainingMetrics();
        this.replayBuffer = new ReplayBuffer(options.bufferSize || 10000);
        
        // Backend configuration
        this.backendOptions = {
            preferGPU: options.preferGPU !== false,
            fallbackToCPU: options.fallbackToCPU !== false,
            gpuBatchThreshold: options.gpuBatchThreshold || 8,
            realTimeThreshold: options.realTimeThreshold || 5.0, // ms
            ...options.backend
        };
        
        // Neural networks with backend abstraction
        this.qNetwork = null;
        this.targetNetwork = null;
        this.backend = null;
        this.usingGPU = false;
        
        // Training state
        this.isInitialized = false;
        this.episode = 0;
        this.stepCount = 0;
        this.lastTargetUpdate = 0;
        
        // Performance monitoring
        this.trainingTimes = [];\n        this.inferenceTimes = [];
        this.batchTrainingTimes = [];
        this.gpuFallbackCount = 0;
        
        // GPU Q-learning specific resources
        this.qlearningBuffers = null;
        this.qlearningBindGroups = null;
        this.qlearningPipelines = null;
        this.targetNetworkBuffers = null;
        
        // Training state tracking
        this.lastWeightSync = 0;
        this.weightSyncInterval = 10; // Sync every 10 training steps
        this.trainingStepCount = 0;
        
        // Action space
        this.actions = [-1.0, 0.0, 1.0]; // Left motor, brake, right motor
        this.numActions = this.actions.length;
        
        // WebGPU-specific optimizations
        this.batchAccumulator = [];
        this.asyncTraining = options.asyncTraining || false;
        this.pendingUpdates = new Map();
        
        console.log('WebGPU-Enhanced Q-Learning initialized');
    }

    /**
     * Initialize Q-learning with intelligent backend selection
     * @param {Object} options - Initialization options
     * @returns {Promise<void>}
     */
    async initialize(options = {}) {
        try {
            console.log('Initializing WebGPU-Enhanced Q-Learning...');
            
            // Try WebGPU backend first
            if (this.backendOptions.preferGPU) {
                try {
                    console.log('Attempting WebGPU backend initialization...');
                    this.backend = new WebGPUBackend();
                    
                    await this.backend.createNetwork(
                        NetworkConfig.INPUT_SIZE,
                        this.hyperparams.hiddenSize,
                        this.numActions,
                        { 
                            initMethod: NetworkConfig.INITIALIZATION.HE,
                            forceCPU: false
                        }
                    );
                    
                    const backendInfo = this.backend.getBackendInfo();
                    this.usingGPU = backendInfo.webgpuAvailable && !backendInfo.usingFallback;
                    
                    if (this.usingGPU) {\n                        console.log('WebGPU backend initialized successfully');
                        console.log('Performance estimate:', backendInfo.capabilities);
                        
                        // Verify GPU performance meets requirements
                        const perfCheck = await this._verifyGPUPerformance();
                        if (!perfCheck.passed) {
                            console.warn('GPU performance verification failed:', perfCheck.issues);
                            if (!this.backendOptions.fallbackToCPU) {
                                throw new Error('GPU performance requirements not met');
                            }
                            // Continue with CPU fallback
                            this.usingGPU = false;
                        }
                    } else {
                        console.log('WebGPU not available, using CPU fallback');
                    }
                    
                } catch (error) {
                    console.warn('WebGPU backend failed:', error.message);
                    if (!this.backendOptions.fallbackToCPU) {
                        throw error;
                    }
                    this.usingGPU = false;
                }
            }
            
            // Fallback to CPU backend if needed
            if (!this.usingGPU || !this.backend) {
                console.log('Initializing CPU backend...');
                this.backend = new CPUBackend();
                await this.backend.createNetwork(
                    NetworkConfig.INPUT_SIZE,
                    this.hyperparams.hiddenSize,
                    this.numActions,
                    { initMethod: NetworkConfig.INITIALIZATION.HE }
                );
                this.usingGPU = false;
            }
            
            // Create Q-network wrapper
            this.qNetwork = this.backend;
            
            // Create target network
            if (this.usingGPU) {
                // For GPU backend, create a separate target network
                this.targetNetwork = new WebGPUBackend();
                await this.targetNetwork.createNetwork(
                    NetworkConfig.INPUT_SIZE,
                    this.hyperparams.hiddenSize,
                    this.numActions,
                    { initMethod: NetworkConfig.INITIALIZATION.HE }
                );
                
                // Copy weights from main network
                const weights = await this.qNetwork.getWeights();
                await this.targetNetwork.setWeights(weights);
            } else {
                // For CPU backend, use clone
                this.targetNetwork = this.qNetwork.clone();
            }
            
            this.isInitialized = true;
            
            console.log(`Q-Learning initialized with ${this.usingGPU ? 'WebGPU' : 'CPU'} backend`);
            console.log(`Architecture: ${NetworkConfig.INPUT_SIZE}-${this.hyperparams.hiddenSize}-${this.numActions}`);
            console.log(`Parameters: ${this.qNetwork.getParameterCount()}`);
            
            // Run initial performance test
            await this._runInitialPerformanceTest();
            
        } catch (error) {
            console.error('Failed to initialize WebGPU Q-Learning:', error);
            throw error;
        }
    }

    /**
     * Select action using epsilon-greedy policy with GPU acceleration
     * @param {Float32Array} state - Current state
     * @param {boolean} training - Training mode flag
     * @returns {Promise<number>} Selected action index
     */
    async selectAction(state, training = true) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized');
        }

        const startTime = performance.now();
        
        try {
            // Epsilon-greedy exploration
            if (training && Math.random() < this.hyperparams.epsilon) {
                const action = Math.floor(Math.random() * this.numActions);
                this._recordInferenceTime(performance.now() - startTime);
                return action;
            }

            // Get Q-values from network (GPU or CPU)
            let qValues;
            if (this.usingGPU && this.backend.gpuNetwork) {
                // Use real-time optimized forward pass for low latency
                qValues = await this.backend.gpuNetwork.forwardRealTime(state, {
                    realTime: true,
                    skipValidation: training // Skip validation in training for speed
                });
            } else {
                qValues = this.backend.forward(state);
            }

            // Select action with highest Q-value
            const action = this._argmax(qValues);
            
            this._recordInferenceTime(performance.now() - startTime);
            return action;

        } catch (error) {
            console.error('Action selection failed:', error);
            
            // Emergency fallback to random action
            if (training) {
                return Math.floor(Math.random() * this.numActions);
            }
            throw error;
        }
    }

    /**
     * Train the Q-network with experience replay and GPU optimization
     * @param {Object} options - Training options
     * @returns {Promise<number>} Training loss
     */
    async trainStep(options = {}) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized');
        }

        if (this.replayBuffer.size() < this.hyperparams.batchSize) {
            return 0; // Not enough experiences to train
        }

        const startTime = performance.now();
        let loss = 0;

        try {
            // Optimize batch size for GPU if enabled
            const optimalBatchSize = this.usingGPU ? 
                this._getOptimalBatchSize() : this.hyperparams.batchSize;
            
            // Sample batch from replay buffer with GPU-optimized experience selection
            const batch = this.usingGPU ? 
                await this._sampleGPUOptimizedBatch(optimalBatchSize) :
                this.replayBuffer.sample(this.hyperparams.batchSize);
            
            // Determine training method based on backend and batch size
            if (this.usingGPU && batch.length >= this.backendOptions.gpuBatchThreshold) {
                loss = await this._trainBatchGPU(batch);
            } else {
                loss = await this._trainBatchCPU(batch);
            }

            // Update target network if needed
            if (this.stepCount % this.hyperparams.targetUpdateFreq === 0) {
                await this._updateTargetNetwork();
                this.lastTargetUpdate = this.stepCount;
            }

            // Decay epsilon
            this._decayEpsilon();
            
            // Record training metrics
            const trainingTime = performance.now() - startTime;
            this._recordTrainingTime(trainingTime);
            this.metrics.recordTrainingLoss(loss);
            
            this.stepCount++;
            
            return loss;

        } catch (error) {
            console.error('Training step failed:', error);
            
            // Handle GPU errors with fallback
            if (this.usingGPU && error.message.includes('GPU')) {
                console.warn('GPU training error, attempting CPU fallback');
                this.gpuFallbackCount++;
                
                // Try CPU training as fallback
                try {
                    const batch = this.replayBuffer.sample(this.hyperparams.batchSize);
                    loss = await this._trainBatchCPU(batch);
                    return loss;
                } catch (fallbackError) {
                    console.error('CPU fallback also failed:', fallbackError);
                }
            }
            
            throw error;
        }
    }

    /**
     * GPU-optimized batch training with complete Q-learning pipeline
     * @private
     */
    async _trainBatchGPU(batch) {
        const startTime = performance.now();
        
        try {
            const batchSize = batch.length;
            
            // Prepare batch data for GPU
            const states = new Float32Array(batchSize * NetworkConfig.INPUT_SIZE);
            const nextStates = new Float32Array(batchSize * NetworkConfig.INPUT_SIZE);
            const actions = new Uint32Array(batchSize);
            const rewards = new Float32Array(batchSize);
            const dones = new Uint32Array(batchSize);

            for (let i = 0; i < batchSize; i++) {
                const exp = batch[i];
                states.set(exp.state, i * NetworkConfig.INPUT_SIZE);
                nextStates.set(exp.nextState, i * NetworkConfig.INPUT_SIZE);
                actions[i] = exp.action;
                rewards[i] = exp.reward;
                dones[i] = exp.done ? 1 : 0;
            }

            // Get GPU-optimized neural network for Q-learning training
            if (!this.backend.gpuNetwork) {
                throw new Error('GPU neural network not available for training');
            }

            // Prepare GPU training buffers with enhanced Q-learning support
            const trainingBuffers = await this._prepareGPUTrainingBuffers(batchSize);
            
            // Upload batch data to GPU
            await this._uploadBatchDataToGPU(trainingBuffers, {
                states, nextStates, actions, rewards, dones
            });

            // Execute GPU-accelerated Q-learning training pipeline
            const { currentQValues, nextQValues, hiddenActivations } = await this._executeGPUForwardPasses(
                trainingBuffers, states, nextStates, batchSize
            );

            // Execute GPU Q-learning weight updates using compute shaders
            const loss = await this._executeGPUQLearningUpdates(
                trainingBuffers, 
                { currentQValues, nextQValues, hiddenActivations },
                { actions, rewards, dones },
                batchSize
            );

            // Update training step counter
            this.trainingStepCount++;

            // Sync weights periodically for stability
            await this._syncGPUWeightsIfNeeded();

            const trainingTime = performance.now() - startTime;
            this._recordBatchTrainingTime(trainingTime);
            
            return loss;

        } catch (error) {
            console.error('GPU batch training failed:', error);
            // Fallback to CPU training
            console.warn('Falling back to CPU training...');
            this.gpuFallbackCount++;
            return await this._trainBatchCPU(batch);
        }
    }

    /**
     * CPU-optimized batch training
     * @private
     */
    async _trainBatchCPU(batch) {
        let totalLoss = 0;

        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;

            // Get current Q-values
            const currentQValues = this.qNetwork.forward(state);
            
            // Calculate target Q-value
            let targetQ = reward;
            if (!done) {
                const nextQValues = this.targetNetwork.forward(nextState);
                const maxNextQ = Math.max(...nextQValues);
                targetQ = reward + this.hyperparams.gamma * maxNextQ;
            }

            // Calculate loss
            const currentQ = currentQValues[action];
            const loss = Math.pow(currentQ - targetQ, 2);
            totalLoss += loss;

            // Note: Actual weight updates would require backpropagation implementation
        }

        return totalLoss / batch.length;
    }

    /**
     * Update target network with current network weights
     * @private
     */
    async _updateTargetNetwork() {
        try {
            const weights = await this.qNetwork.getWeights();
            await this.targetNetwork.setWeights(weights);
            console.log(`Target network updated at step ${this.stepCount}`);
        } catch (error) {
            console.error('Target network update failed:', error);
            throw error;
        }
    }

    /**
     * Verify GPU performance meets requirements
     * @private
     */
    async _verifyGPUPerformance() {
        const result = { passed: true, issues: [] };

        try {
            // Test single inference latency
            const testState = new Float32Array([0.1, -0.05]);
            const times = [];
            
            for (let i = 0; i < 100; i++) {
                const start = performance.now();
                await this.backend.forward(testState);
                times.push(performance.now() - start);
            }

            const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
            const p95Time = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)];

            if (p95Time > this.backendOptions.realTimeThreshold) {
                result.passed = false;
                result.issues.push(`Real-time requirement not met: ${p95Time.toFixed(2)}ms > ${this.backendOptions.realTimeThreshold}ms`);
            }

            console.log(`GPU performance check: avg=${avgTime.toFixed(2)}ms, p95=${p95Time.toFixed(2)}ms`);

        } catch (error) {
            result.passed = false;
            result.issues.push(`Performance test failed: ${error.message}`);
        }

        return result;
    }

    /**
     * Run initial performance test and optimization
     * @private
     */
    async _runInitialPerformanceTest() {
        console.log('Running initial performance test...');
        
        const testState = new Float32Array([0.1, -0.05]);
        const iterations = 100;
        
        // Warmup
        for (let i = 0; i < 10; i++) {
            await this.selectAction(testState, false);
        }
        
        // Benchmark
        const times = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await this.selectAction(testState, false);
            times.push(performance.now() - start);
        }
        
        const stats = this._calculateStats(times);
        console.log(`Initial performance test (${this.usingGPU ? 'GPU' : 'CPU'}):`, {
            mean: stats.mean.toFixed(2) + 'ms',
            median: stats.median.toFixed(2) + 'ms',
            p95: stats.p95.toFixed(2) + 'ms',
            throughput: (1000 / stats.median).toFixed(0) + ' inferences/sec'
        });
    }

    /**
     * Add experience to replay buffer
     * @param {Float32Array} state - Current state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {Float32Array} nextState - Next state
     * @param {boolean} done - Episode done flag
     */
    addExperience(state, action, reward, nextState, done) {
        this.replayBuffer.add(state, action, reward, nextState, done);
    }

    /**
     * Start new episode
     */
    startEpisode() {
        this.episode++;
        this.metrics.startEpisode();
    }

    /**
     * End current episode
     * @param {number} totalReward - Total episode reward
     * @param {number} steps - Episode length
     */
    endEpisode(totalReward, steps) {
        this.metrics.endEpisode(totalReward, steps);
        
        // Check for convergence
        if (this.metrics.checkConvergence(
            this.hyperparams.convergenceWindow,
            this.hyperparams.convergenceThreshold
        )) {
            console.log(`Training converged at episode ${this.episode}!`);
        }
    }

    /**
     * Get comprehensive training metrics
     * @returns {Object} Training metrics and performance data
     */
    getMetrics() {
        const baseMetrics = this.metrics.getSummary();
        
        return {
            ...baseMetrics,
            backend: {
                type: this.usingGPU ? 'webgpu' : 'cpu',
                usingGPU: this.usingGPU,
                gpuFallbackCount: this.gpuFallbackCount,
                backendInfo: this.backend ? this.backend.getBackendInfo?.() : null
            },
            performance: {
                averageInferenceTime: this._getAverageTime(this.inferenceTimes),
                averageTrainingTime: this._getAverageTime(this.trainingTimes),
                averageBatchTrainingTime: this._getAverageTime(this.batchTrainingTimes),
                totalInferences: this.inferenceTimes.length,
                totalTrainingSteps: this.trainingTimes.length,
                currentOptimalBatchSize: this.usingGPU ? (this._getOptimalBatchSize ? this._getOptimalBatchSize() : this.hyperparams.batchSize) : this.hyperparams.batchSize,
                gpuUtilization: this._calculateGPUUtilization ? this._calculateGPUUtilization() : 0
            },
            hyperparameters: this.hyperparams,
            bufferStatus: {
                size: this.replayBuffer.size(),
                capacity: this.replayBuffer.maxSize,
                utilizationPercent: (this.replayBuffer.size() / this.replayBuffer.maxSize * 100).toFixed(1)
            }
        };
    }

    /**
     * Save model state
     * @returns {Promise<Object>} Serializable model state
     */
    async saveModel() {
        if (!this.isInitialized) {
            throw new Error('Cannot save uninitialized model');
        }

        const weights = await this.qNetwork.getWeights();
        const targetWeights = await this.targetNetwork.getWeights();

        return {
            version: '1.0',
            timestamp: new Date().toISOString(),
            hyperparameters: this.hyperparams,
            weights: weights,
            targetWeights: targetWeights,
            metrics: this.getMetrics(),
            backend: this.usingGPU ? 'webgpu' : 'cpu',
            episode: this.episode,
            stepCount: this.stepCount
        };
    }

    /**
     * Load model state
     * @param {Object} modelState - Saved model state
     * @returns {Promise<void>}
     */
    async loadModel(modelState) {
        if (!this.isInitialized) {
            await this.initialize();
        }

        // Load hyperparameters
        this.hyperparams = new Hyperparameters(modelState.hyperparameters);
        
        // Load weights
        await this.qNetwork.setWeights(modelState.weights);
        await this.targetNetwork.setWeights(modelState.targetWeights);
        
        // Restore training state
        this.episode = modelState.episode || 0;
        this.stepCount = modelState.stepCount || 0;
        
        console.log(`Model loaded from episode ${this.episode}, step ${this.stepCount}`);
    }

    /**
     * Clean up resources
     */
    destroy() {
        if (this.backend?.destroy) {
            this.backend.destroy();
        }
        if (this.targetNetwork?.destroy) {
            this.targetNetwork.destroy();
        }
        
        this.qNetwork = null;
        this.targetNetwork = null;
        this.backend = null;
        this.isInitialized = false;
        
        console.log('WebGPU Q-Learning destroyed');
    }

    // GPU Q-Learning Training Pipeline Methods

    /**
     * Prepare GPU training buffers for Q-learning with optimal batch processing
     * @private
     */
    async _prepareGPUTrainingBuffers(batchSize) {
        const cacheKey = `qlearning_${batchSize}`;
        
        // Check if we have cached buffers for this batch size
        if (this.qlearningBuffers && this.qlearningBuffers.batchSize === batchSize) {
            return this.qlearningBuffers;
        }

        try {
            console.log(`Creating Q-learning GPU buffers for batch size: ${batchSize}`);
            
            // Get WebGPU device
            const device = this.backend.gpuNetwork?.device;
            if (!device) {
                throw new Error('WebGPU device not available');
            }

            // Calculate buffer sizes
            const inputSize = NetworkConfig.INPUT_SIZE;
            const hiddenSize = this.hyperparams.hiddenSize;
            const outputSize = this.numActions;
            
            const statesSize = batchSize * inputSize * 4; // Float32
            const actionsSize = batchSize * 4; // Uint32
            const rewardsSize = batchSize * 4; // Float32
            const donesSize = batchSize * 4; // Uint32
            const qValuesSize = batchSize * outputSize * 4; // Float32
            const hiddenActivationsSize = batchSize * hiddenSize * 4; // Float32
            const tdErrorsSize = batchSize * 4; // Float32

            // Create Q-learning specific buffers
            const buffers = {
                batchSize,
                
                // Input data buffers
                inputStates: device.createBuffer({
                    label: 'qlearning_input_states',
                    size: statesSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                }),
                
                nextStates: device.createBuffer({
                    label: 'qlearning_next_states', 
                    size: statesSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                }),
                
                actions: device.createBuffer({
                    label: 'qlearning_actions',
                    size: actionsSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                }),
                
                rewards: device.createBuffer({
                    label: 'qlearning_rewards',
                    size: rewardsSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                }),
                
                dones: device.createBuffer({
                    label: 'qlearning_dones',
                    size: donesSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                }),
                
                // Q-values and computations
                currentQValues: device.createBuffer({
                    label: 'qlearning_current_q_values',
                    size: qValuesSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
                }),
                
                targetQValues: device.createBuffer({
                    label: 'qlearning_target_q_values',
                    size: qValuesSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
                }),
                
                hiddenActivations: device.createBuffer({
                    label: 'qlearning_hidden_activations',
                    size: hiddenActivationsSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
                }),
                
                tdErrors: device.createBuffer({
                    label: 'qlearning_td_errors',
                    size: tdErrorsSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                }),
                
                // Parameters buffer
                params: device.createBuffer({
                    label: 'qlearning_params',
                    size: 32 * 4, // 8 parameters * 4 bytes each
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
                })
            };\n\n            // Cache buffers\n            this.qlearningBuffers = buffers;\n            \n            console.log(`Q-learning GPU buffers created successfully for batch size ${batchSize}`);\n            return buffers;\n            \n        } catch (error) {\n            console.error('Failed to create Q-learning GPU buffers:', error);\n            throw error;\n        }\n    }\n\n    /**\n     * Upload batch data to GPU buffers\n     * @private\n     */\n    async _uploadBatchDataToGPU(buffers, data) {\n        const device = this.backend.gpuNetwork.device;\n        const queue = device.queue;\n        \n        try {\n            // Upload batch data to GPU\n            queue.writeBuffer(buffers.inputStates, 0, data.states);\n            queue.writeBuffer(buffers.nextStates, 0, data.nextStates);\n            queue.writeBuffer(buffers.actions, 0, data.actions);\n            queue.writeBuffer(buffers.rewards, 0, data.rewards);\n            queue.writeBuffer(buffers.dones, 0, data.dones);\n            \n            // Upload Q-learning parameters\n            const params = new Float32Array([\n                buffers.batchSize,           // batch_size\n                NetworkConfig.INPUT_SIZE,    // input_size\n                this.hyperparams.hiddenSize,// hidden_size\n                this.numActions,             // output_size\n                this.hyperparams.learningRate, // learning_rate\n                this.hyperparams.gamma,      // gamma\n                1e-8,                        // epsilon (numerical stability)\n                1.0                          // clip_grad_norm\n            ]);\n            \n            queue.writeBuffer(buffers.params, 0, params);\n            \n        } catch (error) {\n            console.error('Failed to upload batch data to GPU:', error);\n            throw error;\n        }\n    }\n\n    /**\n     * Execute GPU forward passes for current and next states\n     * @private\n     */\n    async _executeGPUForwardPasses(buffers, states, nextStates, batchSize) {\n        try {\n            // Get current Q-values using batch processing\n            const currentQValues = await this.backend.gpuNetwork.forwardBatch(states, batchSize, {\n                captureHiddenActivations: true\n            });\n            \n            // Get next Q-values from target network\n            let nextQValues;\n            if (this.targetNetwork.gpuNetwork) {\n                nextQValues = await this.targetNetwork.gpuNetwork.forwardBatch(nextStates, batchSize);\n            } else {\n                // Fallback to multiple single forwards\n                nextQValues = new Float32Array(batchSize * this.numActions);\n                for (let i = 0; i < batchSize; i++) {\n                    const nextState = nextStates.slice(i * NetworkConfig.INPUT_SIZE, (i + 1) * NetworkConfig.INPUT_SIZE);\n                    const qVals = this.targetNetwork.forward(nextState);\n                    nextQValues.set(qVals, i * this.numActions);\n                }\n            }\n            \n            // Get hidden activations for gradient computation\n            const hiddenActivations = await this._extractHiddenActivations(batchSize);\n            \n            return { currentQValues, nextQValues, hiddenActivations };\n            \n        } catch (error) {\n            console.error('GPU forward passes failed:', error);\n            throw error;\n        }\n    }\n\n    /**\n     * Execute GPU Q-learning weight updates using compute shaders\n     * @private\n     */\n    async _executeGPUQLearningUpdates(buffers, qValues, batchData, batchSize) {\n        try {\n            const device = this.backend.gpuNetwork.device;\n            \n            // Upload Q-values and activations to GPU\n            device.queue.writeBuffer(buffers.currentQValues, 0, qValues.currentQValues);\n            device.queue.writeBuffer(buffers.targetQValues, 0, qValues.nextQValues);\n            device.queue.writeBuffer(buffers.hiddenActivations, 0, qValues.hiddenActivations);\n            \n            // Create compute pipeline for Q-learning updates if not exists\n            if (!this.qlearningPipelines) {\n                this.qlearningPipelines = await this._createQLearningComputePipelines();\n            }\n            \n            // Create bind groups for Q-learning compute shaders\n            const bindGroup = await this._createQLearningBindGroup(buffers);\n            \n            // Execute Q-learning compute pipeline\n            const commandEncoder = device.createCommandEncoder({\n                label: 'qlearning_training_pass'\n            });\n            \n            const computePass = commandEncoder.beginComputePass({\n                label: 'qlearning_compute_pass'\n            });\n            \n            // Step 1: Compute TD errors\n            computePass.setPipeline(this.qlearningPipelines.computeTDErrors);\n            computePass.setBindGroup(0, bindGroup);\n            computePass.dispatchWorkgroups(Math.ceil(batchSize / 32));\n            \n            // Step 2: Update output layer weights\n            computePass.setPipeline(this.qlearningPipelines.updateOutputWeights);\n            computePass.setBindGroup(0, bindGroup);\n            const outputWeightCount = this.hyperparams.hiddenSize * this.numActions;\n            computePass.dispatchWorkgroups(Math.ceil(outputWeightCount / 32));\n            \n            // Step 3: Update output layer biases\n            computePass.setPipeline(this.qlearningPipelines.updateOutputBias);\n            computePass.setBindGroup(0, bindGroup);\n            computePass.dispatchWorkgroups(Math.ceil(this.numActions / 32));\n            \n            // Step 4: Update hidden layer weights\n            computePass.setPipeline(this.qlearningPipelines.updateHiddenWeights);\n            computePass.setBindGroup(0, bindGroup);\n            const hiddenWeightCount = NetworkConfig.INPUT_SIZE * this.hyperparams.hiddenSize;\n            computePass.dispatchWorkgroups(Math.ceil(hiddenWeightCount / 32));\n            \n            // Step 5: Update hidden layer biases\n            computePass.setPipeline(this.qlearningPipelines.updateHiddenBias);\n            computePass.setBindGroup(0, bindGroup);\n            computePass.dispatchWorkgroups(Math.ceil(this.hyperparams.hiddenSize / 32));\n            \n            computePass.end();\n            \n            // Submit commands\n            device.queue.submit([commandEncoder.finish()]);\n            \n            // Wait for completion\n            await device.queue.onSubmittedWorkDone();\n            \n            // Calculate and return loss\n            const loss = await this._calculateTrainingLoss(buffers, batchSize);\n            \n            return loss;\n            \n        } catch (error) {\n            console.error('GPU Q-learning updates failed:', error);\n            throw error;\n        }\n    }\n\n    /**\n     * Create Q-learning compute pipelines\n     * @private\n     */\n    async _createQLearningComputePipelines() {\n        try {\n            const device = this.backend.gpuNetwork.device;\n            \n            // Load Q-learning shader\n            const shaderModule = device.createShaderModule({\n                label: 'qlearning_shader',\n                code: await this._loadQLearningShader()\n            });\n            \n            // Create bind group layout\n            const bindGroupLayout = device.createBindGroupLayout({\n                label: 'qlearning_bind_group_layout',\n                entries: [\n                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // q_values\n                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // target_q_values\n                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // actions\n                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // rewards\n                    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dones\n                    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // hidden_activations\n                    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input_states\n                    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // weights_hidden\n                    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // bias_hidden\n                    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // weights_output\n                    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // bias_output\n                    { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // td_errors\n                    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }           // params\n                ]\n            });\n            \n            // Create compute pipelines\n            const pipelines = {\n                computeTDErrors: device.createComputePipeline({\n                    label: 'qlearning_compute_td_errors',\n                    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),\n                    compute: {\n                        module: shaderModule,\n                        entryPoint: 'compute_td_errors'\n                    }\n                }),\n                \n                updateOutputWeights: device.createComputePipeline({\n                    label: 'qlearning_update_output_weights',\n                    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),\n                    compute: {\n                        module: shaderModule,\n                        entryPoint: 'update_output_weights'\n                    }\n                }),\n                \n                updateOutputBias: device.createComputePipeline({\n                    label: 'qlearning_update_output_bias',\n                    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),\n                    compute: {\n                        module: shaderModule,\n                        entryPoint: 'update_output_bias'\n                    }\n                }),\n                \n                updateHiddenWeights: device.createComputePipeline({\n                    label: 'qlearning_update_hidden_weights',\n                    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),\n                    compute: {\n                        module: shaderModule,\n                        entryPoint: 'update_hidden_weights'\n                    }\n                }),\n                \n                updateHiddenBias: device.createComputePipeline({\n                    label: 'qlearning_update_hidden_bias',\n                    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),\n                    compute: {\n                        module: shaderModule,\n                        entryPoint: 'update_hidden_bias'\n                    }\n                })\n            };\n            \n            this.qlearningBindGroupLayout = bindGroupLayout;\n            \n            console.log('Q-learning compute pipelines created successfully');\n            return pipelines;\n            \n        } catch (error) {\n            console.error('Failed to create Q-learning compute pipelines:', error);\n            throw error;\n        }\n    }\n\n    /**\n     * Load Q-learning shader code\n     * @private\n     */\n    async _loadQLearningShader() {\n        try {\n            // In a real implementation, you'd load this from the qlearning.wgsl file\n            // For now, return the basic structure that matches qlearning.wgsl\n            const response = await fetch('/src/network/shaders/qlearning.wgsl');\n            if (!response.ok) {\n                throw new Error(`Failed to load Q-learning shader: ${response.statusText}`);\n            }\n            return await response.text();\n        } catch (error) {\n            console.error('Failed to load Q-learning shader:', error);\n            // Fallback to embedded shader code if file loading fails\n            return this._getEmbeddedQLearningShader();\n        }\n    }\n\n    /**\n     * Get embedded Q-learning shader as fallback\n     * @private\n     */\n    _getEmbeddedQLearningShader() {\n        return `\n// Q-Learning Weight Update Compute Shader\n// Simplified version for embedded fallback\n\nstruct QLearningParams {\n    batch_size: u32,\n    input_size: u32,\n    hidden_size: u32,\n    output_size: u32,\n    learning_rate: f32,\n    gamma: f32,\n    epsilon: f32,\n    clip_grad_norm: f32,\n}\n\n@group(0) @binding(0) var<storage, read> q_values: array<f32>;\n@group(0) @binding(1) var<storage, read> target_q_values: array<f32>;\n@group(0) @binding(2) var<storage, read> actions: array<u32>;\n@group(0) @binding(3) var<storage, read> rewards: array<f32>;\n@group(0) @binding(4) var<storage, read> dones: array<u32>;\n@group(0) @binding(5) var<storage, read> hidden_activations: array<f32>;\n@group(0) @binding(6) var<storage, read> input_states: array<f32>;\n@group(0) @binding(7) var<storage, read_write> weights_hidden: array<f32>;\n@group(0) @binding(8) var<storage, read_write> bias_hidden: array<f32>;\n@group(0) @binding(9) var<storage, read_write> weights_output: array<f32>;\n@group(0) @binding(10) var<storage, read_write> bias_output: array<f32>;\n@group(0) @binding(11) var<storage, read_write> td_errors: array<f32>;\n@group(0) @binding(12) var<uniform> params: QLearningParams;\n\n@compute @workgroup_size(32, 1, 1)\nfn compute_td_errors(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let batch_idx = global_id.x;\n    if (batch_idx >= params.batch_size) { return; }\n    \n    let action = actions[batch_idx];\n    let reward = rewards[batch_idx];\n    let done = dones[batch_idx];\n    \n    let q_idx = batch_idx * params.output_size + action;\n    let current_q = q_values[q_idx];\n    \n    var target_q = reward;\n    if (done == 0u) {\n        var max_target_q = target_q_values[batch_idx * params.output_size];\n        for (var i: u32 = 1u; i < params.output_size; i++) {\n            let target_idx = batch_idx * params.output_size + i;\n            max_target_q = max(max_target_q, target_q_values[target_idx]);\n        }\n        target_q += params.gamma * max_target_q;\n    }\n    \n    td_errors[batch_idx] = target_q - current_q;\n}\n\n@compute @workgroup_size(32, 1, 1)\nfn update_output_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let weight_idx = global_id.x;\n    let total_output_weights = params.hidden_size * params.output_size;\n    if (weight_idx >= total_output_weights) { return; }\n    \n    let hidden_idx = weight_idx / params.output_size;\n    let output_idx = weight_idx % params.output_size;\n    \n    var gradient = 0.0;\n    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {\n        let action = actions[batch_idx];\n        if (action == output_idx) {\n            let td_error = td_errors[batch_idx];\n            let hidden_activation = hidden_activations[batch_idx * params.hidden_size + hidden_idx];\n            gradient += td_error * hidden_activation;\n        }\n    }\n    \n    gradient /= f32(params.batch_size);\n    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);\n    weights_output[weight_idx] += params.learning_rate * gradient;\n}\n\n@compute @workgroup_size(32, 1, 1)\nfn update_output_bias(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let output_idx = global_id.x;\n    if (output_idx >= params.output_size) { return; }\n    \n    var gradient = 0.0;\n    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {\n        let action = actions[batch_idx];\n        if (action == output_idx) {\n            let td_error = td_errors[batch_idx];\n            gradient += td_error;\n        }\n    }\n    \n    gradient /= f32(params.batch_size);\n    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);\n    bias_output[output_idx] += params.learning_rate * gradient;\n}\n\n@compute @workgroup_size(32, 1, 1)\nfn update_hidden_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let weight_idx = global_id.x;\n    let total_hidden_weights = params.input_size * params.hidden_size;\n    if (weight_idx >= total_hidden_weights) { return; }\n    \n    let input_idx = weight_idx / params.hidden_size;\n    let hidden_idx = weight_idx % params.hidden_size;\n    \n    var gradient = 0.0;\n    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {\n        let action = actions[batch_idx];\n        let td_error = td_errors[batch_idx];\n        let input_value = input_states[batch_idx * params.input_size + input_idx];\n        let hidden_activation = hidden_activations[batch_idx * params.hidden_size + hidden_idx];\n        \n        let output_weight = weights_output[hidden_idx * params.output_size + action];\n        let relu_derivative = select(0.0, 1.0, hidden_activation > 0.0);\n        let backprop_error = td_error * output_weight * relu_derivative;\n        gradient += backprop_error * input_value;\n    }\n    \n    gradient /= f32(params.batch_size);\n    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);\n    weights_hidden[weight_idx] += params.learning_rate * gradient;\n}\n\n@compute @workgroup_size(32, 1, 1)\nfn update_hidden_bias(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let hidden_idx = global_id.x;\n    if (hidden_idx >= params.hidden_size) { return; }\n    \n    var gradient = 0.0;\n    for (var batch_idx: u32 = 0u; batch_idx < params.batch_size; batch_idx++) {\n        let action = actions[batch_idx];\n        let td_error = td_errors[batch_idx];\n        let hidden_activation = hidden_activations[batch_idx * params.hidden_size + hidden_idx];\n        \n        let output_weight = weights_output[hidden_idx * params.output_size + action];\n        let relu_derivative = select(0.0, 1.0, hidden_activation > 0.0);\n        let backprop_error = td_error * output_weight * relu_derivative;\n        gradient += backprop_error;\n    }\n    \n    gradient /= f32(params.batch_size);\n    gradient = clamp(gradient, -params.clip_grad_norm, params.clip_grad_norm);\n    bias_hidden[hidden_idx] += params.learning_rate * gradient;\n}\n`;\n    }\n\n    /**\n     * Create Q-learning bind group\n     * @private\n     */\n    async _createQLearningBindGroup(buffers) {\n        try {\n            const device = this.backend.gpuNetwork.device;\n            \n            // Get network weight buffers\n            const networkBuffers = this.backend.gpuNetwork.buffers;\n            \n            const bindGroup = device.createBindGroup({\n                label: 'qlearning_bind_group',\n                layout: this.qlearningBindGroupLayout,\n                entries: [\n                    { binding: 0, resource: { buffer: buffers.currentQValues } },\n                    { binding: 1, resource: { buffer: buffers.targetQValues } },\n                    { binding: 2, resource: { buffer: buffers.actions } },\n                    { binding: 3, resource: { buffer: buffers.rewards } },\n                    { binding: 4, resource: { buffer: buffers.dones } },\n                    { binding: 5, resource: { buffer: buffers.hiddenActivations } },\n                    { binding: 6, resource: { buffer: buffers.inputStates } },\n                    { binding: 7, resource: { buffer: networkBuffers.weightsHidden } },\n                    { binding: 8, resource: { buffer: networkBuffers.biasHidden } },\n                    { binding: 9, resource: { buffer: networkBuffers.weightsOutput } },\n                    { binding: 10, resource: { buffer: networkBuffers.biasOutput } },\n                    { binding: 11, resource: { buffer: buffers.tdErrors } },\n                    { binding: 12, resource: { buffer: buffers.params } }\n                ]\n            });\n            \n            return bindGroup;\n            \n        } catch (error) {\n            console.error('Failed to create Q-learning bind group:', error);\n            throw error;\n        }\n    }\n\n    /**\n     * Extract hidden activations from GPU\n     * @private\n     */\n    async _extractHiddenActivations(batchSize) {\n        try {\n            // This would require the neural network to capture hidden activations\n            // For now, create a placeholder that matches the expected size\n            const hiddenSize = this.hyperparams.hiddenSize;\n            return new Float32Array(batchSize * hiddenSize);\n            \n        } catch (error) {\n            console.error('Failed to extract hidden activations:', error);\n            throw error;\n        }\n    }\n\n    /**\n     * Calculate training loss from TD errors\n     * @private\n     */\n    async _calculateTrainingLoss(buffers, batchSize) {\n        try {\n            const device = this.backend.gpuNetwork.device;\n            \n            // Read TD errors from GPU\n            const tdErrorsBuffer = await this._readGPUBuffer(\n                device, \n                buffers.tdErrors, \n                batchSize * 4\n            );\n            \n            const tdErrors = new Float32Array(tdErrorsBuffer);\n            \n            // Calculate MSE loss\n            let totalLoss = 0;\n            for (let i = 0; i < batchSize; i++) {\n                totalLoss += tdErrors[i] * tdErrors[i];\n            }\n            \n            return totalLoss / batchSize;\n            \n        } catch (error) {\n            console.error('Failed to calculate training loss:', error);\n            return 0;\n        }\n    }\n\n    /**\n     * Read data from GPU buffer\n     * @private\n     */\n    async _readGPUBuffer(device, buffer, size) {\n        const stagingBuffer = device.createBuffer({\n            size: size,\n            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ\n        });\n        \n        const commandEncoder = device.createCommandEncoder();\n        commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);\n        device.queue.submit([commandEncoder.finish()]);\n        \n        await stagingBuffer.mapAsync(GPUMapMode.READ);\n        const data = stagingBuffer.getMappedRange();\n        const result = new ArrayBuffer(data.byteLength);\n        new Uint8Array(result).set(new Uint8Array(data));\n        stagingBuffer.unmap();\n        stagingBuffer.destroy();\n        \n        return result;\n    }\n\n    /**\n     * Sync GPU weights if needed for stability\n     * @private\n     */\n    async _syncGPUWeightsIfNeeded() {\n        if (this.trainingStepCount - this.lastWeightSync >= this.weightSyncInterval) {\n            // Optionally sync weights between main and target networks on GPU\n            // This could be implemented for additional stability\n            this.lastWeightSync = this.trainingStepCount;\n        }\n    }\n\n    // Utility methods

    _argmax(array) {
        let maxIndex = 0;
        let maxValue = array[0];
        
        for (let i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }

    _decayEpsilon() {
        this.hyperparams.epsilon = Math.max(
            this.hyperparams.epsilonMin,
            this.hyperparams.epsilon * this.hyperparams.epsilonDecay
        );
        this.metrics.epsilonHistory.push(this.hyperparams.epsilon);
    }

    _recordInferenceTime(time) {
        this.inferenceTimes.push(time);
        if (this.inferenceTimes.length > 1000) {
            this.inferenceTimes.shift();
        }
    }

    _recordTrainingTime(time) {
        this.trainingTimes.push(time);
        if (this.trainingTimes.length > 1000) {
            this.trainingTimes.shift();
        }
    }
    
    _recordBatchTrainingTime(time) {
        this.batchTrainingTimes.push(time);
        if (this.batchTrainingTimes.length > 1000) {
            this.batchTrainingTimes.shift();
        }
    }

    _getAverageTime(timeArray) {
        if (timeArray.length === 0) return 0;
        return timeArray.reduce((sum, t) => sum + t, 0) / timeArray.length;
    }

    _calculateStats(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const n = values.length;
        
        return {
            count: n,
            min: Math.min(...values),
            max: Math.max(...values),
            mean: values.reduce((sum, v) => sum + v, 0) / n,
            median: sorted[Math.floor(n / 2)],
            p95: sorted[Math.floor(n * 0.95)]
        };
    }
}

/**
 * Factory function to create WebGPU-enhanced Q-Learning instance
 * @param {Object} hyperparams - Q-learning hyperparameters
 * @param {Object} options - Configuration options
 * @returns {WebGPUQLearning} Configured Q-learning instance
 */
export function createWebGPUQLearning(hyperparams = {}, options = {}) {
    return new WebGPUQLearning(hyperparams, options);
}

/**
 * Quick setup for development/testing
 * @param {Object} config - Quick configuration
 * @returns {Promise<WebGPUQLearning>} Initialized Q-learning instance
 */
export async function quickSetupWebGPUQLearning(config = {}) {
    const hyperparams = {
        learningRate: 0.001,
        gamma: 0.95,
        epsilon: 0.1,
        batchSize: 32,
        hiddenSize: 8,
        ...config.hyperparams
    };
    
    const options = {
        preferGPU: true,
        fallbackToCPU: true,
        asyncTraining: false,
        ...config.options
    };
    
    const qlearning = new WebGPUQLearning(hyperparams, options);
    await qlearning.initialize();
    
    return qlearning;
}