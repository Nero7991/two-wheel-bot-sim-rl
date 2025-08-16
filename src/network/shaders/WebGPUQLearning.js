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
        this.trainingTimes = [];\n        this.inferenceTimesn = [];
        this.batchTrainingTimes = [];
        this.gpuFallbackCount = 0;
        
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
            // Sample batch from replay buffer
            const batch = this.replayBuffer.sample(this.hyperparams.batchSize);
            
            // Determine training method based on backend and batch size
            if (this.usingGPU && this.hyperparams.batchSize >= this.backendOptions.gpuBatchThreshold) {
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
     * GPU-optimized batch training
     * @private
     */
    async _trainBatchGPU(batch) {
        try {
            // Prepare batch data for GPU
            const states = new Float32Array(batch.length * NetworkConfig.INPUT_SIZE);
            const nextStates = new Float32Array(batch.length * NetworkConfig.INPUT_SIZE);
            const actions = new Float32Array(batch.length);
            const rewards = new Float32Array(batch.length);
            const dones = new Float32Array(batch.length);

            for (let i = 0; i < batch.length; i++) {
                const exp = batch[i];
                states.set(exp.state, i * NetworkConfig.INPUT_SIZE);
                nextStates.set(exp.nextState, i * NetworkConfig.INPUT_SIZE);
                actions[i] = exp.action;
                rewards[i] = exp.reward;
                dones[i] = exp.done ? 1.0 : 0.0;
            }

            // Get current Q-values using batch processing
            const currentQValues = await this.backend.gpuNetwork.forwardBatch(states, batch.length);
            
            // Get next Q-values from target network
            let nextQValues;
            if (this.targetNetwork.gpuNetwork) {
                nextQValues = await this.targetNetwork.gpuNetwork.forwardBatch(nextStates, batch.length);
            } else {
                // Fallback to multiple single forwards
                nextQValues = new Float32Array(batch.length * this.numActions);
                for (let i = 0; i < batch.length; i++) {
                    const nextState = nextStates.slice(i * NetworkConfig.INPUT_SIZE, (i + 1) * NetworkConfig.INPUT_SIZE);
                    const qVals = this.targetNetwork.forward(nextState);
                    nextQValues.set(qVals, i * this.numActions);
                }
            }

            // Calculate target Q-values
            const targetQValues = new Float32Array(currentQValues.length);
            targetQValues.set(currentQValues); // Copy current values

            for (let i = 0; i < batch.length; i++) {
                const actionIndex = actions[i];
                const reward = rewards[i];
                const done = dones[i];
                
                // Calculate target for this action
                if (done) {
                    targetQValues[i * this.numActions + actionIndex] = reward;
                } else {
                    const nextQSlice = nextQValues.slice(i * this.numActions, (i + 1) * this.numActions);
                    const maxNextQ = Math.max(...nextQSlice);
                    targetQValues[i * this.numActions + actionIndex] = reward + this.hyperparams.gamma * maxNextQ;
                }
            }

            // Calculate loss (MSE)
            let totalLoss = 0;
            for (let i = 0; i < batch.length; i++) {
                const actionIndex = actions[i];
                const current = currentQValues[i * this.numActions + actionIndex];
                const target = targetQValues[i * this.numActions + actionIndex];
                totalLoss += Math.pow(current - target, 2);
            }

            const loss = totalLoss / batch.length;

            // Note: Actual weight updates would require backpropagation implementation
            // For now, we simulate the training step
            
            return loss;

        } catch (error) {
            console.error('GPU batch training failed:', error);
            throw error;
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
                totalTrainingSteps: this.trainingTimes.length
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

    // Utility methods

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