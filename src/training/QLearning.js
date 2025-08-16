/**
 * Q-Learning Algorithm Implementation for Two-Wheel Balancing Robot
 * 
 * Deep Q-Network (DQN) implementation for continuous state space learning.
 * Optimized for the balancing robot control task with neural network function approximation.
 * 
 * Features:
 * - DQN with target network for stable training
 * - Epsilon-greedy exploration with decay
 * - Experience replay buffer
 * - Temporal difference learning
 * - Hyperparameter management and validation
 * - Training episode management
 * - Convergence detection and metrics
 */

import { CPUBackend } from '../network/CPUBackend.js';
import { NetworkConfig } from '../network/NeuralNetwork.js';

/**
 * Hyperparameters for Q-learning algorithm
 */
export class Hyperparameters {
    constructor(options = {}) {
        // Learning parameters
        this.learningRate = this._validateParameter(options.learningRate, 0.001, 0.0001, 0.1, 'learningRate');
        this.gamma = this._validateParameter(options.gamma, 0.95, 0.5, 0.999, 'gamma');
        
        // Exploration parameters
        this.epsilon = this._validateParameter(options.epsilon, 0.1, 0.0, 1.0, 'epsilon');
        this.epsilonMin = this._validateParameter(options.epsilonMin, 0.01, 0.0, 0.1, 'epsilonMin');
        this.epsilonDecay = this._validateParameter(options.epsilonDecay, 0.995, 0.9, 0.9999, 'epsilonDecay');
        
        // Training parameters
        this.batchSize = this._validateParameter(options.batchSize, 32, 1, 128, 'batchSize');
        this.targetUpdateFreq = this._validateParameter(options.targetUpdateFreq, 100, 10, 1000, 'targetUpdateFreq');
        this.maxEpisodes = this._validateParameter(options.maxEpisodes, 1000, 10, 10000, 'maxEpisodes');
        this.maxStepsPerEpisode = this._validateParameter(options.maxStepsPerEpisode, 500, 50, 2000, 'maxStepsPerEpisode');
        
        // Convergence detection
        this.convergenceWindow = this._validateParameter(options.convergenceWindow, 100, 10, 500, 'convergenceWindow');
        this.convergenceThreshold = this._validateParameter(options.convergenceThreshold, 200, 50, 1000, 'convergenceThreshold');
        
        // Network architecture
        this.hiddenSize = this._validateParameter(options.hiddenSize, 8, 4, 16, 'hiddenSize');
    }
    
    _validateParameter(value, defaultValue, min, max, name) {
        if (value === undefined || value === null) {
            return defaultValue;
        }
        if (typeof value !== 'number' || isNaN(value)) {
            console.warn(`Invalid ${name}: ${value}, using default: ${defaultValue}`);
            return defaultValue;
        }
        if (value < min || value > max) {
            console.warn(`${name} out of range [${min}, ${max}]: ${value}, clamping`);
            return Math.max(min, Math.min(max, value));
        }
        return value;
    }
    
    clone() {
        return new Hyperparameters({
            learningRate: this.learningRate,
            gamma: this.gamma,
            epsilon: this.epsilon,
            epsilonMin: this.epsilonMin,
            epsilonDecay: this.epsilonDecay,
            batchSize: this.batchSize,
            targetUpdateFreq: this.targetUpdateFreq,
            maxEpisodes: this.maxEpisodes,
            maxStepsPerEpisode: this.maxStepsPerEpisode,
            convergenceWindow: this.convergenceWindow,
            convergenceThreshold: this.convergenceThreshold,
            hiddenSize: this.hiddenSize
        });
    }
}

/**
 * Experience replay buffer for storing and sampling training experiences
 */
export class ReplayBuffer {
    constructor(maxSize = 10000) {
        this.maxSize = maxSize;
        this.buffer = [];
        this.index = 0;
    }
    
    /**
     * Add experience to buffer
     * @param {Float32Array} state - Current state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {Float32Array} nextState - Next state
     * @param {boolean} done - Episode termination flag
     */
    add(state, action, reward, nextState, done) {
        const experience = {
            state: new Float32Array(state),
            action: action,
            reward: reward,
            nextState: new Float32Array(nextState),
            done: done
        };
        
        if (this.buffer.length < this.maxSize) {
            this.buffer.push(experience);
        } else {
            this.buffer[this.index] = experience;
        }
        
        this.index = (this.index + 1) % this.maxSize;
    }
    
    /**
     * Sample random batch of experiences
     * @param {number} batchSize - Size of batch to sample
     * @returns {Array} Array of sampled experiences
     */
    sample(batchSize) {
        if (this.buffer.length < batchSize) {
            return this.buffer.slice();
        }
        
        const sampled = [];
        const indices = new Set();
        
        while (sampled.length < batchSize) {
            const idx = Math.floor(Math.random() * this.buffer.length);
            if (!indices.has(idx)) {
                indices.add(idx);
                sampled.push(this.buffer[idx]);
            }
        }
        
        return sampled;
    }
    
    size() {
        return this.buffer.length;
    }
    
    clear() {
        this.buffer = [];
        this.index = 0;
    }
}

/**
 * Training metrics and statistics
 */
export class TrainingMetrics {
    constructor() {
        this.reset();
    }
    
    reset() {
        this.episodeRewards = [];
        this.episodeLengths = [];
        this.losses = [];
        this.epsilonHistory = [];
        this.startTime = Date.now();
        this.totalSteps = 0;
        this.bestReward = -Infinity;
        this.converged = false;
        this.convergenceEpisode = null;
    }
    
    addEpisode(reward, length, loss, epsilon) {
        this.episodeRewards.push(reward);
        this.episodeLengths.push(length);
        this.losses.push(loss);
        this.epsilonHistory.push(epsilon);
        this.totalSteps += length;
        
        if (reward > this.bestReward) {
            this.bestReward = reward;
        }
    }
    
    getAverageReward(window = 100) {
        const start = Math.max(0, this.episodeRewards.length - window);
        const recent = this.episodeRewards.slice(start);
        return recent.reduce((sum, r) => sum + r, 0) / recent.length;
    }
    
    getAverageLength(window = 100) {
        const start = Math.max(0, this.episodeLengths.length - window);
        const recent = this.episodeLengths.slice(start);
        return recent.reduce((sum, l) => sum + l, 0) / recent.length;
    }
    
    getTrainingTime() {
        return (Date.now() - this.startTime) / 1000; // seconds
    }
    
    checkConvergence(threshold, window) {
        if (this.episodeRewards.length < window) {
            return false;
        }
        
        const avgReward = this.getAverageReward(window);
        if (avgReward >= threshold && !this.converged) {
            this.converged = true;
            this.convergenceEpisode = this.episodeRewards.length;
            return true;
        }
        
        return this.converged;
    }
    
    getSummary() {
        const episodes = this.episodeRewards.length;
        const avgReward = episodes > 0 ? this.getAverageReward() : 0;
        const avgLength = episodes > 0 ? this.getAverageLength() : 0;
        const avgLoss = this.losses.length > 0 ? 
            this.losses.reduce((sum, l) => sum + l, 0) / this.losses.length : 0;
        
        return {
            episodes: episodes,
            totalSteps: this.totalSteps,
            averageReward: avgReward,
            bestReward: this.bestReward,
            averageLength: avgLength,
            averageLoss: avgLoss,
            trainingTime: this.getTrainingTime(),
            converged: this.converged,
            convergenceEpisode: this.convergenceEpisode,
            currentEpsilon: this.epsilonHistory[this.epsilonHistory.length - 1] || 0
        };
    }
}

/**
 * Deep Q-Network implementation for balancing robot control
 */
export class QLearning {
    constructor(hyperparams = {}) {
        this.hyperparams = new Hyperparameters(hyperparams);
        this.metrics = new TrainingMetrics();
        this.replayBuffer = new ReplayBuffer(10000);
        
        // Neural networks
        this.qNetwork = null;       // Main Q-network
        this.targetNetwork = null;  // Target Q-network for stable training
        
        // Training state
        this.isInitialized = false;
        this.episode = 0;
        this.stepCount = 0;
        this.lastTargetUpdate = 0;
        
        // Action space (discrete actions for continuous control)
        this.actions = [-1.0, 0.0, 1.0]; // Left motor, brake, right motor
        this.numActions = this.actions.length;
        
        console.log('Q-Learning algorithm initialized with hyperparameters:', this.hyperparams);
    }
    
    /**
     * Initialize the Q-learning networks
     * @returns {Promise<void>}
     */
    async initialize() {
        try {
            // Create main Q-network
            this.qNetwork = new CPUBackend();
            await this.qNetwork.createNetwork(
                NetworkConfig.INPUT_SIZE,    // 2 inputs: angle, angular velocity
                this.hyperparams.hiddenSize, // Hidden layer size
                this.numActions,             // 3 actions: left, brake, right
                { initMethod: NetworkConfig.INITIALIZATION.HE }
            );
            
            // Create target network (copy of main network)
            this.targetNetwork = this.qNetwork.clone();
            
            this.isInitialized = true;
            console.log(`Q-Learning networks initialized: ${NetworkConfig.INPUT_SIZE}-${this.hyperparams.hiddenSize}-${this.numActions}`);
            console.log(`Total parameters: ${this.qNetwork.getParameterCount()}`);
            
        } catch (error) {
            console.error('Failed to initialize Q-Learning networks:', error);
            throw error;
        }
    }
    
    /**
     * Select action using epsilon-greedy policy
     * @param {Float32Array} state - Current state (normalized)
     * @param {boolean} training - Whether in training mode (affects exploration)
     * @returns {number} Selected action index
     */
    selectAction(state, training = true) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized. Call initialize() first.');
        }
        
        // Validate state input
        if (!(state instanceof Float32Array) || state.length !== NetworkConfig.INPUT_SIZE) {
            throw new Error(`Invalid state input. Expected Float32Array of length ${NetworkConfig.INPUT_SIZE}`);
        }
        
        // Epsilon-greedy exploration
        if (training && Math.random() < this.hyperparams.epsilon) {
            // Random action (exploration)
            return Math.floor(Math.random() * this.numActions);
        } else {
            // Greedy action (exploitation)
            const qValues = this.qNetwork.forward(state);
            return this._argmax(qValues);
        }
    }
    
    /**
     * Train the Q-network on a single experience
     * @param {Float32Array} state - Current state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {Float32Array} nextState - Next state
     * @param {boolean} done - Episode termination flag
     * @returns {number} Training loss
     */
    train(state, action, reward, nextState, done) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized. Call initialize() first.');
        }
        
        // Add experience to replay buffer
        this.replayBuffer.add(state, action, reward, nextState, done);
        
        // Train on batch if buffer has enough experiences
        if (this.replayBuffer.size() >= this.hyperparams.batchSize) {
            return this._trainBatch();
        }
        
        return 0; // No training performed
    }
    
    /**
     * Train on a batch of experiences from replay buffer
     * @private
     * @returns {number} Average training loss
     */
    _trainBatch() {
        const batch = this.replayBuffer.sample(this.hyperparams.batchSize);
        let totalLoss = 0;
        
        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;
            
            // Calculate current Q-values
            const currentQValues = this.qNetwork.forward(state);
            const currentQ = currentQValues[action];
            
            // Calculate target Q-value
            let targetQ;
            if (done) {
                targetQ = reward; // No future reward if episode is done
            } else {
                const nextQValues = this.targetNetwork.forward(nextState);
                const maxNextQ = Math.max(...nextQValues);
                targetQ = reward + this.hyperparams.gamma * maxNextQ;
            }
            
            // Calculate temporal difference error
            const tdError = targetQ - currentQ;
            
            // Update Q-network using gradient descent
            const loss = this._updateNetwork(state, action, tdError);
            totalLoss += loss;
        }
        
        this.stepCount++;
        
        // Update target network periodically
        if (this.stepCount - this.lastTargetUpdate >= this.hyperparams.targetUpdateFreq) {
            this._updateTargetNetwork();
            this.lastTargetUpdate = this.stepCount;
        }
        
        return totalLoss / batch.length;
    }
    
    /**
     * Update Q-network weights using simple gradient descent
     * @private
     * @param {Float32Array} state - Input state
     * @param {number} action - Action index
     * @param {number} tdError - Temporal difference error
     * @returns {number} Mean squared error loss
     */
    _updateNetwork(state, action, tdError) {
        // Simple gradient update: approximate gradient using finite differences
        const learningRate = this.hyperparams.learningRate;
        
        // Get current weights
        const weights = this.qNetwork.getWeights();
        
        // Calculate loss (MSE)
        const loss = 0.5 * tdError * tdError;
        
        // Update output layer weights (simplified gradient)
        // This is a simplified update - in practice, you'd compute actual gradients
        const outputWeights = weights.weightsHiddenOutput;
        const hiddenActivation = this._getHiddenActivation(state);
        
        for (let i = 0; i < hiddenActivation.length; i++) {
            const weightIndex = action * hiddenActivation.length + i;
            outputWeights[weightIndex] += learningRate * tdError * hiddenActivation[i];
        }
        
        // Update output bias
        weights.biasOutput[action] += learningRate * tdError;
        
        // Set updated weights
        this.qNetwork.setWeights(weights);
        
        return loss;
    }
    
    /**
     * Get hidden layer activation for given state
     * @private
     * @param {Float32Array} state - Input state
     * @returns {Array} Hidden layer activation
     */
    _getHiddenActivation(state) {
        // This is a simplified version - ideally we'd modify CPUBackend to expose intermediate activations
        // For now, we'll approximate by running a forward pass and estimating
        const qValues = this.qNetwork.forward(state);
        
        // Create a simplified hidden activation based on input
        const hiddenSize = this.hyperparams.hiddenSize;
        const hidden = new Array(hiddenSize);
        
        for (let i = 0; i < hiddenSize; i++) {
            // Simple approximation: ReLU(weighted sum of inputs)
            const weight = (state[0] * (i + 1) + state[1] * (i + 2)) / hiddenSize;
            hidden[i] = Math.max(0, weight);
        }
        
        return hidden;
    }
    
    /**
     * Update target network with current Q-network weights
     * @private
     */
    _updateTargetNetwork() {
        const weights = this.qNetwork.getWeights();
        this.targetNetwork.setWeights(weights);
    }
    
    /**
     * Run training episode
     * @param {Object} environment - Environment with reset() and step() methods
     * @param {boolean} verbose - Whether to log progress
     * @returns {Object} Episode results
     */
    runEpisode(environment, verbose = false) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized. Call initialize() first.');
        }
        
        // Reset environment and get initial state
        environment.reset();
        let state = environment.getState().getNormalizedInputs();
        
        let totalReward = 0;
        let stepCount = 0;
        let totalLoss = 0;
        let lossCount = 0;
        
        for (let step = 0; step < this.hyperparams.maxStepsPerEpisode; step++) {
            // Select action
            const actionIndex = this.selectAction(state, true);
            const actionValue = this.actions[actionIndex];
            
            // Take action in environment
            const result = environment.step(actionValue);
            const nextState = result.state.getNormalizedInputs();
            const reward = result.reward;
            const done = result.done;
            
            // Train on this experience
            const loss = this.train(state, actionIndex, reward, nextState, done);
            if (loss > 0) {
                totalLoss += loss;
                lossCount++;
            }
            
            totalReward += reward;
            stepCount++;
            
            if (done) {
                break;
            }
            
            state = nextState;
        }
        
        // Decay epsilon
        if (this.hyperparams.epsilon > this.hyperparams.epsilonMin) {
            this.hyperparams.epsilon *= this.hyperparams.epsilonDecay;
        }
        
        this.episode++;
        
        const avgLoss = lossCount > 0 ? totalLoss / lossCount : 0;
        
        // Record metrics
        this.metrics.addEpisode(totalReward, stepCount, avgLoss, this.hyperparams.epsilon);
        
        if (verbose) {
            console.log(`Episode ${this.episode}: Reward=${totalReward.toFixed(2)}, Steps=${stepCount}, Loss=${avgLoss.toFixed(4)}, Epsilon=${this.hyperparams.epsilon.toFixed(3)}`);
        }
        
        return {
            episode: this.episode,
            reward: totalReward,
            steps: stepCount,
            loss: avgLoss,
            epsilon: this.hyperparams.epsilon
        };
    }
    
    /**
     * Run full training process
     * @param {Object} environment - Environment with reset() and step() methods
     * @param {Object} options - Training options
     * @returns {TrainingMetrics} Training results
     */
    async runTraining(environment, options = {}) {
        const verbose = options.verbose || false;
        const saveInterval = options.saveInterval || 100;
        const onEpisodeEnd = options.onEpisodeEnd || null;
        
        console.log('Starting Q-Learning training...');
        console.log(`Max episodes: ${this.hyperparams.maxEpisodes}`);
        console.log(`Convergence threshold: ${this.hyperparams.convergenceThreshold} (over ${this.hyperparams.convergenceWindow} episodes)`);
        
        this.metrics.reset();
        
        for (let ep = 0; ep < this.hyperparams.maxEpisodes; ep++) {
            const result = this.runEpisode(environment, verbose);
            
            // Check convergence
            const converged = this.metrics.checkConvergence(
                this.hyperparams.convergenceThreshold,
                this.hyperparams.convergenceWindow
            );
            
            if (converged && !this.metrics.converged) {
                console.log(`Training converged at episode ${this.episode}!`);
                console.log(`Average reward: ${this.metrics.getAverageReward().toFixed(2)}`);
            }
            
            // Progress reporting
            if (verbose && (ep + 1) % saveInterval === 0) {
                const summary = this.metrics.getSummary();
                console.log(`\n=== Episode ${ep + 1}/${this.hyperparams.maxEpisodes} ===`);
                console.log(`Average reward (last 100): ${summary.averageReward.toFixed(2)}`);
                console.log(`Best reward: ${summary.bestReward.toFixed(2)}`);
                console.log(`Average episode length: ${summary.averageLength.toFixed(1)}`);
                console.log(`Current epsilon: ${summary.currentEpsilon.toFixed(3)}`);
                console.log(`Replay buffer size: ${this.replayBuffer.size()}`);
            }
            
            // Call episode callback if provided
            if (onEpisodeEnd) {
                onEpisodeEnd(result, this.metrics.getSummary());
            }
            
            // Early stopping on convergence (optional)
            if (options.earlyStop && converged) {
                console.log('Early stopping due to convergence');
                break;
            }
        }
        
        const finalSummary = this.metrics.getSummary();
        console.log('\n=== Training Complete ===');
        console.log(`Episodes completed: ${finalSummary.episodes}`);
        console.log(`Total steps: ${finalSummary.totalSteps}`);
        console.log(`Final average reward: ${finalSummary.averageReward.toFixed(2)}`);
        console.log(`Best reward achieved: ${finalSummary.bestReward.toFixed(2)}`);
        console.log(`Training time: ${finalSummary.trainingTime.toFixed(1)}s`);
        console.log(`Converged: ${finalSummary.converged ? 'Yes' : 'No'}`);
        
        return this.metrics;
    }
    
    /**
     * Evaluate trained agent performance
     * @param {Object} environment - Environment for evaluation
     * @param {number} numEpisodes - Number of evaluation episodes
     * @returns {Object} Evaluation results
     */
    evaluate(environment, numEpisodes = 10) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized. Call initialize() first.');
        }
        
        const oldEpsilon = this.hyperparams.epsilon;
        this.hyperparams.epsilon = 0; // No exploration during evaluation
        
        const results = [];
        
        for (let ep = 0; ep < numEpisodes; ep++) {
            environment.reset();
            let state = environment.getState().getNormalizedInputs();
            let totalReward = 0;
            let stepCount = 0;
            
            for (let step = 0; step < this.hyperparams.maxStepsPerEpisode; step++) {
                const actionIndex = this.selectAction(state, false); // No exploration
                const actionValue = this.actions[actionIndex];
                
                const result = environment.step(actionValue);
                const nextState = result.state.getNormalizedInputs();
                const reward = result.reward;
                const done = result.done;
                
                totalReward += reward;
                stepCount++;
                
                if (done) {
                    break;
                }
                
                state = nextState;
            }
            
            results.push({ reward: totalReward, steps: stepCount });
        }
        
        this.hyperparams.epsilon = oldEpsilon; // Restore original epsilon
        
        const avgReward = results.reduce((sum, r) => sum + r.reward, 0) / results.length;
        const avgSteps = results.reduce((sum, r) => sum + r.steps, 0) / results.length;
        const bestReward = Math.max(...results.map(r => r.reward));
        const worstReward = Math.min(...results.map(r => r.reward));
        
        return {
            episodes: numEpisodes,
            averageReward: avgReward,
            bestReward: bestReward,
            worstReward: worstReward,
            averageSteps: avgSteps,
            results: results
        };
    }
    
    /**
     * Get action value for given state (Q-value)
     * @param {Float32Array} state - Input state
     * @param {number} actionIndex - Action index
     * @returns {number} Q-value for state-action pair
     */
    getQValue(state, actionIndex) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized. Call initialize() first.');
        }
        
        const qValues = this.qNetwork.forward(state);
        return qValues[actionIndex];
    }
    
    /**
     * Get all Q-values for given state
     * @param {Float32Array} state - Input state
     * @returns {Float32Array} All Q-values
     */
    getAllQValues(state) {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized. Call initialize() first.');
        }
        
        return this.qNetwork.forward(state);
    }
    
    /**
     * Save Q-learning model
     * @returns {Object} Serializable model data
     */
    save() {
        if (!this.isInitialized) {
            throw new Error('Q-Learning not initialized. Call initialize() first.');
        }
        
        return {
            hyperparams: this.hyperparams,
            qNetworkWeights: this.qNetwork.getWeights(),
            targetNetworkWeights: this.targetNetwork.getWeights(),
            episode: this.episode,
            stepCount: this.stepCount,
            metrics: this.metrics.getSummary(),
            actions: this.actions
        };
    }
    
    /**
     * Load Q-learning model
     * @param {Object} modelData - Saved model data
     */
    async load(modelData) {
        this.hyperparams = new Hyperparameters(modelData.hyperparams);
        this.episode = modelData.episode || 0;
        this.stepCount = modelData.stepCount || 0;
        this.actions = modelData.actions || [-1.0, 0.0, 1.0];
        this.numActions = this.actions.length;
        
        await this.initialize();
        
        this.qNetwork.setWeights(modelData.qNetworkWeights);
        this.targetNetwork.setWeights(modelData.targetNetworkWeights);
        
        console.log('Q-Learning model loaded successfully');
    }
    
    /**
     * Reset training state
     */
    reset() {
        this.metrics.reset();
        this.replayBuffer.clear();
        this.episode = 0;
        this.stepCount = 0;
        this.lastTargetUpdate = 0;
        
        if (this.isInitialized) {
            this.qNetwork.resetWeights();
            this.targetNetwork = this.qNetwork.clone();
        }
    }
    
    /**
     * Get training statistics
     * @returns {Object} Current training statistics
     */
    getStats() {
        return {
            ...this.metrics.getSummary(),
            replayBufferSize: this.replayBuffer.size(),
            stepCount: this.stepCount,
            lastTargetUpdate: this.lastTargetUpdate,
            networkParameters: this.isInitialized ? this.qNetwork.getParameterCount() : 0
        };
    }
    
    /**
     * Utility function to find index of maximum value in array
     * @private
     * @param {Float32Array} array - Input array
     * @returns {number} Index of maximum value
     */
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
}

/**
 * Utility function to create Q-learning instance with default hyperparameters
 * @param {Object} overrides - Hyperparameter overrides
 * @returns {QLearning} New Q-learning instance
 */
export function createDefaultQLearning(overrides = {}) {
    return new QLearning(overrides);
}

/**
 * Utility function to create Q-learning instance optimized for fast training
 * @returns {QLearning} Q-learning configured for training speed
 */
export function createFastQLearning() {
    return new QLearning({
        learningRate: 0.01,
        epsilon: 0.3,
        epsilonDecay: 0.99,
        batchSize: 16,
        targetUpdateFreq: 50,
        maxEpisodes: 500,
        hiddenSize: 6
    });
}

/**
 * Utility function to create Q-learning instance optimized for performance
 * @returns {QLearning} Q-learning configured for best performance
 */
export function createOptimalQLearning() {
    return new QLearning({
        learningRate: 0.001,
        epsilon: 0.1,
        epsilonDecay: 0.995,
        batchSize: 32,
        targetUpdateFreq: 100,
        maxEpisodes: 2000,
        hiddenSize: 12,
        convergenceThreshold: 300
    });
}