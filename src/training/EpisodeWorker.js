/**
 * Episode Worker for Parallel Training
 * 
 * Runs complete training episodes in a Web Worker with:
 * - Full physics simulation (BalancingRobot)
 * - Neural network inference (CPU backend)
 * - Experience collection and episode execution
 * 
 * This worker receives episode parameters and neural network weights,
 * runs complete episodes, and returns episode experiences and metrics.
 */

// Worker imports - we need to import modules for physics and neural network
// Note: ES6 imports in workers require module support
let BalancingRobot, RobotState;
let CPUBackend;
let vectorMatrixMultiply, relu, copyArray, heInit;

// Neural network instance for this worker
let neuralNetwork = null;
let workerInitialized = false;

/**
 * Initialize worker with required modules
 * This simulates the imports since Web Workers have limited module support
 */
async function initializeWorker() {
    if (workerInitialized) return;
    
    try {
        // In a real implementation, we'd import the actual modules
        // For now, we'll inline the essential functionality
        
        // Initialize physics simulation capability
        initializePhysics();
        
        // Initialize neural network capability
        initializeNeuralNetwork();
        
        workerInitialized = true;
        console.log('Episode worker initialized successfully');
        
    } catch (error) {
        console.error('Worker initialization failed:', error);
        throw error;
    }
}

/**
 * Initialize physics simulation (simplified for worker)
 */
function initializePhysics() {
    // Simplified RobotState class
    class RobotState {
        constructor(angle = 0, angularVelocity = 0, position = 0, velocity = 0, wheelAngle = 0, wheelVelocity = 0) {
            this.angle = angle;
            this.angularVelocity = angularVelocity;
            this.position = position;
            this.velocity = velocity;
            this.wheelAngle = wheelAngle;
            this.wheelVelocity = wheelVelocity;
        }
        
        clone() {
            return new RobotState(this.angle, this.angularVelocity, this.position, this.velocity, this.wheelAngle, this.wheelVelocity);
        }
        
        getNormalizedInputs(maxAngle = Math.PI / 3, timesteps = 1) {
            // For worker, we'll just repeat current state for all timesteps
            // This is a simplified implementation since workers don't maintain full history
            const normalizedAngle = Math.max(-1, Math.min(1, this.angle / maxAngle));
            const normalizedAngularVelocity = Math.max(-1, Math.min(1, this.angularVelocity / 10));
            
            const inputSize = timesteps * 2;
            const inputs = new Float32Array(inputSize);
            
            // Fill all timesteps with current state (simplified approach)
            for (let i = 0; i < timesteps; i++) {
                inputs[i * 2] = normalizedAngle;
                inputs[i * 2 + 1] = normalizedAngularVelocity;
            }
            
            return inputs;
        }
        
        hasFailed(maxAngle = Math.PI / 3) {
            return Math.abs(this.angle) > maxAngle;
        }
    }
    
    // Simplified BalancingRobot class for worker
    class BalancingRobot {
        constructor(config = {}) {
            this.mass = config.mass || 1.0;
            this.centerOfMassHeight = config.centerOfMassHeight || 0.4;
            this.motorStrength = config.motorStrength || 5.0;
            this.friction = config.friction || 0.02;
            this.damping = config.damping || 0.01;
            this.timestep = config.timestep || 0.006667; // Adjusted timestep
            this.rewardType = config.rewardType || 'simple';
            
            // Configurable angle and motor limits
            this.maxAngle = config.maxAngle || Math.PI / 6; // Default 30 degrees
            this.motorTorqueRange = config.motorTorqueRange || 8.0; // Default ±8.0 Nm
            
            this.gravity = 9.81;
            this.momentOfInertia = this.mass * this.centerOfMassHeight * this.centerOfMassHeight;
            
            this.state = new RobotState();
            this.currentMotorTorque = 0;
            this.stepCount = 0;
            this.totalReward = 0;
        }
        
        reset(initialState = {}) {
            this.state = new RobotState(
                this._normalizeAngle(initialState.angle || 0),
                initialState.angularVelocity || 0,
                initialState.position || 0,
                initialState.velocity || 0,
                initialState.wheelAngle || 0,
                initialState.wheelVelocity || 0
            );
            this.currentMotorTorque = 0;
            this.stepCount = 0;
            this.totalReward = 0;
        }
        
        step(motorTorque) {
            if (this.state.hasFailed(this.maxAngle)) {
                return {
                    state: this.state.clone(),
                    reward: this.rewardType === 'simple' ? 0.0 : -10.0,
                    done: true
                };
            }
            
            // Scale motor torque by the configurable range, then clamp to motor strength
            const scaledTorque = motorTorque * this.motorTorqueRange;
            this.currentMotorTorque = Math.max(-this.motorStrength, Math.min(this.motorStrength, scaledTorque));
            
            this._updatePhysics();
            this.state.angle = this._normalizeAngle(this.state.angle);
            
            const reward = this._calculateReward();
            this.totalReward += reward;
            
            const done = this.state.hasFailed(this.maxAngle);
            this.stepCount++;
            
            return {
                state: this.state.clone(),
                reward: reward,
                done: done
            };
        }
        
        _updatePhysics() {
            const dt = this.timestep;
            
            const gravityTorque = this.mass * this.gravity * this.centerOfMassHeight * Math.sin(this.state.angle);
            const dampingTorque = -this.damping * this.state.angularVelocity;
            const pendulumTorque = -this.currentMotorTorque + gravityTorque + dampingTorque;
            
            const angularAcceleration = pendulumTorque / this.momentOfInertia;
            this.state.angularVelocity += angularAcceleration * dt;
            this.state.angle += this.state.angularVelocity * dt;
            
            const motorForce = this.currentMotorTorque / 0.12; // wheelRadius
            const normalForce = this.mass * this.gravity;
            const frictionForce = -this.friction * this.state.velocity * normalForce;
            const totalForce = motorForce + frictionForce;
            
            const horizontalAcceleration = totalForce / this.mass;
            this.state.velocity += horizontalAcceleration * dt;
            this.state.position += this.state.velocity * dt;
            
            const distanceTraveled = this.state.velocity * dt;
            const wheelRotationChange = distanceTraveled / 0.12;
            this.state.wheelAngle += wheelRotationChange;
            
            while (this.state.wheelAngle > Math.PI * 2) this.state.wheelAngle -= Math.PI * 2;
            while (this.state.wheelAngle < -Math.PI * 2) this.state.wheelAngle += Math.PI * 2;
            
            this.state.wheelVelocity = this.state.velocity / 0.12;
        }
        
        _calculateReward() {
            if (this.rewardType === 'simple') {
                if (this.state.hasFailed(this.maxAngle)) {
                    return 0.0;
                }
                return 1.0;
            } else {
                if (this.state.hasFailed(this.maxAngle)) {
                    return -10.0;
                }
                const angleError = Math.abs(this.state.angle);
                return 1.0 - (angleError / this.maxAngle);
            }
        }
        
        _normalizeAngle(angle) {
            while (angle > Math.PI) angle -= 2 * Math.PI;
            while (angle < -Math.PI) angle += 2 * Math.PI;
            return angle;
        }
        
        getState() {
            return this.state.clone();
        }
        
        getNormalizedInputs(timesteps = 1) {
            return this.state.getNormalizedInputs(this.maxAngle, timesteps);
        }
        
        getStats() {
            return {
                stepCount: this.stepCount,
                totalReward: this.totalReward,
                currentMotorTorque: this.currentMotorTorque,
                simulationTime: this.stepCount * this.timestep
            };
        }
    }
    
    // Make classes available globally in worker
    globalThis.RobotState = RobotState;
    globalThis.BalancingRobot = BalancingRobot;
}

/**
 * Initialize neural network capability (simplified CPU backend)
 */
function initializeNeuralNetwork() {
    // Simple matrix operations
    function matrixMultiply(input, weights, bias, inputSize, outputSize) {
        const output = new Float32Array(outputSize);
        
        for (let i = 0; i < outputSize; i++) {
            let sum = bias[i];
            for (let j = 0; j < inputSize; j++) {
                sum += input[j] * weights[j * outputSize + i];
            }
            output[i] = sum;
        }
        
        return output;
    }
    
    function relu(input) {
        const output = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }
    
    // Simple neural network for worker with variable input size support
    class WorkerNeuralNetwork {
        constructor() {
            this.inputSize = 2;  // Will be updated when weights are set
            this.hiddenSize = 64;
            this.outputSize = 3;
            
            this.weightsInputHidden = null;
            this.biasHidden = null;
            this.weightsHiddenOutput = null;
            this.biasOutput = null;
            
            this.hiddenActivation = null;
            this.outputActivation = null;
        }
        
        setWeights(weights) {
            // Extract architecture from weights
            if (weights.architecture) {
                this.inputSize = weights.architecture.inputSize;
                this.hiddenSize = weights.architecture.hiddenSize;
                this.outputSize = weights.architecture.outputSize;
            }
            
            this.weightsInputHidden = new Float32Array(weights.weightsInputHidden);
            this.biasHidden = new Float32Array(weights.biasHidden);
            this.weightsHiddenOutput = new Float32Array(weights.weightsHiddenOutput);
            this.biasOutput = new Float32Array(weights.biasOutput);
            
            // Allocate activation arrays
            this.hiddenActivation = new Float32Array(this.hiddenSize);
            this.outputActivation = new Float32Array(this.outputSize);
        }
        
        forward(input) {
            // Input to hidden layer
            const hiddenLinear = matrixMultiply(
                input, 
                this.weightsInputHidden, 
                this.biasHidden,
                this.inputSize, 
                this.hiddenSize
            );
            
            // Apply ReLU activation
            const hiddenActivated = relu(hiddenLinear);
            
            // Hidden to output layer
            const output = matrixMultiply(
                hiddenActivated,
                this.weightsHiddenOutput,
                this.biasOutput,
                this.hiddenSize,
                this.outputSize
            );
            
            return output;
        }
    }
    
    // Make neural network available globally
    globalThis.WorkerNeuralNetwork = WorkerNeuralNetwork;
}

/**
 * Run a complete episode with given parameters
 */
function runEpisode(params) {
    const {
        episodeId,
        maxSteps,
        robotConfig,
        neuralNetworkWeights,
        epsilon,
        explorationEnabled,
        timesteps = 1
    } = params;
    
    // Create robot with provided configuration
    const robot = new globalThis.BalancingRobot(robotConfig);
    
    // Initialize robot state
    robot.reset({
        angle: (Math.random() - 0.5) * 0.2, // Small random initial angle
        angularVelocity: 0,
        position: 0,
        velocity: 0
    });
    
    // Update neural network weights
    if (!neuralNetwork) {
        neuralNetwork = new globalThis.WorkerNeuralNetwork();
    }
    neuralNetwork.setWeights(neuralNetworkWeights);
    
    // Episode execution
    const experiences = [];
    const actions = [-1.0, 0.0, 1.0]; // Left, brake, right
    let stepCount = 0;
    let totalReward = 0;
    
    let previousState = null;
    let previousAction = null;
    let previousReward = 0;
    
    while (stepCount < maxSteps) {
        const currentState = robot.getNormalizedInputs(timesteps);
        
        // Select action using neural network
        let actionIndex;
        if (explorationEnabled && Math.random() < epsilon) {
            // Exploration: random action
            actionIndex = Math.floor(Math.random() * 3);
        } else {
            // Exploitation: best action from neural network
            const qValues = neuralNetwork.forward(currentState);
            actionIndex = 0;
            for (let i = 1; i < qValues.length; i++) {
                if (qValues[i] > qValues[actionIndex]) {
                    actionIndex = i;
                }
            }
        }
        
        const motorTorque = actions[actionIndex];
        
        // Execute action
        const result = robot.step(motorTorque);
        
        // Store experience
        if (previousState !== null) {
            experiences.push({
                state: Array.from(previousState),
                action: previousAction,
                reward: previousReward,
                nextState: Array.from(currentState),
                done: result.done
            });
        }
        
        previousState = currentState;
        previousAction = actionIndex;
        previousReward = result.reward;
        totalReward += result.reward;
        stepCount++;
        
        if (result.done) {
            // Add final experience
            experiences.push({
                state: Array.from(currentState),
                action: actionIndex,
                reward: result.reward,
                nextState: Array.from(result.state.getNormalizedInputs(robot.maxAngle, timesteps)),
                done: true
            });
            break;
        }
    }
    
    return {
        episodeId: episodeId,
        totalReward: totalReward,
        stepCount: stepCount,
        experiences: experiences,
        completed: stepCount < maxSteps, // true if episode ended naturally
        finalState: robot.getState()
    };
}

/**
 * Worker message handler
 */
self.onmessage = async function(event) {
    const { type, taskId, ...params } = event.data;
    
    try {
        if (type === 'initialize') {
            await initializeWorker();
            self.postMessage({
                type: 'initialized',
                taskId: taskId,
                success: true
            });
            return;
        }
        
        if (type === 'run_episode') {
            if (!workerInitialized) {
                await initializeWorker();
            }
            
            const result = runEpisode(params);
            
            self.postMessage({
                type: 'episode_complete',
                taskId: taskId,
                result: result
            });
            return;
        }
        
        if (type === 'ping') {
            self.postMessage({
                type: 'pong',
                taskId: taskId,
                timestamp: Date.now()
            });
            return;
        }
        
    } catch (error) {
        self.postMessage({
            type: 'error',
            taskId: taskId,
            error: {
                message: error.message,
                stack: error.stack
            }
        });
    }
};