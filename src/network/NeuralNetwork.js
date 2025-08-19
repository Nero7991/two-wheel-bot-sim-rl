/**
 * Neural Network Interface for Two-Wheel Balancing Robot RL
 * 
 * This interface defines the standard API for neural network implementations
 * supporting both CPU and WebGPU backends for the balancing robot application.
 * 
 * Architecture:
 * - Input: Robot state (angle, angular velocity) - 2 values
 * - Hidden: 4-16 neurons with ReLU activation (configurable)
 * - Output: Action probabilities (left motor, right motor, brake) - 3 values
 * - Total parameters: <200 for embedded deployment
 */

/**
 * Abstract base class for neural network implementations
 */
export class NeuralNetwork {
    constructor() {
        if (this.constructor === NeuralNetwork) {
            throw new Error('NeuralNetwork is an abstract class and cannot be instantiated directly');
        }
    }

    /**
     * Create a neural network with specified architecture
     * @param {number} inputSize - Number of input neurons (2 for robot state)
     * @param {number} hiddenSize - Number of hidden neurons (4-16)
     * @param {number} outputSize - Number of output neurons (3 for actions)
     * @param {Object} options - Additional configuration options
     * @returns {Promise<void>} Promise that resolves when network is created
     */
    async createNetwork(inputSize, hiddenSize, outputSize, options = {}) {
        throw new Error('createNetwork method must be implemented by subclass');
    }

    /**
     * Perform forward pass through the network
     * @param {Float32Array} input - Input values (angle, angular velocity)
     * @returns {Promise<Float32Array>|Float32Array} Output probabilities for actions
     */
    forward(input) {
        throw new Error('forward method must be implemented by subclass');
    }

    /**
     * Get the total number of parameters in the network
     * @returns {number} Total parameter count
     */
    getParameterCount() {
        throw new Error('getParameterCount method must be implemented by subclass');
    }

    /**
     * Get network weights for serialization/export
     * @returns {Object} Object containing weights and biases
     */
    getWeights() {
        throw new Error('getWeights method must be implemented by subclass');
    }

    /**
     * Set network weights from serialized data
     * @param {Object} weights - Object containing weights and biases
     */
    setWeights(weights) {
        throw new Error('setWeights method must be implemented by subclass');
    }

    /**
     * Get network architecture information
     * @returns {Object} Architecture details
     */
    getArchitecture() {
        throw new Error('getArchitecture method must be implemented by subclass');
    }

    /**
     * Validate input dimensions
     * @param {Float32Array} input - Input to validate
     * @throws {Error} If input dimensions are invalid
     */
    validateInput(input) {
        if (!(input instanceof Float32Array)) {
            throw new Error('Input must be a Float32Array');
        }
        if (input.length !== this.inputSize) {
            throw new Error(`Input size mismatch. Expected ${this.inputSize}, got ${input.length}`);
        }
        
        // Check for NaN and infinite values in input
        for (let i = 0; i < input.length; i++) {
            if (!isFinite(input[i])) {
                console.warn(`Invalid input value at index ${i}: ${input[i]}, clamping to valid range`);
                input[i] = Math.max(-1, Math.min(1, isNaN(input[i]) ? 0 : input[i]));
            }
        }
    }

    /**
     * Validate output dimensions
     * @param {Float32Array} output - Output to validate
     * @throws {Error} If output dimensions are invalid
     */
    validateOutput(output) {
        if (!(output instanceof Float32Array)) {
            throw new Error('Output must be a Float32Array');
        }
        if (output.length !== this.outputSize) {
            throw new Error(`Output size mismatch. Expected ${this.outputSize}, got ${output.length}`);
        }
        
        // Check for NaN and infinite values
        for (let i = 0; i < output.length; i++) {
            if (!isFinite(output[i])) {
                console.warn(`Invalid output value at index ${i}: ${output[i]}, replacing with 0`);
                output[i] = 0;
            }
        }
    }
}

/**
 * Network configuration constants
 */
export const NetworkConfig = {
    // Architecture constraints
    INPUT_SIZE: 2,          // Base robot state: angle, angular velocity (can be 2-16 for multi-timestep)
    OUTPUT_SIZE: 3,         // Actions: left motor, right motor, brake
    MIN_HIDDEN_SIZE: 4,     // Minimum hidden layer size
    MAX_HIDDEN_SIZE: 256,   // Maximum hidden layer size (updated for DQN standards)
    MAX_PARAMETERS: 50000,  // Maximum total parameters (updated for DQN web training)
    
    // Activation functions
    ACTIVATION: {
        RELU: 'relu',
        LINEAR: 'linear'
    },
    
    // Weight initialization methods
    INITIALIZATION: {
        XAVIER: 'xavier',
        HE: 'he',
        RANDOM: 'random'
    }
};

/**
 * Utility function to calculate parameter count for given architecture
 * @param {number} inputSize - Number of input neurons
 * @param {number} hiddenSize - Number of hidden neurons
 * @param {number} outputSize - Number of output neurons
 * @returns {number} Total parameter count
 */
export function calculateParameterCount(inputSize, hiddenSize, outputSize) {
    // Input to hidden: weights + biases
    const inputToHidden = (inputSize * hiddenSize) + hiddenSize;
    
    // Hidden to output: weights + biases
    const hiddenToOutput = (hiddenSize * outputSize) + outputSize;
    
    return inputToHidden + hiddenToOutput;
}

/**
 * Validate network architecture constraints (legacy - fixed input size)
 * @param {number} inputSize - Number of input neurons
 * @param {number} hiddenSize - Number of hidden neurons
 * @param {number} outputSize - Number of output neurons
 * @throws {Error} If architecture is invalid
 * @deprecated Use validateVariableArchitecture for multi-timestep support
 */
export function validateArchitecture(inputSize, hiddenSize, outputSize) {
    if (inputSize !== NetworkConfig.INPUT_SIZE) {
        throw new Error(`Input size must be ${NetworkConfig.INPUT_SIZE}, got ${inputSize}`);
    }
    
    if (outputSize !== NetworkConfig.OUTPUT_SIZE) {
        throw new Error(`Output size must be ${NetworkConfig.OUTPUT_SIZE}, got ${outputSize}`);
    }
    
    if (hiddenSize < NetworkConfig.MIN_HIDDEN_SIZE || hiddenSize > NetworkConfig.MAX_HIDDEN_SIZE) {
        throw new Error(`Hidden size must be between ${NetworkConfig.MIN_HIDDEN_SIZE} and ${NetworkConfig.MAX_HIDDEN_SIZE}, got ${hiddenSize}`);
    }
    
    const paramCount = calculateParameterCount(inputSize, hiddenSize, outputSize);
    if (paramCount > NetworkConfig.MAX_PARAMETERS) {
        throw new Error(`Parameter count ${paramCount} exceeds maximum ${NetworkConfig.MAX_PARAMETERS}`);
    }
}