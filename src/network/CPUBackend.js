/**
 * CPU Backend for Neural Network Implementation
 * 
 * Efficient CPU-based neural network for two-wheel balancing robot RL.
 * Optimized for performance with Float32Array operations and designed
 * for embedded deployment constraints (<200 parameters).
 */

import { NeuralNetwork, NetworkConfig, validateArchitecture, calculateParameterCount } from './NeuralNetwork.js';
import { 
    vectorMatrixMultiply, 
    relu, 
    copyArray, 
    heInit, 
    xavierInit, 
    zeroInit,
    Performance 
} from './MatrixUtils.js';

/**
 * CPU-based neural network implementation
 * Architecture: Input(2) -> Hidden(4-16, ReLU) -> Output(3, Linear)
 */
export class CPUBackend extends NeuralNetwork {
    constructor() {
        super();
        
        // Network architecture
        this.inputSize = 0;
        this.hiddenSize = 0;
        this.outputSize = 0;
        this.parameterCount = 0;
        
        // Network weights and biases
        this.weightsInputHidden = null;  // Input to hidden layer weights
        this.biasHidden = null;          // Hidden layer biases
        this.weightsHiddenOutput = null; // Hidden to output layer weights
        this.biasOutput = null;          // Output layer biases
        
        // Temporary arrays for forward pass (reused for efficiency)
        this.hiddenActivation = null;
        this.outputActivation = null;
        
        // Configuration
        this.initMethod = NetworkConfig.INITIALIZATION.HE; // Default for ReLU
        this.isInitialized = false;
    }

    /**
     * Create and initialize the neural network
     * @param {number} inputSize - Number of input neurons (must be 2)
     * @param {number} hiddenSize - Number of hidden neurons (4-16)
     * @param {number} outputSize - Number of output neurons (must be 3)
     * @param {Object} options - Configuration options
     * @param {string} options.initMethod - Weight initialization method ('he', 'xavier', 'random')
     * @param {number} options.seed - Random seed for reproducible initialization
     * @returns {Promise<void>} Promise that resolves when network is created
     */
    async createNetwork(inputSize, hiddenSize, outputSize, options = {}) {
        // Validate architecture constraints
        validateArchitecture(inputSize, hiddenSize, outputSize);
        
        // Set architecture
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.parameterCount = calculateParameterCount(inputSize, hiddenSize, outputSize);
        
        // Set initialization method
        this.initMethod = options.initMethod || NetworkConfig.INITIALIZATION.HE;
        
        // Initialize weights and biases
        this._initializeWeights();
        
        // Allocate temporary arrays for forward pass
        this.hiddenActivation = new Float32Array(hiddenSize);
        this.outputActivation = new Float32Array(outputSize);
        
        this.isInitialized = true;
        
        console.log(`CPU Neural Network created: ${inputSize}-${hiddenSize}-${outputSize} (${this.parameterCount} parameters)`);
    }

    /**
     * Perform forward pass through the network
     * @param {Float32Array} input - Input values (angle, angular velocity)
     * @returns {Float32Array} Output probabilities for actions
     */
    forward(input) {
        if (!this.isInitialized) {
            throw new Error('Network not initialized. Call createNetwork() first.');
        }
        
        // Validate input
        this.validateInput(input);
        
        // Forward pass: Input -> Hidden (with ReLU)
        const hiddenLinear = vectorMatrixMultiply(
            input, 
            this.weightsInputHidden, 
            this.biasHidden,
            this.inputSize, 
            this.hiddenSize
        );
        
        // Apply ReLU activation to hidden layer
        for (let i = 0; i < this.hiddenSize; i++) {
            this.hiddenActivation[i] = Math.max(0, hiddenLinear[i]);
        }
        
        // Forward pass: Hidden -> Output (linear)
        const output = vectorMatrixMultiply(
            this.hiddenActivation,
            this.weightsHiddenOutput,
            this.biasOutput,
            this.hiddenSize,
            this.outputSize
        );
        
        // Validate and return output
        this.validateOutput(output);
        return output;
    }

    /**
     * Get the total number of parameters in the network
     * @returns {number} Total parameter count
     */
    getParameterCount() {
        return this.parameterCount;
    }

    /**
     * Get network weights for serialization/export
     * @returns {Object} Object containing weights and biases
     */
    getWeights() {
        if (!this.isInitialized) {
            throw new Error('Network not initialized');
        }
        
        return {
            architecture: {
                inputSize: this.inputSize,
                hiddenSize: this.hiddenSize,
                outputSize: this.outputSize,
                parameterCount: this.parameterCount
            },
            weightsInputHidden: Array.from(this.weightsInputHidden),
            biasHidden: Array.from(this.biasHidden),
            weightsHiddenOutput: Array.from(this.weightsHiddenOutput),
            biasOutput: Array.from(this.biasOutput),
            initMethod: this.initMethod
        };
    }

    /**
     * Set network weights from serialized data
     * @param {Object} weights - Object containing weights and biases
     */
    setWeights(weights) {
        if (!weights.architecture) {
            throw new Error('Invalid weights object: missing architecture');
        }
        
        const arch = weights.architecture;
        
        // Validate architecture compatibility
        if (!this.isInitialized) {
            // Initialize with the architecture from weights
            this.inputSize = arch.inputSize;
            this.hiddenSize = arch.hiddenSize;
            this.outputSize = arch.outputSize;
            this.parameterCount = arch.parameterCount;
            
            this.hiddenActivation = new Float32Array(this.hiddenSize);
            this.outputActivation = new Float32Array(this.outputSize);
            this.isInitialized = true;
        } else {
            // Verify compatibility
            if (arch.inputSize !== this.inputSize || 
                arch.hiddenSize !== this.hiddenSize || 
                arch.outputSize !== this.outputSize) {
                throw new Error('Weight architecture incompatible with current network');
            }
        }
        
        // Set weights and biases
        this.weightsInputHidden = new Float32Array(weights.weightsInputHidden);
        this.biasHidden = new Float32Array(weights.biasHidden);
        this.weightsHiddenOutput = new Float32Array(weights.weightsHiddenOutput);
        this.biasOutput = new Float32Array(weights.biasOutput);
        this.initMethod = weights.initMethod || this.initMethod;
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
            parameterCount: this.parameterCount,
            activations: {
                hidden: 'relu',
                output: 'linear'
            },
            initMethod: this.initMethod,
            backend: 'cpu',
            isInitialized: this.isInitialized
        };
    }

    /**
     * Benchmark forward pass performance
     * @param {number} iterations - Number of iterations to benchmark
     * @returns {Object} Performance statistics
     */
    benchmark(iterations = 1000) {
        if (!this.isInitialized) {
            throw new Error('Network not initialized');
        }
        
        // Create test input
        const testInput = new Float32Array([0.1, -0.05]); // Sample robot state
        
        return Performance.benchmark(
            (input) => this.forward(input),
            iterations,
            testInput
        );
    }

    /**
     * Initialize network weights and biases
     * @private
     */
    _initializeWeights() {
        switch (this.initMethod) {
            case NetworkConfig.INITIALIZATION.HE:
                // He initialization - optimal for ReLU
                this.weightsInputHidden = heInit(this.inputSize, this.hiddenSize);
                this.weightsHiddenOutput = heInit(this.hiddenSize, this.outputSize);
                break;
                
            case NetworkConfig.INITIALIZATION.XAVIER:
                // Xavier initialization - good for sigmoid/tanh
                this.weightsInputHidden = xavierInit(this.inputSize, this.hiddenSize);
                this.weightsHiddenOutput = xavierInit(this.hiddenSize, this.outputSize);
                break;
                
            default:
                throw new Error(`Unknown initialization method: ${this.initMethod}`);
        }
        
        // Initialize biases to zero (standard practice)
        this.biasHidden = zeroInit(this.hiddenSize);
        this.biasOutput = zeroInit(this.outputSize);
    }

    /**
     * Get memory usage information
     * @returns {Object} Memory usage statistics
     */
    getMemoryUsage() {
        if (!this.isInitialized) {
            return { totalBytes: 0, breakdown: {} };
        }
        
        const bytesPerFloat = 4; // Float32Array uses 4 bytes per element
        
        const breakdown = {
            weightsInputHidden: this.weightsInputHidden.length * bytesPerFloat,
            biasHidden: this.biasHidden.length * bytesPerFloat,
            weightsHiddenOutput: this.weightsHiddenOutput.length * bytesPerFloat,
            biasOutput: this.biasOutput.length * bytesPerFloat,
            hiddenActivation: this.hiddenActivation.length * bytesPerFloat,
            outputActivation: this.outputActivation.length * bytesPerFloat
        };
        
        const totalBytes = Object.values(breakdown).reduce((sum, bytes) => sum + bytes, 0);
        
        return {
            totalBytes,
            totalKB: totalBytes / 1024,
            breakdown,
            parameterBytes: (this.parameterCount * bytesPerFloat),
            parameterKB: (this.parameterCount * bytesPerFloat) / 1024
        };
    }

    /**
     * Create a copy of the network
     * @returns {CPUBackend} Deep copy of this network
     */
    clone() {
        const clone = new CPUBackend();
        
        if (this.isInitialized) {
            clone.createNetwork(this.inputSize, this.hiddenSize, this.outputSize, {
                initMethod: this.initMethod
            });
            clone.setWeights(this.getWeights());
        }
        
        return clone;
    }

    /**
     * Reset network weights to initial random values
     */
    resetWeights() {
        if (!this.isInitialized) {
            throw new Error('Network not initialized');
        }
        
        this._initializeWeights();
    }
}