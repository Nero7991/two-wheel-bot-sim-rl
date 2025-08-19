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
 * Architecture: Input(2-16) -> Hidden(4-256, ReLU) -> Output(3, Linear)
 * Supports variable input sizes for multi-timestep learning
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
     * @param {number} inputSize - Number of input neurons (2-16 for multi-timestep support)
     * @param {number} hiddenSize - Number of hidden neurons (4-256)
     * @param {number} outputSize - Number of output neurons (must be 3)
     * @param {Object} options - Configuration options
     * @param {string} options.initMethod - Weight initialization method ('he', 'xavier', 'random')
     * @param {number} options.seed - Random seed for reproducible initialization
     * @returns {Promise<void>} Promise that resolves when network is created
     */
    async createNetwork(inputSize, hiddenSize, outputSize, options = {}) {
        // Validate basic constraints (but allow variable input size)
        this._validateVariableArchitecture(inputSize, hiddenSize, outputSize);
        
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
     * @param {Float32Array} input - Input values (multi-timestep robot state)
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
        
        // Validate and return output with Q-value clamping
        this.validateOutput(output);
        
        // Clamp Q-values to prevent explosive growth
        for (let i = 0; i < output.length; i++) {
            output[i] = Math.max(-100.0, Math.min(100.0, output[i]));
        }
        
        return output;
    }

    /**
     * Get the most recent hidden layer activations from forward pass
     * @returns {Float32Array} Hidden layer activations
     */
    getHiddenActivations() {
        if (!this.isInitialized) {
            throw new Error('Network not initialized');
        }
        if (!this.hiddenActivation) {
            throw new Error('No forward pass performed yet');
        }
        return this.hiddenActivation.slice();
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
        
        // Create test input matching current network input size
        const testInput = new Float32Array(this.inputSize);
        // Fill with sample robot state values, repeating the pattern for all timesteps
        for (let i = 0; i < this.inputSize; i += 2) {
            testInput[i] = 0.1;      // Sample angle
            testInput[i + 1] = -0.05; // Sample angular velocity
        }
        
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
        
        // Initialize biases
        // Small positive bias for hidden layer to ensure some initial activation with ReLU
        this.biasHidden = new Float32Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            this.biasHidden[i] = 0.01; // Small positive bias
        }
        
        // Zero bias for output layer (common practice for final layer)
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
            // Use He initialization for cloned network instead of preserving 'imported' method
            const cloneInitMethod = (this.initMethod === 'imported') ? 
                NetworkConfig.INITIALIZATION.HE : this.initMethod;
                
            clone.createNetwork(this.inputSize, this.hiddenSize, this.outputSize, {
                initMethod: cloneInitMethod
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
        
        // Save original initialization method
        const originalInitMethod = this.initMethod;
        
        // Use He initialization for reset if current method is not supported
        if (this.initMethod === 'imported' || !this.initMethod || 
            (this.initMethod !== NetworkConfig.INITIALIZATION.HE && 
             this.initMethod !== NetworkConfig.INITIALIZATION.XAVIER)) {
            this.initMethod = NetworkConfig.INITIALIZATION.HE;
        }
        
        this._initializeWeights();
        
        // Restore original initialization method
        this.initMethod = originalInitMethod;
    }

    /**
     * Validate network architecture for variable input sizes
     * @private
     * @param {number} inputSize - Number of input neurons (2-16)
     * @param {number} hiddenSize - Number of hidden neurons (4-256)
     * @param {number} outputSize - Number of output neurons (must be 3)
     */
    _validateVariableArchitecture(inputSize, hiddenSize, outputSize) {
        // Validate input size range (2-16 for 1-8 timesteps)
        if (inputSize < 2 || inputSize > 16) {
            throw new Error(`Input size must be between 2 and 16 (for 1-8 timesteps), got ${inputSize}`);
        }
        
        // Input size must be even (pairs of angle/angular velocity)
        if (inputSize % 2 !== 0) {
            throw new Error(`Input size must be even (pairs of values), got ${inputSize}`);
        }
        
        // Validate output size (must be 3 for actions)
        if (outputSize !== NetworkConfig.OUTPUT_SIZE) {
            throw new Error(`Output size must be ${NetworkConfig.OUTPUT_SIZE}, got ${outputSize}`);
        }
        
        // Validate hidden size range
        if (hiddenSize < NetworkConfig.MIN_HIDDEN_SIZE || hiddenSize > NetworkConfig.MAX_HIDDEN_SIZE) {
            throw new Error(`Hidden size must be between ${NetworkConfig.MIN_HIDDEN_SIZE} and ${NetworkConfig.MAX_HIDDEN_SIZE}, got ${hiddenSize}`);
        }
        
        // Check parameter count constraint
        const paramCount = calculateParameterCount(inputSize, hiddenSize, outputSize);
        if (paramCount > NetworkConfig.MAX_PARAMETERS) {
            throw new Error(`Parameter count ${paramCount} exceeds maximum ${NetworkConfig.MAX_PARAMETERS}`);
        }
    }
}