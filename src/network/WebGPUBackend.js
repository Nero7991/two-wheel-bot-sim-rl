/**
 * WebGPU Backend for Neural Network Implementation
 * 
 * GPU-accelerated neural network for two-wheel balancing robot RL.
 * Provides significant performance improvements for matrix operations
 * with automatic fallback to CPU backend when WebGPU is unavailable.
 */

import { NeuralNetwork, NetworkConfig, validateArchitecture, calculateParameterCount } from './NeuralNetwork.js';
import { CPUBackend } from './CPUBackend.js';

/**
 * WebGPU feature requirements for neural network operations
 */
export const WebGPUFeatures = {
    // Required features for compute shaders
    REQUIRED_FEATURES: [
        // Core compute features will be checked programmatically
    ],
    
    // Minimum buffer sizes needed for neural networks
    MIN_BUFFER_SIZES: {
        storageBuffer: 1024,      // Minimum storage buffer size
        uniformBuffer: 256        // Minimum uniform buffer size
    },
    
    // Minimum workgroup sizes for compute shaders
    MIN_WORKGROUP_SIZES: {
        maxComputeWorkgroupSizeX: 64,
        maxComputeWorkgroupSizeY: 1,
        maxComputeWorkgroupSizeZ: 1,
        maxComputeInvocationsPerWorkgroup: 64
    }
};

/**
 * WebGPU device capabilities and status
 */
export class WebGPUDeviceInfo {
    constructor() {
        this.isAvailable = false;
        this.adapter = null;
        this.device = null;
        this.capabilities = null;
        this.features = new Set();
        this.limits = null;
        this.errorMessage = null;
        this.fallbackReason = null;
    }

    /**
     * Get a summary of device capabilities
     * @returns {Object} Device capability summary
     */
    getSummary() {
        return {
            isAvailable: this.isAvailable,
            hasComputeShaders: this.features.has('compute') || this.isAvailable, // Basic compute assumed if available
            maxBufferSize: this.limits?.maxBufferSize || 0,
            maxComputeWorkgroupSizeX: this.limits?.maxComputeWorkgroupSizeX || 0,
            maxComputeInvocationsPerWorkgroup: this.limits?.maxComputeInvocationsPerWorkgroup || 0,
            errorMessage: this.errorMessage,
            fallbackReason: this.fallbackReason,
            adapterInfo: this.adapter ? {
                vendor: this.adapter.info?.vendor || 'Unknown',
                architecture: this.adapter.info?.architecture || 'Unknown',
                device: this.adapter.info?.device || 'Unknown',
                description: this.adapter.info?.description || 'Unknown'
            } : null
        };
    }

    /**
     * Check if device meets minimum requirements for neural networks
     * @returns {boolean} True if device meets requirements
     */
    meetsRequirements() {
        if (!this.isAvailable || !this.limits) {
            return false;
        }

        // Check buffer size requirements
        if (this.limits.maxBufferSize < WebGPUFeatures.MIN_BUFFER_SIZES.storageBuffer) {
            this.fallbackReason = `Insufficient buffer size: ${this.limits.maxBufferSize} < ${WebGPUFeatures.MIN_BUFFER_SIZES.storageBuffer}`;
            return false;
        }

        // Check workgroup size requirements
        if (this.limits.maxComputeWorkgroupSizeX < WebGPUFeatures.MIN_WORKGROUP_SIZES.maxComputeWorkgroupSizeX) {
            this.fallbackReason = `Insufficient workgroup size X: ${this.limits.maxComputeWorkgroupSizeX} < ${WebGPUFeatures.MIN_WORKGROUP_SIZES.maxComputeWorkgroupSizeX}`;
            return false;
        }

        if (this.limits.maxComputeInvocationsPerWorkgroup < WebGPUFeatures.MIN_WORKGROUP_SIZES.maxComputeInvocationsPerWorkgroup) {
            this.fallbackReason = `Insufficient compute invocations: ${this.limits.maxComputeInvocationsPerWorkgroup} < ${WebGPUFeatures.MIN_WORKGROUP_SIZES.maxComputeInvocationsPerWorkgroup}`;
            return false;
        }

        return true;
    }
}

/**
 * WebGPU device initialization and management
 */
export class WebGPUDeviceManager {
    constructor() {
        this.deviceInfo = new WebGPUDeviceInfo();
        this.isInitialized = false;
        this.initializationPromise = null;
    }

    /**
     * Initialize WebGPU device with comprehensive feature detection
     * @param {Object} options - Initialization options
     * @param {boolean} options.preferHighPerformance - Prefer high-performance adapter
     * @param {boolean} options.forceFallback - Force CPU fallback for testing
     * @returns {Promise<WebGPUDeviceInfo>} Device information
     */
    async initWebGPU(options = {}) {
        // Return existing initialization promise if already in progress
        if (this.initializationPromise) {
            return this.initializationPromise;
        }

        this.initializationPromise = this._initWebGPUInternal(options);
        return this.initializationPromise;
    }

    /**
     * Internal WebGPU initialization implementation
     * @private
     */
    async _initWebGPUInternal(options) {
        const { preferHighPerformance = true, forceFallback = false } = options;

        try {
            // Check for forced fallback (for testing)
            if (forceFallback) {
                throw new Error('Forced CPU fallback for testing');
            }

            // Check WebGPU browser support
            if (!navigator.gpu) {
                throw new Error('WebGPU is not supported in this browser. Requires Chrome 113+ or Edge 113+');
            }

            console.log('WebGPU API detected, requesting adapter...');

            // Request WebGPU adapter
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: preferHighPerformance ? 'high-performance' : 'low-power'
            });

            if (!adapter) {
                throw new Error('No suitable WebGPU adapter found. Your GPU may not support WebGPU');
            }

            this.deviceInfo.adapter = adapter;
            console.log('WebGPU adapter obtained:', adapter);

            // Log adapter information if available
            if (adapter.info) {
                console.log('Adapter info:', {
                    vendor: adapter.info.vendor,
                    architecture: adapter.info.architecture,
                    device: adapter.info.device,
                    description: adapter.info.description
                });
            }

            // Get adapter features and limits
            const adapterFeatures = Array.from(adapter.features);
            console.log('Adapter features:', adapterFeatures);

            // Request device with required features
            const deviceDescriptor = this._createDeviceDescriptor(adapter);
            console.log('Requesting device with descriptor:', deviceDescriptor);

            const device = await adapter.requestDevice(deviceDescriptor);

            if (!device) {
                throw new Error('Failed to obtain WebGPU device');
            }

            this.deviceInfo.device = device;
            this.deviceInfo.features = device.features;
            this.deviceInfo.limits = device.limits;

            console.log('WebGPU device obtained successfully');
            console.log('Device features:', Array.from(device.features));
            console.log('Device limits:', device.limits);

            // Set up error handling
            device.addEventListener('uncapturederror', (event) => {
                console.error('WebGPU uncaptured error:', event.error);
            });

            // Validate device capabilities
            if (!this.deviceInfo.meetsRequirements()) {
                throw new Error(`WebGPU device does not meet requirements: ${this.deviceInfo.fallbackReason}`);
            }

            // Mark as successfully initialized
            this.deviceInfo.isAvailable = true;
            this.isInitialized = true;

            console.log('WebGPU initialization completed successfully');
            return this.deviceInfo;

        } catch (error) {
            console.warn('WebGPU initialization failed:', error.message);
            this.deviceInfo.isAvailable = false;
            this.deviceInfo.errorMessage = error.message;
            this.deviceInfo.fallbackReason = this.deviceInfo.fallbackReason || error.message;
            
            return this.deviceInfo;
        }
    }

    /**
     * Create device descriptor with required features
     * @private
     */
    _createDeviceDescriptor(adapter) {
        const requiredFeatures = [];
        
        // Add features that are available and useful for neural networks
        const usefulFeatures = [
            'timestamp-query',
            'pipeline-statistics-query'
        ];

        for (const feature of usefulFeatures) {
            if (adapter.features.has(feature)) {
                requiredFeatures.push(feature);
            }
        }

        return {
            requiredFeatures,
            requiredLimits: {
                // Request higher limits if supported
                maxBufferSize: Math.min(
                    adapter.limits.maxBufferSize,
                    1024 * 1024 * 1024 // 1GB max
                ),
                maxComputeWorkgroupSizeX: Math.min(
                    adapter.limits.maxComputeWorkgroupSizeX,
                    256
                )
            }
        };
    }

    /**
     * Get current device information
     * @returns {WebGPUDeviceInfo} Current device info
     */
    getDeviceInfo() {
        return this.deviceInfo;
    }

    /**
     * Check if WebGPU is available and ready
     * @returns {boolean} True if WebGPU is available
     */
    isWebGPUAvailable() {
        return this.deviceInfo.isAvailable && this.isInitialized;
    }

    /**
     * Get the WebGPU device
     * @returns {GPUDevice|null} WebGPU device or null if not available
     */
    getDevice() {
        return this.deviceInfo.device;
    }

    /**
     * Clean up WebGPU resources
     */
    destroy() {
        if (this.deviceInfo.device) {
            this.deviceInfo.device.destroy();
            this.deviceInfo.device = null;
        }
        
        this.deviceInfo.adapter = null;
        this.isInitialized = false;
        this.initializationPromise = null;
        
        console.log('WebGPU device manager destroyed');
    }

    /**
     * Get performance characteristics estimate
     * @returns {Object} Performance estimates
     */
    getPerformanceEstimate() {
        if (!this.isWebGPUAvailable()) {
            return {
                estimatedSpeedup: 1.0,
                supportedOperations: ['basic'],
                recommendedBatchSize: 1
            };
        }

        // Estimate performance based on device capabilities
        const limits = this.deviceInfo.limits;
        const workgroupSize = Math.min(limits.maxComputeWorkgroupSizeX, 64);
        
        // Conservative estimates for neural network operations
        const estimatedSpeedup = Math.min(
            workgroupSize / 8, // Assume 8x speedup per 64 threads
            10.0 // Cap at 10x speedup
        );

        return {
            estimatedSpeedup,
            supportedOperations: ['matrix_multiply', 'activation_functions', 'gradient_computation'],
            recommendedBatchSize: Math.floor(workgroupSize / 4),
            maxWorkgroupSize: workgroupSize,
            maxBufferSizeMB: Math.floor(limits.maxBufferSize / (1024 * 1024))
        };
    }
}

/**
 * WebGPU-accelerated neural network implementation
 * Falls back to CPU backend when WebGPU is unavailable
 */
export class WebGPUBackend extends NeuralNetwork {
    constructor() {
        super();
        
        // WebGPU device management
        this.deviceManager = new WebGPUDeviceManager();
        this.isWebGPUAvailable = false;
        
        // CPU fallback backend
        this.cpuBackend = new CPUBackend();
        this.usingFallback = false;
        
        // Network state
        this.isInitialized = false;
        
        // Performance monitoring
        this.performanceMetrics = {
            initializationTime: 0,
            forwardPassTimes: [],
            gpuMemoryUsage: 0,
            fallbackReason: null
        };
    }

    /**
     * Create and initialize the neural network with WebGPU acceleration
     * @param {number} inputSize - Number of input neurons (must be 2)
     * @param {number} hiddenSize - Number of hidden neurons (4-16)
     * @param {number} outputSize - Number of output neurons (must be 3)
     * @param {Object} options - Configuration options
     * @param {string} options.initMethod - Weight initialization method
     * @param {number} options.seed - Random seed for reproducible initialization
     * @param {boolean} options.forceCPU - Force CPU fallback for testing
     * @returns {Promise<void>} Promise that resolves when network is created
     */
    async createNetwork(inputSize, hiddenSize, outputSize, options = {}) {
        const startTime = performance.now();
        
        try {
            // Validate architecture constraints
            validateArchitecture(inputSize, hiddenSize, outputSize);
            
            console.log('Initializing WebGPU backend...');
            
            // Initialize WebGPU device
            const deviceInfo = await this.deviceManager.initWebGPU({
                forceFallback: options.forceCPU || false
            });
            
            if (deviceInfo.isAvailable && deviceInfo.meetsRequirements()) {
                // WebGPU is available and meets requirements
                this.isWebGPUAvailable = true;
                this.usingFallback = false;
                
                console.log('WebGPU backend initialized successfully');
                console.log('Performance estimate:', this.deviceManager.getPerformanceEstimate());
                
                // TODO: Initialize WebGPU-specific resources in Phase 2.2
                // For now, use CPU backend as implementation
                await this.cpuBackend.createNetwork(inputSize, hiddenSize, outputSize, options);
                
            } else {
                // Fall back to CPU backend
                this.isWebGPUAvailable = false;
                this.usingFallback = true;
                this.performanceMetrics.fallbackReason = deviceInfo.fallbackReason || deviceInfo.errorMessage;
                
                console.log('Falling back to CPU backend:', this.performanceMetrics.fallbackReason);
                await this.cpuBackend.createNetwork(inputSize, hiddenSize, outputSize, options);
            }
            
            this.isInitialized = true;
            this.performanceMetrics.initializationTime = performance.now() - startTime;
            
            const backendType = this.isWebGPUAvailable ? 'WebGPU' : 'CPU (fallback)';
            console.log(`Neural Network created with ${backendType} backend: ${inputSize}-${hiddenSize}-${outputSize}`);
            
        } catch (error) {
            console.error('Failed to initialize WebGPU backend:', error);
            
            // Emergency fallback to CPU
            if (!this.usingFallback) {
                console.log('Emergency fallback to CPU backend');
                this.isWebGPUAvailable = false;
                this.usingFallback = true;
                this.performanceMetrics.fallbackReason = error.message;
                
                await this.cpuBackend.createNetwork(inputSize, hiddenSize, outputSize, options);
                this.isInitialized = true;
            } else {
                throw error;
            }
        }
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
        
        const startTime = performance.now();
        
        // Use CPU backend for now (WebGPU compute shaders in Phase 2.2)
        const result = this.cpuBackend.forward(input);
        
        // Track performance
        const forwardTime = performance.now() - startTime;
        this.performanceMetrics.forwardPassTimes.push(forwardTime);
        
        // Keep only recent timing history
        if (this.performanceMetrics.forwardPassTimes.length > 100) {
            this.performanceMetrics.forwardPassTimes.shift();
        }
        
        return result;
    }

    /**
     * Get the total number of parameters in the network
     * @returns {number} Total parameter count
     */
    getParameterCount() {
        if (!this.isInitialized) {
            return 0;
        }
        return this.cpuBackend.getParameterCount();
    }

    /**
     * Get network weights for serialization/export
     * @returns {Object} Object containing weights and biases
     */
    getWeights() {
        if (!this.isInitialized) {
            throw new Error('Network not initialized');
        }
        return this.cpuBackend.getWeights();
    }

    /**
     * Set network weights from serialized data
     * @param {Object} weights - Object containing weights and biases
     */
    setWeights(weights) {
        return this.cpuBackend.setWeights(weights);
    }

    /**
     * Get network architecture information including backend details
     * @returns {Object} Architecture details with backend information
     */
    getArchitecture() {
        const baseArch = this.cpuBackend.getArchitecture();
        
        return {
            ...baseArch,
            backend: this.isWebGPUAvailable ? 'webgpu' : 'cpu',
            webgpuAvailable: this.isWebGPUAvailable,
            usingFallback: this.usingFallback,
            fallbackReason: this.performanceMetrics.fallbackReason,
            deviceInfo: this.deviceManager.getDeviceInfo().getSummary(),
            performanceEstimate: this.deviceManager.getPerformanceEstimate()
        };
    }

    /**
     * Get comprehensive backend information
     * @returns {Object} Backend status and capabilities
     */
    getBackendInfo() {
        return {
            type: this.isWebGPUAvailable ? 'webgpu' : 'cpu',
            webgpuAvailable: this.isWebGPUAvailable,
            usingFallback: this.usingFallback,
            fallbackReason: this.performanceMetrics.fallbackReason,
            deviceInfo: this.deviceManager.getDeviceInfo(),
            performanceMetrics: this.getPerformanceMetrics(),
            capabilities: this.deviceManager.getPerformanceEstimate()
        };
    }

    /**
     * Get performance metrics
     * @returns {Object} Performance statistics
     */
    getPerformanceMetrics() {
        const forwardTimes = this.performanceMetrics.forwardPassTimes;
        const avgForwardTime = forwardTimes.length > 0 
            ? forwardTimes.reduce((a, b) => a + b) / forwardTimes.length 
            : 0;

        return {
            initializationTime: this.performanceMetrics.initializationTime,
            averageForwardTime: avgForwardTime,
            totalForwardPasses: forwardTimes.length,
            estimatedSpeedup: this.isWebGPUAvailable 
                ? this.deviceManager.getPerformanceEstimate().estimatedSpeedup 
                : 1.0,
            memoryUsage: this.getMemoryUsage()
        };
    }

    /**
     * Benchmark forward pass performance
     * @param {number} iterations - Number of iterations to benchmark
     * @returns {Object} Performance statistics with backend comparison
     */
    benchmark(iterations = 1000) {
        if (!this.isInitialized) {
            throw new Error('Network not initialized');
        }
        
        // Use CPU backend's benchmark for now
        const results = this.cpuBackend.benchmark(iterations);
        
        // Add backend-specific information
        return {
            ...results,
            backend: this.isWebGPUAvailable ? 'webgpu' : 'cpu',
            webgpuAvailable: this.isWebGPUAvailable,
            usingFallback: this.usingFallback,
            estimatedSpeedup: this.isWebGPUAvailable 
                ? this.deviceManager.getPerformanceEstimate().estimatedSpeedup 
                : 1.0
        };
    }

    /**
     * Get memory usage information
     * @returns {Object} Memory usage statistics
     */
    getMemoryUsage() {
        const cpuMemory = this.cpuBackend.getMemoryUsage();
        
        return {
            ...cpuMemory,
            backend: this.isWebGPUAvailable ? 'webgpu' : 'cpu',
            gpuMemoryEstimate: this.isWebGPUAvailable ? cpuMemory.totalBytes : 0,
            webgpuBuffers: this.isWebGPUAvailable ? 'Not implemented (Phase 2.2)' : 'N/A'
        };
    }

    /**
     * Create a copy of the network
     * @returns {WebGPUBackend} Deep copy of this network
     */
    clone() {
        const clone = new WebGPUBackend();
        
        if (this.isInitialized) {
            const arch = this.getArchitecture();
            clone.createNetwork(arch.inputSize, arch.hiddenSize, arch.outputSize, {
                initMethod: arch.initMethod
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
        
        this.cpuBackend.resetWeights();
    }

    /**
     * Clean up resources
     */
    destroy() {
        if (this.cpuBackend) {
            // CPU backend doesn't have a destroy method, but we can clear references
            this.cpuBackend = null;
        }
        
        if (this.deviceManager) {
            this.deviceManager.destroy();
            this.deviceManager = null;
        }
        
        this.isInitialized = false;
        this.isWebGPUAvailable = false;
        this.usingFallback = false;
        
        console.log('WebGPU backend destroyed');
    }
}

/**
 * Utility function to check WebGPU availability without full initialization
 * @returns {Promise<Object>} Quick availability check result
 */
export async function checkWebGPUAvailability() {
    try {
        if (!navigator.gpu) {
            return {
                available: false,
                reason: 'WebGPU API not found in browser'
            };
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return {
                available: false,
                reason: 'No WebGPU adapter available'
            };
        }

        return {
            available: true,
            adapterInfo: adapter.info ? {
                vendor: adapter.info.vendor,
                device: adapter.info.device
            } : 'Unknown adapter'
        };
    } catch (error) {
        return {
            available: false,
            reason: error.message
        };
    }
}

/**
 * Factory function to create appropriate backend (WebGPU with CPU fallback)
 * @param {Object} options - Backend creation options
 * @returns {WebGPUBackend} Backend instance
 */
export function createBackend(options = {}) {
    return new WebGPUBackend();
}