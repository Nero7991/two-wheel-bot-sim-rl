/**
 * Network Architecture Presets for Different Bot Models
 * 
 * This module provides predefined network configurations optimized for different
 * types of balancing robots and control scenarios. Each preset includes architecture
 * specifications, hyperparameters, and deployment constraints.
 */

/**
 * Base network architecture configuration
 */
export class NetworkArchitecture {
    constructor(config) {
        this.name = config.name || 'Custom';
        this.description = config.description || '';
        this.inputSize = config.inputSize || 2;
        this.outputSize = config.outputSize || 3;
        this.layers = config.layers || [8]; // Hidden layer sizes
        this.activations = config.activations || ['relu'];
        this.maxParameters = config.maxParameters || 200;
        this.memoryConstraint = config.memoryConstraint || 1024; // KB
        this.targetFrequency = config.targetFrequency || 50; // Hz
        this.deployment = config.deployment || 'web'; // web, embedded, mobile
    }

    /**
     * Get total parameter count for this architecture
     */
    getParameterCount() {
        let totalParams = 0;
        let prevSize = this.inputSize;
        
        for (const layerSize of this.layers) {
            // Weights + biases for each layer
            totalParams += (prevSize * layerSize) + layerSize;
            prevSize = layerSize;
        }
        
        // Final output layer
        totalParams += (prevSize * this.outputSize) + this.outputSize;
        
        return totalParams;
    }

    /**
     * Validate architecture constraints
     */
    validate() {
        const errors = [];
        
        if (this.layers.length === 0) {
            errors.push('At least one hidden layer is required');
        }
        
        if (this.layers.length > 8) {
            errors.push('Maximum 8 layers supported');
        }
        
        for (const [i, size] of this.layers.entries()) {
            if (size < 1 || size > 256) {
                errors.push(`Layer ${i + 1}: Size must be between 1 and 256, got ${size}`);
            }
        }
        
        const paramCount = this.getParameterCount();
        if (paramCount > this.maxParameters) {
            errors.push(`Parameter count ${paramCount} exceeds limit ${this.maxParameters}`);
        }
        
        return {
            valid: errors.length === 0,
            errors: errors,
            parameterCount: paramCount
        };
    }

    /**
     * Create a copy of this architecture
     */
    clone() {
        return new NetworkArchitecture({
            name: this.name,
            description: this.description,
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            layers: [...this.layers],
            activations: [...this.activations],
            maxParameters: this.maxParameters,
            memoryConstraint: this.memoryConstraint,
            targetFrequency: this.targetFrequency,
            deployment: this.deployment
        });
    }
}

/**
 * Predefined network architectures for different bot models
 */
export const NetworkPresets = {
    // Minimal architecture for very constrained embedded systems
    MICRO: new NetworkArchitecture({
        name: 'Micro Bot',
        description: 'Ultra-minimal for 8-bit microcontrollers (< 32KB RAM)',
        layers: [4],
        maxParameters: 50,
        memoryConstraint: 32,
        targetFrequency: 20,
        deployment: 'embedded'
    }),
    
    // Lightweight architecture for basic balancing
    NANO: new NetworkArchitecture({
        name: 'Nano Bot',
        description: 'Lightweight for basic two-wheel balancing (Arduino Nano)',
        layers: [6],
        maxParameters: 80,
        memoryConstraint: 64,
        targetFrequency: 30,
        deployment: 'embedded'
    }),
    
    // Current default architecture
    CLASSIC: new NetworkArchitecture({
        name: 'Classic Bot',
        description: 'Standard two-wheel balancing robot (ESP32/STM32)',
        layers: [8],
        maxParameters: 150,
        memoryConstraint: 256,
        targetFrequency: 50,
        deployment: 'embedded'
    }),
    
    // Enhanced architecture for better performance
    ENHANCED: new NetworkArchitecture({
        name: 'Enhanced Bot',
        description: 'Improved control with deeper network (Raspberry Pi)',
        layers: [12, 8],
        maxParameters: 400,
        memoryConstraint: 512,
        targetFrequency: 50,
        deployment: 'embedded'
    }),
    
    // Advanced architecture for complex behaviors
    ADVANCED: new NetworkArchitecture({
        name: 'Advanced Bot',
        description: 'Multi-layer network for complex maneuvers',
        layers: [16, 12, 8],
        maxParameters: 800,
        memoryConstraint: 1024,
        targetFrequency: 100,
        deployment: 'embedded'
    }),
    
    // High-performance architecture for research
    RESEARCH: new NetworkArchitecture({
        name: 'Research Bot',
        description: 'Deep network for research and experimentation',
        layers: [32, 24, 16, 8],
        maxParameters: 2000,
        memoryConstraint: 4096,
        targetFrequency: 100,
        deployment: 'web'
    }),
    
    // Maximum architecture for web-based training
    MAXIMUM: new NetworkArchitecture({
        name: 'Maximum Bot',
        description: 'Largest network for intensive web training',
        layers: [64, 48, 32, 24, 16],
        maxParameters: 10000,
        memoryConstraint: 16384,
        targetFrequency: 200,
        deployment: 'web'
    }),
    
    // DQN Standard architecture matching PyTorch tutorial
    DQN_STANDARD: new NetworkArchitecture({
        name: 'DQN Standard',
        description: 'PyTorch DQN tutorial standard (proven hyperparameters)',
        layers: [128],
        maxParameters: 1000,
        memoryConstraint: 2048,
        targetFrequency: 100,
        deployment: 'web'
    }),
    
    // DQN* - Enhanced DQN with additional layer
    DQN_STAR: new NetworkArchitecture({
        name: 'DQN*',
        description: 'Enhanced DQN with deeper architecture (128-32-3)',
        layers: [128, 32],
        maxParameters: 5000,
        memoryConstraint: 4096,
        targetFrequency: 100,
        deployment: 'web'
    }),
    
    // Custom starting point
    CUSTOM: new NetworkArchitecture({
        name: 'Custom',
        description: 'User-defined architecture',
        layers: [8],
        maxParameters: 1000,
        memoryConstraint: 2048,
        targetFrequency: 50,
        deployment: 'web'
    })
};

/**
 * Get preset by name
 */
export function getPreset(name) {
    return NetworkPresets[name.toUpperCase()];
}

/**
 * Get all preset names
 */
export function getPresetNames() {
    return Object.keys(NetworkPresets);
}

/**
 * Get presets suitable for a specific deployment target
 */
export function getPresetsForDeployment(deployment) {
    return Object.entries(NetworkPresets)
        .filter(([name, preset]) => preset.deployment === deployment)
        .map(([name, preset]) => ({ name, preset }));
}

/**
 * Create a custom architecture with validation
 */
export function createCustomArchitecture(config) {
    const arch = new NetworkArchitecture(config);
    const validation = arch.validate();
    
    if (!validation.valid) {
        throw new Error(`Invalid architecture: ${validation.errors.join(', ')}`);
    }
    
    return arch;
}

/**
 * Architecture optimization utilities
 */
export class ArchitectureOptimizer {
    /**
     * Suggest optimal architecture for given constraints
     */
    static suggestArchitecture(constraints) {
        const {
            maxParameters = 200,
            minLayers = 1,
            maxLayers = 3,
            targetAccuracy = 0.8,
            deployment = 'embedded'
        } = constraints;
        
        // Find presets that fit constraints
        const suitablePresets = Object.entries(NetworkPresets)
            .filter(([name, preset]) => {
                return preset.getParameterCount() <= maxParameters &&
                       preset.layers.length >= minLayers &&
                       preset.layers.length <= maxLayers &&
                       preset.deployment === deployment;
            })
            .sort((a, b) => b[1].getParameterCount() - a[1].getParameterCount());
        
        if (suitablePresets.length > 0) {
            return suitablePresets[0][1];
        }
        
        // Generate custom architecture
        const layers = [];
        let remainingParams = maxParameters - 6; // Reserve for output layer
        let currentLayer = Math.min(16, Math.floor(remainingParams / 10));
        
        for (let i = 0; i < maxLayers && remainingParams > 0; i++) {
            const layerSize = Math.max(4, Math.min(currentLayer, Math.floor(remainingParams / (maxLayers - i))));
            layers.push(layerSize);
            remainingParams -= layerSize * (i === 0 ? 2 : layers[i-1]);
            currentLayer = Math.max(4, Math.floor(currentLayer * 0.7));
        }
        
        return new NetworkArchitecture({
            name: 'Auto-Generated',
            description: `Optimized for ${maxParameters} parameters`,
            layers: layers,
            maxParameters: maxParameters,
            deployment: deployment
        });
    }
    
    /**
     * Optimize existing architecture for better performance
     */
    static optimizeArchitecture(architecture, performance) {
        const optimized = architecture.clone();
        
        // Simple optimization heuristics based on performance
        if (performance.accuracy < 0.6) {
            // Increase capacity
            optimized.layers = optimized.layers.map(size => Math.min(256, Math.floor(size * 1.2)));
        } else if (performance.accuracy > 0.9 && performance.speed < 50) {
            // Reduce capacity for speed
            optimized.layers = optimized.layers.map(size => Math.max(4, Math.floor(size * 0.8)));
        }
        
        return optimized;
    }
}

/**
 * Export utilities for backward compatibility
 */
export function validateArchitecture(inputSize, layers, outputSize, maxParams = 200) {
    const arch = new NetworkArchitecture({
        inputSize,
        outputSize,
        layers: Array.isArray(layers) ? layers : [layers],
        maxParameters: maxParams
    });
    
    return arch.validate();
}

export function calculateParameterCount(inputSize, layers, outputSize) {
    const arch = new NetworkArchitecture({
        inputSize,
        outputSize,
        layers: Array.isArray(layers) ? layers : [layers]
    });
    
    return arch.getParameterCount();
}