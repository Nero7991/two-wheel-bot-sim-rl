/**
 * Shader Manager for WebGPU Neural Network Operations
 * 
 * Handles compilation, pipeline creation, and execution of WGSL compute shaders
 * for neural network operations in the two-wheel balancing robot RL application.
 */

/**
 * Shader compilation and management utilities
 */
export class ShaderManager {
    constructor(device) {
        this.device = device;
        this.shaderModules = new Map();
        this.computePipelines = new Map();
        this.bindGroupLayouts = new Map();
        this.uniformBuffers = new Map();
        this.storageBuffers = new Map();
        
        // Shader source cache
        this.shaderSources = new Map();
        
        // Performance monitoring
        this.compilationTimes = new Map();
        this.pipelineCreationTimes = new Map();
    }

    /**
     * Load and compile all neural network shaders
     * @returns {Promise<void>} Promise that resolves when all shaders are loaded
     */
    async loadShaders() {
        const startTime = performance.now();
        
        try {
            // Load shader source files
            await this._loadShaderSources();
            
            // Compile shader modules
            await this._compileShaderModules();
            
            // Create compute pipelines
            await this._createComputePipelines();
            
            const totalTime = performance.now() - startTime;
            console.log(`Shaders loaded and compiled in ${totalTime.toFixed(2)}ms`);
            
        } catch (error) {
            console.error('Failed to load shaders:', error);
            throw error;
        }
    }

    /**
     * Load shader source code from files
     * @private
     */
    async _loadShaderSources() {
        const shaderFiles = {
            'matmul': './shaders/matmul.wgsl',
            'relu': './shaders/relu.wgsl', 
            'qlearning': './shaders/qlearning.wgsl'
        };

        const loadPromises = Object.entries(shaderFiles).map(async ([name, path]) => {
            try {
                const response = await fetch(path);
                if (!response.ok) {
                    throw new Error(`Failed to load shader ${name}: ${response.statusText}`);
                }
                const source = await response.text();
                this.shaderSources.set(name, source);
                console.log(`Loaded shader source: ${name}`);
            } catch (error) {
                console.error(`Failed to load shader ${name}:`, error);
                // Fall back to inline shader sources
                this.shaderSources.set(name, this._getInlineShaderSource(name));
            }
        });

        await Promise.all(loadPromises);
    }

    /**
     * Get inline shader source as fallback
     * @private
     */
    _getInlineShaderSource(name) {
        // Simplified inline versions for fallback
        const inlineShaders = {
            'matmul': `
                @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
                @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
                @group(0) @binding(2) var<storage, read> bias: array<f32>;
                @group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;
                @group(0) @binding(4) var<uniform> params: vec4<u32>;
                
                @compute @workgroup_size(64, 1, 1)
                fn matmul_simple(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    let M = params.x;
                    let K = params.y;
                    let N = params.z;
                    
                    if (index >= M * N) { return; }
                    
                    let row = index / N;
                    let col = index % N;
                    
                    var sum: f32 = 0.0;
                    for (var k: u32 = 0u; k < K; k++) {
                        sum += matrixA[row * K + k] * matrixB[k * N + col];
                    }
                    
                    matrixC[index] = sum + bias[col];
                }
            `,
            'relu': `
                @group(0) @binding(0) var<storage, read> input_data: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
                @group(0) @binding(2) var<uniform> params: vec4<u32>;
                
                @compute @workgroup_size(64, 1, 1)
                fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    let size = params.x;
                    
                    if (index >= size) { return; }
                    
                    output_data[index] = max(0.0, input_data[index]);
                }
            `,
            'qlearning': `
                @group(0) @binding(0) var<storage, read> q_values: array<f32>;
                @group(0) @binding(1) var<storage, read> target_q_values: array<f32>;
                @group(0) @binding(2) var<storage, read> actions: array<u32>;
                @group(0) @binding(3) var<storage, read_write> td_errors: array<f32>;
                @group(0) @binding(4) var<uniform> params: vec4<f32>;
                
                @compute @workgroup_size(32, 1, 1)
                fn compute_td_errors(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let batch_idx = global_id.x;
                    let batch_size = u32(params.x);
                    let gamma = params.y;
                    
                    if (batch_idx >= batch_size) { return; }
                    
                    let action = actions[batch_idx];
                    let current_q = q_values[batch_idx * 3u + action];
                    let target_q = target_q_values[batch_idx * 3u + action];
                    
                    td_errors[batch_idx] = target_q - current_q;
                }
            `
        };

        return inlineShaders[name] || '';
    }

    /**
     * Compile shader modules from source code
     * @private
     */
    async _compileShaderModules() {
        for (const [name, source] of this.shaderSources) {
            const startTime = performance.now();
            
            try {
                const shaderModule = this.device.createShaderModule({
                    label: `${name}_shader`,
                    code: source
                });

                // Check for compilation errors
                const compilationInfo = await shaderModule.getCompilationInfo();
                
                if (compilationInfo.messages.length > 0) {
                    console.warn(`Shader ${name} compilation messages:`, compilationInfo.messages);
                    
                    // Check for errors
                    const errors = compilationInfo.messages.filter(msg => msg.type === 'error');
                    if (errors.length > 0) {
                        throw new Error(`Shader ${name} compilation failed: ${errors.map(e => e.message).join(', ')}`);
                    }
                }

                this.shaderModules.set(name, shaderModule);
                
                const compilationTime = performance.now() - startTime;
                this.compilationTimes.set(name, compilationTime);
                
                console.log(`Compiled shader ${name} in ${compilationTime.toFixed(2)}ms`);
                
            } catch (error) {
                console.error(`Failed to compile shader ${name}:`, error);
                throw error;
            }
        }
    }

    /**
     * Create compute pipelines for all operations
     * @private
     */
    async _createComputePipelines() {
        // Matrix multiplication pipelines
        await this._createMatMulPipelines();
        
        // Activation function pipelines
        await this._createActivationPipelines();
        
        // Q-learning pipelines
        await this._createQLearningPipelines();
    }

    /**
     * Create matrix multiplication pipelines
     * @private
     */
    async _createMatMulPipelines() {
        const matmulModule = this.shaderModules.get('matmul');
        if (!matmulModule) {
            throw new Error('Matrix multiplication shader module not found');
        }

        // Bind group layout for matrix multiplication
        const matmulBindGroupLayout = this.device.createBindGroupLayout({
            label: 'matmul_bind_group_layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // matrixA
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // matrixB
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // bias
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // matrixC
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },          // params
            ]
        });

        this.bindGroupLayouts.set('matmul', matmulBindGroupLayout);

        // Create pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'matmul_pipeline_layout',
            bindGroupLayouts: [matmulBindGroupLayout]
        });

        // Create compute pipelines for different entry points
        const entryPoints = ['main', 'matmul_simple', 'matmul_batch'];
        
        for (const entryPoint of entryPoints) {
            try {
                const pipeline = this.device.createComputePipeline({
                    label: `matmul_${entryPoint}_pipeline`,
                    layout: pipelineLayout,
                    compute: {
                        module: matmulModule,
                        entryPoint: entryPoint
                    }
                });

                this.computePipelines.set(`matmul_${entryPoint}`, pipeline);
                console.log(`Created matrix multiplication pipeline: ${entryPoint}`);
                
            } catch (error) {
                console.warn(`Failed to create pipeline for ${entryPoint}, skipping:`, error.message);
            }
        }
    }

    /**
     * Create activation function pipelines
     * @private
     */
    async _createActivationPipelines() {
        const reluModule = this.shaderModules.get('relu');
        if (!reluModule) {
            throw new Error('ReLU shader module not found');
        }

        // Bind group layout for activation functions
        const activationBindGroupLayout = this.device.createBindGroupLayout({
            label: 'activation_bind_group_layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },          // params
            ]
        });

        this.bindGroupLayouts.set('activation', activationBindGroupLayout);

        // Create pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'activation_pipeline_layout',
            bindGroupLayouts: [activationBindGroupLayout]
        });

        // Create pipelines for different activation functions
        const entryPoints = ['relu', 'leaky_relu', 'relu_derivative', 'relu_inplace', 'softmax'];
        
        for (const entryPoint of entryPoints) {
            try {
                const pipeline = this.device.createComputePipeline({
                    label: `${entryPoint}_pipeline`,
                    layout: pipelineLayout,
                    compute: {
                        module: reluModule,
                        entryPoint: entryPoint
                    }
                });

                this.computePipelines.set(entryPoint, pipeline);
                console.log(`Created activation pipeline: ${entryPoint}`);
                
            } catch (error) {
                console.warn(`Failed to create pipeline for ${entryPoint}, skipping:`, error.message);
            }
        }
    }

    /**
     * Create Q-learning pipelines
     * @private
     */
    async _createQLearningPipelines() {
        const qlearningModule = this.shaderModules.get('qlearning');
        if (!qlearningModule) {
            throw new Error('Q-learning shader module not found');
        }

        // Bind group layout for Q-learning
        const qlearningBindGroupLayout = this.device.createBindGroupLayout({
            label: 'qlearning_bind_group_layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // q_values
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // target_q_values
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // actions
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // rewards
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dones
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // hidden_activations
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // input_states
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // weights_hidden
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // bias_hidden
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // weights_output
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // bias_output
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // td_errors
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },         // params
            ]
        });

        this.bindGroupLayouts.set('qlearning', qlearningBindGroupLayout);

        // Create pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'qlearning_pipeline_layout',
            bindGroupLayouts: [qlearningBindGroupLayout]
        });

        // Create pipelines for Q-learning operations
        const entryPoints = [
            'compute_td_errors',
            'update_output_weights',
            'update_output_bias',
            'update_hidden_weights',
            'update_hidden_bias'
        ];
        
        for (const entryPoint of entryPoints) {
            try {
                const pipeline = this.device.createComputePipeline({
                    label: `qlearning_${entryPoint}_pipeline`,
                    layout: pipelineLayout,
                    compute: {
                        module: qlearningModule,
                        entryPoint: entryPoint
                    }
                });

                this.computePipelines.set(`qlearning_${entryPoint}`, pipeline);
                console.log(`Created Q-learning pipeline: ${entryPoint}`);
                
            } catch (error) {
                console.warn(`Failed to create pipeline for ${entryPoint}, skipping:`, error.message);
            }
        }
    }

    /**
     * Get a compute pipeline by name
     * @param {string} pipelineName - Name of the pipeline
     * @returns {GPUComputePipeline} The requested compute pipeline
     */
    getPipeline(pipelineName) {
        const pipeline = this.computePipelines.get(pipelineName);
        if (!pipeline) {
            throw new Error(`Pipeline ${pipelineName} not found. Available pipelines: ${Array.from(this.computePipelines.keys()).join(', ')}`);
        }
        return pipeline;
    }

    /**
     * Get a bind group layout by name
     * @param {string} layoutName - Name of the bind group layout
     * @returns {GPUBindGroupLayout} The requested bind group layout
     */
    getBindGroupLayout(layoutName) {
        const layout = this.bindGroupLayouts.get(layoutName);
        if (!layout) {
            throw new Error(`Bind group layout ${layoutName} not found. Available layouts: ${Array.from(this.bindGroupLayouts.keys()).join(', ')}`);
        }
        return layout;
    }

    /**
     * Get compilation and creation performance metrics
     * @returns {Object} Performance metrics
     */
    getPerformanceMetrics() {
        const totalCompilationTime = Array.from(this.compilationTimes.values())
            .reduce((sum, time) => sum + time, 0);
        
        const totalPipelineTime = Array.from(this.pipelineCreationTimes.values())
            .reduce((sum, time) => sum + time, 0);

        return {
            totalCompilationTime,
            totalPipelineTime,
            shadersCompiled: this.shaderModules.size,
            pipelinesCreated: this.computePipelines.size,
            compilationTimes: Object.fromEntries(this.compilationTimes),
            pipelineCreationTimes: Object.fromEntries(this.pipelineCreationTimes),
            availablePipelines: Array.from(this.computePipelines.keys()),
            availableLayouts: Array.from(this.bindGroupLayouts.keys())
        };
    }

    /**
     * Validate shader compilation status
     * @returns {Object} Validation results
     */
    validateShaders() {
        const validation = {
            allShadersCompiled: true,
            missingShaders: [],
            availableShaders: Array.from(this.shaderModules.keys()),
            availablePipelines: Array.from(this.computePipelines.keys())
        };

        const requiredShaders = ['matmul', 'relu', 'qlearning'];
        const requiredPipelines = [
            'matmul_simple', 'relu', 'qlearning_compute_td_errors'
        ];

        for (const shader of requiredShaders) {
            if (!this.shaderModules.has(shader)) {
                validation.allShadersCompiled = false;
                validation.missingShaders.push(shader);
            }
        }

        validation.allPipelinesCreated = requiredPipelines.every(
            pipeline => this.computePipelines.has(pipeline)
        );

        return validation;
    }

    /**
     * Clean up resources
     */
    destroy() {
        // Clear all caches
        this.shaderModules.clear();
        this.computePipelines.clear();
        this.bindGroupLayouts.clear();
        this.shaderSources.clear();
        this.compilationTimes.clear();
        this.pipelineCreationTimes.clear();

        console.log('Shader manager destroyed');
    }
}