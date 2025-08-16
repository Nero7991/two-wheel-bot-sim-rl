/**
 * Buffer Manager for WebGPU Neural Network Operations
 * 
 * Manages GPU buffers for neural network operations including matrices, weights,
 * biases, and temporary computation buffers. Optimized for the two-wheel balancing
 * robot RL application with small neural networks.
 */

/**
 * GPU buffer management for neural network operations
 */
export class BufferManager {
    constructor(device) {
        this.device = device;
        
        // Buffer storage
        this.buffers = new Map();
        this.bindGroups = new Map();
        
        // Buffer pools for reuse
        this.bufferPools = new Map();
        
        // Memory tracking
        this.totalMemoryUsed = 0;
        this.bufferSizes = new Map();
        
        // Configuration
        this.maxBufferSize = 1024 * 1024 * 64; // 64MB max per buffer
        this.alignment = 256; // Buffer alignment requirement
        
        // Performance tracking
        this.bufferCreationTimes = [];
        this.memoryTransferTimes = [];
    }

    /**
     * Create buffers for neural network architecture
     * @param {Object} architecture - Network architecture specification
     * @param {number} architecture.inputSize - Input layer size (2)
     * @param {number} architecture.hiddenSize - Hidden layer size (4-16)
     * @param {number} architecture.outputSize - Output layer size (3)
     * @param {number} architecture.batchSize - Batch size for training (1-32)
     * @returns {Object} Created buffer references
     */
    createNetworkBuffers(architecture) {
        const { inputSize, hiddenSize, outputSize, batchSize = 1 } = architecture;
        
        console.log(`Creating buffers for ${inputSize}-${hiddenSize}-${outputSize} network, batch size: ${batchSize}`);
        
        const buffers = {};
        
        // Calculate buffer sizes
        const sizes = this._calculateBufferSizes(architecture);
        
        // Create input/output buffers
        buffers.input = this._createBuffer('input', sizes.input, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        buffers.hidden = this._createBuffer('hidden', sizes.hidden, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        buffers.output = this._createBuffer('output', sizes.output, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        
        // Create weight buffers
        buffers.weightsHidden = this._createBuffer('weightsHidden', sizes.weightsHidden, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        buffers.weightsOutput = this._createBuffer('weightsOutput', sizes.weightsOutput, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        
        // Create bias buffers
        buffers.biasHidden = this._createBuffer('biasHidden', sizes.biasHidden, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        buffers.biasOutput = this._createBuffer('biasOutput', sizes.biasOutput, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        
        // Create uniform buffers for parameters
        buffers.matmulParams = this._createBuffer('matmulParams', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        buffers.activationParams = this._createBuffer('activationParams', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        
        // Create staging buffers for CPU-GPU data transfer
        buffers.stagingInput = this._createBuffer('stagingInput', sizes.input, 
            GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC);
        buffers.stagingOutput = this._createBuffer('stagingOutput', sizes.output, 
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        
        // Create buffers for training (Q-learning)
        if (batchSize > 1) {
            buffers.qValues = this._createBuffer('qValues', sizes.qValues, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            buffers.targetQValues = this._createBuffer('targetQValues', sizes.qValues, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            buffers.actions = this._createBuffer('actions', batchSize * 4, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            buffers.rewards = this._createBuffer('rewards', batchSize * 4, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            buffers.dones = this._createBuffer('dones', batchSize * 4, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            buffers.tdErrors = this._createBuffer('tdErrors', batchSize * 4, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            buffers.qlearningParams = this._createBuffer('qlearningParams', 32, 
                GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        }
        
        console.log(`Created ${Object.keys(buffers).length} buffers, total memory: ${this._formatBytes(this.totalMemoryUsed)}`);
        
        return buffers;
    }

    /**
     * Calculate required buffer sizes for architecture
     * @private
     */
    _calculateBufferSizes(architecture) {
        const { inputSize, hiddenSize, outputSize, batchSize = 1 } = architecture;
        
        const bytesPerFloat = 4;
        const sizes = {};
        
        // Layer data sizes (with batch dimension)
        sizes.input = this._alignSize(batchSize * inputSize * bytesPerFloat);
        sizes.hidden = this._alignSize(batchSize * hiddenSize * bytesPerFloat);
        sizes.output = this._alignSize(batchSize * outputSize * bytesPerFloat);
        
        // Weight matrix sizes
        sizes.weightsHidden = this._alignSize(inputSize * hiddenSize * bytesPerFloat);
        sizes.weightsOutput = this._alignSize(hiddenSize * outputSize * bytesPerFloat);
        
        // Bias vector sizes
        sizes.biasHidden = this._alignSize(hiddenSize * bytesPerFloat);
        sizes.biasOutput = this._alignSize(outputSize * bytesPerFloat);
        
        // Training-specific sizes
        sizes.qValues = this._alignSize(batchSize * outputSize * bytesPerFloat);
        
        return sizes;
    }

    /**
     * Align buffer size to device requirements
     * @private
     */
    _alignSize(size) {
        return Math.ceil(size / this.alignment) * this.alignment;
    }

    /**
     * Create a GPU buffer with specified parameters
     * @private
     */
    _createBuffer(name, size, usage) {
        const startTime = performance.now();
        
        if (size > this.maxBufferSize) {
            throw new Error(`Buffer ${name} size ${size} exceeds maximum ${this.maxBufferSize}`);
        }
        
        const buffer = this.device.createBuffer({
            label: `neural_network_${name}`,
            size: size,
            usage: usage
        });
        
        // Store buffer reference and size
        this.buffers.set(name, buffer);
        this.bufferSizes.set(name, size);
        this.totalMemoryUsed += size;
        
        const creationTime = performance.now() - startTime;
        this.bufferCreationTimes.push(creationTime);
        
        console.log(`Created buffer ${name}: ${this._formatBytes(size)}, usage: ${this._formatUsage(usage)}`);
        
        return buffer;
    }

    /**
     * Create bind groups for shader operations
     * @param {Object} buffers - Buffer references
     * @param {Object} layouts - Bind group layouts from ShaderManager
     * @returns {Object} Created bind groups
     */
    createBindGroups(buffers, layouts) {
        const bindGroups = {};
        
        // Matrix multiplication bind group
        bindGroups.matmul = this.device.createBindGroup({
            label: 'matmul_bind_group',
            layout: layouts.matmul,
            entries: [
                { binding: 0, resource: { buffer: buffers.input } },
                { binding: 1, resource: { buffer: buffers.weightsHidden } },
                { binding: 2, resource: { buffer: buffers.biasHidden } },
                { binding: 3, resource: { buffer: buffers.hidden } },
                { binding: 4, resource: { buffer: buffers.matmulParams } }
            ]
        });
        
        // Activation function bind group
        bindGroups.activation = this.device.createBindGroup({
            label: 'activation_bind_group',
            layout: layouts.activation,
            entries: [
                { binding: 0, resource: { buffer: buffers.hidden } },
                { binding: 1, resource: { buffer: buffers.hidden } }, // In-place operation
                { binding: 2, resource: { buffer: buffers.activationParams } }
            ]
        });
        
        // Output layer bind group
        bindGroups.output = this.device.createBindGroup({
            label: 'output_bind_group',
            layout: layouts.matmul,
            entries: [
                { binding: 0, resource: { buffer: buffers.hidden } },
                { binding: 1, resource: { buffer: buffers.weightsOutput } },
                { binding: 2, resource: { buffer: buffers.biasOutput } },
                { binding: 3, resource: { buffer: buffers.output } },
                { binding: 4, resource: { buffer: buffers.matmulParams } }
            ]
        });
        
        // Q-learning bind group (if training buffers exist)
        if (buffers.qValues) {
            bindGroups.qlearning = this.device.createBindGroup({
                label: 'qlearning_bind_group',
                layout: layouts.qlearning,
                entries: [
                    { binding: 0, resource: { buffer: buffers.qValues } },
                    { binding: 1, resource: { buffer: buffers.targetQValues } },
                    { binding: 2, resource: { buffer: buffers.actions } },
                    { binding: 3, resource: { buffer: buffers.rewards } },
                    { binding: 4, resource: { buffer: buffers.dones } },
                    { binding: 5, resource: { buffer: buffers.hidden } },
                    { binding: 6, resource: { buffer: buffers.input } },
                    { binding: 7, resource: { buffer: buffers.weightsHidden } },
                    { binding: 8, resource: { buffer: buffers.biasHidden } },
                    { binding: 9, resource: { buffer: buffers.weightsOutput } },
                    { binding: 10, resource: { buffer: buffers.biasOutput } },
                    { binding: 11, resource: { buffer: buffers.tdErrors } },
                    { binding: 12, resource: { buffer: buffers.qlearningParams } }
                ]
            });
        }
        
        // Store bind groups
        Object.entries(bindGroups).forEach(([name, bindGroup]) => {
            this.bindGroups.set(name, bindGroup);
        });
        
        console.log(`Created ${Object.keys(bindGroups).length} bind groups`);
        
        return bindGroups;
    }

    /**
     * Upload data to GPU buffer
     * @param {string} bufferName - Name of the buffer
     * @param {ArrayBuffer|TypedArray} data - Data to upload
     * @param {number} offset - Offset in buffer (bytes)
     */
    async uploadData(bufferName, data, offset = 0) {
        const startTime = performance.now();
        
        const buffer = this.buffers.get(bufferName);
        if (!buffer) {
            throw new Error(`Buffer ${bufferName} not found`);
        }
        
        // Convert to Uint8Array if needed
        let uint8Data;
        if (data instanceof ArrayBuffer) {
            uint8Data = new Uint8Array(data);
        } else if (ArrayBuffer.isView(data)) {
            uint8Data = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        } else {
            throw new Error('Data must be ArrayBuffer or TypedArray');
        }
        
        // Write data to buffer
        this.device.queue.writeBuffer(buffer, offset, uint8Data);
        
        const transferTime = performance.now() - startTime;
        this.memoryTransferTimes.push(transferTime);
        
        console.log(`Uploaded ${this._formatBytes(uint8Data.length)} to ${bufferName} in ${transferTime.toFixed(2)}ms`);
    }

    /**
     * Download data from GPU buffer
     * @param {string} bufferName - Name of the buffer
     * @param {number} size - Size to read (bytes)
     * @param {number} offset - Offset in buffer (bytes)
     * @returns {Promise<ArrayBuffer>} Downloaded data
     */
    async downloadData(bufferName, size, offset = 0) {
        const startTime = performance.now();
        
        const buffer = this.buffers.get(bufferName);
        if (!buffer) {
            throw new Error(`Buffer ${bufferName} not found`);
        }
        
        // Create a staging buffer for reading
        const stagingBuffer = this.device.createBuffer({
            label: `staging_${bufferName}_read`,
            size: size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        
        // Copy data to staging buffer
        const commandEncoder = this.device.createCommandEncoder({
            label: `copy_${bufferName}_to_staging`
        });
        
        commandEncoder.copyBufferToBuffer(buffer, offset, stagingBuffer, 0, size);
        
        const commands = commandEncoder.finish();
        this.device.queue.submit([commands]);
        
        // Map and read data
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = stagingBuffer.getMappedRange().slice();
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        const transferTime = performance.now() - startTime;
        this.memoryTransferTimes.push(transferTime);
        
        console.log(`Downloaded ${this._formatBytes(size)} from ${bufferName} in ${transferTime.toFixed(2)}ms`);
        
        return arrayBuffer;
    }

    /**
     * Update uniform buffer with parameters
     * @param {string} bufferName - Name of uniform buffer
     * @param {Object} params - Parameters to upload
     */
    updateUniformBuffer(bufferName, params) {
        const buffer = this.buffers.get(bufferName);
        if (!buffer) {
            throw new Error(`Uniform buffer ${bufferName} not found`);
        }
        
        // Convert parameters to appropriate format
        let data;
        if (bufferName === 'matmulParams') {
            // Matrix multiplication parameters: M, K, N, padding
            data = new Uint32Array([params.M || 0, params.K || 0, params.N || 0, 0]);
        } else if (bufferName === 'activationParams') {
            // Activation parameters: size, padding...
            data = new Uint32Array([params.size || 0, 0, 0, 0]);
        } else if (bufferName === 'qlearningParams') {
            // Q-learning parameters: batch_size, input_size, hidden_size, output_size, learning_rate, gamma, epsilon, clip_grad_norm
            data = new Float32Array([
                params.batch_size || 1,
                params.input_size || 2,
                params.hidden_size || 8,
                params.output_size || 3,
                params.learning_rate || 0.01,
                params.gamma || 0.99,
                params.epsilon || 1e-8,
                params.clip_grad_norm || 1.0
            ]);
        } else {
            throw new Error(`Unknown uniform buffer format for ${bufferName}`);
        }
        
        this.device.queue.writeBuffer(buffer, 0, data);
    }

    /**
     * Get buffer by name
     * @param {string} name - Buffer name
     * @returns {GPUBuffer} The requested buffer
     */
    getBuffer(name) {
        const buffer = this.buffers.get(name);
        if (!buffer) {
            throw new Error(`Buffer ${name} not found. Available buffers: ${Array.from(this.buffers.keys()).join(', ')}`);
        }
        return buffer;
    }

    /**
     * Get bind group by name
     * @param {string} name - Bind group name
     * @returns {GPUBindGroup} The requested bind group
     */
    getBindGroup(name) {
        const bindGroup = this.bindGroups.get(name);
        if (!bindGroup) {
            throw new Error(`Bind group ${name} not found. Available bind groups: ${Array.from(this.bindGroups.keys()).join(', ')}`);
        }
        return bindGroup;
    }

    /**
     * Get memory usage statistics
     * @returns {Object} Memory usage information
     */
    getMemoryUsage() {
        const bufferInfo = Array.from(this.buffers.entries()).map(([name, buffer]) => ({
            name,
            size: this.bufferSizes.get(name),
            sizeFormatted: this._formatBytes(this.bufferSizes.get(name))
        }));

        const avgCreationTime = this.bufferCreationTimes.length > 0
            ? this.bufferCreationTimes.reduce((a, b) => a + b) / this.bufferCreationTimes.length
            : 0;

        const avgTransferTime = this.memoryTransferTimes.length > 0
            ? this.memoryTransferTimes.reduce((a, b) => a + b) / this.memoryTransferTimes.length
            : 0;

        return {
            totalMemoryUsed: this.totalMemoryUsed,
            totalMemoryFormatted: this._formatBytes(this.totalMemoryUsed),
            bufferCount: this.buffers.size,
            bindGroupCount: this.bindGroups.size,
            bufferInfo,
            performance: {
                avgCreationTime: avgCreationTime.toFixed(2) + 'ms',
                avgTransferTime: avgTransferTime.toFixed(2) + 'ms',
                totalCreations: this.bufferCreationTimes.length,
                totalTransfers: this.memoryTransferTimes.length
            }
        };
    }

    /**
     * Format buffer usage flags for display
     * @private
     */
    _formatUsage(usage) {
        const flags = [];
        if (usage & GPUBufferUsage.VERTEX) flags.push('VERTEX');
        if (usage & GPUBufferUsage.INDEX) flags.push('INDEX');
        if (usage & GPUBufferUsage.UNIFORM) flags.push('UNIFORM');
        if (usage & GPUBufferUsage.STORAGE) flags.push('STORAGE');
        if (usage & GPUBufferUsage.COPY_SRC) flags.push('COPY_SRC');
        if (usage & GPUBufferUsage.COPY_DST) flags.push('COPY_DST');
        if (usage & GPUBufferUsage.MAP_READ) flags.push('MAP_READ');
        if (usage & GPUBufferUsage.MAP_WRITE) flags.push('MAP_WRITE');
        return flags.join(' | ');
    }

    /**
     * Format byte size for display
     * @private
     */
    _formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Clean up all buffers and resources
     */
    destroy() {
        // Destroy all buffers
        for (const [name, buffer] of this.buffers) {
            try {
                buffer.destroy();
            } catch (error) {
                console.warn(`Failed to destroy buffer ${name}:`, error);
            }
        }
        
        // Clear all maps
        this.buffers.clear();
        this.bindGroups.clear();
        this.bufferPools.clear();
        this.bufferSizes.clear();
        
        // Reset counters
        this.totalMemoryUsed = 0;
        this.bufferCreationTimes = [];
        this.memoryTransferTimes = [];
        
        console.log('Buffer manager destroyed');
    }
}