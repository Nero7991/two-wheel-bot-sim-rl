/**
 * Enhanced Buffer Manager for WebGPU Neural Network Operations
 * 
 * Advanced GPU buffer management with pooling, async operations, and optimization
 * for neural network operations. Provides efficient memory management for the
 * two-wheel balancing robot RL application with production-ready features.
 * 
 * Features:
 * - Buffer pooling and reuse for performance optimization
 * - Async buffer operations with proper error handling
 * - Memory usage tracking and validation
 * - Batch operation support for training efficiency
 * - Staging buffer management for optimal CPU-GPU transfers
 * - Buffer size validation and limits checking
 */

/**
 * Buffer usage types for different operations
 */
export const BufferUsageType = {
    STORAGE_READ_WRITE: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    STORAGE_READ_ONLY: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    STORAGE_WRITE_ONLY: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    UNIFORM: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    STAGING_UPLOAD: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    STAGING_DOWNLOAD: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    VERTEX: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    INDEX: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
};

/**
 * Buffer pool entry for reuse management
 */
class BufferPoolEntry {
    constructor(buffer, size, usage, lastUsed = Date.now()) {
        this.buffer = buffer;
        this.size = size;
        this.usage = usage;
        this.lastUsed = lastUsed;
        this.isInUse = false;
        this.useCount = 0;
    }

    markUsed() {
        this.isInUse = true;
        this.lastUsed = Date.now();
        this.useCount++;
    }

    markAvailable() {
        this.isInUse = false;
        this.lastUsed = Date.now();
    }
}

/**
 * Enhanced GPU buffer management for neural network operations
 */
export class BufferManager {
    constructor(device, options = {}) {
        this.device = device;
        this.isDestroyed = false;
        
        // Configuration
        this.config = {
            maxBufferSize: options.maxBufferSize || 1024 * 1024 * 64, // 64MB default
            maxTotalMemory: options.maxTotalMemory || 1024 * 1024 * 512, // 512MB total limit
            alignment: options.alignment || 256,
            poolEnabled: options.poolEnabled !== false,
            poolMaxAge: options.poolMaxAge || 30000, // 30 seconds
            poolMaxSize: options.poolMaxSize || 50,
            asyncTimeout: options.asyncTimeout || 10000, // 10 seconds
            enableValidation: options.enableValidation !== false,
            enableProfiling: options.enableProfiling !== false
        };
        
        // Named buffer storage
        this.buffers = new Map();
        this.bindGroups = new Map();
        this.bufferMetadata = new Map();
        
        // Enhanced buffer pools for reuse
        this.bufferPools = new Map(); // key: "size_usage", value: BufferPoolEntry[]
        this.poolStats = {
            hits: 0,
            misses: 0,
            evictions: 0,
            totalReused: 0
        };
        
        // Memory tracking
        this.memoryUsage = {
            totalAllocated: 0,
            totalActive: 0,
            totalPooled: 0,
            bufferCount: 0,
            pooledCount: 0
        };
        
        // Performance tracking
        this.performanceMetrics = {
            bufferCreationTimes: [],
            memoryTransferTimes: [],
            mappingTimes: [],
            asyncOperationTimes: [],
            errorCount: 0,
            warningCount: 0
        };
        
        // Async operation tracking
        this.pendingOperations = new Map();
        this.operationCounter = 0;
        
        // Validation and error handling
        this.validationEnabled = this.config.enableValidation;
        this.errorLog = [];
        
        // Buffer type tracking for optimization
        this.bufferTypeStats = new Map();
        
        // Setup periodic pool cleanup
        this._setupPoolCleanup();
        
        console.log('Enhanced BufferManager initialized with config:', this.config);
    }

    /**
     * Create a GPU buffer with comprehensive validation and pooling
     * @param {GPUDevice} device - WebGPU device
     * @param {ArrayBuffer|TypedArray|number} data - Data to upload or size in bytes
     * @param {number} usage - Buffer usage flags (use BufferUsageType constants)
     * @param {Object} options - Buffer creation options
     * @param {string} options.label - Buffer label for debugging
     * @param {boolean} options.allowReuse - Allow buffer reuse from pool (default: true)
     * @param {boolean} options.persistent - Keep buffer even when not in use
     * @returns {Promise<GPUBuffer>} Created or reused buffer
     */
    async createBuffer(device, data, usage, options = {}) {
        this._validateNotDestroyed();
        
        const {
            label = 'unnamed_buffer',
            allowReuse = true,
            persistent = false
        } = options;
        
        const startTime = performance.now();
        
        try {
            // Determine buffer size
            let size;
            let initialData = null;
            
            if (typeof data === 'number') {
                size = data;
            } else {
                if (data instanceof ArrayBuffer) {
                    size = data.byteLength;
                    initialData = new Uint8Array(data);
                } else if (ArrayBuffer.isView(data)) {
                    size = data.byteLength;
                    initialData = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
                } else {
                    throw new Error('Data must be ArrayBuffer, TypedArray, or number (size)');
                }
            }
            
            // Validate buffer size
            this._validateBufferSize(size, label);
            
            // Align size
            const alignedSize = this._alignSize(size);
            
            // Try to get buffer from pool if allowed
            let buffer = null;
            if (allowReuse && this.config.poolEnabled) {
                buffer = this._getBufferFromPool(alignedSize, usage);
            }
            
            // Create new buffer if not found in pool
            if (!buffer) {
                this._validateMemoryLimit(alignedSize);
                
                buffer = device.createBuffer({
                    label: `enhanced_${label}`,
                    size: alignedSize,
                    usage: usage
                });
                
                this.poolStats.misses++;
                this._updateMemoryUsage(alignedSize, 'allocate');
                
                console.log(`Created new buffer ${label}: ${this._formatBytes(alignedSize)}`);
            } else {
                this.poolStats.hits++;
                this.poolStats.totalReused++;
                console.log(`Reused buffer ${label}: ${this._formatBytes(alignedSize)}`);
            }
            
            // Upload initial data if provided
            if (initialData) {
                await this.updateBuffer(device.queue, buffer, initialData, 0, { label });
            }
            
            // Track buffer metadata
            this._trackBufferMetadata(buffer, {
                label,
                size: alignedSize,
                usage,
                persistent,
                createdAt: Date.now(),
                accessCount: 0
            });
            
            const creationTime = performance.now() - startTime;
            this.performanceMetrics.bufferCreationTimes.push(creationTime);
            
            return buffer;
            
        } catch (error) {
            this._handleError(`Failed to create buffer ${label}`, error);
            throw error;
        }
    }
    
    /**
     * Update buffer data with efficient staging and validation
     * @param {GPUQueue} queue - WebGPU queue
     * @param {GPUBuffer} buffer - Target buffer
     * @param {ArrayBuffer|TypedArray} data - Data to upload
     * @param {number} offset - Offset in buffer (bytes)
     * @param {Object} options - Update options
     * @param {string} options.label - Operation label for debugging
     * @param {boolean} options.useStaging - Use staging buffer for large uploads
     * @returns {Promise<void>}
     */
    async updateBuffer(queue, buffer, data, offset = 0, options = {}) {
        this._validateNotDestroyed();
        
        const { label = 'buffer_update', useStaging = false } = options;
        const startTime = performance.now();
        const operationId = this._generateOperationId();
        
        try {
            // Convert data to Uint8Array
            let uint8Data;
            if (data instanceof ArrayBuffer) {
                uint8Data = new Uint8Array(data);
            } else if (ArrayBuffer.isView(data)) {
                uint8Data = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            } else {
                throw new Error('Data must be ArrayBuffer or TypedArray');
            }
            
            // Validate operation
            this._validateBufferUpdate(buffer, uint8Data, offset);
            
            // Track pending operation
            this.pendingOperations.set(operationId, {
                type: 'update',
                buffer,
                size: uint8Data.length,
                startTime
            });
            
            // Choose upload method based on size and options
            if (useStaging && uint8Data.length > 64 * 1024) { // 64KB threshold
                await this._updateBufferStaging(queue, buffer, uint8Data, offset, label);
            } else {
                queue.writeBuffer(buffer, offset, uint8Data);
            }
            
            // Update buffer access tracking
            this._updateBufferAccess(buffer);
            
            const transferTime = performance.now() - startTime;
            this.performanceMetrics.memoryTransferTimes.push(transferTime);
            
            if (this.config.enableProfiling) {
                console.log(`Updated buffer ${label}: ${this._formatBytes(uint8Data.length)} in ${transferTime.toFixed(2)}ms`);
            }
            
        } catch (error) {
            this._handleError(`Failed to update buffer ${label}`, error);
            throw error;
        } finally {
            this.pendingOperations.delete(operationId);
        }
    }
    
    /**
     * Read buffer data with async operations and validation
     * @param {GPUDevice} device - WebGPU device
     * @param {GPUBuffer} buffer - Source buffer
     * @param {number} size - Size to read (bytes)
     * @param {number} offset - Offset in buffer (bytes)
     * @param {Object} options - Read options
     * @param {string} options.label - Operation label for debugging
     * @param {number} options.timeout - Operation timeout (ms)
     * @returns {Promise<ArrayBuffer>} Downloaded data
     */
    async readBuffer(device, buffer, size, offset = 0, options = {}) {
        this._validateNotDestroyed();
        
        const {
            label = 'buffer_read',
            timeout = this.config.asyncTimeout
        } = options;
        
        const startTime = performance.now();
        const operationId = this._generateOperationId();
        
        try {
            // Validate read operation
            this._validateBufferRead(buffer, size, offset);
            
            // Track pending operation
            this.pendingOperations.set(operationId, {
                type: 'read',
                buffer,
                size,
                startTime
            });
            
            // Create staging buffer for reading
            const stagingBuffer = await this.createBuffer(device, size, BufferUsageType.STAGING_DOWNLOAD, {
                label: `staging_${label}_read`,
                allowReuse: true
            });
            
            // Copy data to staging buffer
            const commandEncoder = device.createCommandEncoder({
                label: `copy_${label}_to_staging`
            });
            
            commandEncoder.copyBufferToBuffer(buffer, offset, stagingBuffer, 0, size);
            const commands = commandEncoder.finish();
            device.queue.submit([commands]);
            
            // Map and read data with timeout
            const mapPromise = stagingBuffer.mapAsync(GPUMapMode.READ);
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error(`Buffer read timeout after ${timeout}ms`)), timeout);
            });
            
            await Promise.race([mapPromise, timeoutPromise]);
            
            const mappingStartTime = performance.now();
            const arrayBuffer = stagingBuffer.getMappedRange().slice();
            stagingBuffer.unmap();
            
            const mappingTime = performance.now() - mappingStartTime;
            this.performanceMetrics.mappingTimes.push(mappingTime);
            
            // Return staging buffer to pool
            this._returnBufferToPool(stagingBuffer, size, BufferUsageType.STAGING_DOWNLOAD);
            
            // Update buffer access tracking
            this._updateBufferAccess(buffer);
            
            const transferTime = performance.now() - startTime;
            this.performanceMetrics.memoryTransferTimes.push(transferTime);
            this.performanceMetrics.asyncOperationTimes.push(transferTime);
            
            if (this.config.enableProfiling) {
                console.log(`Read buffer ${label}: ${this._formatBytes(size)} in ${transferTime.toFixed(2)}ms`);
            }
            
            return arrayBuffer;
            
        } catch (error) {
            this._handleError(`Failed to read buffer ${label}`, error);
            throw error;
        } finally {
            this.pendingOperations.delete(operationId);
        }
    }
    
    /**
     * Create buffers for neural network architecture with enhanced features
     * @param {Object} architecture - Network architecture specification
     * @param {number} architecture.inputSize - Input layer size (2)
     * @param {number} architecture.hiddenSize - Hidden layer size (4-16)
     * @param {number} architecture.outputSize - Output layer size (3)
     * @param {number} architecture.batchSize - Batch size for training (1-32)
     * @param {Object} options - Buffer creation options
     * @returns {Promise<Object>} Created buffer references
     */
    async createNetworkBuffers(architecture, options = {}) {
        const { inputSize, hiddenSize, outputSize, batchSize = 1 } = architecture;
        const { persistent = false, allowReuse = true } = options;
        
        console.log(`Creating enhanced buffers for ${inputSize}-${hiddenSize}-${outputSize} network, batch size: ${batchSize}`);
        
        const buffers = {};
        const startTime = performance.now();
        
        try {
            // Calculate buffer sizes
            const sizes = this._calculateBufferSizes(architecture);
            
            // Create input/output buffers with enhanced features
            buffers.input = await this.createBuffer(this.device, sizes.input, BufferUsageType.STORAGE_READ_ONLY, {
                label: 'neural_input',
                persistent,
                allowReuse
            });
            
            buffers.hidden = await this.createBuffer(this.device, sizes.hidden, BufferUsageType.STORAGE_READ_WRITE, {
                label: 'neural_hidden',
                persistent,
                allowReuse
            });
            
            buffers.output = await this.createBuffer(this.device, sizes.output, BufferUsageType.STORAGE_WRITE_ONLY, {
                label: 'neural_output',
                persistent,
                allowReuse
            });
            
            // Create weight buffers with read-write access
            buffers.weightsHidden = await this.createBuffer(this.device, sizes.weightsHidden, BufferUsageType.STORAGE_READ_WRITE, {
                label: 'weights_hidden',
                persistent: true, // Weights should persist
                allowReuse
            });
            
            buffers.weightsOutput = await this.createBuffer(this.device, sizes.weightsOutput, BufferUsageType.STORAGE_READ_WRITE, {
                label: 'weights_output',
                persistent: true,
                allowReuse
            });
            
            // Create bias buffers
            buffers.biasHidden = await this.createBuffer(this.device, sizes.biasHidden, BufferUsageType.STORAGE_READ_WRITE, {
                label: 'bias_hidden',
                persistent: true,
                allowReuse
            });
            
            buffers.biasOutput = await this.createBuffer(this.device, sizes.biasOutput, BufferUsageType.STORAGE_READ_WRITE, {
                label: 'bias_output',
                persistent: true,
                allowReuse
            });
            
            // Create uniform buffers for parameters
            buffers.matmulParams = await this.createBuffer(this.device, 16, BufferUsageType.UNIFORM, {
                label: 'matmul_params',
                persistent,
                allowReuse
            });
            
            buffers.activationParams = await this.createBuffer(this.device, 16, BufferUsageType.UNIFORM, {
                label: 'activation_params',
                persistent,
                allowReuse
            });
            
            // Create staging buffers for efficient CPU-GPU data transfer
            buffers.stagingInput = await this.createBuffer(this.device, sizes.input, BufferUsageType.STAGING_UPLOAD, {
                label: 'staging_input',
                persistent: false,
                allowReuse: true
            });
            
            buffers.stagingOutput = await this.createBuffer(this.device, sizes.output, BufferUsageType.STAGING_DOWNLOAD, {
                label: 'staging_output',
                persistent: false,
                allowReuse: true
            });
            
            // Create buffers for training (Q-learning) with batch support
            if (batchSize > 1) {
                buffers.qValues = await this.createBuffer(this.device, sizes.qValues, BufferUsageType.STORAGE_READ_WRITE, {
                    label: `q_values_batch_${batchSize}`,
                    persistent,
                    allowReuse
                });
                
                buffers.targetQValues = await this.createBuffer(this.device, sizes.qValues, BufferUsageType.STORAGE_READ_WRITE, {
                    label: `target_q_values_batch_${batchSize}`,
                    persistent,
                    allowReuse
                });
                
                buffers.actions = await this.createBuffer(this.device, batchSize * 4, BufferUsageType.STORAGE_READ_ONLY, {
                    label: `actions_batch_${batchSize}`,
                    persistent,
                    allowReuse
                });
                
                buffers.rewards = await this.createBuffer(this.device, batchSize * 4, BufferUsageType.STORAGE_READ_ONLY, {
                    label: `rewards_batch_${batchSize}`,
                    persistent,
                    allowReuse
                });
                
                buffers.dones = await this.createBuffer(this.device, batchSize * 4, BufferUsageType.STORAGE_READ_ONLY, {
                    label: `dones_batch_${batchSize}`,
                    persistent,
                    allowReuse
                });
                
                buffers.tdErrors = await this.createBuffer(this.device, batchSize * 4, BufferUsageType.STORAGE_WRITE_ONLY, {
                    label: `td_errors_batch_${batchSize}`,
                    persistent,
                    allowReuse
                });
                
                buffers.qlearningParams = await this.createBuffer(this.device, 32, BufferUsageType.UNIFORM, {
                    label: `qlearning_params_batch_${batchSize}`,
                    persistent,
                    allowReuse
                });
            }
            
            // Store named buffer references
            Object.entries(buffers).forEach(([name, buffer]) => {
                this.buffers.set(name, buffer);
            });
            
            const creationTime = performance.now() - startTime;
            const bufferCount = Object.keys(buffers).length;
            const totalMemory = this.memoryUsage.totalAllocated;
            
            console.log(`Created ${bufferCount} enhanced buffers in ${creationTime.toFixed(2)}ms`);
            console.log(`Total memory allocated: ${this._formatBytes(totalMemory)}`);
            console.log(`Buffer pool stats: ${this.poolStats.hits} hits, ${this.poolStats.misses} misses`);
            
            return buffers;
            
        } catch (error) {
            this._handleError('Failed to create network buffers', error);
            throw error;
        }
    }

    /**
     * Calculate required buffer sizes for architecture with enhanced validation
     * @private
     */
    _calculateBufferSizes(architecture) {
        const { inputSize, hiddenSize, outputSize, batchSize = 1 } = architecture;
        
        // Validate architecture parameters
        if (!inputSize || !hiddenSize || !outputSize || !batchSize) {
            throw new Error('Invalid architecture: all sizes must be positive');
        }
        
        if (batchSize > 1024) {
            console.warn(`Large batch size ${batchSize} detected, may impact performance`);
        }
        
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
        
        // Calculate total memory requirement
        const totalMemory = Object.values(sizes).reduce((sum, size) => sum + size, 0);
        
        if (this.validationEnabled && totalMemory > this.config.maxTotalMemory) {
            throw new Error(`Total buffer memory ${this._formatBytes(totalMemory)} exceeds limit ${this._formatBytes(this.config.maxTotalMemory)}`);
        }
        
        console.log(`Calculated buffer sizes for ${inputSize}-${hiddenSize}-${outputSize} (batch: ${batchSize}):`);
        Object.entries(sizes).forEach(([name, size]) => {
            console.log(`  ${name}: ${this._formatBytes(size)}`);
        });
        console.log(`  Total: ${this._formatBytes(totalMemory)}`);
        
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

/**
 * Utility functions for buffer management
 */

/**
 * Create a buffer with validation and error handling
 * @param {GPUDevice} device - WebGPU device
 * @param {ArrayBuffer|TypedArray|number} data - Data or size
 * @param {number} usage - Buffer usage flags
 * @param {Object} options - Creation options
 * @returns {Promise<GPUBuffer>} Created buffer
 */
export async function createBuffer(device, data, usage, options = {}) {
    // Create a temporary buffer manager for standalone usage
    const bufferManager = new BufferManager(device, {
        enableValidation: options.enableValidation !== false,
        enableProfiling: options.enableProfiling !== false
    });
    
    try {
        const buffer = await bufferManager.createBuffer(device, data, usage, options);
        return buffer;
    } finally {
        // Clean up temporary buffer manager (but not the buffer)
        bufferManager.buffers.clear();
        bufferManager.bufferMetadata.clear();
    }
}

/**
 * Update buffer data with efficient upload
 * @param {GPUQueue} queue - WebGPU queue
 * @param {GPUBuffer} buffer - Target buffer
 * @param {ArrayBuffer|TypedArray} data - Data to upload
 * @param {number} offset - Offset in buffer
 * @param {Object} options - Update options
 * @returns {Promise<void>}
 */
export async function updateBuffer(queue, buffer, data, offset = 0, options = {}) {
    const startTime = performance.now();
    
    try {
        // Convert data to Uint8Array
        let uint8Data;
        if (data instanceof ArrayBuffer) {
            uint8Data = new Uint8Array(data);
        } else if (ArrayBuffer.isView(data)) {
            uint8Data = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        } else {
            throw new Error('Data must be ArrayBuffer or TypedArray');
        }
        
        // Use appropriate upload method based on size
        if (uint8Data.length > 64 * 1024 && options.useStaging) {
            // Use staging buffer for large uploads
            const device = queue.getDevice ? queue.getDevice() : options.device;
            if (!device) {
                throw new Error('Device required for staging buffer operations');
            }
            
            const stagingBuffer = device.createBuffer({
                label: 'staging_upload',
                size: uint8Data.length,
                usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
            });
            
            await stagingBuffer.mapAsync(GPUMapMode.WRITE);
            const mappedRange = new Uint8Array(stagingBuffer.getMappedRange());
            mappedRange.set(uint8Data);
            stagingBuffer.unmap();
            
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(stagingBuffer, 0, buffer, offset, uint8Data.length);
            queue.submit([commandEncoder.finish()]);
            
            stagingBuffer.destroy();
        } else {
            // Direct upload for smaller data
            queue.writeBuffer(buffer, offset, uint8Data);
        }
        
        const transferTime = performance.now() - startTime;
        
        if (options.enableProfiling) {
            console.log(`Buffer update: ${(uint8Data.length / 1024).toFixed(2)}KB in ${transferTime.toFixed(2)}ms`);
        }
        
    } catch (error) {
        console.error('Failed to update buffer:', error);
        throw error;
    }
}

/**
 * Read buffer data with async operations
 * @param {GPUDevice} device - WebGPU device
 * @param {GPUBuffer} buffer - Source buffer
 * @param {number} size - Size to read
 * @param {number} offset - Offset in buffer
 * @param {Object} options - Read options
 * @returns {Promise<ArrayBuffer>} Downloaded data
 */
export async function readBuffer(device, buffer, size, offset = 0, options = {}) {
    const startTime = performance.now();
    
    try {
        // Create staging buffer for reading
        const stagingBuffer = device.createBuffer({
            label: 'staging_download',
            size: size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        
        // Copy data to staging buffer
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(buffer, offset, stagingBuffer, 0, size);
        device.queue.submit([commandEncoder.finish()]);
        
        // Map and read data
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = stagingBuffer.getMappedRange().slice();
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        
        const transferTime = performance.now() - startTime;
        
        if (options.enableProfiling) {
            console.log(`Buffer read: ${(size / 1024).toFixed(2)}KB in ${transferTime.toFixed(2)}ms`);
        }
        
        return arrayBuffer;
        
    } catch (error) {
        console.error('Failed to read buffer:', error);
        throw error;
    }
}

/**
 * Validate buffer operations for safety
 * @param {GPUBuffer} buffer - Buffer to validate
 * @param {number} size - Operation size
 * @param {number} offset - Operation offset
 * @throws {Error} If validation fails
 */
export function validateBufferOperation(buffer, size, offset = 0) {
    if (!buffer) {
        throw new Error('Buffer is null or undefined');
    }
    
    if (size <= 0) {
        throw new Error(`Invalid size: ${size}`);
    }
    
    if (offset < 0) {
        throw new Error(`Invalid offset: ${offset}`);
    }
    
    // Note: We can't check buffer.size directly as it's not exposed in WebGPU API
    // Size validation should be done by the BufferManager that tracks metadata
}