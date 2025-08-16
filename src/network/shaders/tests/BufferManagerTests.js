/**
 * Comprehensive Test Suite for Enhanced BufferManager
 * 
 * Tests all aspects of the enhanced buffer management system including:
 * - Buffer creation and pooling
 * - Async operations with error handling
 * - Memory tracking and validation
 * - Batch operations for training efficiency
 * - Performance monitoring and optimization
 */

import { BufferManager, BufferUsageType, createBuffer, updateBuffer, readBuffer, validateBufferOperation } from '../BufferManager.js';

/**
 * Mock WebGPU device for testing
 */
class MockGPUDevice {
    constructor() {
        this.buffers = new Map();
        this.bufferCounter = 0;
        this.queue = new MockGPUQueue();
        this.destroyed = false;
    }

    createBuffer({ label, size, usage }) {
        if (this.destroyed) {
            throw new Error('Device destroyed');
        }
        
        const bufferId = `buffer_${++this.bufferCounter}`;
        const buffer = new MockGPUBuffer(bufferId, size, usage, label);
        this.buffers.set(bufferId, buffer);
        return buffer;
    }

    createCommandEncoder({ label = 'test_encoder' } = {}) {
        return new MockGPUCommandEncoder();
    }

    destroy() {
        this.destroyed = true;
        for (const buffer of this.buffers.values()) {
            buffer.destroy();
        }
        this.buffers.clear();
    }
}

class MockGPUQueue {
    writeBuffer(buffer, offset, data) {
        if (buffer.destroyed) {
            throw new Error('Buffer destroyed');
        }
        // Simulate write operation
        buffer.lastWriteTime = Date.now();
        buffer.lastWriteSize = data.length;
    }

    submit(commands) {
        // Simulate command submission
        return Promise.resolve();
    }
}

class MockGPUBuffer {
    constructor(id, size, usage, label) {
        this.id = id;
        this.size = size;
        this.usage = usage;
        this.label = label;
        this.destroyed = false;
        this.mapped = false;
        this.lastWriteTime = null;
        this.lastWriteSize = 0;
        this.mapData = null;
    }

    destroy() {
        this.destroyed = true;
    }

    async mapAsync(mode) {
        if (this.destroyed) {
            throw new Error('Buffer destroyed');
        }
        this.mapped = true;
        this.mapData = new ArrayBuffer(this.size);
        return Promise.resolve();
    }

    getMappedRange() {
        if (!this.mapped) {
            throw new Error('Buffer not mapped');
        }
        return this.mapData;
    }

    unmap() {
        this.mapped = false;
        this.mapData = null;
    }
}

class MockGPUCommandEncoder {
    constructor() {
        this.commands = [];
    }

    copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
        this.commands.push({
            type: 'copyBufferToBuffer',
            src: src.id,
            srcOffset,
            dst: dst.id,
            dstOffset,
            size
        });
    }

    finish() {
        return { commands: this.commands };
    }
}

/**
 * Test suite for BufferManager
 */
export class BufferManagerTests {
    constructor() {
        this.testResults = [];
        this.mockDevice = null;
        this.bufferManager = null;
    }

    /**
     * Run all tests
     */
    async runAllTests() {
        console.log('Starting BufferManager comprehensive test suite...');
        
        const tests = [
            'testBasicBufferCreation',
            'testBufferPooling',
            'testAsyncOperations',
            'testMemoryTracking',
            'testValidation',
            'testErrorHandling',
            'testBatchOperations',
            'testPerformanceMonitoring',
            'testNetworkBufferCreation',
            'testUtilityFunctions',
            'testMemoryLimits',
            'testLargeDataOperations',
            'testConcurrentOperations',
            'testBufferReuse',
            'testCleanupAndDestroy'
        ];

        let passed = 0;
        let failed = 0;

        for (const testName of tests) {
            try {
                await this.setup();
                await this[testName]();
                this.testResults.push({ test: testName, status: 'PASSED', error: null });
                passed++;
                console.log(`✓ ${testName}`);
            } catch (error) {
                this.testResults.push({ test: testName, status: 'FAILED', error: error.message });
                failed++;
                console.error(`✗ ${testName}: ${error.message}`);
            } finally {
                await this.cleanup();
            }
        }

        console.log(`\nTest Results: ${passed} passed, ${failed} failed`);
        return { passed, failed, results: this.testResults };
    }

    /**
     * Setup test environment
     */
    async setup() {
        this.mockDevice = new MockGPUDevice();
        this.bufferManager = new BufferManager(this.mockDevice, {
            maxBufferSize: 1024 * 1024, // 1MB for testing
            maxTotalMemory: 10 * 1024 * 1024, // 10MB for testing
            poolEnabled: true,
            poolMaxAge: 1000, // 1 second for faster testing
            poolMaxSize: 10,
            enableValidation: true,
            enableProfiling: true
        });
    }

    /**
     * Cleanup test environment
     */
    async cleanup() {
        if (this.bufferManager) {
            this.bufferManager.destroy();
            this.bufferManager = null;
        }
        if (this.mockDevice) {
            this.mockDevice.destroy();
            this.mockDevice = null;
        }
    }

    /**
     * Test basic buffer creation
     */
    async testBasicBufferCreation() {
        const testData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
        
        const buffer = await this.bufferManager.createBuffer(
            this.mockDevice,
            testData,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'test_buffer' }
        );

        if (!buffer) {
            throw new Error('Buffer creation failed');
        }

        if (buffer.destroyed) {
            throw new Error('Buffer should not be destroyed');
        }

        if (!buffer.label.includes('test_buffer')) {
            throw new Error('Buffer label not set correctly');
        }

        const memoryUsage = this.bufferManager.getMemoryUsage();
        if (memoryUsage.buffers.count === 0) {
            throw new Error('Buffer not tracked in memory usage');
        }
    }

    /**
     * Test buffer pooling functionality
     */
    async testBufferPooling() {
        const size = 1024;
        const usage = BufferUsageType.STORAGE_READ_WRITE;

        // Create first buffer
        const buffer1 = await this.bufferManager.createBuffer(
            this.mockDevice,
            size,
            usage,
            { label: 'pool_test_1' }
        );

        // Return to pool
        this.bufferManager._returnBufferToPool(buffer1, size, usage);

        // Create second buffer (should reuse from pool)
        const buffer2 = await this.bufferManager.createBuffer(
            this.mockDevice,
            size,
            usage,
            { label: 'pool_test_2' }
        );

        const memoryUsage = this.bufferManager.getMemoryUsage();
        if (memoryUsage.pool.hits === 0) {
            throw new Error('Buffer pool not working - no hits recorded');
        }

        if (memoryUsage.pool.totalReused === 0) {
            throw new Error('Buffer reuse not tracked');
        }
    }

    /**
     * Test async operations
     */
    async testAsyncOperations() {
        const testData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
        
        const buffer = await this.bufferManager.createBuffer(
            this.mockDevice,
            testData.byteLength * 2, // Make buffer larger
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'async_test' }
        );

        // Test async update
        await this.bufferManager.updateBuffer(
            this.mockDevice.queue,
            buffer,
            testData,
            0,
            { label: 'async_update_test' }
        );

        // Test async read
        const readData = await this.bufferManager.readBuffer(
            this.mockDevice,
            buffer,
            testData.byteLength,
            0,
            { label: 'async_read_test' }
        );

        if (!readData) {
            throw new Error('Async read failed');
        }

        if (readData.byteLength !== testData.byteLength) {
            throw new Error('Read data size mismatch');
        }
    }

    /**
     * Test memory tracking
     */
    async testMemoryTracking() {
        const initialUsage = this.bufferManager.getMemoryUsage();
        const initialMemory = initialUsage.memory.totalActive;

        const bufferSize = 1024;
        const buffer = await this.bufferManager.createBuffer(
            this.mockDevice,
            bufferSize,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'memory_tracking_test' }
        );

        const afterUsage = this.bufferManager.getMemoryUsage();
        const afterMemory = afterUsage.memory.totalActive;

        if (afterMemory <= initialMemory) {
            throw new Error('Memory tracking not working - no increase detected');
        }

        const expectedIncrease = this.bufferManager._alignSize(bufferSize);
        const actualIncrease = afterMemory - initialMemory;

        if (actualIncrease !== expectedIncrease) {
            throw new Error(`Memory increase mismatch: expected ${expectedIncrease}, got ${actualIncrease}`);
        }
    }

    /**
     * Test validation functionality
     */
    async testValidation() {
        // Test invalid buffer size
        try {
            await this.bufferManager.createBuffer(
                this.mockDevice,
                -1,
                BufferUsageType.STORAGE_READ_WRITE,
                { label: 'invalid_size_test' }
            );
            throw new Error('Should have thrown error for negative size');
        } catch (error) {
            if (!error.message.includes('Invalid buffer size')) {
                throw new Error('Wrong error message for invalid size');
            }
        }

        // Test buffer size limit
        try {
            await this.bufferManager.createBuffer(
                this.mockDevice,
                this.bufferManager.config.maxBufferSize + 1,
                BufferUsageType.STORAGE_READ_WRITE,
                { label: 'oversized_test' }
            );
            throw new Error('Should have thrown error for oversized buffer');
        } catch (error) {
            if (!error.message.includes('exceeds maximum')) {
                throw new Error('Wrong error message for oversized buffer');
            }
        }
    }

    /**
     * Test error handling
     */
    async testErrorHandling() {
        const buffer = await this.bufferManager.createBuffer(
            this.mockDevice,
            1024,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'error_test' }
        );

        // Test invalid data type
        try {
            await this.bufferManager.updateBuffer(
                this.mockDevice.queue,
                buffer,
                "invalid data",
                0,
                { label: 'invalid_data_test' }
            );
            throw new Error('Should have thrown error for invalid data type');
        } catch (error) {
            if (!error.message.includes('Data must be ArrayBuffer or TypedArray')) {
                throw new Error('Wrong error message for invalid data');
            }
        }

        const metrics = this.bufferManager.getPerformanceMetrics();
        if (metrics.errors.count === 0) {
            throw new Error('Error not tracked in performance metrics');
        }
    }

    /**
     * Test batch operations
     */
    async testBatchOperations() {
        const operations = [
            {
                type: 'create',
                data: 1024,
                usage: BufferUsageType.STORAGE_READ_WRITE,
                options: { label: 'batch_create_1' }
            },
            {
                type: 'create',
                data: 2048,
                usage: BufferUsageType.STORAGE_READ_ONLY,
                options: { label: 'batch_create_2' }
            },
            {
                type: 'create',
                data: 512,
                usage: BufferUsageType.UNIFORM,
                options: { label: 'batch_create_3' }
            }
        ];

        const results = await this.bufferManager.batchOperations(operations);

        if (results.length !== operations.length) {
            throw new Error('Batch operation result count mismatch');
        }

        for (const result of results) {
            if (result.error) {
                throw new Error(`Batch operation failed: ${result.error.message}`);
            }
            if (!result.result) {
                throw new Error('Batch operation result missing');
            }
        }
    }

    /**
     * Test performance monitoring
     */
    async testPerformanceMonitoring() {
        // Create a buffer to generate metrics
        await this.bufferManager.createBuffer(
            this.mockDevice,
            1024,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'performance_test' }
        );

        const metrics = this.bufferManager.getPerformanceMetrics();
        
        if (metrics.bufferCreation.count === 0) {
            throw new Error('Performance metrics not tracking buffer creation');
        }

        if (metrics.bufferCreation.avg <= 0) {
            throw new Error('Invalid average creation time');
        }

        const memoryUsage = this.bufferManager.getMemoryUsage();
        if (!memoryUsage.performance.avgCreationTime) {
            throw new Error('Memory usage not tracking performance');
        }
    }

    /**
     * Test network buffer creation
     */
    async testNetworkBufferCreation() {
        const architecture = {
            inputSize: 2,
            hiddenSize: 8,
            outputSize: 3,
            batchSize: 4
        };

        const buffers = await this.bufferManager.createNetworkBuffers(architecture);

        const expectedBuffers = [
            'input', 'hidden', 'output',
            'weightsHidden', 'weightsOutput',
            'biasHidden', 'biasOutput',
            'matmulParams', 'activationParams',
            'stagingInput', 'stagingOutput',
            'qValues', 'targetQValues', 'actions', 'rewards', 'dones', 'tdErrors', 'qlearningParams'
        ];

        for (const bufferName of expectedBuffers) {
            if (!buffers[bufferName]) {
                throw new Error(`Network buffer ${bufferName} not created`);
            }
        }

        // Check if buffers are tracked
        for (const bufferName of expectedBuffers) {
            if (!this.bufferManager.buffers.has(bufferName)) {
                throw new Error(`Network buffer ${bufferName} not tracked`);
            }
        }
    }

    /**
     * Test utility functions
     */
    async testUtilityFunctions() {
        // Test standalone createBuffer function
        const buffer = await createBuffer(
            this.mockDevice,
            1024,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'utility_test' }
        );

        if (!buffer) {
            throw new Error('Utility createBuffer failed');
        }

        // Test standalone updateBuffer function
        const testData = new Float32Array([1, 2, 3, 4]);
        await updateBuffer(this.mockDevice.queue, buffer, testData);

        // Test standalone readBuffer function
        const readData = await readBuffer(this.mockDevice, buffer, testData.byteLength);
        
        if (!readData) {
            throw new Error('Utility readBuffer failed');
        }

        // Test validation function
        validateBufferOperation(buffer, 1024, 0);

        try {
            validateBufferOperation(null, 1024, 0);
            throw new Error('Should have thrown error for null buffer');
        } catch (error) {
            if (!error.message.includes('null or undefined')) {
                throw new Error('Wrong validation error message');
            }
        }
    }

    /**
     * Test memory limits
     */
    async testMemoryLimits() {
        const maxMemory = this.bufferManager.config.maxTotalMemory;
        const bufferSize = Math.floor(maxMemory / 3); // Create buffers that total more than limit

        // Create buffers up to near the limit
        await this.bufferManager.createBuffer(
            this.mockDevice,
            bufferSize,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'limit_test_1' }
        );

        await this.bufferManager.createBuffer(
            this.mockDevice,
            bufferSize,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'limit_test_2' }
        );

        // This should fail due to memory limit
        try {
            await this.bufferManager.createBuffer(
                this.mockDevice,
                bufferSize,
                BufferUsageType.STORAGE_READ_WRITE,
                { label: 'limit_test_3_should_fail' }
            );
            throw new Error('Should have thrown error for exceeding memory limit');
        } catch (error) {
            if (!error.message.includes('exceed memory limit')) {
                throw new Error('Wrong error message for memory limit');
            }
        }
    }

    /**
     * Test large data operations
     */
    async testLargeDataOperations() {
        const largeData = new Float32Array(16 * 1024); // 64KB
        for (let i = 0; i < largeData.length; i++) {
            largeData[i] = i;
        }

        const buffer = await this.bufferManager.createBuffer(
            this.mockDevice,
            largeData.byteLength,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'large_data_test' }
        );

        // Test large update with staging
        await this.bufferManager.updateBuffer(
            this.mockDevice.queue,
            buffer,
            largeData,
            0,
            { label: 'large_update', useStaging: true }
        );

        // Test large read
        const readData = await this.bufferManager.readBuffer(
            this.mockDevice,
            buffer,
            largeData.byteLength,
            0,
            { label: 'large_read' }
        );

        if (readData.byteLength !== largeData.byteLength) {
            throw new Error('Large data read size mismatch');
        }
    }

    /**
     * Test concurrent operations
     */
    async testConcurrentOperations() {
        const promises = [];
        const bufferCount = 5;

        // Create multiple buffers concurrently
        for (let i = 0; i < bufferCount; i++) {
            const promise = this.bufferManager.createBuffer(
                this.mockDevice,
                1024 + i * 256,
                BufferUsageType.STORAGE_READ_WRITE,
                { label: `concurrent_test_${i}` }
            );
            promises.push(promise);
        }

        const buffers = await Promise.all(promises);

        if (buffers.length !== bufferCount) {
            throw new Error('Concurrent buffer creation failed');
        }

        for (const buffer of buffers) {
            if (!buffer || buffer.destroyed) {
                throw new Error('Concurrent buffer creation produced invalid buffer');
            }
        }
    }

    /**
     * Test buffer reuse patterns
     */
    async testBufferReuse() {
        const size = 1024;
        const usage = BufferUsageType.STORAGE_READ_WRITE;

        // Create and return several buffers to build up pool
        const buffers = [];
        for (let i = 0; i < 3; i++) {
            const buffer = await this.bufferManager.createBuffer(
                this.mockDevice,
                size,
                usage,
                { label: `reuse_test_${i}` }
            );
            buffers.push(buffer);
        }

        // Return all to pool
        for (const buffer of buffers) {
            this.bufferManager._returnBufferToPool(buffer, size, usage);
        }

        const initialStats = this.bufferManager.getMemoryUsage().pool;

        // Create new buffers - should reuse from pool
        for (let i = 0; i < 3; i++) {
            await this.bufferManager.createBuffer(
                this.mockDevice,
                size,
                usage,
                { label: `reuse_new_${i}` }
            );
        }

        const finalStats = this.bufferManager.getMemoryUsage().pool;
        
        if (finalStats.hits <= initialStats.hits) {
            throw new Error('Buffer reuse not working - no additional hits');
        }

        if (finalStats.totalReused <= initialStats.totalReused) {
            throw new Error('Buffer reuse count not increasing');
        }
    }

    /**
     * Test cleanup and destroy
     */
    async testCleanupAndDestroy() {
        // Create some buffers and operations
        await this.bufferManager.createBuffer(
            this.mockDevice,
            1024,
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'cleanup_test' }
        );

        const beforeMemory = this.bufferManager.getMemoryUsage();
        if (beforeMemory.buffers.count === 0) {
            throw new Error('No buffers created for cleanup test');
        }

        // Destroy buffer manager
        this.bufferManager.destroy();

        // Check if properly cleaned up
        if (!this.bufferManager.isDestroyed) {
            throw new Error('BufferManager not marked as destroyed');
        }

        // Try to use destroyed manager
        try {
            await this.bufferManager.createBuffer(
                this.mockDevice,
                1024,
                BufferUsageType.STORAGE_READ_WRITE,
                { label: 'should_fail' }
            );
            throw new Error('Should not be able to create buffer after destroy');
        } catch (error) {
            if (!error.message.includes('destroyed')) {
                throw new Error('Wrong error message for destroyed manager');
            }
        }
    }
}

/**
 * Performance benchmark for buffer operations
 */
export class BufferManagerBenchmark {
    constructor() {
        this.mockDevice = null;
        this.bufferManager = null;
    }

    async setup() {
        this.mockDevice = new MockGPUDevice();
        this.bufferManager = new BufferManager(this.mockDevice, {
            maxBufferSize: 10 * 1024 * 1024, // 10MB
            maxTotalMemory: 100 * 1024 * 1024, // 100MB
            poolEnabled: true,
            enableProfiling: true
        });
    }

    async cleanup() {
        if (this.bufferManager) {
            this.bufferManager.destroy();
        }
        if (this.mockDevice) {
            this.mockDevice.destroy();
        }
    }

    async runBenchmark() {
        console.log('Running BufferManager performance benchmark...');
        
        await this.setup();

        try {
            const results = {};

            // Benchmark buffer creation
            results.bufferCreation = await this.benchmarkBufferCreation();
            
            // Benchmark memory operations
            results.memoryOperations = await this.benchmarkMemoryOperations();
            
            // Benchmark batch operations
            results.batchOperations = await this.benchmarkBatchOperations();
            
            // Benchmark pool efficiency
            results.poolEfficiency = await this.benchmarkPoolEfficiency();

            console.log('Benchmark Results:', results);
            return results;

        } finally {
            await this.cleanup();
        }
    }

    async benchmarkBufferCreation() {
        const iterations = 100;
        const sizes = [1024, 4096, 16384, 65536]; // 1KB to 64KB
        const results = {};

        for (const size of sizes) {
            const startTime = performance.now();
            
            for (let i = 0; i < iterations; i++) {
                await this.bufferManager.createBuffer(
                    this.mockDevice,
                    size,
                    BufferUsageType.STORAGE_READ_WRITE,
                    { label: `bench_create_${size}_${i}` }
                );
            }
            
            const endTime = performance.now();
            const avgTime = (endTime - startTime) / iterations;
            
            results[`${size}B`] = {
                avgTime: avgTime.toFixed(3) + 'ms',
                totalTime: (endTime - startTime).toFixed(3) + 'ms',
                iterations
            };
        }

        return results;
    }

    async benchmarkMemoryOperations() {
        const buffer = await this.bufferManager.createBuffer(
            this.mockDevice,
            64 * 1024, // 64KB
            BufferUsageType.STORAGE_READ_WRITE,
            { label: 'bench_memory' }
        );

        const testSizes = [1024, 4096, 16384]; // 1KB to 16KB
        const results = {};

        for (const size of testSizes) {
            const data = new Float32Array(size / 4);
            for (let i = 0; i < data.length; i++) {
                data[i] = Math.random();
            }

            // Benchmark writes
            const writeStart = performance.now();
            for (let i = 0; i < 10; i++) {
                await this.bufferManager.updateBuffer(
                    this.mockDevice.queue,
                    buffer,
                    data,
                    0,
                    { label: `bench_write_${size}` }
                );
            }
            const writeTime = (performance.now() - writeStart) / 10;

            // Benchmark reads
            const readStart = performance.now();
            for (let i = 0; i < 10; i++) {
                await this.bufferManager.readBuffer(
                    this.mockDevice,
                    buffer,
                    size,
                    0,
                    { label: `bench_read_${size}` }
                );
            }
            const readTime = (performance.now() - readStart) / 10;

            results[`${size}B`] = {
                avgWriteTime: writeTime.toFixed(3) + 'ms',
                avgReadTime: readTime.toFixed(3) + 'ms',
                throughputMBps: ((size / (1024 * 1024)) / (writeTime / 1000)).toFixed(2)
            };
        }

        return results;
    }

    async benchmarkBatchOperations() {
        const batchSizes = [5, 10, 20, 50];
        const results = {};

        for (const batchSize of batchSizes) {
            const operations = [];
            for (let i = 0; i < batchSize; i++) {
                operations.push({
                    type: 'create',
                    data: 1024 + i * 256,
                    usage: BufferUsageType.STORAGE_READ_WRITE,
                    options: { label: `batch_bench_${batchSize}_${i}` }
                });
            }

            const startTime = performance.now();
            await this.bufferManager.batchOperations(operations);
            const endTime = performance.now();

            const totalTime = endTime - startTime;
            const avgTime = totalTime / batchSize;

            results[`batch_${batchSize}`] = {
                totalTime: totalTime.toFixed(3) + 'ms',
                avgTimePerOp: avgTime.toFixed(3) + 'ms',
                opsPerSecond: (1000 / avgTime).toFixed(0)
            };
        }

        return results;
    }

    async benchmarkPoolEfficiency() {
        const size = 4096;
        const usage = BufferUsageType.STORAGE_READ_WRITE;
        const iterations = 50;

        // First run without pool
        this.bufferManager.config.poolEnabled = false;
        const withoutPoolStart = performance.now();
        
        for (let i = 0; i < iterations; i++) {
            const buffer = await this.bufferManager.createBuffer(
                this.mockDevice,
                size,
                usage,
                { label: `no_pool_${i}` }
            );
            // Simulate immediate destruction
            buffer.destroy();
        }
        
        const withoutPoolTime = performance.now() - withoutPoolStart;

        // Reset and run with pool
        await this.cleanup();
        await this.setup();
        this.bufferManager.config.poolEnabled = true;

        const withPoolStart = performance.now();
        
        for (let i = 0; i < iterations; i++) {
            const buffer = await this.bufferManager.createBuffer(
                this.mockDevice,
                size,
                usage,
                { label: `with_pool_${i}` }
            );
            // Return to pool for reuse
            this.bufferManager._returnBufferToPool(buffer, size, usage);
        }
        
        const withPoolTime = performance.now() - withPoolStart;

        const poolStats = this.bufferManager.getMemoryUsage().pool;

        return {
            withoutPool: {
                totalTime: withoutPoolTime.toFixed(3) + 'ms',
                avgTime: (withoutPoolTime / iterations).toFixed(3) + 'ms'
            },
            withPool: {
                totalTime: withPoolTime.toFixed(3) + 'ms',
                avgTime: (withPoolTime / iterations).toFixed(3) + 'ms',
                hitRate: poolStats.hitRate,
                totalReused: poolStats.totalReused
            },
            improvement: {
                speedup: (withoutPoolTime / withPoolTime).toFixed(2) + 'x',
                timeSaved: (withoutPoolTime - withPoolTime).toFixed(3) + 'ms'
            }
        };
    }
}

// Export test runner function
export async function runBufferManagerTests() {
    const testSuite = new BufferManagerTests();
    const results = await testSuite.runAllTests();
    
    const benchmark = new BufferManagerBenchmark();
    const benchmarkResults = await benchmark.runBenchmark();
    
    return {
        tests: results,
        benchmark: benchmarkResults
    };
}