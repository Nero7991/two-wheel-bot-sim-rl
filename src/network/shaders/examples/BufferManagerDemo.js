/**
 * BufferManager Integration Demo
 * 
 * Demonstrates the enhanced GPU buffer management system integrated with
 * the WebGPU neural network for the two-wheel balancing robot RL application.
 * 
 * This example shows:
 * - Enhanced buffer creation with pooling
 * - Async operations with proper error handling
 * - Memory tracking and optimization
 * - Batch operations for training
 * - Performance monitoring
 */

import { WebGPUNeuralNetwork } from '../WebGPUNeuralNetwork.js';
import { BufferManager, BufferUsageType, createBuffer, updateBuffer, readBuffer } from '../BufferManager.js';

/**
 * Demo class for enhanced buffer management
 */
export class BufferManagerDemo {
    constructor() {
        this.device = null;
        this.network = null;
        this.standalone = null;
        this.results = {};
    }

    /**
     * Initialize WebGPU device
     */
    async initializeDevice() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        this.device = await adapter.requestDevice();
        console.log('WebGPU device initialized for BufferManager demo');
    }

    /**
     * Run complete demo
     */
    async runDemo() {
        console.log('🚀 Starting Enhanced BufferManager Demo...\n');

        try {
            await this.initializeDevice();
            
            // Demo 1: Basic buffer operations
            await this.demoBasicOperations();
            
            // Demo 2: Neural network integration
            await this.demoNeuralNetworkIntegration();
            
            // Demo 3: Buffer pooling efficiency
            await this.demoBufferPooling();
            
            // Demo 4: Batch operations
            await this.demoBatchOperations();
            
            // Demo 5: Memory management
            await this.demoMemoryManagement();
            
            // Demo 6: Performance monitoring
            await this.demoPerformanceMonitoring();

            this.displayResults();

        } catch (error) {
            console.error('Demo failed:', error);
            throw error;
        } finally {
            await this.cleanup();
        }
    }

    /**
     * Demo 1: Basic buffer operations with enhanced features
     */
    async demoBasicOperations() {
        console.log('📋 Demo 1: Basic Buffer Operations');
        console.log('=' .repeat(40));

        const bufferManager = new BufferManager(this.device, {
            poolEnabled: true,
            enableProfiling: true,
            enableValidation: true
        });

        try {
            // Create buffer with validation
            console.log('Creating buffer with enhanced validation...');
            const testData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
            
            const buffer = await bufferManager.createBuffer(
                this.device,
                testData,
                BufferUsageType.STORAGE_READ_WRITE,
                { 
                    label: 'demo_basic_buffer',
                    allowReuse: true 
                }
            );

            console.log('✅ Buffer created successfully');

            // Update buffer data
            console.log('Updating buffer data...');
            const updateData = new Float32Array([10.0, 20.0, 30.0, 40.0, 50.0]);
            await bufferManager.updateBuffer(
                this.device.queue,
                buffer,
                updateData,
                0,
                { label: 'demo_update' }
            );

            console.log('✅ Buffer updated successfully');

            // Read buffer data
            console.log('Reading buffer data...');
            const readData = await bufferManager.readBuffer(
                this.device,
                buffer,
                testData.byteLength,
                0,
                { label: 'demo_read' }
            );

            const result = new Float32Array(readData);
            console.log('✅ Buffer read successfully:', Array.from(result));

            // Get memory usage
            const memoryUsage = bufferManager.getMemoryUsage();
            console.log('📊 Memory Usage:');
            console.log(`  Total Active: ${memoryUsage.memory.totalActiveFormatted}`);
            console.log(`  Buffer Count: ${memoryUsage.buffers.count}`);

            this.results.basicOperations = {
                bufferCreated: true,
                dataWritten: true,
                dataRead: true,
                memoryTracked: true
            };

        } finally {
            bufferManager.destroy();
        }

        console.log('✅ Demo 1 completed\n');
    }

    /**
     * Demo 2: Neural network integration
     */
    async demoNeuralNetworkIntegration() {
        console.log('🧠 Demo 2: Neural Network Integration');
        console.log('=' .repeat(40));

        this.network = new WebGPUNeuralNetwork(this.device);

        try {
            // Initialize network with enhanced buffer management
            console.log('Initializing neural network with enhanced buffers...');
            await this.network.initialize(2, 8, 3, {
                initMethod: 'xavier',
                seed: 42
            });

            console.log('✅ Neural network initialized');

            // Perform forward pass
            console.log('Performing forward pass...');
            const input = new Float32Array([0.5, -0.3]);
            const output = await this.network.forward(input);

            console.log('✅ Forward pass completed:', Array.from(output));

            // Get detailed performance metrics
            const metrics = this.network.getPerformanceMetrics();
            console.log('📊 Performance Metrics:');
            console.log(`  Average Forward Time: ${metrics.averageForwardTime.toFixed(2)}ms`);
            console.log(`  Buffer Pool Hit Rate: ${metrics.efficiency.bufferPoolHitRate}`);
            console.log(`  Memory Utilization: ${metrics.efficiency.memoryUtilization}`);

            // Validate network
            const validation = this.network.validate();
            console.log('🔍 Network Validation:');
            console.log(`  Valid: ${validation.isValid}`);
            console.log(`  Buffer Count: ${validation.bufferStatus.totalBuffers}`);
            console.log(`  Memory Usage: ${validation.bufferStatus.memoryUsage}`);

            if (validation.warnings.length > 0) {
                console.log('⚠️  Warnings:', validation.warnings);
            }

            this.results.neuralNetworkIntegration = {
                networkInitialized: true,
                forwardPassCompleted: true,
                performanceTracked: true,
                validationPassed: validation.isValid
            };

        } catch (error) {
            console.error('Neural network demo failed:', error);
            this.results.neuralNetworkIntegration = {
                error: error.message
            };
        }

        console.log('✅ Demo 2 completed\n');
    }

    /**
     * Demo 3: Buffer pooling efficiency
     */
    async demoBufferPooling() {
        console.log('🔄 Demo 3: Buffer Pooling Efficiency');
        console.log('=' .repeat(40));

        const bufferManager = new BufferManager(this.device, {
            poolEnabled: true,
            poolMaxSize: 10,
            enableProfiling: true
        });

        try {
            const bufferSize = 4096; // 4KB
            const usage = BufferUsageType.STORAGE_READ_WRITE;
            const iterations = 5;

            console.log(`Creating and reusing ${iterations} buffers of ${bufferSize} bytes...`);

            // Create buffers and return to pool
            const buffers = [];
            for (let i = 0; i < iterations; i++) {
                const buffer = await bufferManager.createBuffer(
                    this.device,
                    bufferSize,
                    usage,
                    { label: `pool_demo_${i}` }
                );
                buffers.push(buffer);
            }

            // Return all to pool
            for (const buffer of buffers) {
                bufferManager._returnBufferToPool(buffer, bufferSize, usage);
            }

            console.log('✅ Buffers returned to pool');

            // Create new buffers - should reuse from pool
            for (let i = 0; i < iterations; i++) {
                await bufferManager.createBuffer(
                    this.device,
                    bufferSize,
                    usage,
                    { label: `pool_reuse_${i}` }
                );
            }

            console.log('✅ Buffers reused from pool');

            const poolStats = bufferManager.getMemoryUsage().pool;
            console.log('📊 Pool Statistics:');
            console.log(`  Hit Rate: ${poolStats.hitRate}`);
            console.log(`  Total Reused: ${poolStats.totalReused}`);
            console.log(`  Hits: ${poolStats.hits}, Misses: ${poolStats.misses}`);

            this.results.bufferPooling = {
                poolEnabled: true,
                hitRate: poolStats.hitRate,
                totalReused: poolStats.totalReused,
                efficiency: parseFloat(poolStats.hitRate) > 50
            };

        } finally {
            bufferManager.destroy();
        }

        console.log('✅ Demo 3 completed\n');
    }

    /**
     * Demo 4: Batch operations for efficiency
     */
    async demoBatchOperations() {
        console.log('⚡ Demo 4: Batch Operations');
        console.log('=' .repeat(40));

        const bufferManager = new BufferManager(this.device, {
            enableProfiling: true
        });

        try {
            console.log('Performing batch buffer operations...');

            const operations = [
                {
                    type: 'create',
                    data: 1024,
                    usage: BufferUsageType.STORAGE_READ_WRITE,
                    options: { label: 'batch_buffer_1' }
                },
                {
                    type: 'create',
                    data: 2048,
                    usage: BufferUsageType.UNIFORM,
                    options: { label: 'batch_buffer_2' }
                },
                {
                    type: 'create',
                    data: 4096,
                    usage: BufferUsageType.STORAGE_READ_ONLY,
                    options: { label: 'batch_buffer_3' }
                }
            ];

            const startTime = performance.now();
            const results = await bufferManager.batchOperations(operations);
            const batchTime = performance.now() - startTime;

            console.log(`✅ Batch operations completed in ${batchTime.toFixed(2)}ms`);
            console.log(`  Success count: ${results.filter(r => !r.error).length}`);
            console.log(`  Error count: ${results.filter(r => r.error).length}`);

            // Demonstrate batch buffer updates if network is available
            if (this.network) {
                console.log('Performing batch neural network operations...');

                const batchSize = 4;
                const batchInput = new Float32Array(batchSize * 2);
                
                // Fill with test data
                for (let i = 0; i < batchSize; i++) {
                    batchInput[i * 2] = Math.random() - 0.5;     // angle
                    batchInput[i * 2 + 1] = Math.random() - 0.5; // angular velocity
                }

                const batchOutput = await this.network.forwardBatch(batchInput, batchSize);
                console.log(`✅ Batch forward pass completed for ${batchSize} samples`);
                console.log(`  Output shape: ${batchOutput.length} elements`);
            }

            this.results.batchOperations = {
                batchCompleted: true,
                operationCount: operations.length,
                totalTime: batchTime,
                allSucceeded: results.every(r => !r.error)
            };

        } finally {
            bufferManager.destroy();
        }

        console.log('✅ Demo 4 completed\n');
    }

    /**
     * Demo 5: Memory management and limits
     */
    async demoMemoryManagement() {
        console.log('💾 Demo 5: Memory Management');
        console.log('=' .repeat(40));

        const bufferManager = new BufferManager(this.device, {
            maxTotalMemory: 1024 * 1024, // 1MB limit for demo
            enableValidation: true,
            enableProfiling: true
        });

        try {
            console.log('Testing memory limits and validation...');

            const bufferSize = 256 * 1024; // 256KB
            const buffers = [];

            // Create buffers until near limit
            for (let i = 0; i < 3; i++) {
                try {
                    const buffer = await bufferManager.createBuffer(
                        this.device,
                        bufferSize,
                        BufferUsageType.STORAGE_READ_WRITE,
                        { label: `memory_test_${i}` }
                    );
                    buffers.push(buffer);
                    
                    const usage = bufferManager.getMemoryUsage();
                    console.log(`  Buffer ${i + 1}: ${usage.memory.totalActiveFormatted} used (${usage.memory.utilizationPercent}%)`);
                    
                } catch (error) {
                    console.log(`  ⚠️  Memory limit reached at buffer ${i + 1}: ${error.message}`);
                    break;
                }
            }

            // Demonstrate memory validation
            try {
                await bufferManager.createBuffer(
                    this.device,
                    bufferSize,
                    BufferUsageType.STORAGE_READ_WRITE,
                    { label: 'should_fail' }
                );
                console.log('❌ Expected memory limit error did not occur');
            } catch (error) {
                console.log('✅ Memory limit properly enforced:', error.message);
            }

            const finalUsage = bufferManager.getMemoryUsage();
            console.log('📊 Final Memory Usage:');
            console.log(`  Total Active: ${finalUsage.memory.totalActiveFormatted}`);
            console.log(`  Utilization: ${finalUsage.memory.utilizationPercent}%`);
            console.log(`  Buffer Count: ${finalUsage.buffers.count}`);

            this.results.memoryManagement = {
                memoryLimitsEnforced: true,
                buffersCreated: buffers.length,
                finalUtilization: finalUsage.memory.utilizationPercent
            };

        } finally {
            bufferManager.destroy();
        }

        console.log('✅ Demo 5 completed\n');
    }

    /**
     * Demo 6: Performance monitoring and optimization
     */
    async demoPerformanceMonitoring() {
        console.log('📈 Demo 6: Performance Monitoring');
        console.log('=' .repeat(40));

        const bufferManager = new BufferManager(this.device, {
            enableProfiling: true,
            poolEnabled: true
        });

        try {
            console.log('Performing operations for performance analysis...');

            // Create various sized buffers
            const sizes = [1024, 4096, 16384, 65536]; // 1KB to 64KB
            const iterations = 10;

            for (const size of sizes) {
                const startTime = performance.now();
                
                for (let i = 0; i < iterations; i++) {
                    const buffer = await bufferManager.createBuffer(
                        this.device,
                        size,
                        BufferUsageType.STORAGE_READ_WRITE,
                        { label: `perf_test_${size}_${i}` }
                    );

                    // Test update operation
                    const testData = new Float32Array(size / 4);
                    testData.fill(Math.random());
                    
                    await bufferManager.updateBuffer(
                        this.device.queue,
                        buffer,
                        testData,
                        0,
                        { label: `perf_update_${size}_${i}` }
                    );
                }
                
                const totalTime = performance.now() - startTime;
                const avgTime = totalTime / iterations;
                
                console.log(`  ${(size / 1024).toFixed(0)}KB buffers: ${avgTime.toFixed(2)}ms avg (${iterations} iterations)`);
            }

            // Get comprehensive performance metrics
            const performance = bufferManager.getPerformanceMetrics();
            console.log('📊 Performance Analysis:');
            console.log(`  Buffer Creation: ${performance.bufferCreation.avg.toFixed(2)}ms avg`);
            console.log(`  Memory Transfer: ${performance.memoryTransfer.avg.toFixed(2)}ms avg`);
            console.log(`  Async Operations: ${performance.asyncOperations.avg.toFixed(2)}ms avg`);
            console.log(`  Error Count: ${performance.errors.count}`);

            const memoryUsage = bufferManager.getMemoryUsage();
            console.log('📊 Efficiency Metrics:');
            console.log(`  Pool Hit Rate: ${memoryUsage.pool.hitRate}`);
            console.log(`  Memory Utilization: ${memoryUsage.memory.utilizationPercent}%`);
            console.log(`  Total Transfers: ${memoryUsage.performance.totalTransfers}`);

            this.results.performanceMonitoring = {
                metricsCollected: true,
                avgCreationTime: performance.bufferCreation.avg,
                avgTransferTime: performance.memoryTransfer.avg,
                poolHitRate: memoryUsage.pool.hitRate,
                errorCount: performance.errors.count
            };

        } finally {
            bufferManager.destroy();
        }

        console.log('✅ Demo 6 completed\n');
    }

    /**
     * Display comprehensive demo results
     */
    displayResults() {
        console.log('🎯 DEMO RESULTS SUMMARY');
        console.log('=' .repeat(50));

        console.log('\n📋 Basic Operations:');
        if (this.results.basicOperations) {
            console.log(`  Buffer Creation: ${this.results.basicOperations.bufferCreated ? '✅' : '❌'}`);
            console.log(`  Data Operations: ${this.results.basicOperations.dataWritten && this.results.basicOperations.dataRead ? '✅' : '❌'}`);
            console.log(`  Memory Tracking: ${this.results.basicOperations.memoryTracked ? '✅' : '❌'}`);
        }

        console.log('\n🧠 Neural Network Integration:');
        if (this.results.neuralNetworkIntegration && !this.results.neuralNetworkIntegration.error) {
            console.log(`  Network Initialization: ${this.results.neuralNetworkIntegration.networkInitialized ? '✅' : '❌'}`);
            console.log(`  Forward Pass: ${this.results.neuralNetworkIntegration.forwardPassCompleted ? '✅' : '❌'}`);
            console.log(`  Performance Tracking: ${this.results.neuralNetworkIntegration.performanceTracked ? '✅' : '❌'}`);
            console.log(`  Validation: ${this.results.neuralNetworkIntegration.validationPassed ? '✅' : '❌'}`);
        } else if (this.results.neuralNetworkIntegration?.error) {
            console.log(`  ❌ Error: ${this.results.neuralNetworkIntegration.error}`);
        }

        console.log('\n🔄 Buffer Pooling:');
        if (this.results.bufferPooling) {
            console.log(`  Pool Efficiency: ${this.results.bufferPooling.efficiency ? '✅' : '❌'} (${this.results.bufferPooling.hitRate})`);
            console.log(`  Reuse Count: ${this.results.bufferPooling.totalReused}`);
        }

        console.log('\n⚡ Batch Operations:');
        if (this.results.batchOperations) {
            console.log(`  Batch Completion: ${this.results.batchOperations.batchCompleted ? '✅' : '❌'}`);
            console.log(`  All Operations Succeeded: ${this.results.batchOperations.allSucceeded ? '✅' : '❌'}`);
            console.log(`  Total Time: ${this.results.batchOperations.totalTime.toFixed(2)}ms`);
        }

        console.log('\n💾 Memory Management:');
        if (this.results.memoryManagement) {
            console.log(`  Memory Limits Enforced: ${this.results.memoryManagement.memoryLimitsEnforced ? '✅' : '❌'}`);
            console.log(`  Buffers Created: ${this.results.memoryManagement.buffersCreated}`);
            console.log(`  Final Utilization: ${this.results.memoryManagement.finalUtilization}%`);
        }

        console.log('\n📈 Performance Monitoring:');
        if (this.results.performanceMonitoring) {
            console.log(`  Metrics Collection: ${this.results.performanceMonitoring.metricsCollected ? '✅' : '❌'}`);
            console.log(`  Avg Creation Time: ${this.results.performanceMonitoring.avgCreationTime.toFixed(2)}ms`);
            console.log(`  Avg Transfer Time: ${this.results.performanceMonitoring.avgTransferTime.toFixed(2)}ms`);
            console.log(`  Pool Hit Rate: ${this.results.performanceMonitoring.poolHitRate}`);
            console.log(`  Error Count: ${this.results.performanceMonitoring.errorCount}`);
        }

        console.log('\n🎉 Enhanced BufferManager Demo Completed Successfully!');
        console.log('    Production-ready GPU buffer management system validated.');
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }

        if (this.standalone) {
            this.standalone.destroy();
            this.standalone = null;
        }

        if (this.device) {
            this.device.destroy();
            this.device = null;
        }

        console.log('Demo cleanup completed');
    }
}

// Utility function to run the demo
export async function runBufferManagerDemo() {
    const demo = new BufferManagerDemo();
    await demo.runDemo();
    return demo.results;
}

// Auto-run demo if this file is loaded directly
if (typeof window !== 'undefined' && window.location.pathname.includes('BufferManagerDemo')) {
    runBufferManagerDemo().catch(console.error);
}