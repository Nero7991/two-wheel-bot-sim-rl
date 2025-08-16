/**
 * Main entry point for Two-Wheel Balancing Robot RL Web Application
 * WebGPU-accelerated machine learning environment
 */

// Module imports (will be implemented in subsequent phases)
// import { PhysicsEngine } from './physics/PhysicsEngine.js';
// import { NeuralNetwork } from './network/NeuralNetwork.js';
// import { TrainingManager } from './training/TrainingManager.js';
// import { Visualizer } from './visualization/Visualizer.js';
// import { ModelExporter } from './export/ModelExporter.js';

class TwoWheelBotRL {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.webgpuDevice = null;
        this.isInitialized = false;
        this.animationId = null;
        
        // Performance tracking
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 0;
        
        // Application state
        this.isTraining = false;
        this.isPaused = false;
        this.trainingStep = 0;
        this.episodeCount = 0;
        this.bestScore = 0;
    }

    async initialize() {
        console.log('Initializing Two-Wheel Balancing Robot RL Environment...');
        
        try {
            // Show loading screen
            this.showLoading(true);
            
            // Initialize canvas and context
            this.initializeCanvas();
            
            // Check WebGPU availability and initialize
            await this.initializeWebGPU();
            
            // Initialize UI controls
            this.initializeControls();
            
            // Start render loop
            this.startRenderLoop();
            
            this.isInitialized = true;
            console.log('Application initialized successfully!');
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showError('Failed to initialize application: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    initializeCanvas() {
        this.canvas = document.getElementById('simulation-canvas');
        if (!this.canvas) {
            throw new Error('Canvas element not found');
        }
        
        this.ctx = this.canvas.getContext('2d');
        if (!this.ctx) {
            throw new Error('Failed to get 2D context');
        }
        
        // Set canvas size to fill available space
        this.resizeCanvas();
        
        // Add resize listener
        window.addEventListener('resize', () => this.resizeCanvas());
        
        console.log('Canvas initialized:', this.canvas.width, 'x', this.canvas.height);
    }

    async initializeWebGPU() {
        const statusElement = document.getElementById('webgpu-text');
        
        try {
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported in this browser');
            }
            
            statusElement.textContent = 'Requesting WebGPU adapter...';
            const adapter = await navigator.gpu.requestAdapter();
            
            if (!adapter) {
                throw new Error('No suitable WebGPU adapter found');
            }
            
            statusElement.textContent = 'Requesting WebGPU device...';
            this.webgpuDevice = await adapter.requestDevice();
            
            statusElement.textContent = 'WebGPU Available';
            statusElement.className = 'status-available';
            
            console.log('WebGPU initialized successfully');
            console.log('Adapter info:', adapter);
            console.log('Device limits:', this.webgpuDevice.limits);
            
        } catch (error) {
            console.warn('WebGPU initialization failed:', error.message);
            statusElement.textContent = 'WebGPU Unavailable (CPU fallback)';
            statusElement.className = 'status-unavailable';
            
            // Continue with CPU fallback
            this.webgpuDevice = null;
        }
    }

    initializeControls() {
        // Training controls
        document.getElementById('start-training').addEventListener('click', () => {
            this.startTraining();
        });
        
        document.getElementById('pause-training').addEventListener('click', () => {
            this.pauseTraining();
        });
        
        document.getElementById('reset-environment').addEventListener('click', () => {
            this.resetEnvironment();
        });
        
        // Visualization controls
        document.getElementById('toggle-physics').addEventListener('click', () => {
            console.log('Toggle physics debug view');
            // Will be implemented in physics module
        });
        
        document.getElementById('toggle-network').addEventListener('click', () => {
            console.log('Toggle network visualization');
            // Will be implemented in visualization module
        });
        
        document.getElementById('toggle-metrics').addEventListener('click', () => {
            console.log('Toggle metrics display');
            // Will be implemented in visualization module
        });
        
        // Model management controls
        document.getElementById('save-model').addEventListener('click', () => {
            console.log('Save model');
            // Will be implemented in export module
        });
        
        document.getElementById('load-model').addEventListener('click', () => {
            console.log('Load model');
            // Will be implemented in export module
        });
        
        document.getElementById('export-model').addEventListener('click', () => {
            console.log('Export model');
            // Will be implemented in export module
        });
        
        console.log('Controls initialized');
    }

    startTraining() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        this.isPaused = false;
        
        document.getElementById('start-training').disabled = true;
        document.getElementById('pause-training').disabled = false;
        
        console.log('Training started');
        // Training logic will be implemented in training module
    }

    pauseTraining() {
        if (!this.isTraining) return;
        
        this.isPaused = !this.isPaused;
        
        const pauseButton = document.getElementById('pause-training');
        pauseButton.textContent = this.isPaused ? 'Resume Training' : 'Pause Training';
        
        console.log('Training', this.isPaused ? 'paused' : 'resumed');
    }

    resetEnvironment() {
        this.isTraining = false;
        this.isPaused = false;
        this.trainingStep = 0;
        this.episodeCount = 0;
        
        document.getElementById('start-training').disabled = false;
        document.getElementById('pause-training').disabled = true;
        document.getElementById('pause-training').textContent = 'Pause Training';
        
        this.updateUI();
        
        console.log('Environment reset');
        // Reset logic will be implemented in physics and training modules
    }

    resizeCanvas() {
        const container = document.getElementById('main-container');
        const controlsPanel = document.getElementById('controls-panel');
        
        const availableWidth = container.clientWidth - controlsPanel.clientWidth;
        const availableHeight = container.clientHeight;
        
        this.canvas.width = Math.max(800, availableWidth - 2); // Account for border
        this.canvas.height = Math.max(600, availableHeight - 2);
        
        console.log('Canvas resized to:', this.canvas.width, 'x', this.canvas.height);
    }

    startRenderLoop() {
        const render = (timestamp) => {
            this.update(timestamp);
            this.draw();
            this.updateFPS(timestamp);
            
            this.animationId = requestAnimationFrame(render);
        };
        
        this.animationId = requestAnimationFrame(render);
        console.log('Render loop started');
    }

    update(timestamp) {
        // Update simulation state
        if (this.isTraining && !this.isPaused) {
            this.trainingStep++;
            
            // Placeholder training logic
            if (this.trainingStep % 1000 === 0) {
                this.episodeCount++;
                this.updateUI();
            }
        }
        
        // Update physics, neural network, etc. (will be implemented in modules)
    }

    draw() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw placeholder content
        this.drawPlaceholder();
        
        // Draw UI overlays (will be implemented in visualization module)
    }

    drawPlaceholder() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Draw a simple robot placeholder
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.fillRect(centerX - 20, centerY - 10, 40, 20);
        
        // Draw wheels
        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(centerX - 15, centerY + 15, 8, 0, Math.PI * 2);
        this.ctx.arc(centerX + 15, centerY + 15, 8, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw status text
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Two-Wheel Balancing Robot', centerX, centerY - 40);
        this.ctx.fillText('Physics and ML modules will be implemented in next phases', centerX, centerY + 60);
    }

    updateFPS(timestamp) {
        this.frameCount++;
        
        if (timestamp - this.lastFpsTime >= 1000) {
            this.fps = Math.round(this.frameCount * 1000 / (timestamp - this.lastFpsTime));
            this.frameCount = 0;
            this.lastFpsTime = timestamp;
            
            document.getElementById('fps-counter').textContent = `FPS: ${this.fps}`;
        }
    }

    updateUI() {
        document.getElementById('training-step').textContent = `Step: ${this.trainingStep.toLocaleString()}`;
        document.getElementById('episode-count').textContent = `Episode: ${this.episodeCount}`;
        document.getElementById('best-score').textContent = `Best Score: ${this.bestScore}`;
    }

    showLoading(show) {
        const loadingElement = document.getElementById('loading');
        loadingElement.style.display = show ? 'flex' : 'none';
    }

    showError(message) {
        alert('Error: ' + message);
        console.error(message);
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        if (this.webgpuDevice) {
            this.webgpuDevice.destroy();
            this.webgpuDevice = null;
        }
        
        console.log('Application destroyed');
    }
}

// Application initialization
let app = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        app = new TwoWheelBotRL();
        await app.initialize();
    } catch (error) {
        console.error('Failed to start application:', error);
    }
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (app) {
        app.destroy();
    }
});

// Export for potential external access
window.TwoWheelBotRL = TwoWheelBotRL;