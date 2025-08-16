/**
 * Main entry point for Two-Wheel Balancing Robot RL Web Application
 * WebGPU-accelerated machine learning environment with 2D visualization
 */

// Core module imports
import { createRenderer } from './visualization/Renderer.js';
import { createDefaultRobot } from './physics/BalancingRobot.js';
import { createDefaultQLearning } from './training/QLearning.js';

// Module imports (will be implemented in subsequent phases)
// import { ModelExporter } from './export/ModelExporter.js';

class TwoWheelBotRL {
    constructor() {
        this.canvas = null;
        this.renderer = null;
        this.webgpuDevice = null;
        this.isInitialized = false;
        
        // Core components
        this.robot = null;
        this.qlearning = null;
        
        // Application state
        this.isTraining = false;
        this.isPaused = false;
        this.trainingStep = 0;
        this.episodeCount = 0;
        this.bestScore = 0;
        this.currentReward = 0;
        
        // Demo modes
        this.demoMode = 'physics'; // 'physics', 'training', 'evaluation'
        
        // Physics simulation timing
        this.lastPhysicsUpdate = 0;
        this.physicsUpdateInterval = 20; // 50 Hz physics updates
    }

    async initialize() {
        console.log('Initializing Two-Wheel Balancing Robot RL Environment...');
        
        try {
            // Show loading screen
            this.showLoading(true);
            
            // Initialize canvas and renderer
            this.initializeRenderer();
            
            // Initialize physics and ML components
            await this.initializeComponents();
            
            // Check WebGPU availability and initialize
            await this.initializeWebGPU();
            
            // Initialize UI controls
            this.initializeControls();
            
            // Start simulation and rendering
            this.startSimulation();
            
            this.isInitialized = true;
            console.log('Application initialized successfully!');
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showError('Failed to initialize application: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    initializeRenderer() {
        this.canvas = document.getElementById('simulation-canvas');
        if (!this.canvas) {
            throw new Error('Canvas element not found');
        }
        
        // Set canvas size to fill available space
        this.resizeCanvas();
        
        // Initialize 2D renderer
        this.renderer = createRenderer(this.canvas, {
            showGrid: true,
            showDebugInfo: true,
            showPerformance: true,
            targetFPS: 60
        });
        
        // Add resize listener
        window.addEventListener('resize', () => this.resizeCanvas());
        
        console.log('Renderer initialized:', this.canvas.width, 'x', this.canvas.height);
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
            this.switchDemoMode('physics');
        });
        
        document.getElementById('toggle-network').addEventListener('click', () => {
            this.renderer.toggleUI('robot');
            console.log('Toggled robot info display');
        });
        
        document.getElementById('toggle-metrics').addEventListener('click', () => {
            this.renderer.toggleUI('training');
            console.log('Toggled training metrics display');
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

    async startTraining() {
        if (this.isTraining) return;
        
        // Initialize Q-learning if not already done
        if (!this.qlearning) {
            await this.initializeQLearning();
        }
        
        this.isTraining = true;
        this.isPaused = false;
        this.demoMode = 'training';
        
        // Reset for new training session
        this.episodeCount = 0;
        this.trainingStep = 0;
        this.bestScore = -Infinity;
        this.startNewEpisode();
        
        document.getElementById('start-training').disabled = true;
        document.getElementById('pause-training').disabled = false;
        
        console.log('Training started');
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
        this.bestScore = 0;
        this.currentReward = 0;
        
        // Reset robot state
        if (this.robot) {
            this.robot.reset({
                angle: (Math.random() - 0.5) * 0.1,
                angularVelocity: 0,
                position: 0,
                velocity: 0
            });
        }
        
        document.getElementById('start-training').disabled = false;
        document.getElementById('pause-training').disabled = true;
        document.getElementById('pause-training').textContent = 'Pause Training';
        
        this.updateUI();
        
        console.log('Environment reset');
    }

    resizeCanvas() {
        const container = document.getElementById('main-container');
        const controlsPanel = document.getElementById('controls-panel');
        
        const availableWidth = container.clientWidth - controlsPanel.clientWidth;
        const availableHeight = container.clientHeight;
        
        const newWidth = Math.max(800, availableWidth - 2);
        const newHeight = Math.max(600, availableHeight - 2);
        
        if (this.renderer) {
            this.renderer.resize(newWidth, newHeight);
        } else {
            this.canvas.width = newWidth;
            this.canvas.height = newHeight;
        }
        
        console.log('Canvas resized to:', newWidth, 'x', newHeight);
    }

    startSimulation() {
        // Start the renderer
        this.renderer.start();
        
        // Start physics simulation loop
        const simulate = (timestamp) => {
            if (!this.isInitialized) return;
            
            // Update physics at fixed intervals
            if (timestamp - this.lastPhysicsUpdate >= this.physicsUpdateInterval) {
                this.updatePhysics();
                this.updateRenderer();
                this.updateUI();
                this.lastPhysicsUpdate = timestamp;
            }
            
            requestAnimationFrame(simulate);
        };
        
        requestAnimationFrame(simulate);
        console.log('Simulation started');
    }

    updatePhysics() {
        if (!this.robot) return;
        
        let motorTorque = 0;
        
        // Get motor torque based on current mode
        switch (this.demoMode) {
            case 'physics':
                motorTorque = this.getPhysicsDemoTorque();
                break;
            case 'training':
                if (this.isTraining && !this.isPaused) {
                    motorTorque = this.getTrainingTorque();
                }
                break;
            case 'evaluation':
                motorTorque = this.getEvaluationTorque();
                break;
        }
        
        // Step physics simulation
        const result = this.robot.step(motorTorque);
        this.currentReward = result.reward;
        
        // Handle episode completion for training/evaluation
        if (result.done && (this.demoMode === 'training' || this.demoMode === 'evaluation')) {
            this.handleEpisodeEnd(result);
        }
        
        // Update training step counter
        if (this.isTraining && !this.isPaused) {
            this.trainingStep++;
        }
    }

    updateRenderer() {
        if (!this.renderer || !this.robot) return;
        
        const state = this.robot.getState();
        const config = this.robot.getConfig();
        const stats = this.robot.getStats();
        
        // Update robot visualization
        this.renderer.updateRobot(state, config, stats.currentMotorTorque);
        
        // Update training metrics
        const trainingMetrics = {
            episode: this.episodeCount,
            step: this.trainingStep,
            reward: this.currentReward,
            totalReward: stats.totalReward,
            bestReward: this.bestScore,
            epsilon: this.qlearning ? this.qlearning.hyperparams.epsilon : 0,
            isTraining: this.isTraining,
            trainingMode: this.demoMode
        };
        
        this.renderer.updateTraining(trainingMetrics);
    }

    updateUI() {
        document.getElementById('training-step').textContent = `Step: ${this.trainingStep.toLocaleString()}`;
        document.getElementById('episode-count').textContent = `Episode: ${this.episodeCount}`;
        document.getElementById('best-score').textContent = `Best Score: ${this.bestScore.toFixed(1)}`;
        
        // Update FPS from renderer performance
        if (this.renderer) {
            const perfMetrics = this.renderer.performance.getMetrics();
            document.getElementById('fps-counter').textContent = `FPS: ${perfMetrics.fps}`;
        }
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
        if (this.renderer) {
            this.renderer.destroy();
            this.renderer = null;
        }
        
        if (this.webgpuDevice) {
            this.webgpuDevice.destroy();
            this.webgpuDevice = null;
        }
        
        this.robot = null;
        this.qlearning = null;
        
        console.log('Application destroyed');
    }
    
    /**
     * Initialize physics and ML components
     */
    async initializeComponents() {
        // Initialize robot physics
        this.robot = createDefaultRobot({
            mass: 1.0,
            centerOfMassHeight: 0.4,
            motorStrength: 5.0,
            friction: 0.1,
            damping: 0.05
        });
        
        // Reset robot to initial state
        this.robot.reset({
            angle: (Math.random() - 0.5) * 0.1, // Small random initial angle
            angularVelocity: 0,
            position: 0,
            velocity: 0
        });
        
        console.log('Physics components initialized');
    }
    
    /**
     * Initialize Q-learning for training
     */
    async initializeQLearning() {
        this.qlearning = createDefaultQLearning({
            learningRate: 0.001,
            epsilon: 0.3,
            epsilonDecay: 0.995,
            maxEpisodes: 1000,
            maxStepsPerEpisode: 1000,
            hiddenSize: 8
        });
        
        await this.qlearning.initialize();
        console.log('Q-learning initialized');
    }
    
    /**
     * Get motor torque for physics demo (simple balancing controller)
     */
    getPhysicsDemoTorque() {
        const state = this.robot.getState();
        
        // Simple PD controller for demonstration
        const kp = 50; // Proportional gain
        const kd = 10; // Derivative gain
        
        const torque = -(kp * state.angle + kd * state.angularVelocity);
        
        // Add some noise for more interesting behavior
        const noise = (Math.random() - 0.5) * 0.5;
        
        return Math.max(-5, Math.min(5, torque + noise));
    }
    
    /**
     * Get motor torque from Q-learning during training
     */
    getTrainingTorque() {
        if (!this.qlearning) return 0;
        
        const state = this.robot.getState();
        const normalizedState = state.getNormalizedInputs();
        
        // Select action using epsilon-greedy policy
        const actionIndex = this.qlearning.selectAction(normalizedState, true);
        const actions = [-3.0, 0.0, 3.0]; // Left, brake, right
        
        return actions[actionIndex];
    }
    
    /**
     * Get motor torque from trained Q-learning (evaluation mode)
     */
    getEvaluationTorque() {
        if (!this.qlearning) return 0;
        
        const state = this.robot.getState();
        const normalizedState = state.getNormalizedInputs();
        
        // Select best action (no exploration)
        const actionIndex = this.qlearning.selectAction(normalizedState, false);
        const actions = [-3.0, 0.0, 3.0];
        
        return actions[actionIndex];
    }
    
    /**
     * Start a new training episode
     */
    startNewEpisode() {
        this.trainingStep = 0;
        
        // Reset robot with small random perturbation
        this.robot.reset({
            angle: (Math.random() - 0.5) * 0.2,
            angularVelocity: (Math.random() - 0.5) * 0.5,
            position: 0,
            velocity: 0
        });
        
        console.log(`Starting episode ${this.episodeCount + 1}`);
    }
    
    /**
     * Handle end of training/evaluation episode
     */
    handleEpisodeEnd(result) {
        const totalReward = this.robot.getStats().totalReward;
        
        // Update best score
        if (totalReward > this.bestScore) {
            this.bestScore = totalReward;
        }
        
        this.episodeCount++;
        
        if (this.demoMode === 'training' && this.isTraining) {
            console.log(`Episode ${this.episodeCount} completed: Reward=${totalReward.toFixed(2)}, Steps=${this.trainingStep}`);
            
            // Start next episode after brief pause
            setTimeout(() => {
                if (this.isTraining && this.episodeCount < 100) { // Limit for demo
                    this.startNewEpisode();
                }
            }, 100);
        } else {
            // Evaluation mode or demo - restart after longer pause
            setTimeout(() => {
                this.robot.reset({
                    angle: (Math.random() - 0.5) * 0.2,
                    angularVelocity: 0,
                    position: 0,
                    velocity: 0
                });
                this.trainingStep = 0;
            }, 1000);
        }
    }
    
    /**
     * Switch demo mode
     */
    async switchDemoMode(newMode) {
        console.log(`Switching from ${this.demoMode} to ${newMode} mode`);
        
        // Stop training if switching away from training mode
        if (this.demoMode === 'training' && newMode !== 'training') {
            this.isTraining = false;
            this.isPaused = false;
            document.getElementById('start-training').disabled = false;
            document.getElementById('pause-training').disabled = true;
            document.getElementById('pause-training').textContent = 'Pause Training';
        }
        
        this.demoMode = newMode;
        
        // Initialize Q-learning if switching to training or evaluation mode
        if ((newMode === 'training' || newMode === 'evaluation') && !this.qlearning) {
            await this.initializeQLearning();
        }
        
        // Reset environment for new mode
        this.resetEnvironment();
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