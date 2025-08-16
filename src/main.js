/**
 * Main entry point for Two-Wheel Balancing Robot RL Web Application
 * WebGPU-accelerated machine learning environment with 2D visualization
 */

// Core module imports
import { createRenderer } from './visualization/Renderer.js';
import { createDefaultRobot } from './physics/BalancingRobot.js';
import { createDefaultQLearning } from './training/QLearning.js';
import { WebGPUBackend, checkWebGPUAvailability } from './network/WebGPUBackend.js';

// Module imports (will be implemented in subsequent phases)
// import { ModelExporter } from './export/ModelExporter.js';

/**
 * UI Controls Manager for parameter management and validation
 */
class UIControls {
    constructor(app) {
        this.app = app;
        this.parameters = {
            trainingSpeed: 1.0,
            hiddenNeurons: 8,
            learningRate: 0.001,
            epsilon: 0.3,
            robotMass: 1.0,
            robotHeight: 0.4,
            motorStrength: 5.0
        };
        
        // Parameter validation ranges
        this.validationRanges = {
            trainingSpeed: { min: 0.1, max: 10.0 },
            hiddenNeurons: { min: 4, max: 16 },
            learningRate: { min: 0.0001, max: 0.01 },
            epsilon: { min: 0.0, max: 1.0 },
            robotMass: { min: 0.5, max: 3.0 },
            robotHeight: { min: 0.2, max: 0.8 },
            motorStrength: { min: 1.0, max: 10.0 }
        };
        
        this.loadParameters();
    }
    
    initialize() {
        this.setupSliderControls();
        this.setupKeyboardShortcuts();
        this.updateAllDisplays();
        console.log('UI Controls initialized');
    }
    
    setupSliderControls() {
        // Training speed control
        const trainingSpeedSlider = document.getElementById('training-speed');
        const trainingSpeedValue = document.getElementById('training-speed-value');
        
        trainingSpeedSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('trainingSpeed', value);
            trainingSpeedValue.textContent = `${value.toFixed(1)}x`;
            this.app.setTrainingSpeed(value);
        });
        
        // Hidden neurons control
        const hiddenNeuronsSlider = document.getElementById('hidden-neurons');
        const hiddenNeuronsValue = document.getElementById('hidden-neurons-value');
        
        hiddenNeuronsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.setParameter('hiddenNeurons', value);
            hiddenNeuronsValue.textContent = value.toString();
            this.app.updateNetworkArchitecture(value);
        });
        
        // Learning rate control
        const learningRateSlider = document.getElementById('learning-rate');
        const learningRateValue = document.getElementById('learning-rate-value');
        
        learningRateSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('learningRate', value);
            learningRateValue.textContent = value.toFixed(4);
            this.app.updateLearningRate(value);
        });
        
        // Epsilon control
        const epsilonSlider = document.getElementById('epsilon');
        const epsilonValue = document.getElementById('epsilon-value');
        
        epsilonSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('epsilon', value);
            epsilonValue.textContent = value.toFixed(2);
            this.app.updateEpsilon(value);
        });
        
        // Robot mass control
        const robotMassSlider = document.getElementById('robot-mass');
        const robotMassValue = document.getElementById('robot-mass-value');
        
        robotMassSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('robotMass', value);
            robotMassValue.textContent = `${value.toFixed(1)} kg`;
            this.app.updateRobotMass(value);
        });
        
        // Robot height control
        const robotHeightSlider = document.getElementById('robot-height');
        const robotHeightValue = document.getElementById('robot-height-value');
        
        robotHeightSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('robotHeight', value);
            robotHeightValue.textContent = `${value.toFixed(2)} m`;
            this.app.updateRobotHeight(value);
        });
        
        // Motor strength control
        const motorStrengthSlider = document.getElementById('motor-strength');
        const motorStrengthValue = document.getElementById('motor-strength-value');
        
        motorStrengthSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('motorStrength', value);
            motorStrengthValue.textContent = `${value.toFixed(1)} Nm`;
            this.app.updateMotorStrength(value);
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore if user is typing in an input field
            if (e.target.tagName === 'INPUT') return;
            
            switch (e.key.toLowerCase()) {
                case ' ': // Spacebar - pause/resume training
                    e.preventDefault();
                    if (this.app.isTraining) {
                        this.app.pauseTraining();
                    }
                    break;
                case 's': // S - start training
                    e.preventDefault();
                    this.app.startTraining();
                    break;
                case 'r': // R - reset environment
                    e.preventDefault();
                    this.app.resetEnvironment();
                    break;
                case '1': // 1-3 for demo modes
                    e.preventDefault();
                    this.app.switchDemoMode('physics');
                    break;
                case '2':
                    e.preventDefault();
                    this.app.switchDemoMode('training');
                    break;
                case '3':
                    e.preventDefault();
                    this.app.switchDemoMode('evaluation');
                    break;
                case 'h': // H - toggle help/debug info
                    e.preventDefault();
                    this.app.renderer.toggleUI('robot');
                    break;
            }
        });
    }
    
    setParameter(paramName, value) {
        // Validate parameter range
        const range = this.validationRanges[paramName];
        if (range) {
            value = Math.max(range.min, Math.min(range.max, value));
        }
        
        this.parameters[paramName] = value;
        this.saveParameters();
        
        console.log(`Parameter ${paramName} set to ${value}`);
    }
    
    getParameter(paramName) {
        return this.parameters[paramName];
    }
    
    updateAllDisplays() {
        // Update all slider values and displays
        document.getElementById('training-speed').value = this.parameters.trainingSpeed;
        document.getElementById('training-speed-value').textContent = `${this.parameters.trainingSpeed.toFixed(1)}x`;
        
        document.getElementById('hidden-neurons').value = this.parameters.hiddenNeurons;
        document.getElementById('hidden-neurons-value').textContent = this.parameters.hiddenNeurons.toString();
        
        document.getElementById('learning-rate').value = this.parameters.learningRate;
        document.getElementById('learning-rate-value').textContent = this.parameters.learningRate.toFixed(4);
        
        document.getElementById('epsilon').value = this.parameters.epsilon;
        document.getElementById('epsilon-value').textContent = this.parameters.epsilon.toFixed(2);
        
        document.getElementById('robot-mass').value = this.parameters.robotMass;
        document.getElementById('robot-mass-value').textContent = `${this.parameters.robotMass.toFixed(1)} kg`;
        
        document.getElementById('robot-height').value = this.parameters.robotHeight;
        document.getElementById('robot-height-value').textContent = `${this.parameters.robotHeight.toFixed(2)} m`;
        
        document.getElementById('motor-strength').value = this.parameters.motorStrength;
        document.getElementById('motor-strength-value').textContent = `${this.parameters.motorStrength.toFixed(1)} Nm`;
    }
    
    saveParameters() {
        try {
            localStorage.setItem('twowheelbot-rl-parameters', JSON.stringify(this.parameters));
        } catch (error) {
            console.warn('Failed to save parameters to localStorage:', error);
        }
    }
    
    loadParameters() {
        try {
            const saved = localStorage.getItem('twowheelbot-rl-parameters');
            if (saved) {
                const loadedParams = JSON.parse(saved);
                this.parameters = { ...this.parameters, ...loadedParams };
                console.log('Parameters loaded from localStorage');
            }
        } catch (error) {
            console.warn('Failed to load parameters from localStorage:', error);
        }
    }
    
    resetToDefaults() {
        this.parameters = {
            trainingSpeed: 1.0,
            hiddenNeurons: 8,
            learningRate: 0.001,
            epsilon: 0.3,
            robotMass: 1.0,
            robotHeight: 0.4,
            motorStrength: 5.0
        };
        this.updateAllDisplays();
        this.saveParameters();
        console.log('Parameters reset to defaults');
    }
}

class TwoWheelBotRL {
    constructor() {
        this.canvas = null;
        this.renderer = null;
        this.webgpuBackend = null;
        this.isInitialized = false;
        
        // Core components
        this.robot = null;
        this.qlearning = null;
        this.uiControls = null;
        
        // Application state
        this.isTraining = false;
        this.isPaused = false;
        this.trainingStep = 0;
        this.episodeCount = 0;
        this.bestScore = 0;
        this.currentReward = 0;
        
        // Training control
        this.trainingSpeed = 1.0;
        this.targetPhysicsStepsPerFrame = 1;
        
        // Demo modes
        this.demoMode = 'physics'; // 'physics', 'training', 'evaluation'
        
        // Physics simulation timing
        this.lastPhysicsUpdate = 0;
        this.physicsUpdateInterval = 20; // 50 Hz physics updates
        
        // Performance monitoring
        this.frameTimeHistory = [];
        this.lastFrameTime = 0;
        this.performanceCheckInterval = 60; // Check every 60 frames (~1 second)
        
        // WebGPU status tracking
        this.webgpuStatus = {
            checked: false,
            available: false,
            deviceInfo: null,
            error: null
        };
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
            
            // Initialize WebGPU backend with CPU fallback
            await this.initializeWebGPUBackend();
            
            // Initialize UI controls
            this.initializeControls();
            
            // Initialize UI Controls Manager
            this.uiControls = new UIControls(this);
            this.uiControls.initialize();
            
            // Update WebGPU status display
            this.updateWebGPUStatusDisplay();
            
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

    async initializeWebGPUBackend() {
        const statusElement = document.getElementById('webgpu-text');
        
        try {
            console.log('Initializing WebGPU backend...');
            statusElement.textContent = 'Initializing WebGPU...';
            
            // Create WebGPU backend instance
            this.webgpuBackend = new WebGPUBackend();
            
            // Quick availability check for UI feedback
            const quickCheck = await checkWebGPUAvailability();
            this.webgpuStatus.checked = true;
            this.webgpuStatus.available = quickCheck.available;
            
            if (quickCheck.available) {
                statusElement.textContent = 'WebGPU Available';
                statusElement.className = 'status-available';
                console.log('WebGPU availability confirmed:', quickCheck.adapterInfo);
            } else {
                statusElement.textContent = 'WebGPU Unavailable (CPU fallback)';
                statusElement.className = 'status-unavailable';
                console.log('WebGPU not available:', quickCheck.reason);
                this.webgpuStatus.error = quickCheck.reason;
            }
            
            // Store status for detailed UI display
            this.webgpuStatus.deviceInfo = quickCheck;
            
        } catch (error) {
            console.warn('WebGPU backend initialization failed:', error.message);
            statusElement.textContent = 'WebGPU Error (CPU fallback)';
            statusElement.className = 'status-unavailable';
            
            this.webgpuStatus.checked = true;
            this.webgpuStatus.available = false;
            this.webgpuStatus.error = error.message;
            
            // Create fallback backend
            this.webgpuBackend = new WebGPUBackend();
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
        
        document.getElementById('reset-parameters').addEventListener('click', () => {
            if (this.uiControls) {
                this.uiControls.resetToDefaults();
                console.log('Parameters reset to defaults');
            }
        });
        
        // WebGPU status refresh
        document.getElementById('refresh-webgpu').addEventListener('click', () => {
            this.updateWebGPUStatusDisplay();
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
            
            // Performance monitoring
            if (this.lastFrameTime > 0) {
                const frameTime = timestamp - this.lastFrameTime;
                this.frameTimeHistory.push(frameTime);
                
                // Keep only recent history
                if (this.frameTimeHistory.length > this.performanceCheckInterval) {
                    this.frameTimeHistory.shift();
                }
                
                // Auto-adjust training speed if performance is poor
                if (this.frameTimeHistory.length === this.performanceCheckInterval) {
                    const avgFrameTime = this.frameTimeHistory.reduce((a, b) => a + b) / this.frameTimeHistory.length;
                    const targetFrameTime = 16.67; // 60 FPS target
                    
                    if (avgFrameTime > targetFrameTime * 1.5 && this.targetPhysicsStepsPerFrame > 1) {
                        // Performance is poor, reduce training speed
                        this.setTrainingSpeed(Math.max(1.0, this.trainingSpeed * 0.8));
                        console.warn(`Performance degraded (${avgFrameTime.toFixed(1)}ms/frame), reducing training speed to ${this.trainingSpeed.toFixed(1)}x`);
                    }
                }
            }
            this.lastFrameTime = timestamp;
            
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
        
        // Run multiple physics steps per frame for training speed control
        // Limit steps to maintain UI responsiveness during intensive training
        const maxStepsPerFrame = Math.min(this.targetPhysicsStepsPerFrame, 5);
        const stepsToRun = this.isTraining && !this.isPaused ? maxStepsPerFrame : 1;
        
        for (let step = 0; step < stepsToRun; step++) {
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
                break; // Don't continue stepping after episode ends
            }
            
            // Update training step counter
            if (this.isTraining && !this.isPaused) {
                this.trainingStep++;
            }
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
        document.getElementById('current-reward').textContent = `Current Reward: ${this.currentReward.toFixed(1)}`;
        
        // Update epsilon display
        if (this.qlearning) {
            document.getElementById('epsilon-display').textContent = `Epsilon: ${this.qlearning.hyperparams.epsilon.toFixed(3)}`;
        }
        
        // Update training status indicator
        const statusIndicator = document.getElementById('training-status-indicator');
        if (this.isTraining && !this.isPaused) {
            statusIndicator.className = 'status-indicator training';
        } else if (this.isTraining && this.isPaused) {
            statusIndicator.className = 'status-indicator paused';
        } else {
            statusIndicator.className = 'status-indicator stopped';
        }
        
        // Update FPS from renderer performance
        if (this.renderer) {
            const perfMetrics = this.renderer.performance.getMetrics();
            document.getElementById('fps-counter').textContent = `FPS: ${perfMetrics.fps}`;
        }
        
        // Update backend performance info
        this.updateBackendPerformanceDisplay();
    }
    
    /**
     * Update WebGPU status display with detailed information
     */
    updateWebGPUStatusDisplay() {
        const backendTypeElement = document.getElementById('backend-type');
        const deviceInfoElement = document.getElementById('device-info');
        const performanceEstimateElement = document.getElementById('performance-estimate');
        
        if (!this.webgpuBackend) {
            backendTypeElement.textContent = 'Backend: Not initialized';
            deviceInfoElement.textContent = 'WebGPU backend not yet initialized';
            performanceEstimateElement.textContent = '';
            return;
        }
        
        const backendInfo = this.webgpuBackend.getBackendInfo();
        const deviceInfo = backendInfo.deviceInfo;
        
        // Update backend type
        const backendType = backendInfo.webgpuAvailable ? 'WebGPU' : 'CPU (Fallback)';
        backendTypeElement.textContent = `Backend: ${backendType}`;
        
        // Update device information
        if (backendInfo.webgpuAvailable && deviceInfo.adapterInfo) {
            const adapter = deviceInfo.adapterInfo;
            deviceInfoElement.innerHTML = `
                <div>Vendor: ${adapter.vendor}</div>
                <div>Device: ${adapter.device}</div>
                <div>Architecture: ${adapter.architecture}</div>
            `;
        } else if (backendInfo.fallbackReason) {
            deviceInfoElement.innerHTML = `
                <div style="color: #ff6666;">Fallback reason:</div>
                <div>${backendInfo.fallbackReason}</div>
            `;
        } else {
            deviceInfoElement.textContent = 'No detailed device information available';
        }
        
        // Update performance estimate
        if (backendInfo.capabilities && backendInfo.webgpuAvailable) {
            const caps = backendInfo.capabilities;
            performanceEstimateElement.innerHTML = `
                <div>Estimated speedup: ${caps.estimatedSpeedup.toFixed(1)}x</div>
                <div>Max workgroup size: ${caps.maxWorkgroupSize || 'Unknown'}</div>
                <div>Max buffer size: ${caps.maxBufferSizeMB || 'Unknown'} MB</div>
            `;
        } else {
            performanceEstimateElement.textContent = 'CPU backend - no hardware acceleration';
        }
        
        console.log('WebGPU status display updated');
    }
    
    /**
     * Update backend performance display
     */
    updateBackendPerformanceDisplay() {
        const performanceElement = document.getElementById('backend-performance');
        
        if (!this.webgpuBackend) {
            performanceElement.textContent = '';
            return;
        }
        
        const perfMetrics = this.webgpuBackend.getPerformanceMetrics();
        
        if (perfMetrics.totalForwardPasses > 0) {
            const avgTime = perfMetrics.averageForwardTime.toFixed(2);
            const speedup = perfMetrics.estimatedSpeedup.toFixed(1);
            performanceElement.textContent = `Avg forward pass: ${avgTime}ms | Est. speedup: ${speedup}x`;
        } else {
            performanceElement.textContent = 'No performance data yet';
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
        
        if (this.webgpuBackend) {
            this.webgpuBackend.destroy();
            this.webgpuBackend = null;
        }
        
        this.robot = null;
        this.qlearning = null;
        
        console.log('Application destroyed');
    }
    
    /**
     * Initialize physics and ML components
     */
    async initializeComponents() {
        // Get parameters from UI controls (will be loaded from localStorage if available)
        const uiControls = new UIControls(this);
        
        // Initialize robot physics with parameters
        this.robot = createDefaultRobot({
            mass: uiControls.getParameter('robotMass'),
            centerOfMassHeight: uiControls.getParameter('robotHeight'),
            motorStrength: uiControls.getParameter('motorStrength'),
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
     * Initialize Q-learning for training with WebGPU backend
     */
    async initializeQLearning() {
        // Use parameters from UI controls if available
        const params = this.uiControls ? {
            learningRate: this.uiControls.getParameter('learningRate'),
            epsilon: this.uiControls.getParameter('epsilon'),
            epsilonDecay: 0.995,
            maxEpisodes: 1000,
            maxStepsPerEpisode: 1000,
            hiddenSize: this.uiControls.getParameter('hiddenNeurons')
        } : {
            learningRate: 0.001,
            epsilon: 0.3,
            epsilonDecay: 0.995,
            maxEpisodes: 1000,
            maxStepsPerEpisode: 1000,
            hiddenSize: 8
        };
        
        // Create Q-learning with WebGPU backend
        this.qlearning = createDefaultQLearning(params);
        
        // Initialize with WebGPU backend if available
        if (this.webgpuBackend) {
            console.log('Initializing Q-learning with WebGPU backend support...');
            // Note: Q-learning will use the backend internally
            // For now, the integration is prepared for when Q-learning supports custom backends
        }
        
        await this.qlearning.initialize();
        console.log('Q-learning initialized with parameters:', params);
        
        // Log backend information
        if (this.webgpuBackend) {
            const backendInfo = this.webgpuBackend.getBackendInfo();
            console.log('Neural network backend:', backendInfo.type);
            if (backendInfo.usingFallback) {
                console.log('Fallback reason:', backendInfo.fallbackReason);
            }
        }
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
    
    /**
     * Parameter update methods called by UI controls
     */
    setTrainingSpeed(speed) {
        this.trainingSpeed = Math.max(0.1, Math.min(10.0, speed));
        this.targetPhysicsStepsPerFrame = Math.round(this.trainingSpeed);
        console.log(`Training speed set to ${speed}x (${this.targetPhysicsStepsPerFrame} steps/frame)`);
    }
    
    updateNetworkArchitecture(hiddenSize) {
        console.log(`Network architecture update requested: ${hiddenSize} hidden neurons`);
        // Note: This requires reinitializing the Q-learning network
        // For now, we'll log the request and apply it on next training start
        if (this.qlearning) {
            this.qlearning.hyperparams.hiddenSize = hiddenSize;
            console.log('Network architecture will be updated on next training initialization');
        }
    }
    
    updateLearningRate(learningRate) {
        if (this.qlearning) {
            this.qlearning.hyperparams.learningRate = learningRate;
            console.log(`Learning rate updated to ${learningRate}`);
        }
    }
    
    updateEpsilon(epsilon) {
        if (this.qlearning) {
            this.qlearning.hyperparams.epsilon = epsilon;
            console.log(`Epsilon updated to ${epsilon}`);
        }
    }
    
    updateRobotMass(mass) {
        if (this.robot) {
            const config = this.robot.getConfig();
            config.mass = mass;
            this.robot.updateConfig(config);
            console.log(`Robot mass updated to ${mass} kg`);
        }
    }
    
    updateRobotHeight(height) {
        if (this.robot) {
            const config = this.robot.getConfig();
            config.centerOfMassHeight = height;
            this.robot.updateConfig(config);
            console.log(`Robot height updated to ${height} m`);
        }
    }
    
    updateMotorStrength(strength) {
        if (this.robot) {
            const config = this.robot.getConfig();
            config.motorStrength = strength;
            this.robot.updateConfig(config);
            console.log(`Motor strength updated to ${strength} Nm`);
        }
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