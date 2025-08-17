/**
 * Main entry point for Two-Wheel Balancing Robot RL Web Application
 * WebGPU-accelerated machine learning environment with 2D visualization
 */

// Import WebGPU polyfill first to ensure constants are available
import './network/shaders/webgpu-polyfill.js';

// Core module imports
import { createRenderer } from './visualization/Renderer.js';
import { createPerformanceCharts } from './visualization/Charts.js';
import { NetworkPresets, getPreset, createCustomArchitecture } from './network/NetworkPresets.js';
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
            motorStrength: 5.0,
            wheelFriction: 0.3
        };
        
        // Network architecture configuration
        this.networkConfig = {
            preset: 'DQN_STANDARD',
            customLayers: [8],
            currentArchitecture: null
        };
        
        // Parameter validation ranges
        this.validationRanges = {
            trainingSpeed: { min: 0.1, max: 100.0 },
            hiddenNeurons: { min: 4, max: 16 },
            learningRate: { min: 0.0001, max: 0.01 },
            epsilon: { min: 0.0, max: 1.0 },
            robotMass: { min: 0.5, max: 3.0 },
            robotHeight: { min: 0.2, max: 0.8 },
            motorStrength: { min: 1.0, max: 10.0 },
            wheelFriction: { min: 0.0, max: 1.0 }
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
        
        // Speed preset buttons
        document.getElementById('speed-1x').addEventListener('click', () => {
            this.setSpeedPreset(1.0);
        });
        document.getElementById('speed-2x').addEventListener('click', () => {
            this.setSpeedPreset(2.0);
        });
        document.getElementById('speed-10x').addEventListener('click', () => {
            this.setSpeedPreset(10.0);
        });
        document.getElementById('speed-100x').addEventListener('click', () => {
            this.setSpeedPreset(100.0);
        });
        
        // Debug speed control for manual testing
        const debugSpeedSlider = document.getElementById('debug-speed');
        const debugSpeedValue = document.getElementById('debug-speed-value');
        
        debugSpeedSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.app.setDebugSpeed(value);
            debugSpeedValue.textContent = `${value.toFixed(1)}x`;
        });
        
        // Network configuration controls
        this.setupNetworkConfiguration();
        
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
        
        // Wheel friction control
        const wheelFrictionSlider = document.getElementById('wheel-friction');
        const wheelFrictionValue = document.getElementById('wheel-friction-value');

        if (wheelFrictionSlider) {
            wheelFrictionSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.setParameter('wheelFriction', value);
                wheelFrictionValue.textContent = value.toFixed(2);
                this.app.updateWheelFriction(value);
            });
        }
    }
    
    setupNetworkConfiguration() {
        // Initialize current architecture from preset
        this.networkConfig.currentArchitecture = getPreset(this.networkConfig.preset);
        
        // Network preset selector
        const presetSelect = document.getElementById('network-preset');
        const presetDescription = document.getElementById('preset-description');
        const customArchitecture = document.getElementById('custom-architecture');
        
        if (presetSelect) {
            presetSelect.addEventListener('change', (e) => {
                const presetName = e.target.value;
                this.networkConfig.preset = presetName;
                
                if (presetName === 'CUSTOM') {
                    customArchitecture.style.display = 'block';
                    this.setupCustomArchitecture();
                } else {
                    customArchitecture.style.display = 'none';
                    const preset = getPreset(presetName);
                    this.networkConfig.currentArchitecture = preset;
                    presetDescription.textContent = preset.description;
                    this.updateArchitectureDisplay();
                    this.app.updateNetworkArchitecture(preset);
                }
            });
            
            // Set initial description
            const initialPreset = getPreset(this.networkConfig.preset);
            presetDescription.textContent = initialPreset.description;
            this.updateArchitectureDisplay();
        }
        
        // Layer count control for custom architecture
        const layerCountSlider = document.getElementById('layer-count');
        const layerCountValue = document.getElementById('layer-count-value');
        
        if (layerCountSlider) {
            layerCountSlider.addEventListener('input', (e) => {
                const layerCount = parseInt(e.target.value);
                layerCountValue.textContent = layerCount;
                this.updateCustomLayers(layerCount);
            });
        }
    }
    
    setupCustomArchitecture() {
        const layerCount = parseInt(document.getElementById('layer-count').value);
        this.updateCustomLayers(layerCount);
    }
    
    updateCustomLayers(layerCount) {
        const layerConfigs = document.getElementById('layer-configs');
        layerConfigs.innerHTML = '';
        
        // Ensure we have the right number of layers
        while (this.networkConfig.customLayers.length < layerCount) {
            this.networkConfig.customLayers.push(8);
        }
        while (this.networkConfig.customLayers.length > layerCount) {
            this.networkConfig.customLayers.pop();
        }
        
        // Create layer configuration controls
        for (let i = 0; i < layerCount; i++) {
            const layerDiv = document.createElement('div');
            layerDiv.className = 'layer-config';
            
            layerDiv.innerHTML = `
                <label>Layer ${i + 1}:</label>
                <input type="range" id="layer-${i}-size" min="1" max="256" step="1" value="${this.networkConfig.customLayers[i]}">
                <span class="layer-value" id="layer-${i}-value">${this.networkConfig.customLayers[i]}</span>
            `;
            
            layerConfigs.appendChild(layerDiv);
            
            // Add event listener for this layer
            const slider = layerDiv.querySelector('input');
            const valueSpan = layerDiv.querySelector('.layer-value');
            
            slider.addEventListener('input', (e) => {
                const size = parseInt(e.target.value);
                this.networkConfig.customLayers[i] = size;
                valueSpan.textContent = size;
                this.updateCustomArchitecture();
            });
        }
        
        this.updateCustomArchitecture();
    }
    
    updateCustomArchitecture() {
        try {
            // Create custom architecture
            const customArch = createCustomArchitecture({
                name: 'Custom',
                description: 'User-defined architecture',
                layers: [...this.networkConfig.customLayers],
                maxParameters: 10000, // Allow large networks for web training
                deployment: 'web'
            });
            
            this.networkConfig.currentArchitecture = customArch;
            this.updateArchitectureDisplay();
            this.app.updateNetworkArchitecture(customArch);
            
        } catch (error) {
            console.error('Invalid custom architecture:', error);
            this.showArchitectureError(error.message);
        }
    }
    
    updateArchitectureDisplay() {
        const arch = this.networkConfig.currentArchitecture;
        if (!arch) return;
        
        const archDisplay = document.getElementById('architecture-display');
        const paramCount = document.getElementById('parameter-count');
        const memoryEstimate = document.getElementById('memory-estimate');
        
        if (archDisplay) {
            const layerSizes = arch.layers.join(' → ');
            archDisplay.textContent = `Input(${arch.inputSize}) → ${layerSizes} → Output(${arch.outputSize})`;
        }
        
        if (paramCount) {
            const params = arch.getParameterCount();
            paramCount.textContent = `Total Parameters: ${params.toLocaleString()}`;
            paramCount.className = params > 1000 ? 'validation-warning' : '';
        }
        
        if (memoryEstimate) {
            const memoryKB = (arch.getParameterCount() * 4) / 1024; // 4 bytes per float
            memoryEstimate.textContent = `Memory: ~${memoryKB.toFixed(1)} KB`;
        }
        
        // Clear any previous errors
        this.clearArchitectureError();
    }
    
    showArchitectureError(message) {
        const summary = document.querySelector('.architecture-summary');
        if (summary) {
            let errorDiv = summary.querySelector('.validation-error');
            if (!errorDiv) {
                errorDiv = document.createElement('div');
                errorDiv.className = 'validation-error';
                summary.appendChild(errorDiv);
            }
            errorDiv.textContent = `Error: ${message}`;
        }
    }
    
    clearArchitectureError() {
        const errorDiv = document.querySelector('.architecture-summary .validation-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
    
    getNetworkArchitecture() {
        return this.networkConfig.currentArchitecture;
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
                case '4': // 4 - toggle user control (arrows)
                    e.preventDefault();
                    this.app.toggleUserControl();
                    break;
                case 'h': // H - toggle help/debug info
                    e.preventDefault();
                    this.app.renderer.toggleUI('robot');
                    break;
            }
        });
        
        // Manual control arrow keys - separate handlers for keydown/keyup
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (!this.app.userControlEnabled) return; // Check userControlEnabled instead
            
            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.app.manualControl.leftPressed = true;
                    this.app.updateManualTorque();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.app.manualControl.rightPressed = true;
                    this.app.updateManualTorque();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    if (!this.app.isTraining) {
                        this.app.resetRobotPosition();
                    }
                    break;
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (!this.app.userControlEnabled) return; // Check userControlEnabled instead
            
            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.app.manualControl.leftPressed = false;
                    this.app.updateManualTorque();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.app.manualControl.rightPressed = false;
                    this.app.updateManualTorque();
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
        
        // Network configuration is now handled by preset system
        
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
        
        if (document.getElementById('wheel-friction')) {
            document.getElementById('wheel-friction').value = this.parameters.wheelFriction;
            document.getElementById('wheel-friction-value').textContent = this.parameters.wheelFriction.toFixed(2);
        }
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
    
    setSpeedPreset(speed) {
        this.setParameter('trainingSpeed', speed);
        document.getElementById('training-speed').value = speed;
        document.getElementById('training-speed-value').textContent = `${speed.toFixed(1)}x`;
        this.app.setTrainingSpeed(speed);
        console.log(`Speed preset set to ${speed}x`);
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
        this.performanceCharts = null;
        
        // Application state
        this.isTraining = false;
        this.isPaused = false;
        this.trainingStep = 0;
        this.episodeCount = 0;
        this.bestScore = 0;
        this.currentReward = 0;
        this.lastQValue = 0;
        
        // Training control
        this.trainingSpeed = 1.0;
        this.targetPhysicsStepsPerFrame = 1;
        this.episodeEnded = false; // Flag to prevent multiple episode end calls
        
        // Debug control speed
        this.debugSpeed = 1.0;
        this.debugLastAction = 'None';
        this.debugCurrentReward = 0;
        
        // Demo modes
        this.demoMode = 'physics'; // 'physics', 'training', 'evaluation', 'manual'
        
        // Model management state
        this.currentModelName = 'No model loaded';
        this.modelHasUnsavedChanges = false;
        this.trainingStartTime = null;
        
        // Manual control state
        this.manualControl = {
            enabled: false,
            leftPressed: false,
            rightPressed: false,
            upPressed: false,
            manualTorque: 0
        };
        
        // PD controller toggle state
        this.pdControllerEnabled = true; // Start enabled in physics mode
        
        // User control toggle state (separate from manual mode)
        this.userControlEnabled = false;
        
        // Network architecture (will be set by UI)
        this.currentNetworkArchitecture = null;
        
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
        
        // Q-learning training state
        this.previousState = null;
        this.previousAction = undefined;
        this.previousReward = 0;
        this.previousDone = false;
        this.lastTrainingLoss = 0;
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
            
            // Initialize UI state
            this.updatePDControllerUI();
            this.updateUserControlUI();
            this.updateModelDisplay('No model loaded', null);
            
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
        
        // Initialize performance charts
        const chartsPanel = document.getElementById('charts-panel');
        if (chartsPanel) {
            this.performanceCharts = createPerformanceCharts(chartsPanel);
            this.performanceCharts.start();
            console.log('Performance charts initialized');
        }
        
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
            
            // Hide status text after 10 seconds
            setTimeout(() => {
                const statusContainer = document.getElementById('webgpu-status');
                if (statusContainer) {
                    statusContainer.style.display = 'none';
                }
            }, 10000);
            
            // Store status for detailed UI display
            this.webgpuStatus.deviceInfo = quickCheck;
            
        } catch (error) {
            console.warn('WebGPU backend initialization failed:', error.message);
            statusElement.textContent = 'WebGPU Error (CPU fallback)';
            statusElement.className = 'status-unavailable';
            
            // Hide status text after 10 seconds (error case)
            setTimeout(() => {
                const statusContainer = document.getElementById('webgpu-status');
                if (statusContainer) {
                    statusContainer.style.display = 'none';
                }
            }, 10000);
            
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
            this.saveModelToLocalStorage();
        });
        
        document.getElementById('test-model-btn')?.addEventListener('click', () => {
            this.switchDemoMode('evaluation');
            console.log('Testing trained model');
        });
        
        document.getElementById('export-model').addEventListener('click', () => {
            this.exportModelToCpp();
        });
        
        document.getElementById('import-model').addEventListener('click', () => {
            this.importModelFromCpp();
        });
        
        document.getElementById('reset-parameters').addEventListener('click', () => {
            this.resetModelParameters();
        });
        
        // WebGPU status refresh
        document.getElementById('refresh-webgpu').addEventListener('click', () => {
            this.updateWebGPUStatusDisplay();
        });
        
        // Zoom controls
        document.getElementById('zoom-in-btn')?.addEventListener('click', () => {
            if (this.renderer) {
                this.renderer.transform.zoomIn();
                console.log('Zoomed in to', this.renderer.transform.getZoomLevel());
            }
        });

        document.getElementById('zoom-out-btn')?.addEventListener('click', () => {
            if (this.renderer) {
                this.renderer.transform.zoomOut();
                console.log('Zoomed out to', this.renderer.transform.getZoomLevel());
            }
        });

        document.getElementById('zoom-reset-btn')?.addEventListener('click', () => {
            if (this.renderer) {
                this.renderer.transform.resetZoom();
                console.log('Zoom reset to 1.0');
            }
        });
        
        // PD Controller toggle
        document.getElementById('toggle-pd-controller')?.addEventListener('click', () => {
            this.togglePDController();
        });
        
        // User Control toggle
        document.getElementById('toggle-user-control')?.addEventListener('click', () => {
            this.toggleUserControl();
        });
        
        // Setup collapsible sections
        this.setupCollapsibleSections();
        
        // Initialize model list
        this.refreshModelList();
        
        console.log('Controls initialized');
    }

    /**
     * Setup collapsible control sections
     */
    setupCollapsibleSections() {
        // Find all control section headers and add click handlers
        const sections = document.querySelectorAll('.control-section h3');
        
        sections.forEach(header => {
            header.addEventListener('click', () => {
                const section = header.parentElement;
                section.classList.toggle('collapsed');
                
                // Optional: Save section states to localStorage
                const sectionTitle = header.textContent.trim();
                const isCollapsed = section.classList.contains('collapsed');
                localStorage.setItem(`section-${sectionTitle}`, isCollapsed.toString());
                
                console.log(`Toggled section: ${sectionTitle} (${isCollapsed ? 'collapsed' : 'expanded'})`);
            });
        });
        
        // Restore section states from localStorage
        sections.forEach(header => {
            const sectionTitle = header.textContent.trim();
            const savedState = localStorage.getItem(`section-${sectionTitle}`);
            
            if (savedState !== null) {
                const section = header.parentElement;
                const shouldCollapse = savedState === 'true';
                
                if (shouldCollapse) {
                    section.classList.add('collapsed');
                } else {
                    section.classList.remove('collapsed');
                }
            }
        });
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
        
        // Track training start time for model naming
        this.trainingStartTime = new Date();
        const timestamp = this.trainingStartTime.toISOString().replace(/[:.]/g, '-').slice(0, -5);
        
        // Update model name to indicate active training
        if (this.currentModelName === 'No model loaded' || !this.currentModelName.startsWith('TwoWheelBot_DQN_')) {
            this.currentModelName = `Unsaved_DQN_${timestamp}`;
        }
        this.modelHasUnsavedChanges = true;
        
        // Don't reset if we have a loaded model and are continuing training
        if (!this.qlearning || this.episodeCount === 0) {
            // Reset for new training session
            this.episodeCount = 0;
            this.trainingStep = 0;
            this.bestScore = -Infinity;
        }
        
        // Update model display
        this.updateTrainingModelDisplay();
        
        this.startNewEpisode();
        
        document.getElementById('start-training').disabled = true;
        document.getElementById('pause-training').disabled = false;
        
        // Update PD controller UI (disabled during training)
        this.updatePDControllerUI();
        
        console.log('Training started');
    }

    pauseTraining() {
        if (!this.isTraining) return;
        
        this.isPaused = !this.isPaused;
        
        const pauseButton = document.getElementById('pause-training');
        pauseButton.textContent = this.isPaused ? 'Resume Training' : 'Pause Training';
        
        // Update model display to show paused/training status
        this.updateTrainingModelDisplay();
        
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
        
        // Update PD controller UI (re-enabled after training stops)
        this.updatePDControllerUI();
        
        this.updateUI();
        
        console.log('Environment reset');
    }

    resizeCanvas() {
        const container = document.getElementById('main-container');
        const controlsPanel = document.getElementById('controls-panel');
        
        // Calculate available space for canvas
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        const controlsPanelWidth = controlsPanel ? controlsPanel.offsetWidth : 300;
        
        // Canvas should take all available width minus controls panel
        const availableWidth = containerWidth - controlsPanelWidth;
        const availableHeight = containerHeight;
        
        // Use device pixel ratio for crisp rendering on high-DPI displays
        const dpr = window.devicePixelRatio || 1;
        
        // Set canvas display size (CSS pixels)
        this.canvas.style.width = availableWidth + 'px';
        this.canvas.style.height = availableHeight + 'px';
        
        // Set canvas actual size (device pixels)
        const actualWidth = availableWidth * dpr;
        const actualHeight = availableHeight * dpr;
        
        this.canvas.width = actualWidth;
        this.canvas.height = actualHeight;
        
        // Update renderer with new dimensions
        if (this.renderer) {
            this.renderer.resize(actualWidth, actualHeight);
            // Reset the canvas context scale after renderer resize
            const ctx = this.canvas.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        
        console.log('Canvas resized to:', actualWidth, 'x', actualHeight, `(${availableWidth}x${availableHeight} display)`);
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
                
                // Auto-adjust training speed if performance is extremely poor (disabled for manual speed control)
                if (this.frameTimeHistory.length === this.performanceCheckInterval) {
                    const avgFrameTime = this.frameTimeHistory.reduce((a, b) => a + b) / this.frameTimeHistory.length;
                    const targetFrameTime = 16.67; // 60 FPS target
                    
                    // Only auto-adjust if frame time is extremely poor (>100ms = <10 FPS) and speed is very high
                    // This prevents unwanted auto-reduction during intentional high-speed training
                    if (avgFrameTime > 100 && this.targetPhysicsStepsPerFrame > 50) {
                        this.setTrainingSpeed(Math.max(10.0, this.trainingSpeed * 0.9));
                        console.warn(`Extremely poor performance (${avgFrameTime.toFixed(1)}ms/frame), reducing training speed to ${this.trainingSpeed.toFixed(1)}x`);
                    }
                }
            }
            this.lastFrameTime = timestamp;
            
            // Update physics at fixed intervals, with debug speed control for manual mode
            let updateInterval = this.physicsUpdateInterval;
            if (this.demoMode === 'manual' || this.demoMode === 'physics') {
                // Apply debug speed: slower speed = longer interval, faster speed = shorter interval
                updateInterval = this.physicsUpdateInterval / this.debugSpeed;
            }
            
            if (timestamp - this.lastPhysicsUpdate >= updateInterval) {
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
        const maxStepsPerFrame = Math.min(this.targetPhysicsStepsPerFrame, 100);
        const stepsToRun = this.isTraining && !this.isPaused ? maxStepsPerFrame : 1;
        
        for (let step = 0; step < stepsToRun; step++) {
            let motorTorque = 0;
            let actionIndex = 0;
            
            // Get current state for Q-learning
            const currentState = this.robot.getState();
            const normalizedState = currentState.getNormalizedInputs();
            
            // Get motor torque based on current mode
            switch (this.demoMode) {
                case 'physics':
                    // In physics mode: user arrows take precedence over PD controller
                    if (this.userControlEnabled && (this.manualControl.leftPressed || this.manualControl.rightPressed)) {
                        motorTorque = this.getManualTorque();
                    } else if (this.pdControllerEnabled) {
                        motorTorque = this.getPhysicsDemoTorque();
                    }
                    break;
                case 'training':
                    if (this.isTraining && !this.isPaused) {
                        const trainingResult = this.getTrainingTorqueWithAction();
                        motorTorque = trainingResult.torque;
                        actionIndex = trainingResult.actionIndex;
                    } else {
                        // When training paused/stopped: user arrows take precedence over PD controller
                        if (this.userControlEnabled && (this.manualControl.leftPressed || this.manualControl.rightPressed)) {
                            motorTorque = this.getManualTorque();
                        } else if (this.pdControllerEnabled) {
                            motorTorque = this.getPhysicsDemoTorque();
                        }
                    }
                    break;
                case 'evaluation':
                    // Allow user control override during evaluation for robustness testing
                    if (this.userControlEnabled && (this.manualControl.leftPressed || this.manualControl.rightPressed)) {
                        motorTorque = this.getManualTorque();
                        this.debugLastAction = `User: ${motorTorque.toFixed(2)}`;
                    } else {
                        motorTorque = this.getEvaluationTorque();
                    }
                    break;
                case 'manual':
                    // Manual mode: arrows take precedence over PD controller
                    if (this.manualControl.leftPressed || this.manualControl.rightPressed) {
                        motorTorque = this.getManualTorque();
                    } else if (this.pdControllerEnabled) {
                        motorTorque = this.getPhysicsDemoTorque();
                    } else {
                        motorTorque = this.getManualTorque(); // This will be 0 if no keys pressed
                    }
                    break;
            }
            
            // Step physics simulation
            const result = this.robot.step(motorTorque);
            this.currentReward = result.reward;
            
            // Update debug reward display during manual control modes and evaluation
            if (this.demoMode === 'manual' || this.demoMode === 'physics' || this.demoMode === 'evaluation') {
                this.debugCurrentReward = result.reward;
            }
            
            // Q-learning training integration
            if (this.demoMode === 'training' && this.isTraining && !this.isPaused && this.qlearning) {
                // Get next state after physics step
                const nextState = result.state.getNormalizedInputs();
                
                // Train on previous experience if we have one
                if (this.previousState && this.previousAction !== undefined) {
                    const loss = this.qlearning.train(
                        this.previousState,      // Previous state
                        this.previousAction,     // Previous action  
                        this.previousReward,     // ✅ FIXED: Reward from previous action
                        nextState,              // Current state (next state)
                        this.previousDone       // ✅ FIXED: Done from previous step
                    );
                    
                    // Store training loss for metrics with NaN protection
                    this.lastTrainingLoss = isFinite(loss) ? loss : 0;
                }
                
                // Store current experience for next iteration (unless episode is done)
                if (!result.done) {
                    this.previousState = normalizedState.slice();
                    this.previousAction = actionIndex;
                    this.previousReward = result.reward;  // Store current reward for next iteration
                    this.previousDone = result.done;
                } else {
                    // Episode ended - train on final experience
                    const finalLoss = this.qlearning.train(
                        normalizedState,        // Current state  
                        actionIndex,           // Current action
                        result.reward,         // ✅ FIXED: Current reward
                        nextState,            // Final state (after step)
                        result.done           // ✅ FIXED: Current done status
                    );
                    this.lastTrainingLoss = isFinite(finalLoss) ? finalLoss : 0;
                    
                    // Clear previous state to start fresh next episode
                    this.previousState = null;
                    this.previousAction = undefined;
                    this.previousReward = 0;
                    this.previousDone = false;
                }
            }
            
            // Handle episode completion for training/evaluation (only once per episode)
            // Skip episode completion when training is paused to prevent falling/resetting
            if (result.done && (this.demoMode === 'training' || this.demoMode === 'evaluation') && !this.isPaused && !this.episodeEnded) {
                this.episodeEnded = true; // Set flag to prevent multiple calls
                this.handleEpisodeEnd(result);
                return; // Exit physics update completely to avoid multiple episode ends
            }
            
            // Update training step counter
            if (this.isTraining && !this.isPaused) {
                this.trainingStep++;
                
                // Check for episode termination based on step count (8000 steps max)
                if (this.trainingStep >= 8000 && !this.episodeEnded && (this.demoMode === 'training' || this.demoMode === 'evaluation')) {
                    console.log(`Episode terminated at step ${this.trainingStep} (reached max steps)`);
                    this.episodeEnded = true;
                    // Create a synthetic "done" result to trigger episode end
                    const syntheticResult = {
                        state: this.robot.getState(),
                        reward: 1.0, // Positive reward for reaching max steps
                        done: true
                    };
                    this.handleEpisodeEnd(syntheticResult);
                    return;
                }
                
                // Check for robot moving off canvas in all modes
                const robotState = this.robot.getState();
                const bounds = this.renderer ? this.renderer.getPhysicsBounds() : null;
                
                if (bounds && (robotState.position < bounds.minX || robotState.position > bounds.maxX)) {
                    // Handle based on mode
                    if ((this.demoMode === 'training' || this.demoMode === 'evaluation') && !this.episodeEnded) {
                        console.log(`Episode terminated at step ${this.trainingStep} (robot moved off canvas: position=${robotState.position.toFixed(2)}m, bounds=[${bounds.minX.toFixed(2)}, ${bounds.maxX.toFixed(2)}])`);
                        this.episodeEnded = true;
                        // Create a synthetic "done" result with penalty for going off canvas
                        const syntheticResult = {
                            state: this.robot.getState(),
                            reward: -1.0, // Penalty for moving off canvas
                            done: true
                        };
                        this.handleEpisodeEnd(syntheticResult);
                        return;
                    } else if (this.demoMode === 'manual' || this.demoMode === 'physics') {
                        // Reset robot to origin for manual/physics modes
                        console.log(`Robot moved off canvas in ${this.demoMode} mode, resetting to origin`);
                        this.robot.reset({
                            angle: 0,
                            angularVelocity: 0,
                            position: 0,
                            velocity: 0
                        });
                    }
                }
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
        
        // Update Q-learning metrics display
        if (this.qlearning) {
            document.getElementById('epsilon-display').textContent = `Epsilon: ${this.qlearning.hyperparams.epsilon.toFixed(3)}`;
            document.getElementById('consecutive-episodes').textContent = `Consecutive Max Episodes: ${this.qlearning.consecutiveMaxEpisodes}/20`;
        }
        
        // Update training loss display
        document.getElementById('training-loss').textContent = `Training Loss: ${this.lastTrainingLoss.toFixed(4)}`;
        
        // Update Q-value display
        document.getElementById('qvalue-estimate').textContent = `Q-Value: ${this.lastQValue.toFixed(3)}`;
        
        // Update debug display during manual control and evaluation
        if (this.demoMode === 'manual' || this.demoMode === 'physics' || this.demoMode === 'evaluation') {
            const robotState = this.robot ? this.robot.getState() : null;
            if (robotState) {
                document.getElementById('debug-last-action').textContent = this.debugLastAction;
                document.getElementById('debug-current-reward').textContent = this.debugCurrentReward.toFixed(3);
                document.getElementById('debug-robot-angle').textContent = `${(robotState.angle * 180 / Math.PI).toFixed(2)}°`;
                document.getElementById('debug-angular-velocity').textContent = `${robotState.angularVelocity.toFixed(3)} rad/s`;
            }
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
        
        // Performance charts are updated only on episode completion in handleEpisodeEnd()
        
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
            maxStepsPerEpisode: 8000,
            hiddenSize: this.uiControls.getParameter('hiddenNeurons')
        } : {
            learningRate: 0.001,  // Increased now that reward timing is fixed
            epsilon: 0.3,         // Moderate exploration
            epsilonDecay: 0.995,  // Slower epsilon decay for stability
            gamma: 0.95,          // Standard discount factor
            batchSize: 8,         // Larger batch for stability
            targetUpdateFreq: 100, // Standard update frequency
            maxEpisodes: 1000,
            maxStepsPerEpisode: 8000,
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
        const kp = 20; // Proportional gain
        const kd = 5;  // Derivative gain
        
        const torque = (kp * state.angle + kd * state.angularVelocity);
        
        // Add some noise for more interesting behavior
        const noise = (Math.random() - 0.5) * 0.5;
        
        return Math.max(-5, Math.min(5, torque + noise));
    }
    
    /**
     * Get motor torque from Q-learning during training
     */
    getTrainingTorque() {
        const result = this.getTrainingTorqueWithAction();
        return result.torque;
    }
    
    /**
     * Get motor torque and action index from Q-learning during training
     */
    getTrainingTorqueWithAction() {
        if (!this.qlearning) return { torque: 0, actionIndex: 0 };
        
        const state = this.robot.getState();
        const normalizedState = state.getNormalizedInputs();
        
        // Get Q-values for current state
        const qValues = this.qlearning.getAllQValues(normalizedState);
        
        // Select action using epsilon-greedy policy
        const actionIndex = this.qlearning.selectAction(normalizedState, true);
        const actions = [-1.0, 0.0, 1.0]; // Left, brake, right (standardized with QLearning.js)
        
        // Update last Q-value for display (max Q-value) with NaN protection
        if (qValues && qValues.length > 0) {
            const validQValues = Array.from(qValues).filter(val => isFinite(val));
            this.lastQValue = validQValues.length > 0 ? Math.max(...validQValues) : 0;
            
            // Debug: Log Q-values periodically to see if they're changing
            if (this.trainingStep % 100 === 0) {
                console.log(`Step ${this.trainingStep} Q-values:`, 
                    Array.from(qValues).map(v => v.toFixed(4)), 
                    'Max:', this.lastQValue.toFixed(4));
            }
        } else {
            this.lastQValue = 0;
        }
        
        return {
            torque: actions[actionIndex],
            actionIndex: actionIndex
        };
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
        const actions = [-1.0, 0.0, 1.0]; // Standardized action values
        
        return actions[actionIndex];
    }
    
    /**
     * Start a new training episode
     */
    startNewEpisode() {
        this.trainingStep = 0;
        this.episodeEnded = false; // Reset episode end flag
        
        // Reset Q-learning training state
        this.previousState = null;
        this.previousAction = undefined;
        this.previousReward = 0;
        this.previousDone = false;
        this.lastTrainingLoss = 0;
        
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
        
        if (this.demoMode === 'training' && this.isTraining) {
            // Only increment episode count during actual training
            this.episodeCount++;
            console.log(`Episode ${this.episodeCount} completed: Reward=${totalReward.toFixed(2)}, Steps=${this.trainingStep}`);
            
            // Update model display with latest stats
            this.updateTrainingModelDisplay();
            
            // Check for training completion before auto-pause
            if (this.qlearning && this.qlearning.trainingCompleted) {
                this.pauseTraining();
                console.log(`🏆 Training automatically completed! Model has consistently balanced for 20 consecutive episodes.`);
                alert(`🏆 Training Successfully Completed!\n\nYour model has consistently balanced for 20 consecutive episodes of 8,000 steps each.\n\nThe robot is now fully trained and ready for deployment!`);
                return; // Exit early to prevent auto-pause at 10k episodes
            }
            
            // Auto-pause training at 10,000 episodes to prevent runaway sessions (if not already completed)
            if (this.episodeCount >= 10000) {
                this.pauseTraining();
                console.log(`🛑 Training auto-paused at ${this.episodeCount} episodes to prevent runaway session`);
                alert(`Training automatically paused at ${this.episodeCount} episodes.\n\nThis prevents runaway training sessions. You can resume if needed.`);
            }
            
            // Update performance charts with episode completion data
            if (this.performanceCharts && this.qlearning) {
                // Get current Q-value estimate for balanced state
                const balancedState = new Float32Array([0.0, 0.0]); // Perfectly balanced
                const qValues = this.qlearning.getAllQValues(balancedState);
                const avgQValue = Array.from(qValues).reduce((a, b) => a + b, 0) / qValues.length;
                
                this.performanceCharts.updateMetrics({
                    episode: this.episodeCount,
                    reward: totalReward,
                    loss: this.lastTrainingLoss || 0,
                    qValue: avgQValue,
                    epsilon: this.qlearning.hyperparams.epsilon
                });
            }
            
            // Start next episode after brief pause
            setTimeout(() => {
                if (this.isTraining) {
                    this.startNewEpisode();
                }
            }, 100);
        } else {
            // Evaluation mode or demo - restart after longer pause (no episode count increment)
            console.log(`Evaluation run completed: Reward=${totalReward.toFixed(2)}`);
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
        
        // Disable manual control when leaving manual mode
        if (this.demoMode === 'manual' && newMode !== 'manual') {
            this.manualControl.enabled = false;
            this.manualControl.leftPressed = false;
            this.manualControl.rightPressed = false;
            this.manualControl.manualTorque = 0;
        }
        
        this.demoMode = newMode;
        
        // Enable manual control when entering manual mode
        if (newMode === 'manual') {
            this.manualControl.enabled = true;
            console.log('Manual control enabled. Use arrow keys: ← → to balance, ↑ to reset');
        }
        
        // Update PD controller UI based on new mode
        this.updatePDControllerUI();
        this.updateUserControlUI();
        
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
        this.trainingSpeed = Math.max(0.1, Math.min(100.0, speed));
        this.targetPhysicsStepsPerFrame = Math.round(this.trainingSpeed);
        console.log(`Training speed set to ${speed}x (${this.targetPhysicsStepsPerFrame} steps/frame)`);
    }
    
    setDebugSpeed(speed) {
        this.debugSpeed = Math.max(0.1, Math.min(2.0, speed));
        console.log(`Debug speed set to ${speed}x`);
    }
    
    updateNetworkArchitecture(architecture) {
        if (typeof architecture === 'number') {
            // Backward compatibility - convert single hiddenSize to architecture
            console.log(`Network architecture update requested: ${architecture} hidden neurons`);
            this.currentNetworkArchitecture = {
                layers: [architecture],
                inputSize: 2,
                outputSize: 3,
                getParameterCount: () => (2 * architecture + architecture) + (architecture * 3 + 3)
            };
        } else {
            // New architecture object
            console.log(`Network architecture update requested: ${architecture.name} (${architecture.layers.join('→')} layers)`);
            this.currentNetworkArchitecture = architecture;
        }
        
        // Note: This requires reinitializing the Q-learning network
        // For now, we'll log the request and apply it on next training start
        if (this.qlearning) {
            // Update hyperparams for backward compatibility
            this.qlearning.hyperparams.hiddenSize = this.currentNetworkArchitecture.layers[0] || 8;
            this.qlearning.hyperparams.networkArchitecture = this.currentNetworkArchitecture;
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
    
    updateWheelFriction(friction) {
        if (this.robot) {
            const config = this.robot.getConfig();
            config.wheelFriction = friction;
            this.robot.updateConfig(config);
            console.log(`Wheel friction updated to ${friction}`);
        }
    }
    
    /**
     * Update manual control torque based on pressed keys
     */
    updateManualTorque() {
        const maxTorque = this.robot?.getConfig()?.motorStrength || 5.0;
        
        if (this.manualControl.leftPressed && this.manualControl.rightPressed) {
            // Both pressed - no movement
            this.manualControl.manualTorque = 0;
            this.debugLastAction = 'Left+Right (Cancel)';
        } else if (this.manualControl.leftPressed) {
            // Left pressed - negative torque (move left)
            this.manualControl.manualTorque = -maxTorque * 0.8;
            this.debugLastAction = 'Left';
        } else if (this.manualControl.rightPressed) {
            // Right pressed - positive torque (move right)  
            this.manualControl.manualTorque = maxTorque * 0.8;
            this.debugLastAction = 'Right';
        } else {
            // No keys pressed
            this.manualControl.manualTorque = 0;
            this.debugLastAction = 'None';
        }
    }
    
    /**
     * Get manual control torque
     */
    getManualTorque() {
        return this.manualControl.manualTorque;
    }
    
    /**
     * Toggle PD controller on/off
     */
    togglePDController() {
        // Don't allow PD controller during training or evaluation
        if ((this.demoMode === 'training' && this.isTraining) || this.demoMode === 'evaluation') {
            console.log('PD Controller cannot be enabled during training or model testing');
            return;
        }
        
        this.pdControllerEnabled = !this.pdControllerEnabled;
        this.updatePDControllerUI();
        
        console.log('PD Controller', this.pdControllerEnabled ? 'enabled' : 'disabled');
    }
    
    /**
     * Update PD controller button UI
     */
    updatePDControllerUI() {
        const button = document.getElementById('toggle-pd-controller');
        const text = document.getElementById('pd-controller-text');
        const status = document.getElementById('pd-controller-status');
        
        if (!button || !text || !status) return;
        
        // Check if PD controller should be disabled
        const shouldDisable = (this.demoMode === 'training' && this.isTraining) || this.demoMode === 'evaluation';
        
        if (shouldDisable) {
            button.disabled = true;
            text.textContent = 'PD Controller';
            status.textContent = '(Disabled during training/testing)';
            status.style.color = '#666';
            this.pdControllerEnabled = false;
        } else {
            button.disabled = false;
            text.textContent = this.pdControllerEnabled ? 'Disable PD Controller' : 'Enable PD Controller';
            status.textContent = this.pdControllerEnabled ? '(Active)' : '(Disabled)';
            status.style.color = this.pdControllerEnabled ? '#00ff88' : '#888';
            status.style.marginLeft = '0'; // Remove margin since using <br>
        }
    }
    
    /**
     * Toggle user control (arrow keys) on/off
     */
    toggleUserControl() {
        this.userControlEnabled = !this.userControlEnabled;
        this.updateUserControlUI();
        
        // Clear any pressed keys when disabling
        if (!this.userControlEnabled) {
            this.manualControl.leftPressed = false;
            this.manualControl.rightPressed = false;
            this.manualControl.manualTorque = 0;
        }
        
        console.log('User Control (arrows)', this.userControlEnabled ? 'enabled' : 'disabled');
    }
    
    /**
     * Update user control button UI
     */
    updateUserControlUI() {
        const button = document.getElementById('toggle-user-control');
        const text = document.getElementById('user-control-text');
        const status = document.getElementById('user-control-status');
        
        if (!button || !text || !status) return;
        
        button.disabled = false;
        text.textContent = this.userControlEnabled ? 'Disable User Control' : 'Enable User Control';
        status.textContent = this.userControlEnabled ? '(Active - Use ← →)' : '(Disabled)';
        status.style.color = this.userControlEnabled ? '#00ff88' : '#888';
        status.style.marginLeft = '0'; // Remove margin since using <br>
    }
    
    /**
     * Reset robot position for manual control
     */
    resetRobotPosition() {
        if (this.robot) {
            this.robot.reset({
                angle: 0,
                angularVelocity: 0,
                position: 0,
                velocity: 0
            });
            console.log('Robot position reset');
        }
    }
    
    /**
     * Model Management Methods
     */
    
    saveModelToLocalStorage() {
        if (!this.qlearning || !this.qlearning.isInitialized) {
            alert('No trained model to save. Please train the model first.');
            return;
        }
        
        // Generate model name with timestamp
        const now = new Date();
        const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const modelName = `TwoWheelBot_DQN_${timestamp}`;
        
        // Get model data
        const modelData = this.qlearning.save();
        
        // Add metadata
        const saveData = {
            name: modelName,
            timestamp: now.toISOString(),
            type: 'DQN',
            architecture: this.qlearning.getStats(),
            episodesTrained: this.episodeCount,
            bestScore: this.bestScore,
            trainingCompleted: this.qlearning.trainingCompleted,
            consecutiveMaxEpisodes: this.qlearning.consecutiveMaxEpisodes,
            modelData: modelData
        };
        
        // Save to localStorage
        try {
            localStorage.setItem(`model_${modelName}`, JSON.stringify(saveData));
            
            // Save model name to list of saved models
            let savedModels = JSON.parse(localStorage.getItem('savedModelsList') || '[]');
            if (!savedModels.includes(modelName)) {
                savedModels.push(modelName);
                localStorage.setItem('savedModelsList', JSON.stringify(savedModels));
            }
            
            // Update UI and clear unsaved flag
            this.currentModelName = modelName;
            this.modelHasUnsavedChanges = false;
            this.updateModelDisplay(modelName, saveData);
            
            // Refresh the model list to show the new/updated model
            this.refreshModelList();
            
            alert(`Model saved successfully as:\n${modelName}`);
            console.log('Model saved:', modelName);
        } catch (error) {
            alert('Failed to save model: ' + error.message);
            console.error('Save model error:', error);
        }
    }
    
    refreshModelList() {
        // Get list of saved models
        const savedModels = JSON.parse(localStorage.getItem('savedModelsList') || '[]');
        const listContainer = document.getElementById('saved-models-list');
        const noModelsMsg = document.getElementById('no-saved-models');
        
        if (!listContainer || !noModelsMsg) return;
        
        // Clear existing list
        listContainer.innerHTML = '';
        
        if (savedModels.length === 0) {
            noModelsMsg.style.display = 'block';
            return;
        }
        
        noModelsMsg.style.display = 'none';
        
        // Create list items for each model
        savedModels.forEach(modelName => {
            try {
                const saveData = JSON.parse(localStorage.getItem(`model_${modelName}`));
                if (!saveData) return;
                
                const modelItem = this.createModelListItem(modelName, saveData);
                listContainer.appendChild(modelItem);
            } catch (error) {
                console.error(`Error loading model ${modelName}:`, error);
            }
        });
    }
    
    createModelListItem(modelName, saveData) {
        const isCurrent = this.currentModelName === modelName;
        
        // Create model item container
        const modelItem = document.createElement('div');
        modelItem.className = `model-item ${isCurrent ? 'current' : ''}`;
        modelItem.id = `model-item-${modelName}`;
        
        // Create header with name and actions
        const header = document.createElement('div');
        header.className = 'model-item-header';
        
        const nameDiv = document.createElement('div');
        nameDiv.className = 'model-item-name';
        nameDiv.innerHTML = modelName;
        if (isCurrent) {
            nameDiv.innerHTML += '<span class="current-badge">LOADED</span>';
        }
        
        const actions = document.createElement('div');
        actions.className = 'model-item-actions';
        
        if (isCurrent) {
            // Unload button for current model
            const unloadBtn = document.createElement('button');
            unloadBtn.textContent = 'Unload';
            unloadBtn.onclick = () => this.unloadModel();
            actions.appendChild(unloadBtn);
        } else {
            // Load button for other models
            const loadBtn = document.createElement('button');
            loadBtn.textContent = 'Load';
            loadBtn.className = 'primary';
            loadBtn.onclick = () => this.loadModelByName(modelName);
            actions.appendChild(loadBtn);
        }
        
        // Delete button for all models
        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.className = 'danger';
        deleteBtn.onclick = () => this.deleteModel(modelName);
        actions.appendChild(deleteBtn);
        
        header.appendChild(nameDiv);
        header.appendChild(actions);
        
        // Create details section
        const details = document.createElement('div');
        details.className = 'model-item-details';
        
        // Extract model info
        const architecture = saveData.modelData?.qNetworkWeights?.architecture;
        const episodes = saveData.episodesTrained || 0;
        const bestScore = saveData.bestScore || 0;
        const timestamp = saveData.timestamp ? new Date(saveData.timestamp).toLocaleString() : 'Unknown';
        const completed = saveData.trainingCompleted ? 'Yes' : 'No';
        
        details.innerHTML = `
            <div class="model-item-detail"><strong>Architecture:</strong> ${architecture ? `${architecture.inputSize}-${architecture.hiddenSize}-${architecture.outputSize}` : 'Unknown'}</div>
            <div class="model-item-detail"><strong>Episodes:</strong> ${episodes} | <strong>Best Score:</strong> ${bestScore.toFixed(0)}</div>
            <div class="model-item-detail"><strong>Training Complete:</strong> ${completed}</div>
            <div class="model-item-detail"><strong>Saved:</strong> ${timestamp}</div>
        `;
        
        modelItem.appendChild(header);
        modelItem.appendChild(details);
        
        return modelItem;
    }
    
    async loadModelByName(modelName) {
        try {
            const saveData = JSON.parse(localStorage.getItem(`model_${modelName}`));
            
            if (!saveData || !saveData.modelData) {
                alert('Model data is corrupted or missing.');
                return;
            }
            
            // Load model into Q-learning
            await this.loadModel(saveData);
            
            // Refresh the model list to update UI
            this.refreshModelList();
            
            alert(`Model loaded successfully:\n${modelName}`);
            console.log('Model loaded:', modelName);
        } catch (error) {
            alert('Failed to load model: ' + error.message);
            console.error('Load model error:', error);
        }
    }
    
    unloadModel() {
        if (confirm('Are you sure you want to unload the current model?\nAny unsaved changes will be lost.')) {
            // Reset Q-learning
            if (this.qlearning) {
                this.qlearning.reset();
            }
            
            // Reset state
            this.currentModelName = 'No model loaded';
            this.modelHasUnsavedChanges = false;
            this.episodeCount = 0;
            this.bestScore = 0;
            
            // Update display
            this.updateModelDisplay('No model loaded', null);
            this.refreshModelList();
            
            console.log('Model unloaded');
        }
    }
    
    deleteModel(modelName) {
        const isCurrent = this.currentModelName === modelName;
        const confirmMsg = isCurrent ? 
            `Are you sure you want to delete the currently loaded model "${modelName}"?\nThis action cannot be undone.` :
            `Are you sure you want to delete "${modelName}"?\nThis action cannot be undone.`;
        
        if (confirm(confirmMsg)) {
            try {
                // Remove from localStorage
                localStorage.removeItem(`model_${modelName}`);
                
                // Update saved models list
                let savedModels = JSON.parse(localStorage.getItem('savedModelsList') || '[]');
                savedModels = savedModels.filter(name => name !== modelName);
                localStorage.setItem('savedModelsList', JSON.stringify(savedModels));
                
                // If deleting current model, unload it
                if (isCurrent) {
                    this.currentModelName = 'No model loaded';
                    this.modelHasUnsavedChanges = false;
                    this.updateModelDisplay('No model loaded', null);
                }
                
                // Refresh the model list
                this.refreshModelList();
                
                console.log(`Model deleted: ${modelName}`);
                alert(`Model "${modelName}" has been deleted.`);
            } catch (error) {
                alert('Failed to delete model: ' + error.message);
                console.error('Delete model error:', error);
            }
        }
    }
    
    async loadModel(saveData) {
        // Extract architecture from saved model
        const savedArchitecture = saveData.modelData?.qNetworkWeights?.architecture;
        if (!savedArchitecture) {
            throw new Error('Invalid model data: missing architecture information');
        }
        
        // Check if current Q-learning exists and has compatible architecture
        let needsReinit = false;
        if (this.qlearning && this.qlearning.isInitialized) {
            const currentArch = this.qlearning.qNetwork.getArchitecture();
            
            if (currentArch.inputSize !== savedArchitecture.inputSize ||
                currentArch.hiddenSize !== savedArchitecture.hiddenSize ||
                currentArch.outputSize !== savedArchitecture.outputSize) {
                
                console.log(`Model architecture mismatch:`);
                console.log(`  Current: ${currentArch.inputSize}-${currentArch.hiddenSize}-${currentArch.outputSize}`);
                console.log(`  Loading: ${savedArchitecture.inputSize}-${savedArchitecture.hiddenSize}-${savedArchitecture.outputSize}`);
                
                needsReinit = true;
            }
        }
        
        // Initialize Q-learning if not exists or architecture mismatch
        if (!this.qlearning || needsReinit) {
            // Update UI controls to match loaded model architecture
            if (this.uiControls) {
                this.uiControls.setParameter('hiddenNeurons', savedArchitecture.hiddenSize);
            }
            
            await this.initializeQLearning({
                hiddenSize: savedArchitecture.hiddenSize
            });
        }
        
        // Load the model data (Q-learning.load now handles architecture validation internally)
        await this.qlearning.load(saveData.modelData);
        
        // Restore training state
        this.episodeCount = saveData.episodesTrained || 0;
        this.bestScore = saveData.bestScore || 0;
        
        // Set current model name and clear unsaved flag
        this.currentModelName = saveData.name;
        this.modelHasUnsavedChanges = false;
        
        // Update UI to reflect loaded model architecture
        this.updateModelDisplay(saveData.name, saveData);
        
        console.log(`Model loaded: ${saveData.name} with architecture ${savedArchitecture.inputSize}-${savedArchitecture.hiddenSize}-${savedArchitecture.outputSize}`);
    }
    
    exportModelToCpp() {
        if (!this.qlearning || !this.qlearning.isInitialized) {
            alert('No trained model to export. Please train the model first.');
            return;
        }
        
        // Generate filename with timestamp
        const now = new Date();
        const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const filename = `two_wheel_bot_dqn_${timestamp}.cpp`;
        
        // Get model weights
        const weights = this.qlearning.qNetwork.getWeights();
        const architecture = this.qlearning.qNetwork.getArchitecture();
        
        // Generate C++ code
        let cppCode = this.generateCppCode(weights, architecture, timestamp);
        
        // Create download
        const blob = new Blob([cppCode], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        alert(`Model exported as C++ file:\n${filename}`);
        console.log('Model exported:', filename);
    }
    
    importModelFromCpp() {
        const fileInput = document.getElementById('cpp-file-input');
        
        // Create new file input handler
        const handleFileSelect = (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const cppContent = e.target.result;
                    console.log('Parsing C++ file:', file.name);
                    const importedModel = this.parseCppModel(cppContent, file.name);
                    console.log('Parsed model:', importedModel);
                    
                    await this.loadImportedModel(importedModel);
                    
                    // Verify it was saved to localStorage
                    const savedModels = JSON.parse(localStorage.getItem('savedModelsList') || '[]');
                    console.log('Updated saved models list:', savedModels);
                    
                    alert(`Model imported successfully from:\n${file.name}\n\nModel name: ${importedModel.name}`);
                    console.log('Model imported successfully:', file.name);
                } catch (error) {
                    alert('Failed to import model: ' + error.message);
                    console.error('Import error:', error);
                }
            };
            
            reader.readAsText(file);
            
            // Reset file input
            fileInput.value = '';
            fileInput.removeEventListener('change', handleFileSelect);
        };
        
        // Add event listener and trigger file dialog
        fileInput.addEventListener('change', handleFileSelect);
        fileInput.click();
    }
    
    parseCppModel(cppContent, filename) {
        // Extract architecture information
        const inputSizeMatch = cppContent.match(/static const int INPUT_SIZE = (\d+);/);
        const hiddenSizeMatch = cppContent.match(/static const int HIDDEN_SIZE = (\d+);/);
        const outputSizeMatch = cppContent.match(/static const int OUTPUT_SIZE = (\d+);/);
        
        if (!inputSizeMatch || !hiddenSizeMatch || !outputSizeMatch) {
            throw new Error('Could not extract network architecture from C++ file');
        }
        
        const inputSize = parseInt(inputSizeMatch[1]);
        const hiddenSize = parseInt(hiddenSizeMatch[1]);
        const outputSize = parseInt(outputSizeMatch[1]);
        
        // Extract weights arrays
        const weightsInputHidden = this.extractWeightsArray(cppContent, 'weightsInputHidden');
        const biasHidden = this.extractWeightsArray(cppContent, 'biasHidden');
        const weightsHiddenOutput = this.extractWeightsArray(cppContent, 'weightsHiddenOutput');
        const biasOutput = this.extractWeightsArray(cppContent, 'biasOutput');
        
        // Validate dimensions
        if (weightsInputHidden.length !== inputSize * hiddenSize) {
            throw new Error(`Expected ${inputSize * hiddenSize} input-to-hidden weights, got ${weightsInputHidden.length}`);
        }
        if (biasHidden.length !== hiddenSize) {
            throw new Error(`Expected ${hiddenSize} hidden biases, got ${biasHidden.length}`);
        }
        if (weightsHiddenOutput.length !== hiddenSize * outputSize) {
            throw new Error(`Expected ${hiddenSize * outputSize} hidden-to-output weights, got ${weightsHiddenOutput.length}`);
        }
        if (biasOutput.length !== outputSize) {
            throw new Error(`Expected ${outputSize} output biases, got ${biasOutput.length}`);
        }
        
        // Extract timestamp from filename or comment
        const timestampMatch = filename.match(/(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})/) || 
                              cppContent.match(/Generated: (\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})/);
        const timestamp = timestampMatch ? timestampMatch[1] : new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        
        // Calculate parameter count
        const parameterCount = weightsInputHidden.length + biasHidden.length + 
                              weightsHiddenOutput.length + biasOutput.length;
        
        return {
            name: `Imported_DQN_${timestamp}`,
            architecture: { inputSize, hiddenSize, outputSize },
            weights: {
                architecture: {
                    inputSize: inputSize,
                    hiddenSize: hiddenSize,
                    outputSize: outputSize,
                    parameterCount: parameterCount
                },
                weightsInputHidden: Array.from(weightsInputHidden),
                biasHidden: Array.from(biasHidden),
                weightsHiddenOutput: Array.from(weightsHiddenOutput),
                biasOutput: Array.from(biasOutput),
                initMethod: 'imported'
            },
            filename: filename,
            importDate: new Date().toISOString()
        };
    }
    
    extractWeightsArray(cppContent, arrayName) {
        // Find the array declaration
        const arrayPattern = new RegExp(`const float ${arrayName}\\[.*?\\] = \\{([^}]+)\\};`, 's');
        const match = cppContent.match(arrayPattern);
        
        if (!match) {
            throw new Error(`Could not find ${arrayName} array in C++ file`);
        }
        
        // Extract values between braces
        const arrayContent = match[1];
        
        // Parse individual values, handling potential line breaks and formatting
        const values = arrayContent
            .split(',')
            .map(s => s.trim())
            .filter(s => s.length > 0)
            .map(s => {
                // Remove 'f' suffix if present
                const cleanValue = s.replace(/f$/, '');
                const value = parseFloat(cleanValue);
                if (isNaN(value)) {
                    throw new Error(`Invalid weight value: ${s}`);
                }
                return value;
            });
        
        return values;
    }
    
    async loadImportedModel(importedModel) {
        // Initialize Q-learning if not already done
        if (!this.qlearning) {
            await this.initializeQLearning();
        }
        
        // Verify architecture compatibility
        const currentArch = this.qlearning.qNetwork.getArchitecture();
        const importedArch = importedModel.architecture;
        
        if (currentArch.inputSize !== importedArch.inputSize ||
            currentArch.hiddenSize !== importedArch.hiddenSize ||
            currentArch.outputSize !== importedArch.outputSize) {
            
            const recreate = confirm(
                `Architecture mismatch:\n` +
                `Current: ${currentArch.inputSize}-${currentArch.hiddenSize}-${currentArch.outputSize}\n` +
                `Imported: ${importedArch.inputSize}-${importedArch.hiddenSize}-${importedArch.outputSize}\n\n` +
                `Recreate network with imported architecture?`
            );
            
            if (!recreate) {
                throw new Error('Import cancelled - architecture mismatch');
            }
            
            // Recreate Q-learning with imported architecture
            await this.initializeQLearning({
                hiddenSize: importedArch.hiddenSize
            });
        }
        
        // Load weights into network
        this.qlearning.qNetwork.setWeights(importedModel.weights);
        
        // Update target network to match
        this.qlearning.targetNetwork = this.qlearning.qNetwork.clone();
        
        // Reset training state for imported model
        this.episodeCount = 0;
        this.trainingStep = 0;
        this.bestScore = 0;
        
        // Set model name and clear unsaved flag
        this.currentModelName = importedModel.name;
        this.modelHasUnsavedChanges = false;
        
        // Create complete model data for saving
        const modelData = {
            hyperparams: this.qlearning.hyperparams,
            qNetworkWeights: importedModel.weights,  // Use the imported weights with architecture
            targetNetworkWeights: importedModel.weights,  // Same weights for target network
            episode: 0,
            stepCount: 0,
            metrics: { episodes: 0, totalSteps: 0, averageReward: 0, bestReward: 0 },
            actions: this.qlearning.actions
        };
        
        // Create save data compatible with localStorage format
        const saveData = {
            name: importedModel.name,
            timestamp: importedModel.importDate,
            type: 'Imported DQN',
            architecture: importedArch,
            episodesTrained: 0,
            bestScore: 0,
            trainingCompleted: false,
            consecutiveMaxEpisodes: 0,
            modelData: modelData,
            importedFrom: importedModel.filename
        };
        
        // Save imported model to localStorage so it appears in Load Model list
        try {
            localStorage.setItem(`model_${importedModel.name}`, JSON.stringify(saveData));
            
            // Add to saved models list
            let savedModels = JSON.parse(localStorage.getItem('savedModelsList') || '[]');
            if (!savedModels.includes(importedModel.name)) {
                savedModels.push(importedModel.name);
                localStorage.setItem('savedModelsList', JSON.stringify(savedModels));
            }
            console.log('Imported model saved to localStorage:', importedModel.name);
        } catch (error) {
            console.warn('Could not save imported model to localStorage:', error);
        }
        
        // Update UI with imported model info (use the actual imported model as current)
        this.updateModelDisplay(importedModel.name, saveData);
        
        // Refresh the model list to show the imported model
        this.refreshModelList();
        
        console.log('Imported model loaded:', importedModel.name);
    }
    
    generateCppCode(weights, architecture, timestamp) {
        const { inputSize, hiddenSize, outputSize } = architecture;
        
        return `/**
 * Two-Wheel Balancing Robot DQN Model
 * Generated: ${timestamp}
 * Architecture: ${inputSize}-${hiddenSize}-${outputSize}
 * 
 * This file contains the trained neural network weights for deployment
 * on embedded systems (Arduino, ESP32, STM32, etc.)
 */

#include <math.h>

class TwoWheelBotDQN {
private:
    static const int INPUT_SIZE = ${inputSize};
    static const int HIDDEN_SIZE = ${hiddenSize};
    static const int OUTPUT_SIZE = ${outputSize};
    
    // Network weights (stored in program memory to save RAM)
    const float weightsInputHidden[INPUT_SIZE * HIDDEN_SIZE] = {
        ${this.formatWeights(weights.weightsInputHidden, 8)}
    };
    
    const float biasHidden[HIDDEN_SIZE] = {
        ${this.formatWeights(weights.biasHidden, 8)}
    };
    
    const float weightsHiddenOutput[HIDDEN_SIZE * OUTPUT_SIZE] = {
        ${this.formatWeights(weights.weightsHiddenOutput, 8)}
    };
    
    const float biasOutput[OUTPUT_SIZE] = {
        ${this.formatWeights(weights.biasOutput, 8)}
    };
    
    // Activation function (ReLU)
    float relu(float x) {
        return x > 0 ? x : 0;
    }
    
public:
    /**
     * Get action from current state
     * @param angle Robot angle in radians
     * @param angularVelocity Angular velocity in rad/s
     * @return Action index (0=left, 1=brake, 2=right)
     */
    int getAction(float angle, float angularVelocity) {
        // Normalize inputs
        float input[INPUT_SIZE];
        input[0] = constrain(angle / (M_PI / 3), -1.0, 1.0);
        input[1] = constrain(angularVelocity / 10.0, -1.0, 1.0);
        
        // Hidden layer computation
        float hidden[HIDDEN_SIZE];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            hidden[h] = biasHidden[h];
            for (int i = 0; i < INPUT_SIZE; i++) {
                hidden[h] += input[i] * weightsInputHidden[i * HIDDEN_SIZE + h];
            }
            hidden[h] = relu(hidden[h]);
        }
        
        // Output layer computation
        float output[OUTPUT_SIZE];
        float maxValue = -1e10;
        int bestAction = 0;
        
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            output[o] = biasOutput[o];
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                output[o] += hidden[h] * weightsHiddenOutput[h * OUTPUT_SIZE + o];
            }
            
            // Track best action
            if (output[o] > maxValue) {
                maxValue = output[o];
                bestAction = o;
            }
        }
        
        return bestAction;
    }
    
    /**
     * Get motor torque for action
     * @param action Action index
     * @return Motor torque (-1.0 to 1.0)
     */
    float getMotorTorque(int action) {
        const float actions[3] = {-1.0, 0.0, 1.0};
        return actions[action];
    }
    
private:
    float constrain(float value, float min, float max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
};

// Usage example:
// TwoWheelBotDQN bot;
// int action = bot.getAction(angle, angularVelocity);
// float torque = bot.getMotorTorque(action);
`;
    }
    
    formatWeights(weights, itemsPerLine) {
        const formatted = [];
        for (let i = 0; i < weights.length; i += itemsPerLine) {
            const line = weights.slice(i, i + itemsPerLine)
                .map(w => w.toFixed(6) + 'f')
                .join(', ');
            formatted.push('        ' + line);
        }
        return formatted.join(',\n');
    }
    
    resetModelParameters() {
        if (!this.qlearning || !this.qlearning.isInitialized) {
            alert('No model to reset. Please initialize training first.');
            return;
        }
        
        if (!confirm('Are you sure you want to reset all model parameters? This will lose all training progress.')) {
            return;
        }
        
        // Reset Q-learning
        this.qlearning.reset();
        
        // Reset training state
        this.episodeCount = 0;
        this.trainingStep = 0;
        this.bestScore = 0;
        
        // Reset model name
        this.currentModelName = 'No model loaded';
        this.modelHasUnsavedChanges = false;
        
        // Update UI
        this.updateModelDisplay('Untrained Model', null);
        
        alert('Model parameters have been reset.');
        console.log('Model parameters reset');
    }
    
    updateModelDisplay(modelName, saveData) {
        const nameElement = document.getElementById('current-model-name');
        const statsElement = document.getElementById('model-stats');
        
        if (nameElement) {
            nameElement.textContent = modelName || 'No model loaded';
        }
        
        if (statsElement && saveData) {
            const stats = [];
            if (saveData.episodesTrained) {
                stats.push(`Episodes: ${saveData.episodesTrained}`);
            }
            if (saveData.bestScore) {
                stats.push(`Best Score: ${saveData.bestScore.toFixed(1)}`);
            }
            if (saveData.trainingCompleted) {
                stats.push('✓ Training Completed');
            }
            if (saveData.timestamp) {
                const date = new Date(saveData.timestamp);
                stats.push(`Saved: ${date.toLocaleDateString()} ${date.toLocaleTimeString()}`);
            }
            statsElement.textContent = stats.join(' | ');
        } else if (statsElement) {
            statsElement.textContent = '';
        }
    }
    
    updateTrainingModelDisplay() {
        const nameElement = document.getElementById('current-model-name');
        const statsElement = document.getElementById('model-stats');
        
        if (nameElement) {
            let displayName = this.currentModelName;
            if (this.modelHasUnsavedChanges && !displayName.includes('(unsaved)')) {
                displayName += ' (unsaved)';
            }
            nameElement.textContent = displayName;
        }
        
        if (statsElement) {
            const stats = [];
            stats.push(`Episodes: ${this.episodeCount}`);
            if (this.bestScore && this.bestScore > -Infinity) {
                stats.push(`Best: ${this.bestScore.toFixed(1)}`);
            }
            if (this.qlearning) {
                if (this.qlearning.trainingCompleted) {
                    stats.push('✓ Completed');
                } else if (this.qlearning.consecutiveMaxEpisodes > 0) {
                    stats.push(`Progress: ${this.qlearning.consecutiveMaxEpisodes}/20`);
                }
                if (this.isTraining && !this.isPaused) {
                    stats.push('🔴 Training');
                } else if (this.isPaused) {
                    stats.push('⏸️ Paused');
                }
            }
            statsElement.textContent = stats.join(' | ');
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