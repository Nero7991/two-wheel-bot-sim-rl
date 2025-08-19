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
import { SystemCapabilities, ParallelQLearning } from './training/ParallelTraining.js';
import { TrainingPerformanceTracker, SmartRenderingManager } from './training/PerformanceTracker.js';

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
            
            // Core Learning Parameters
            learningRate: 0.0020,
            gamma: 0.99,
            
            // Exploration Parameters
            epsilon: 0.9,
            epsilonMin: 0.01,
            epsilonDecay: 2500,
            
            // Training Parameters
            batchSize: 128,
            targetUpdateFreq: 100,
            maxEpisodes: 1000,
            maxStepsPerEpisode: 1000,
            
            // Robot Physics Parameters
            robotMass: 1.0,
            robotHeight: 0.4,
            motorStrength: 5.0,
            wheelFriction: 0.3,
            maxAngle: Math.PI / 6, // 30 degrees
            motorTorqueRange: 8.0
        };
        
        // Network architecture configuration
        this.networkConfig = {
            preset: 'DQN_STANDARD',
            customLayers: [8],
            currentArchitecture: null
        };
        
        // Parameter validation ranges
        this.validationRanges = {
            trainingSpeed: { min: 0.1, max: 1000.0 },
            hiddenNeurons: { min: 4, max: 16 },
            learningRate: { min: 0.0001, max: 0.01 },
            gamma: { min: 0.8, max: 0.999 },
            epsilon: { min: 0.0, max: 1.0 },
            epsilonMin: { min: 0.0, max: 0.1 },
            epsilonDecay: { min: 1000, max: 10000 },
            batchSize: { min: 4, max: 512 },
            targetUpdateFreq: { min: 10, max: 1000 },
            maxEpisodes: { min: 100, max: 1000000 },
            maxStepsPerEpisode: { min: 50, max: 5000 },
            robotMass: { min: 0.5, max: 3.0 },
            robotHeight: { min: 0.2, max: 0.8 },
            motorStrength: { min: 1.0, max: 20.0 },
            wheelFriction: { min: 0.0, max: 1.0 },
            maxAngle: { min: Math.PI / 180, max: Math.PI / 3 }, // 1 to 60 degrees
            motorTorqueRange: { min: 0.5, max: 10.0 }
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
        
        if (trainingSpeedSlider) {
            trainingSpeedSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.setParameter('trainingSpeed', value);
                this.app.setTrainingSpeed(value);
            });
        }
        
        // Speed preset buttons
        const speedButtons = [
            { id: 'speed-1x', value: 1.0 },
            { id: 'speed-2x', value: 2.0 },
            { id: 'speed-10x', value: 10.0 },
            { id: 'speed-20x', value: 20.0 },
            { id: 'speed-50x', value: 50.0 }
        ];
        
        speedButtons.forEach(button => {
            const element = document.getElementById(button.id);
            if (element) {
                element.addEventListener('click', () => {
                    this.setSpeedPreset(button.value);
                });
            }
        });
        
        // Parallel training button
        const parallelButton = document.getElementById('speed-parallel');
        if (parallelButton) {
            parallelButton.addEventListener('click', () => {
                this.enableParallelTraining();
            });
        }
        
        // Debug speed control for manual testing
        const debugSpeedSlider = document.getElementById('debug-speed');
        const debugSpeedValue = document.getElementById('debug-speed-value');
        
        if (debugSpeedSlider) {
            debugSpeedSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.app.setDebugSpeed(value);
                debugSpeedValue.textContent = `${value.toFixed(1)}x`;
            });
        }
        
        // Network configuration controls
        this.setupNetworkConfiguration();
        
        // Learning rate control
        const learningRateSlider = document.getElementById('learning-rate');
        const learningRateValue = document.getElementById('learning-rate-value');
        
        learningRateSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('learningRate', value);
            this.app.updateLearningRate(value);
        });
        
        // Epsilon control
        const epsilonSlider = document.getElementById('epsilon');
        const epsilonValue = document.getElementById('epsilon-value');
        
        epsilonSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('epsilon', value);
            this.app.updateEpsilon(value);
        });
        
        // Gamma control
        const gammaSlider = document.getElementById('gamma');
        const gammaValue = document.getElementById('gamma-value');
        
        gammaSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('gamma', value);
            this.app.updateGamma(value);
        });
        
        // Epsilon Min control
        const epsilonMinSlider = document.getElementById('epsilon-min');
        const epsilonMinValue = document.getElementById('epsilon-min-value');
        
        epsilonMinSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('epsilonMin', value);
            this.app.updateEpsilonMin(value);
        });
        
        // Epsilon Decay control
        const epsilonDecaySlider = document.getElementById('epsilon-decay');
        const epsilonDecayValue = document.getElementById('epsilon-decay-value');
        
        epsilonDecaySlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.setParameter('epsilonDecay', value);
            this.app.updateEpsilonDecay(value);
        });
        
        // Batch Size control
        const batchSizeSlider = document.getElementById('batch-size');
        const batchSizeValue = document.getElementById('batch-size-value');
        
        batchSizeSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.setParameter('batchSize', value);
            this.app.updateBatchSize(value);
        });
        
        // Target Update Frequency control
        const targetUpdateFreqSlider = document.getElementById('target-update-freq');
        const targetUpdateFreqValue = document.getElementById('target-update-freq-value');
        
        targetUpdateFreqSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.setParameter('targetUpdateFreq', value);
            this.app.updateTargetUpdateFreq(value);
        });
        
        // Max Episodes control
        const maxEpisodesSlider = document.getElementById('max-episodes');
        const maxEpisodesValue = document.getElementById('max-episodes-value');
        
        maxEpisodesSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.setParameter('maxEpisodes', value);
            this.app.updateMaxEpisodes(value);
        });
        
        // Max Steps Per Episode control
        const maxStepsPerEpisodeSlider = document.getElementById('max-steps-per-episode');
        const maxStepsPerEpisodeValue = document.getElementById('max-steps-per-episode-value');
        
        maxStepsPerEpisodeSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.setParameter('maxStepsPerEpisode', value);
            this.app.updateMaxStepsPerEpisode(value);
        });
        
        // Robot mass control
        const robotMassSlider = document.getElementById('robot-mass');
        const robotMassValue = document.getElementById('robot-mass-value');
        
        robotMassSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('robotMass', value);
            this.app.updateRobotMass(value);
        });
        
        // Robot height control
        const robotHeightSlider = document.getElementById('robot-height');
        const robotHeightValue = document.getElementById('robot-height-value');
        
        robotHeightSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('robotHeight', value);
            this.app.updateRobotHeight(value);
        });
        
        // Motor strength control
        const motorStrengthSlider = document.getElementById('motor-strength');
        const motorStrengthValue = document.getElementById('motor-strength-value');
        
        motorStrengthSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setParameter('motorStrength', value);
            this.app.updateMotorStrength(value);
        });
        
        // Wheel friction control
        const wheelFrictionSlider = document.getElementById('wheel-friction');
        const wheelFrictionValue = document.getElementById('wheel-friction-value');

        if (wheelFrictionSlider) {
            wheelFrictionSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.setParameter('wheelFriction', value);
                this.app.updateWheelFriction(value);
            });
        }
        
        // Max angle control
        const maxAngleSlider = document.getElementById('max-angle');
        const maxAngleValue = document.getElementById('max-angle-value');
        
        if (maxAngleSlider) {
            maxAngleSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.setParameter('maxAngle', value);
                this.app.updateMaxAngle(value);
            });
        }
        
        // Motor torque range control
        const motorTorqueRangeSlider = document.getElementById('motor-torque-range');
        const motorTorqueRangeValue = document.getElementById('motor-torque-range-value');
        
        if (motorTorqueRangeSlider) {
            motorTorqueRangeSlider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.setParameter('motorTorqueRange', value);
                this.app.updateMotorTorqueRange(value);
            });
        }
        
        // History timesteps control for multi-timestep learning
        const historyTimestepsSlider = document.getElementById('history-timesteps');
        const historyTimestepsValue = document.getElementById('history-timesteps-value');
        
        if (historyTimestepsSlider) {
            historyTimestepsSlider.addEventListener('input', (e) => {
                const timesteps = parseInt(e.target.value);
                historyTimestepsValue.textContent = timesteps;
                
                // Update robot's timestep setting
                if (this.app && this.app.robot) {
                    this.app.robot.setHistoryTimesteps(timesteps);
                    console.log(`History timesteps updated to: ${timesteps} (${timesteps * 2} inputs)`);
                    
                    // If Q-learning is initialized, reinitialize with new input size
                    if (this.app.qlearning) {
                        const currentEpisode = this.app.qlearning.episode;
                        console.log(`Reinitializing Q-learning for ${timesteps} timesteps...`);
                        
                        // Reinitialize Q-learning with new input size
                        this.app.initializeQLearning().then(() => {
                            console.log(`Q-learning reinitialized for ${timesteps} timesteps`);
                        }).catch(error => {
                            console.error('Failed to reinitialize Q-learning:', error);
                        });
                    }
                }
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
                case 's': // S - start/stop training
                    e.preventDefault();
                    if (this.app.isTraining) {
                        this.app.stopTraining();
                    } else {
                        this.app.startTraining();
                    }
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
        
        // Manual control arrow keys - separate handlers for keydown/keyup
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (!this.app.userControlEnabled) return; // Check userControlEnabled instead
            
            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.app.manualControl.leftPressed = true;
                    this.app.updateManualTorque();
                    // Add visual feedback to on-screen button
                    document.getElementById('control-left')?.classList.add('active');
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.app.manualControl.rightPressed = true;
                    this.app.updateManualTorque();
                    // Add visual feedback to on-screen button
                    document.getElementById('control-right')?.classList.add('active');
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    if (!this.app.isTraining) {
                        this.app.resetRobotPosition();
                    }
                    // Add visual feedback to on-screen button
                    document.getElementById('control-reset')?.classList.add('active');
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
                    // Remove visual feedback from on-screen button
                    document.getElementById('control-left')?.classList.remove('active');
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.app.manualControl.rightPressed = false;
                    this.app.updateManualTorque();
                    // Remove visual feedback from on-screen button
                    document.getElementById('control-right')?.classList.remove('active');
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    // Remove visual feedback from on-screen button
                    document.getElementById('control-reset')?.classList.remove('active');
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
        this.updateAllDisplays();
        
        // Update robot configuration for real-time parameter changes
        if (this.app && this.app.robot) {
            if (paramName === 'maxAngle') {
                this.app.updateMaxAngle(value);
            } else if (paramName === 'motorTorqueRange') {
                this.app.updateMotorTorqueRange(value);
            }
        }
        
        console.log(`Parameter ${paramName} set to ${value}`);
    }
    
    getParameter(paramName) {
        return this.parameters[paramName];
    }
    
    updateAllDisplays() {
        // Update all slider values and displays
        document.getElementById('training-speed').value = this.parameters.trainingSpeed;
        document.getElementById('training-speed-value').textContent = `${this.parameters.trainingSpeed.toFixed(1)}x`;
        
        // Core Learning Parameters
        document.getElementById('learning-rate').value = this.parameters.learningRate;
        document.getElementById('learning-rate-value').textContent = this.parameters.learningRate.toFixed(4);
        
        document.getElementById('gamma').value = this.parameters.gamma;
        document.getElementById('gamma-value').textContent = this.parameters.gamma.toFixed(3);
        
        // Exploration Parameters
        document.getElementById('epsilon').value = this.parameters.epsilon;
        document.getElementById('epsilon-value').textContent = this.parameters.epsilon.toFixed(2);
        
        document.getElementById('epsilon-min').value = this.parameters.epsilonMin;
        document.getElementById('epsilon-min-value').textContent = this.parameters.epsilonMin.toFixed(3);
        
        document.getElementById('epsilon-decay').value = this.parameters.epsilonDecay;
        document.getElementById('epsilon-decay-value').textContent = this.parameters.epsilonDecay.toString();
        
        // Training Parameters  
        document.getElementById('batch-size').value = this.parameters.batchSize;
        document.getElementById('batch-size-value').textContent = this.parameters.batchSize.toString();
        
        document.getElementById('target-update-freq').value = this.parameters.targetUpdateFreq;
        document.getElementById('target-update-freq-value').textContent = this.parameters.targetUpdateFreq.toString();
        
        document.getElementById('max-episodes').value = this.parameters.maxEpisodes;
        document.getElementById('max-episodes-value').textContent = this.parameters.maxEpisodes.toString();
        
        document.getElementById('max-steps-per-episode').value = this.parameters.maxStepsPerEpisode;
        document.getElementById('max-steps-per-episode-value').textContent = this.parameters.maxStepsPerEpisode.toString();
        
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
        
        if (document.getElementById('max-angle')) {
            document.getElementById('max-angle').value = this.parameters.maxAngle;
            document.getElementById('max-angle-value').textContent = `${(this.parameters.maxAngle * 180 / Math.PI).toFixed(1)}°`;
        }
        
        if (document.getElementById('motor-torque-range')) {
            document.getElementById('motor-torque-range').value = this.parameters.motorTorqueRange;
            document.getElementById('motor-torque-range-value').textContent = `±${this.parameters.motorTorqueRange.toFixed(1)} Nm`;
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
        this.parallelModeEnabled = false; // Disable parallel mode when setting speed preset
        this.app.setParallelMode(false);
        console.log(`Speed preset set to ${speed}x`);
    }
    
    enableParallelTraining() {
        this.parallelModeEnabled = true;
        document.getElementById('training-speed-value').textContent = 'Parallel Mode';
        this.app.setParallelMode(true);
        console.log('Parallel training mode enabled');
        
        // If currently training, switch to parallel immediately
        if (this.isTraining) {
            console.log('Switching current training to parallel mode');
        }
    }
    
    resetToDefaults() {
        this.parameters = {
            trainingSpeed: 1.0,
            hiddenNeurons: 8,
            
            // Core Learning Parameters
            learningRate: 0.0020,
            gamma: 0.99,
            
            // Exploration Parameters
            epsilon: 0.9,
            epsilonMin: 0.01,
            epsilonDecay: 2500,
            
            // Training Parameters
            batchSize: 128,
            targetUpdateFreq: 100,
            maxEpisodes: 1000,
            maxStepsPerEpisode: 1000,
            
            // Robot Physics Parameters
            robotMass: 1.0,
            robotHeight: 0.4,
            motorStrength: 5.0,
            wheelFriction: 0.3
        };
        this.updateAllDisplays();
        this.saveParameters();
        console.log('Parameters reset to defaults');
    }
    
    /**
     * Sync parameters from Q-learning back to UI (for two-way binding)
     * Call this when parameters change internally (e.g., epsilon decay)
     */
    syncParametersFromQLearning() {
        if (!this.app.qlearning) return;
        
        const hyperparams = this.app.qlearning.hyperparams;
        
        // Update internal parameters to match Q-learning
        this.parameters.learningRate = hyperparams.learningRate;
        this.parameters.gamma = hyperparams.gamma;
        this.parameters.epsilon = hyperparams.epsilon;
        this.parameters.epsilonMin = hyperparams.epsilonMin;
        this.parameters.epsilonDecay = hyperparams.epsilonDecay;
        this.parameters.batchSize = hyperparams.batchSize;
        this.parameters.targetUpdateFreq = hyperparams.targetUpdateFreq;
        this.parameters.maxEpisodes = hyperparams.maxEpisodes;
        this.parameters.maxStepsPerEpisode = hyperparams.maxStepsPerEpisode;
        
        // Update UI elements to reflect changes
        this.updateParameterDisplays([
            'epsilon', 'gamma', 'learningRate', 'epsilonMin', 'epsilonDecay',
            'batchSize', 'targetUpdateFreq', 'maxEpisodes', 'maxStepsPerEpisode'
        ]);
    }
    
    /**
     * Update specific parameter displays (more efficient than updateAllDisplays)
     */
    updateParameterDisplays(paramNames) {
        paramNames.forEach(paramName => {
            const elementId = this.getElementIdForParameter(paramName);
            const valueId = `${elementId}-value`;
            
            const element = document.getElementById(elementId);
            const valueElement = document.getElementById(valueId);
            
            if (element && valueElement) {
                element.value = this.parameters[paramName];
                valueElement.textContent = this.formatParameterValue(paramName, this.parameters[paramName]);
            }
        });
    }
    
    /**
     * Get HTML element ID for a parameter name
     */
    getElementIdForParameter(paramName) {
        const mapping = {
            'learningRate': 'learning-rate',
            'gamma': 'gamma',
            'epsilon': 'epsilon',
            'epsilonMin': 'epsilon-min',
            'epsilonDecay': 'epsilon-decay',
            'batchSize': 'batch-size',
            'targetUpdateFreq': 'target-update-freq',
            'maxEpisodes': 'max-episodes',
            'maxStepsPerEpisode': 'max-steps-per-episode'
        };
        return mapping[paramName] || paramName;
    }
    
    /**
     * Format parameter value for display
     */
    formatParameterValue(paramName, value) {
        switch (paramName) {
            case 'learningRate':
                return value.toFixed(4);
            case 'gamma':
                return value.toFixed(3);
            case 'epsilon':
                return value.toFixed(2);
            case 'epsilonMin':
                return value.toFixed(3);
            case 'epsilonDecay':
            case 'batchSize':
            case 'targetUpdateFreq':
            case 'maxEpisodes':
            case 'maxStepsPerEpisode':
                return value.toString();
            default:
                return value.toString();
        }
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
        
        // Parallel training
        this.systemCapabilities = null;
        this.parallelQLearning = null;
        
        // Performance tracking
        this.performanceTracker = new TrainingPerformanceTracker();
        this.smartRenderingManager = new SmartRenderingManager(this.performanceTracker, {
            targetFPS: 60,
            mode: 'interval' // Use setInterval for consistent 60 FPS
        });
        
        // Application state with debugging
        this._isTraining = false;
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
        this.parallelModeEnabled = false; // Track if parallel training is manually enabled
        
        // Free run mode settings
        this.resetAngleDegrees = 0.0; // Default reset angle in degrees
        this.maxOffsetRangeDegrees = 5.0; // Default max offset range for training
        
        // Debug control speed
        this.debugSpeed = 1.0;
        this.debugLastAction = 'None';
        this.debugCurrentReward = 0;
        
        // Demo modes
        this.demoMode = 'freerun'; // 'freerun', 'training'
        
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
        
        // Simulation loop management
        this.simulationInterval = null;
        
        // PD controller toggle state
        this.pdControllerEnabled = true; // Start enabled in free run mode
        
        // User control only mode
        this.userControlOnlyEnabled = false; // When true, disables all assistance
        
        // User control toggle state (separate from manual mode)
        this.userControlEnabled = true; // Always enabled in free run mode
        
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
            
            // Initialize parallel training system
            this.systemCapabilities = new SystemCapabilities();
            console.log(`System capabilities: ${this.systemCapabilities.coreCount} cores, using ${this.systemCapabilities.maxWorkers} workers`);
            
            // Update WebGPU status display
            this.updateWebGPUStatusDisplay();
            
            // Update CPU cores status display
            this.updateCPUCoresStatusDisplay();
            
            // Start simulation and rendering
            this.startSimulation();
            
            // Initialize UI state
            this.updatePDControllerUI();
            this.updateUserControlOnlyUI();
            this.updateFreeRunSpeedUI();
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

    // Add debugging getter and setter for isTraining to track when it changes
    get isTraining() {
        return this._isTraining;
    }

    set isTraining(value) {
        if (this._isTraining !== value) {
            console.log(`🔄 isTraining changed from ${this._isTraining} to ${value}`, new Error().stack);
        }
        this._isTraining = value;
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
            if (this.isTraining) {
                this.stopTraining();
            } else {
                this.startTraining();
            }
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
            this.testModel();
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
        
        // User Control Only toggle
        document.getElementById('toggle-user-control-only')?.addEventListener('click', () => {
            this.toggleUserControlOnly();
        });
        
        // Reward Function dropdown
        document.getElementById('reward-function')?.addEventListener('change', (e) => {
            this.setRewardFunction(e.target.value);
        });
        
        // On-screen controls for free run mode
        this.setupOnScreenControls();
        
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
        
        // CRITICAL FIX: Read current training speed from UI slider before starting
        const speedSlider = document.getElementById('training-speed');
        if (speedSlider) {
            const currentSpeed = parseFloat(speedSlider.value);
            this.setTrainingSpeed(currentSpeed);
            console.log(`Training will start at current slider speed: ${currentSpeed}x`);
        }
        
        this.isTraining = true;
        this.isPaused = false;
        this.demoMode = 'training';
        this.updateFreeRunSpeedUI();
        
        // Ensure robot has current offset range for training
        if (this.robot && this.maxOffsetRangeDegrees > 0) {
            this.robot.setTrainingOffsetRange(this.maxOffsetRangeDegrees);
            console.log(`Training offset range set to: ±${this.maxOffsetRangeDegrees.toFixed(1)}°`);
        }
        
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
        
        // Update start button to become stop button
        const startButton = document.getElementById('start-training');
        startButton.textContent = 'Stop Training';
        startButton.classList.remove('primary');
        startButton.classList.add('danger');
        startButton.disabled = false;
        
        document.getElementById('pause-training').disabled = false;
        
        // Update PD controller UI (disabled during training)
        this.updatePDControllerUI();
        
        console.log('Training started');
    }

    stopTraining() {
        if (!this.isTraining) return;
        
        // Stop training
        this.isTraining = false;
        this.isPaused = false;
        this.demoMode = 'freerun';
        
        // Reset training metrics while preserving the model
        this.episodeCount = 0;
        this.trainingStep = 0;
        this.bestScore = 0;
        this.currentReward = 0;
        
        // Reset Q-learning training state but keep the trained weights
        if (this.qlearning) {
            // Reset training counters and epsilon without resetting weights
            this.qlearning.episode = 0;
            this.qlearning.stepCount = 0;
            this.qlearning.globalStepCount = 0;
            this.qlearning.lastTargetUpdate = 0;
            
            // Reset epsilon to initial value for potential restart
            this.qlearning.hyperparams.epsilon = this.qlearning.initialEpsilon || 0.9;
            
            // Clear replay buffer
            this.qlearning.replayBuffer.clear();
        }
        
        // Reset performance tracking
        if (this.performanceTracker) {
            this.performanceTracker.reset();
        }
        
        // Clear performance charts if reset method exists
        if (this.performanceCharts && typeof this.performanceCharts.reset === 'function') {
            this.performanceCharts.reset();
        }
        
        // Reset previous training state
        this.previousState = null;
        this.previousAction = undefined;
        this.previousReward = 0;
        this.previousDone = false;
        this.lastTrainingLoss = 0;
        
        // Update UI buttons
        const startButton = document.getElementById('start-training');
        const pauseButton = document.getElementById('pause-training');
        
        startButton.textContent = 'Start Training';
        startButton.classList.remove('danger');
        startButton.classList.add('primary');
        
        pauseButton.textContent = 'Pause Training';
        pauseButton.disabled = true;
        
        // Update training status display
        this.updateTrainingModelDisplay();
        this.updatePDControllerUI();
        this.updateUserControlOnlyUI();
        this.updateOnScreenControlsVisibility();
        
        // Reset robot to clean state
        this.robot.reset();
        
        console.log('Training stopped and reset');
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
        // Stop all training and parallel processing
        this.isTraining = false;
        this.isPaused = false;
        this.parallelModeEnabled = false;
        this.demoMode = 'freerun'; // Reset to free run mode
        this.pdControllerEnabled = true; // Reset to PD controller enabled
        this.userControlOnlyEnabled = false; // Reset User Control Only
        
        // Reset all training counters and metrics
        this.trainingStep = 0;
        this.episodeCount = 0;
        this.bestScore = 0;
        this.currentReward = 0;
        this.episodeEnded = false;
        
        // Reset Q-learning training state
        this.previousState = null;
        this.previousAction = undefined;
        this.previousReward = 0;
        this.previousDone = false;
        this.lastTrainingLoss = 0;
        
        // Reset Q-learning if it exists
        if (this.qlearning) {
            this.qlearning.reset();
            console.log('Q-learning network reset');
        }
        
        // Reset parallel training if it exists
        if (this.parallelQLearning) {
            this.parallelQLearning.cleanup();
            this.parallelQLearning = null;
            console.log('Parallel training system reset');
        }
        
        // Reset performance charts
        if (this.performanceCharts) {
            this.performanceCharts.reset();
            console.log('Performance charts reset');
        }
        
        // Reset robot state
        if (this.robot) {
            let resetAngle = 0; // Default: perfectly balanced
            
            // In free run mode, use exact angle from slider
            if (this.demoMode === 'freerun' && Math.abs(this.resetAngleDegrees) > 0) {
                const angleDegrees = this.resetAngleDegrees || 0.0;
                resetAngle = angleDegrees * Math.PI / 180; // Use exact angle from slider
                console.log(`Reset with angle: ${(resetAngle * 180 / Math.PI).toFixed(2)}° (set to ${angleDegrees.toFixed(1)}°)`);
            }
            
            this.robot.reset({
                angle: resetAngle,
                angularVelocity: 0,
                position: 0,
                velocity: 0
            });
        }
        
        // Reset UI states
        document.getElementById('start-training').disabled = false;
        document.getElementById('pause-training').disabled = true;
        document.getElementById('pause-training').textContent = 'Pause Training';
        
        // Reset parallel training UI
        const parallelBtn = document.getElementById('parallel-training');
        if (parallelBtn) {
            parallelBtn.textContent = 'Parallel';
            parallelBtn.classList.remove('active');
        }
        
        // Update various UI components
        this.updateFreeRunSpeedUI();
        this.updatePDControllerUI();
        this.updateUserControlOnlyUI();
        this.updateOnScreenControlsVisibility();
        this.updateUI();
        
        console.log('Environment completely reset - all training state cleared');
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
        // Clean up any existing simulation loop
        this.stopSimulation();
        
        // Start the renderer
        this.renderer.start();
        
        // Choose simulation mode based on SmartRenderingManager configuration
        if (this.smartRenderingManager.renderingMode === 'interval') {
            this.startHighPerformanceSimulation();
        } else {
            this.startDisplaySyncedSimulation();
        }
    }
    
    stopSimulation() {
        // Clean up interval-based simulation
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
            this.simulationInterval = null;
            console.log('High-performance simulation stopped');
        }
        
        // Note: RAF-based simulation will naturally stop when requestAnimationFrame stops being called
    }
    
    startDisplaySyncedSimulation() {
        // Traditional RAF-based simulation (60 FPS max, display-synced)
        const simulate = (timestamp) => {
            if (!this.isInitialized) return;
            
            this.runSimulationFrame(timestamp || performance.now());
            requestAnimationFrame(simulate);
        };
        
        requestAnimationFrame(simulate);
        console.log('Simulation started (display-synced, ~60 FPS)');
    }
    
    startHighPerformanceSimulation() {
        // Adaptive high-performance simulation based on training speed
        const baseTargetInterval = 1000 / this.smartRenderingManager.actualTargetFPS; // 8.33ms for 120 FPS
        
        // Adjust interval based on training speed - REMOVE throttling for max performance
        let adaptedInterval;
        if (this.trainingSpeed <= 20) {
            adaptedInterval = baseTargetInterval; // Full 120 FPS for low speeds
        } else if (this.trainingSpeed <= 100) {
            adaptedInterval = baseTargetInterval; // Keep 120 FPS for medium speeds too
        } else {
            // For very high speeds, run as fast as possible with minimal interval
            adaptedInterval = 1; // 1ms = ~1000 FPS maximum theoretical rate
        }
        
        let lastTimestamp = performance.now();
        let simulationCallCount = 0;
        
        const simulate = async () => {
            if (!this.isInitialized) return;
            
            const currentTimestamp = performance.now();
            const actualInterval = currentTimestamp - lastTimestamp;
            
            // Log performance every 60 calls to detect interval backup
            simulationCallCount++;
            if (simulationCallCount % 60 === 0) {
                const intervalBackup = actualInterval > (adaptedInterval * 1.5) ? '🚨 BACKUP!' : '✅';
                console.log(`⏱️  TIMING: ${this.trainingSpeed}x speed | Target: ${adaptedInterval.toFixed(1)}ms | Actual: ${actualInterval.toFixed(1)}ms ${intervalBackup}`);
            }
            
            await this.runSimulationFrame(currentTimestamp);
            lastTimestamp = currentTimestamp;
        };
        
        // Use adapted interval to prevent backup at high speeds
        this.simulationInterval = setInterval(simulate, adaptedInterval);
        console.log(`Simulation started (adaptive, ${(1000/adaptedInterval).toFixed(0)} FPS for ${this.trainingSpeed}x speed)`);
    }
    
    async runSimulationFrame(timestamp) {
        // Performance monitoring
        if (this.lastFrameTime > 0) {
            const frameTime = timestamp - this.lastFrameTime;
            this.frameTimeHistory.push(frameTime);
            
            // Keep only recent history
            if (this.frameTimeHistory.length > this.performanceCheckInterval) {
                this.frameTimeHistory.shift();
            }
            
            // Debug performance degradation: log when frame time gets bad
            if (frameTime > 50 && Math.random() < 0.1) { // 10% chance to log when frame time > 50ms
                console.warn(`🐌 PERFORMANCE: Frame time ${frameTime.toFixed(1)}ms, History length: ${this.frameTimeHistory.length}, Physics steps: ${this.targetPhysicsStepsPerFrame}`);
            }
            
            // Auto-adjust training speed if performance is extremely poor (disabled for manual speed control)
            if (this.frameTimeHistory.length === this.performanceCheckInterval) {
                const avgFrameTime = this.frameTimeHistory.reduce((a, b) => a + b) / this.frameTimeHistory.length;
                const targetFrameTime = 1000 / this.smartRenderingManager.actualTargetFPS;
                
                // Only auto-adjust if frame time is catastrophically poor (>200ms = <5 FPS)
                // and we're running an extreme number of physics steps, and rendering is active
                const isRenderingActive = this.smartRenderingManager.shouldRenderFrame(this.trainingSpeed);
                if (avgFrameTime > 200 && this.targetPhysicsStepsPerFrame > 500 && isRenderingActive) {
                    this.setTrainingSpeed(Math.max(50.0, this.trainingSpeed * 0.8));
                    console.warn(`Catastrophic performance (${avgFrameTime.toFixed(1)}ms/frame), reducing training speed to ${this.trainingSpeed.toFixed(1)}x`);
                }
            }
        }
        this.lastFrameTime = timestamp;
        
        // Update physics at appropriate intervals based on mode and speed
        let updateInterval;
        if (this.demoMode === 'freerun') {
            // Free run mode: controlled by debugSpeed (Free Run Speed slider)
            // debugSpeed ranges from 0.1x to 2x, so adjust interval accordingly
            // Base interval is 16.67ms (60fps), adjust inversely with speed
            updateInterval = Math.max(1, 16.67 / this.debugSpeed);
        } else if (this.demoMode === 'training' && (this.parallelModeEnabled || this.trainingSpeed > 100)) {
            // For parallel mode or high-speed training, run physics as fast as possible
            updateInterval = 0; // No throttling - run every frame
        } else {
            // Normal throttling for low/medium speeds in training mode only
            updateInterval = this.physicsUpdateInterval;
        }
        
        if (timestamp - this.lastPhysicsUpdate >= updateInterval) {
            await this.updatePhysics();
            
            // Smart rendering - skip rendering during high-speed training or parallel mode
            const effectiveSpeed = this.parallelModeEnabled ? 1000 : this.trainingSpeed;
            const shouldRender = this.smartRenderingManager.shouldRenderFrame(effectiveSpeed);
            if (shouldRender) {
                this.updateRenderer();
                this.updateUI();
            }
            
            this.lastPhysicsUpdate = timestamp;
        }
    }

    async updatePhysics() {
        if (!this.robot) return;
        
        const physicsStartTime = performance.now();
        
        // Run multiple physics steps per frame for speed control
        let maxStepsPerFrame;
        if (this.demoMode === 'freerun') {
            // Free run mode: controlled by debugSpeed (Free Run Speed slider)
            // debugSpeed ranges from 0.1x to 2x
            maxStepsPerFrame = Math.max(1, Math.round(this.debugSpeed * 3)); // 1-6 steps based on speed
        } else if (this.parallelModeEnabled) {
            // Parallel mode: main thread runs at maximum speed regardless of slider
            maxStepsPerFrame = 1000; // Max speed for parallel training
        } else if (this.trainingSpeed <= 100) {
            // For speeds ≤100x, ensure minimum steps for smooth motion (continuous rendering active)
            maxStepsPerFrame = Math.max(3, this.targetPhysicsStepsPerFrame);
        } else if (this.trainingSpeed <= 200) {
            // For moderate high speeds, limit to prevent frame drops during sparse rendering
            maxStepsPerFrame = Math.min(this.targetPhysicsStepsPerFrame, 500);
        } else {
            // For extreme speeds, allow higher limits since rendering is minimal
            maxStepsPerFrame = Math.min(this.targetPhysicsStepsPerFrame, 1000);
        }
        
        // Use optimized steps for all modes except when explicitly paused
        const stepsToRun = this.isPaused ? 1 : maxStepsPerFrame;
        
        // Profiling variables
        let robotStepTime = 0;
        let trainingTime = 0;
        let yieldTime = 0;
        let otherTime = 0;
        
        for (let step = 0; step < stepsToRun; step++) {
            const stepStartTime = performance.now();
            
            // Yield control to event loop every 50 steps to prevent blocking
            if (step > 0 && step % 50 === 0) {
                const yieldStart = performance.now();
                await new Promise(resolve => setTimeout(resolve, 0));
                yieldTime += performance.now() - yieldStart;
            }
            
            let motorTorque = 0;
            let actionIndex = 0;
            
            // Get current normalized state for Q-learning (includes multi-timestep logic)
            const normalizedState = this.robot.getNormalizedInputs();
            
            // Get motor torque based on current mode
            switch (this.demoMode) {
                case 'freerun':
                    // Free run mode: user input takes precedence, then assistance modes
                    if (this.manualControl.leftPressed || this.manualControl.rightPressed) {
                        // User input always takes precedence when active
                        motorTorque = this.getManualTorque();
                    } else if (this.userControlOnlyEnabled) {
                        // User Control Only mode - no assistance
                        motorTorque = 0;
                    } else if (this.pdControllerEnabled) {
                        // PD controller assistance
                        motorTorque = this.getPhysicsDemoTorque();
                    } else if (this.qlearning) {
                        // Model control available if model loaded
                        motorTorque = this.getEvaluationTorque();
                    } else {
                        motorTorque = 0; // No control active
                    }
                    break;
                case 'training':
                    if (this.isTraining && !this.isPaused) {
                        const trainingResult = this.getTrainingTorqueWithAction();
                        motorTorque = trainingResult.torque;
                        actionIndex = trainingResult.actionIndex;
                    } else {
                        // When training paused/stopped: user arrows take precedence, then assistance modes
                        if (this.userControlEnabled && (this.manualControl.leftPressed || this.manualControl.rightPressed)) {
                            motorTorque = this.getManualTorque();
                        } else if (this.userControlOnlyEnabled) {
                            motorTorque = 0; // User Control Only mode
                        } else if (this.pdControllerEnabled) {
                            motorTorque = this.getPhysicsDemoTorque();
                        } else if (this.qlearning) {
                            motorTorque = this.getEvaluationTorque();
                        } else {
                            motorTorque = 0;
                        }
                    }
                    break;
            }
            
            // Step physics simulation
            const robotStepStart = performance.now();
            const result = this.robot.step(motorTorque);
            robotStepTime += performance.now() - robotStepStart;
            this.currentReward = result.reward;
            
            // Check for physics instability that could cause silent training failures
            if (!this.robot.isStable() || !isFinite(result.reward)) {
                console.error('Physics simulation became unstable:', {
                    robotState: this.robot.getState(),
                    reward: result.reward,
                    motorTorque: motorTorque,
                    episode: this.episodeCount,
                    step: this.trainingStep
                });
                
                if (this.demoMode === 'training' && this.isTraining) {
                    // Force episode end due to instability
                    console.warn('Ending episode due to physics instability');
                    this.episodeEnded = true;
                    this.handleEpisodeEnd({
                        ...result,
                        done: true,
                        reward: -10.0 // Penalty for instability
                    });
                    return;
                }
            }
            
            // Update debug reward display during manual control modes and evaluation
            if (this.demoMode === 'freerun') {
                this.debugCurrentReward = result.reward;
            }
            
            // Q-learning training integration
            if (this.demoMode === 'training' && this.isTraining && !this.isPaused && this.qlearning) {
                // Get next state after physics step (includes multi-timestep logic)
                const nextState = this.robot.getNormalizedInputs();
                
                // Train on previous experience if we have one
                // At high speeds, reduce training frequency to prevent bottlenecks
                const shouldTrain = this._shouldTrainThisStep(step, stepsToRun);
                
                if (this.previousState && this.previousAction !== undefined && shouldTrain) {
                    const trainingStart = performance.now();
                    const loss = this.qlearning.train(
                        this.previousState,      // Previous state
                        this.previousAction,     // Previous action  
                        this.previousReward,     // ✅ FIXED: Reward from previous action
                        nextState,              // Current state (next state)
                        this.previousDone       // ✅ FIXED: Done from previous step
                    );
                    trainingTime += performance.now() - trainingStart;
                    
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
                    // Episode ended - always train on final experience (critical for episode completion)
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
            if (result.done && (this.demoMode === 'training') && !this.isPaused && !this.episodeEnded) {
                this.episodeEnded = true; // Set flag to prevent multiple calls
                this.handleEpisodeEnd(result);
                return; // Exit physics update completely to avoid multiple episode ends
            }
            
            // CRITICAL: Break out of physics loop if episode ended to prevent infinite loops
            if (this.episodeEnded) {
                break; // Exit the multi-step physics loop immediately
            }
            
            // Update training step counter
            if (this.isTraining && !this.isPaused) {
                this.trainingStep++;
                
                // Performance tracking - record step
                this.performanceTracker.recordStep();
                
                // Check for episode termination based on step count (uses dynamic maxStepsPerEpisode)
                const maxSteps = this.uiControls?.getParameter('maxStepsPerEpisode') || 1000;
                if (this.trainingStep >= maxSteps && !this.episodeEnded && (this.demoMode === 'training')) {
                    console.log(`Episode terminated at step ${this.trainingStep} (reached max steps)`);
                    this.episodeEnded = true;
                    // Create a synthetic "done" result to trigger episode end
                    const syntheticResult = {
                        state: this.robot.getState(),
                        reward: 1.0, // Positive reward for reaching max steps
                        done: true
                    };
                    this.handleEpisodeEnd(syntheticResult);
                    break; // Exit the physics loop immediately
                }
                
                // Check for robot moving off canvas in all modes
                const robotState = this.robot.getState();
                const bounds = this.renderer ? this.renderer.getPhysicsBounds() : null;
                
                if (bounds && (robotState.position < bounds.minX || robotState.position > bounds.maxX)) {
                    // Handle based on mode
                    if ((this.demoMode === 'training') && !this.episodeEnded) {
                        console.log(`Episode terminated at step ${this.trainingStep} (robot moved off canvas: position=${robotState.position.toFixed(2)}m, bounds=[${bounds.minX.toFixed(2)}, ${bounds.maxX.toFixed(2)}])`);
                        this.episodeEnded = true;
                        // Create a synthetic "done" result with penalty for going off canvas
                        const syntheticResult = {
                            state: this.robot.getState(),
                            reward: -1.0, // Penalty for moving off canvas
                            done: true
                        };
                        this.handleEpisodeEnd(syntheticResult);
                        break; // Exit the physics loop immediately
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
        
        // Profiling output every 100 physics steps for performance analysis
        const totalPhysicsTime = performance.now() - physicsStartTime;
        if (this.trainingSpeed >= 40 && Math.random() < 0.01) { // 1% chance to log at high speeds
            console.log(`🔍 PROFILING ${this.trainingSpeed}x (${stepsToRun} steps): Total=${totalPhysicsTime.toFixed(1)}ms | Robot=${robotStepTime.toFixed(1)}ms | Training=${trainingTime.toFixed(1)}ms | Yield=${yieldTime.toFixed(1)}ms`);
            console.log(`⏱️  Per step: Robot=${(robotStepTime/stepsToRun).toFixed(2)}ms | Training=${(trainingTime/stepsToRun).toFixed(2)}ms`);
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
        
        // Update Q-learning metrics display and sync UI controls
        if (this.qlearning) {
            document.getElementById('epsilon-display').textContent = `Epsilon: ${this.qlearning.hyperparams.epsilon.toFixed(3)}`;
            document.getElementById('consecutive-episodes').textContent = `Consecutive Max Episodes: ${this.qlearning.consecutiveMaxEpisodes}/20`;
            
            // Sync UI controls with current Q-learning parameters (especially epsilon after decay)
            if (this.uiControls) {
                this.uiControls.syncParametersFromQLearning();
            }
        }
        
        // Update training loss display
        document.getElementById('training-loss').textContent = `Training Loss: ${this.lastTrainingLoss.toFixed(4)}`;
        
        // Update Q-value display
        document.getElementById('qvalue-estimate').textContent = `Q-Value: ${this.lastQValue.toFixed(3)}`;
        
        // Update history timesteps debug display
        this.updateHistoryDebugDisplay();
        
        // Update debug display during manual control and evaluation
        if (this.demoMode === 'freerun') {
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
     * Update the history debug display showing current timestep values
     */
    updateHistoryDebugDisplay() {
        const debugContent = document.getElementById('history-debug-content');
        if (!debugContent || !this.robot) return;
        
        const robotState = this.robot.getState();
        const timesteps = this.robot.historyTimesteps || 1;
        
        // Get current normalized inputs
        const networkInput = this.robot.getNormalizedInputs();
        
        // Format current state
        let displayLines = [];
        
        if (this.robot.rewardType === 'offset-adaptive') {
            const measuredAngle = this.robot.getMeasuredAngle();
            const trueAngle = robotState.angle;
            const offset = this.robot.angleOffset;
            
            displayLines.push(`True: angle=${trueAngle.toFixed(3)}, angVel=${robotState.angularVelocity.toFixed(3)}`);
            displayLines.push(`Measured: angle=${measuredAngle.toFixed(3)}, offset=${offset.toFixed(3)}`);
        } else {
            displayLines.push(`Current: angle=${robotState.angle.toFixed(3)}, angVel=${robotState.angularVelocity.toFixed(3)}`);
        }
        
        // Format network input (showing all timesteps vertically)
        displayLines.push(`Network Input [${timesteps}x]:`);
        for (let i = 0; i < networkInput.length; i += 2) {
            const angle = networkInput[i];
            const angVel = networkInput[i + 1];
            const timestepIndex = (i / 2) + 1;
            displayLines.push(`  T${timestepIndex}: [${angle.toFixed(2)}, ${angVel.toFixed(2)}]`);
        }
        
        // Show timestep info
        displayLines.push(`Timesteps: ${timesteps} (${networkInput.length} inputs total)`);
        
        // Show state history stats if available
        if (this.robot.stateHistory && timesteps > 1) {
            const stats = this.robot.stateHistory.getStats();
            displayLines.push(`Avg Angle: ${stats.averageAngle.toFixed(3)}, StdDev: ${stats.angleStdDev.toFixed(3)}`);
        }
        
        debugContent.innerHTML = displayLines.map(line => `<div>${line}</div>`).join('');
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
    
    updateCPUCoresStatusDisplay() {
        const coresInfoElement = document.getElementById('cores-info');
        const workersInfoElement = document.getElementById('workers-info');
        const parallelSpeedupElement = document.getElementById('parallel-speedup');
        
        if (!this.systemCapabilities) {
            coresInfoElement.textContent = 'CPU Cores: Not detected';
            workersInfoElement.textContent = 'Parallel training unavailable';
            parallelSpeedupElement.textContent = '';
            return;
        }
        
        // Update core count
        const coreCount = this.systemCapabilities.coreCount;
        const targetCores = this.systemCapabilities.targetCores;
        const maxWorkers = this.systemCapabilities.maxWorkers;
        
        coresInfoElement.textContent = `CPU Cores: ${coreCount} detected`;
        
        // Update worker information
        if (maxWorkers > 1) {
            workersInfoElement.innerHTML = `
                <div>Using ${targetCores} cores (50% of ${coreCount})</div>
                <div>Worker threads: ${maxWorkers}</div>
            `;
        } else {
            workersInfoElement.innerHTML = `
                <div style="color: #ff6666;">Single core detected</div>
                <div>Parallel training disabled</div>
            `;
        }
        
        // Update speedup estimate
        if (this.parallelQLearning) {
            const perfStats = this.parallelQLearning.getPerformanceStats();
            if (perfStats.speedupFactor > 1) {
                parallelSpeedupElement.innerHTML = `
                    <div>Training speedup: ${perfStats.speedupFactor.toFixed(1)}x</div>
                    <div>Parallel episodes: ${perfStats.totalParallelEpisodes}</div>
                `;
            } else {
                parallelSpeedupElement.textContent = 'Speedup measurement in progress...';
            }
        } else if (maxWorkers > 1) {
            parallelSpeedupElement.textContent = `Estimated speedup: ${Math.min(maxWorkers, 4).toFixed(1)}x`;
        } else {
            parallelSpeedupElement.textContent = 'No parallel acceleration';
        }
        
        console.log('CPU cores status display updated');
    }
    
    /**
     * Update training performance metrics display
     */
    updatePerformanceMetricsDisplay() {
        const episodesPerMinuteElement = document.getElementById('episodes-per-minute');
        const stepsPerSecondElement = document.getElementById('steps-per-second');
        const renderingModeElement = document.getElementById('rendering-mode');
        const trainingEfficiencyElement = document.getElementById('training-efficiency');
        
        if (!this.performanceTracker) {
            return;
        }
        
        const displayStats = this.performanceTracker.getDisplayStats();
        
        // Update performance metrics
        if (episodesPerMinuteElement) {
            episodesPerMinuteElement.textContent = `Episodes/min: ${displayStats.episodesPerMinute}`;
        }
        
        if (stepsPerSecondElement) {
            stepsPerSecondElement.textContent = `Steps/sec: ${displayStats.stepsPerSecond}`;
        }
        
        if (renderingModeElement) {
            const mode = displayStats.renderingMode;
            let color = '#ffaa00'; // Default orange
            
            if (mode.includes('Full')) {
                color = '#00ff88'; // Green for full rendering
            } else if (mode.includes('Sparse')) {
                color = '#ffaa00'; // Orange for sparse rendering
            } else if (mode.includes('Minimal')) {
                color = '#ff6666'; // Red for minimal rendering
            }
            
            renderingModeElement.innerHTML = `<span style="color: ${color}">Rendering: ${mode}</span>`;
        }
        
        if (trainingEfficiencyElement) {
            const efficiency = parseFloat(displayStats.trainingEfficiency.replace('%', ''));
            let color = '#00d4ff'; // Default blue
            
            if (efficiency >= 80) {
                color = '#00ff88'; // Green for high efficiency
            } else if (efficiency >= 50) {
                color = '#ffaa00'; // Orange for medium efficiency
            } else if (efficiency > 0) {
                color = '#ff6666'; // Red for low efficiency
            }
            
            trainingEfficiencyElement.innerHTML = `<span style="color: ${color}">Training efficiency: ${displayStats.trainingEfficiency}</span>`;
        }
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
        // Scale timestep to account for multiple physics steps per frame
        // Default timestep of 0.02s is for 1 step per 20ms (50 Hz)
        // With minimum 3 steps per frame at 60 FPS, we need 0.02/3 = 0.0067s per step
        const baseTimestep = 0.02; // 20ms timestep for real-time physics
        const minStepsPerFrame = 3; // Our minimum steps per frame for smooth motion
        const adjustedTimestep = baseTimestep / minStepsPerFrame;
        
        this.robot = createDefaultRobot({
            mass: uiControls.getParameter('robotMass'),
            centerOfMassHeight: uiControls.getParameter('robotHeight'),
            motorStrength: uiControls.getParameter('motorStrength'),
            friction: 0.1,
            damping: 0.05,
            timestep: adjustedTimestep,
            maxAngle: uiControls.getParameter('maxAngle'),
            motorTorqueRange: uiControls.getParameter('motorTorqueRange'),
            rewardType: 'complex' // Default to proportional reward
        });
        
        // Set training offset range if available
        if (this.maxOffsetRangeDegrees > 0) {
            this.robot.setTrainingOffsetRange(this.maxOffsetRangeDegrees);
        }
        
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
     * Initialize Q-learning for training with variable timestep support
     */
    async initializeQLearning(overrideParams = {}) {
        // Get current timestep settings from robot
        const timesteps = this.robot ? this.robot.historyTimesteps : 1;
        const inputSize = timesteps * 2; // 2 values per timestep (angle, angular velocity)
        
        // Use parameters from UI controls if available
        const params = this.uiControls ? {
            learningRate: this.uiControls.getParameter('learningRate'),
            epsilon: this.uiControls.getParameter('epsilon'),
            epsilonDecay: 0.995,
            maxEpisodes: 1000,
            maxStepsPerEpisode: 1000,
            hiddenSize: this.uiControls.getParameter('hiddenNeurons'),
            ...overrideParams
        } : {
            learningRate: 0.0020,  // Optimized default for better convergence
            epsilon: 0.3,         // Moderate exploration
            epsilonDecay: 0.995,  // Slower epsilon decay for stability
            gamma: 0.95,          // Standard discount factor
            batchSize: 8,         // Larger batch for stability
            targetUpdateFreq: 100, // Standard update frequency
            maxEpisodes: 1000,
            maxStepsPerEpisode: 1000,
            hiddenSize: 8,
            ...overrideParams
        };
        
        // Create Q-learning with WebGPU backend
        this.qlearning = createDefaultQLearning(params);
        
        // Initialize with WebGPU backend if available
        if (this.webgpuBackend) {
            console.log('Initializing Q-learning with WebGPU backend support...');
            // Note: Q-learning will use the backend internally
            // For now, the integration is prepared for when Q-learning supports custom backends
        }
        
        // Initialize with calculated input size for current timestep setting
        await this.qlearning.initialize(inputSize);
        console.log(`Q-learning initialized for ${timesteps} timesteps (${inputSize} inputs) with parameters:`, params);
        
        // Initialize parallel training wrapper
        if (this.systemCapabilities && this.systemCapabilities.maxWorkers > 1) {
            this.parallelQLearning = new ParallelQLearning(this.qlearning, this.robot);
            await this.parallelQLearning.initialize();
            console.log(`Parallel training initialized with ${this.systemCapabilities.maxWorkers} workers`);
        } else {
            console.log('Parallel training disabled (insufficient cores)');
        }
        
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
        
        // Use measured angle (includes sensor offset) instead of true angle
        const measuredAngle = this.robot.getMeasuredAngle();
        const torque = (kp * measuredAngle + kd * state.angularVelocity);
        
        // Debug offset effect (only log occasionally to avoid spam)
        if (Math.abs(this.robot.angleOffset) > 0.01 && Math.random() < 0.01) {
            console.log(`PD Controller: true=${(state.angle * 180 / Math.PI).toFixed(1)}°, measured=${(measuredAngle * 180 / Math.PI).toFixed(1)}°, offset=${(this.robot.angleOffset * 180 / Math.PI).toFixed(1)}°`);
        }
        
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
        
        // Get normalized state directly from robot (includes multi-timestep logic)
        const normalizedState = this.robot.getNormalizedInputs();
        
        // Debug offset effect on neural network (log occasionally to avoid spam)
        if (Math.abs(this.robot.angleOffset) > 0.01 && Math.random() < 0.01) {
            const trueAngle = this.robot.getState().angle;
            const measuredAngle = this.robot.getMeasuredAngle();
            console.log(`Neural Network Input: true=${(trueAngle * 180 / Math.PI).toFixed(1)}°, measured=${(measuredAngle * 180 / Math.PI).toFixed(1)}°, offset=${(this.robot.angleOffset * 180 / Math.PI).toFixed(1)}°, normalized=[${normalizedState[0].toFixed(3)}, ${normalizedState[1].toFixed(3)}]`);
        }
        
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
        
        // Get normalized state directly from robot (includes multi-timestep logic)
        const normalizedState = this.robot.getNormalizedInputs();
        
        // Select best action (no exploration)
        const actionIndex = this.qlearning.selectAction(normalizedState, false);
        const actions = [-1.0, 0.0, 1.0]; // Standardized action values
        
        return actions[actionIndex];
    }
    
    /**
     * Decide whether to run single episode or parallel batch
     */
    async startNextEpisodeOrBatch() {
        try {
            // Check if parallel training is available and beneficial
            const shouldUseParallel = this.shouldUseParallelTraining();
            
            if (shouldUseParallel) {
                // Pure parallel training - let workers handle everything for maximum speed
                await this.runParallelEpisodeBatch();
            } else {
                this.startNewEpisode();
            }
        } catch (error) {
            console.error('Critical training error in startNextEpisodeOrBatch:', error);
            
            // Reset training state to prevent getting stuck
            this.isTraining = false;
            this.isPaused = false;
            this.demoMode = 'freerun';
            
            // Update UI to reflect stopped state
            const startButton = document.getElementById('start-training');
            const pauseButton = document.getElementById('pause-training');
            
            if (startButton) startButton.disabled = false;
            if (pauseButton) {
                pauseButton.disabled = true;
                pauseButton.textContent = 'Pause Training';
            }
            
            // Update training display
            this.updateTrainingModelDisplay();
            this.updateOnScreenControlsVisibility();
            
            // Show error to user
            alert('Training stopped due to an error. Please check the console for details and try starting training again.');
        }
    }
    
    /**
     * Determine if parallel training should be used
     */
    shouldUseParallelTraining() {
        // Use parallel training if:
        // 1. Parallel training is available
        // 2. Parallel mode is manually enabled via the Parallel button
        
        if (!this.parallelQLearning || !this.parallelQLearning.parallelManager.parallelEnabled) {
            return false;
        }
        
        // Use parallel training only when explicitly enabled by user
        return this.parallelModeEnabled;
    }
    
    /**
     * Run a batch of episodes in parallel
     */
    async runParallelEpisodeBatch() {
        console.log('🚀 Running parallel episode batch...');
        
        try {
            // Determine batch size based on worker configuration
            const batchSize = this.parallelQLearning.parallelManager.config.batchSize;
            
            // Use full batch size for maximum parallel performance
            const effectiveBatchSize = batchSize;
            
            // Performance tracking - start batch
            const batchStartTime = Date.now();
            this.performanceTracker.startEpisode();
            
            // Run parallel episodes
            const results = await this.parallelQLearning.parallelManager.runEpisodes(effectiveBatchSize);
            
            // Process results and update charts for each individual episode
            let totalReward = 0;
            let totalSteps = 0;
            let bestEpisodeReward = -Infinity;
            
            console.log(`📊 Processing ${results.length} individual episode results...`);
            
            for (let i = 0; i < results.length; i++) {
                const result = results[i];
                if (result) {
                    const episodeReward = result.totalReward || 0;
                    const episodeSteps = result.stepCount || 0;
                    
                    totalReward += episodeReward;
                    totalSteps += episodeSteps;
                    
                    // Track best episode in this batch
                    if (episodeReward > bestEpisodeReward) {
                        bestEpisodeReward = episodeReward;
                    }
                    
                    // Update episode count for each completed episode
                    this.episodeCount++;
                    
                    // Update best score if this episode was better
                    if (episodeReward > this.bestScore) {
                        this.bestScore = episodeReward;
                    }
                    
                    // Update performance charts for EACH individual episode
                    if (this.performanceCharts && this.qlearning) {
                        // Create balanced state matching current timestep configuration
                        const timesteps = this.robot ? this.robot.historyTimesteps : 1;
                        const inputSize = timesteps * 2;
                        const balancedState = new Float32Array(inputSize).fill(0.0);
                        const qValues = this.qlearning.getAllQValues(balancedState);
                        const avgQValue = Array.from(qValues).reduce((a, b) => a + b, 0) / qValues.length;
                        
                        this.performanceCharts.updateMetrics({
                            episode: this.episodeCount,
                            reward: episodeReward, // Individual episode reward
                            loss: this.lastTrainingLoss || 0,
                            qValue: avgQValue,
                            epsilon: this.qlearning.hyperparams.epsilon
                        });
                    }
                    
                    // Add worker experiences to replay buffer for training
                    if (result.experiences && result.experiences.length > 0) {
                        for (const experience of result.experiences) {
                            this.qlearning.replayBuffer.addExperience(
                                new Float32Array(experience.state),
                                experience.action,
                                experience.reward,
                                new Float32Array(experience.nextState),
                                experience.done
                            );
                        }
                    }
                    
                    console.log(`  Episode ${this.episodeCount}: reward=${episodeReward.toFixed(2)}, steps=${episodeSteps}, experiences=${result.experiences?.length || 0}`);
                    
                    // Small delay between individual episode updates to make charts visible
                    await new Promise(resolve => setTimeout(resolve, 2));
                }
            }
            
            // Train the network on collected experiences
            let totalTrainingLoss = 0;
            const totalExperiencesCollected = results.reduce((sum, result) => sum + (result?.experiences?.length || 0), 0);
            
            if (this.qlearning.replayBuffer.size() >= this.qlearning.hyperparams.batchSize) {
                // Scale training steps based on how much new data we collected
                // More experiences = more training to utilize the data effectively
                const baseTrainingSteps = effectiveBatchSize;
                const experienceMultiplier = Math.max(1, Math.ceil(totalExperiencesCollected / 32)); // Scale up for lots of data
                const trainingSteps = Math.min(baseTrainingSteps * experienceMultiplier, totalExperiencesCollected);
                
                for (let i = 0; i < trainingSteps; i++) {
                    const loss = this.qlearning._trainBatch();
                    totalTrainingLoss += loss || 0;
                    
                    // Standard yield frequency for performance
                    const yieldFrequency = 2;
                    if (i % yieldFrequency === 0) {
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                }
                this.lastTrainingLoss = trainingSteps > 0 ? totalTrainingLoss / trainingSteps : 0;
            }
            
            // Performance tracking - end batch
            const batchEndTime = Date.now();
            const batchTrainingTime = batchEndTime - batchStartTime;
            this.performanceTracker.endEpisode(totalSteps, batchTrainingTime);
            
            // Update episode timing statistics (simulate episode completion)
            const now = Date.now();
            if (!this.episodeTimings) {
                this.episodeTimings = {
                    lastEpisodeTime: now,
                    episodeCount: 0,
                    recentEpisodes: [],
                    lastLogTime: now
                };
            }
            
            // Add timing for the entire batch (simulating rapid episode completion)
            const avgEpisodeTime = batchTrainingTime / batchSize;
            for (let i = 0; i < batchSize; i++) {
                this.episodeTimings.recentEpisodes.push(avgEpisodeTime);
                this.episodeTimings.episodeCount++;
            }
            this.episodeTimings.lastEpisodeTime = now;
            
            // Keep only last 10 episodes for rate calculation
            while (this.episodeTimings.recentEpisodes.length > 10) {
                this.episodeTimings.recentEpisodes.shift();
            }
            
            // Calculate and log performance metrics (every 5 episodes)
            if (this.episodeTimings.episodeCount % 5 === 0) {
                const avgIntervalMs = this.episodeTimings.recentEpisodes.reduce((a, b) => a + b, 0) / this.episodeTimings.recentEpisodes.length;
                const episodesPerSecond = 1000 / avgIntervalMs;
                const episodesPerMinute = episodesPerSecond * 60;
                const episodeDurationSec = avgIntervalMs / 1000;
                
                console.log(`🔥 PARALLEL PERFORMANCE: ${this.trainingSpeed}x speed → ${episodesPerSecond.toFixed(2)} eps/sec (${episodesPerMinute.toFixed(0)} eps/min)`);
                console.log(`📊 BATCH ${Math.floor(this.episodeCount/batchSize)}: Duration=${(batchTrainingTime/1000).toFixed(2)}s, Episodes=${batchSize}, Best Reward=${bestEpisodeReward.toFixed(2)}`);
            }
            
            // Charts are now updated individually for each episode above
            // No need for batch-level chart update
            
            // Update target network periodically
            if (this.qlearning.stepCount % this.qlearning.hyperparams.targetUpdateFreq === 0) {
                this.qlearning.updateTargetNetwork();
            }
            
            
            // Update statistics for logging
            const avgReward = totalReward / results.length;
            const avgSteps = totalSteps / results.length;
            
            console.log(`✅ Parallel batch completed: ${results.length} episodes, avg reward: ${avgReward.toFixed(2)}, best: ${bestEpisodeReward.toFixed(2)}, avg steps: ${avgSteps.toFixed(0)}`);
            console.log(`📊 Training data: ${totalExperiencesCollected} experiences collected, ${trainingSteps || 0} training steps performed (${((trainingSteps || 0) / Math.max(1, totalExperiencesCollected) * 100).toFixed(1)}% utilization)`);
            
            // Yield control to UI and continue training after brief pause
            await new Promise(resolve => setTimeout(resolve, 50)); // Longer pause to allow UI updates
            
            // Continue training if still in training mode
            if (this.isTraining) {
                this.startNextEpisodeOrBatch();
            }
            
        } catch (error) {
            console.warn('Parallel batch failed, falling back to single episode:', error);
            this.startNewEpisode();
        }
    }
    
    /**
     * Start a new training episode
     */
    startNewEpisode() {
        this.trainingStep = 0;
        this.episodeEnded = false; // Reset episode end flag
        
        // Performance tracking - start episode
        this.performanceTracker.startEpisode();
        this.episodeStartTime = Date.now();
        
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
        
        // Performance tracking - end episode
        const episodeEndTime = Date.now();
        const episodeTrainingTime = episodeEndTime - this.episodeStartTime;
        this.performanceTracker.endEpisode(this.trainingStep, episodeTrainingTime);
        
        // Update best score
        if (totalReward > this.bestScore) {
            this.bestScore = totalReward;
        }
        
        if (this.demoMode === 'training' && this.isTraining) {
            // Only increment episode count during actual training
            this.episodeCount++;
            
            // Performance monitoring for episode completion rate
            const now = Date.now();
            if (!this.episodeTimings) {
                this.episodeTimings = {
                    lastEpisodeTime: now,
                    episodeCount: 0,
                    recentEpisodes: [],
                    lastLogTime: now
                };
            }
            
            const timeSinceLastEpisode = now - this.episodeTimings.lastEpisodeTime;
            this.episodeTimings.recentEpisodes.push(timeSinceLastEpisode);
            this.episodeTimings.lastEpisodeTime = now;
            this.episodeTimings.episodeCount++;
            
            // Keep only last 10 episodes for rate calculation
            if (this.episodeTimings.recentEpisodes.length > 10) {
                this.episodeTimings.recentEpisodes.shift();
            }
            
            // Log performance every 5 episodes for more frequent monitoring
            if (this.episodeTimings.episodeCount % 5 === 0) {
                const avgIntervalMs = this.episodeTimings.recentEpisodes.reduce((a, b) => a + b, 0) / this.episodeTimings.recentEpisodes.length;
                const episodesPerSecond = 1000 / avgIntervalMs;
                const episodesPerMinute = episodesPerSecond * 60;
                
                // Calculate episode duration in seconds
                const episodeDurationMs = now - this.episodeStartTime;
                const episodeDurationSec = episodeDurationMs / 1000;
                
                console.log(`🔥 PERFORMANCE: ${this.trainingSpeed}x speed → ${episodesPerSecond.toFixed(2)} eps/sec (${episodesPerMinute.toFixed(0)} eps/min)`);
                console.log(`📊 EPISODE ${this.episodeCount}: Duration=${episodeDurationSec.toFixed(2)}s, Steps=${this.trainingStep}, Reward=${totalReward.toFixed(2)}`);
                
                // Check if episode duration is scaling unexpectedly with speed
                const expectedDuration = 8000 / (50 * this.trainingSpeed); // 8000 steps at 50Hz base rate
                const durationRatio = episodeDurationSec / expectedDuration;
                if (durationRatio > 2.0) {
                    console.warn(`⚠️  SLOW EPISODE: Taking ${durationRatio.toFixed(2)}x longer than expected!`);
                }
            }
            
            // Update model display with latest stats
            this.updateTrainingModelDisplay();
            
            // Update performance metrics display
            this.updatePerformanceMetricsDisplay();
            
            
            
            // Update performance charts with episode completion data
            if (this.performanceCharts && this.qlearning) {
                // Get current Q-value estimate for balanced state
                // Create balanced state matching current timestep configuration
                const timesteps = this.robot ? this.robot.historyTimesteps : 1;
                const inputSize = timesteps * 2;
                const balancedState = new Float32Array(inputSize).fill(0.0); // All timesteps balanced
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
                try {
                    console.log(`🔄 Episode ${this.episodeCount} completed. Training state: isTraining=${this.isTraining}, isPaused=${this.isPaused}`);
                    if (this.isTraining) {
                        console.log(`✅ Starting next episode/batch...`);
                        this.startNextEpisodeOrBatch();
                    } else {
                        console.warn(`⚠️ Training stopped unexpectedly! isTraining=${this.isTraining}, isPaused=${this.isPaused}`);
                        // This is where the bug might be - training stopped but UI not updated
                        // Force UI update to reflect actual state
                        const startButton = document.getElementById('start-training');
                        const pauseButton = document.getElementById('pause-training');
                        
                        if (startButton) startButton.disabled = false;
                        if (pauseButton) {
                            pauseButton.disabled = true;
                            pauseButton.textContent = 'Pause Training';
                        }
                        
                        this.updateTrainingModelDisplay();
                        console.log('🛠️ UI state corrected to match actual training state');
                    }
                } catch (error) {
                    console.error('Error in setTimeout callback for next episode:', error);
                    // Reset training state if error occurs
                    this.isTraining = false;
                    this.isPaused = false;
                    this.demoMode = 'freerun';
                    
                    // Update UI
                    const startButton = document.getElementById('start-training');
                    const pauseButton = document.getElementById('pause-training');
                    
                    if (startButton) startButton.disabled = false;
                    if (pauseButton) {
                        pauseButton.disabled = true;
                        pauseButton.textContent = 'Pause Training';
                    }
                    
                    this.updateOnScreenControlsVisibility();
                    
                    this.updateTrainingModelDisplay();
                    alert('Training stopped due to an error. Please check the console and try starting training again.');
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
        this.updateRewardTypeUI();
        this.updateOnScreenControlsVisibility();
        
        // Initialize Q-learning if switching to training or evaluation mode
        if ((newMode === 'training' || newMode === 'evaluation') && !this.qlearning) {
            await this.initializeQLearning();
        }
        
        // Reset environment for new mode
        this.resetEnvironment();
    }
    
    /**
     * Determine if Q-learning training should occur this physics step
     * Reduces training frequency at high speeds to prevent bottlenecks
     * @param {number} currentStep - Current step in the physics loop
     * @param {number} totalSteps - Total steps to run this frame
     * @returns {boolean} Whether to train this step
     */
    _shouldTrainThisStep(currentStep, totalSteps) {
        // Always train on episode boundaries (handled separately)
        // For regular steps, use intelligent training frequency based on speed
        
        if (totalSteps <= 10) {
            // Low speed: train every step for maximum learning
            return true;
        } else if (totalSteps <= 50) {
            // Medium speed: train every other step
            return currentStep % 2 === 0;
        } else if (totalSteps <= 100) {
            // High speed: train every 4th step
            return currentStep % 4 === 0;
        } else if (totalSteps <= 500) {
            // Very high speed: train every 10th step
            return currentStep % 10 === 0;
        } else {
            // Extreme speed: train every 20th step
            return currentStep % 20 === 0;
        }
    }

    /**
     * Parameter update methods called by UI controls
     */
    setTrainingSpeed(speed) {
        this.trainingSpeed = Math.max(0.1, Math.min(1000.0, speed));
        this.targetPhysicsStepsPerFrame = Math.round(this.trainingSpeed);
        console.log(`Training speed set to ${speed}x (${this.targetPhysicsStepsPerFrame} steps/frame)`);
        
        // CRITICAL FIX: Restart simulation with adapted interval for the new speed
        if (this.isInitialized && this.smartRenderingManager.renderingMode === 'interval') {
            console.log('Restarting simulation with adapted interval for new speed...');
            this.startSimulation(); // This will call stopSimulation() and restart with correct interval
        }
    }
    
    setDebugSpeed(speed) {
        this.debugSpeed = Math.max(0.1, Math.min(2.0, speed));
        console.log(`Debug speed set to ${speed}x`);
    }
    
    setParallelMode(enabled) {
        this.parallelModeEnabled = enabled;
        console.log(`Parallel training mode ${enabled ? 'enabled' : 'disabled'}`);
    }
    
    testModel() {
        // Check if model is available
        if (!this.qlearning) {
            alert('No model available for testing!\n\nPlease train a model first or load a saved model.');
            return;
        }
        
        // Stop any current training
        if (this.isTraining) {
            this.pauseTraining();
        }
        
        // Switch to free run mode
        this.demoMode = 'freerun';
        this.isTraining = false;
        this.isPaused = false;
        
        // Disable PD controller and User Control Only to enable model control
        this.pdControllerEnabled = false;
        this.userControlOnlyEnabled = false;
        
        // Update UI states
        this.updateFreeRunSpeedUI();
        this.updatePDControllerUI();
        this.updateUserControlOnlyUI();
        
        // Reset robot to a clean state for testing
        if (this.robot) {
            this.robot.reset({
                angle: (Math.random() - 0.5) * 0.1,
                angularVelocity: 0,
                position: 0,
                velocity: 0
            });
        }
        
        console.log('Testing model in free run mode - Model Control active');
    }
    
    updateFreeRunSpeedUI() {
        const slider = document.getElementById('debug-speed');
        const label = document.querySelector('label[for="debug-speed"]');
        
        if (slider && label) {
            const isEnabled = this.demoMode === 'freerun';
            slider.disabled = !isEnabled;
            slider.style.opacity = isEnabled ? '1' : '0.5';
            label.style.opacity = isEnabled ? '1' : '0.5';
        }
        
        // Show/hide on-screen controls based on mode
        this.updateOnScreenControlsVisibility();
    }
    
    setupOnScreenControls() {
        // Left control button
        document.getElementById('control-left')?.addEventListener('mousedown', () => {
            if (!this.userControlEnabled) return;
            this.manualControl.leftPressed = true;
            this.updateManualTorque();
        });
        document.getElementById('control-left')?.addEventListener('mouseup', () => {
            if (!this.userControlEnabled) return;
            this.manualControl.leftPressed = false;
            this.updateManualTorque();
        });
        document.getElementById('control-left')?.addEventListener('mouseleave', () => {
            if (!this.userControlEnabled) return;
            this.manualControl.leftPressed = false;
            this.updateManualTorque();
        });
        
        // Right control button
        document.getElementById('control-right')?.addEventListener('mousedown', () => {
            if (!this.userControlEnabled) return;
            this.manualControl.rightPressed = true;
            this.updateManualTorque();
        });
        document.getElementById('control-right')?.addEventListener('mouseup', () => {
            if (!this.userControlEnabled) return;
            this.manualControl.rightPressed = false;
            this.updateManualTorque();
        });
        document.getElementById('control-right')?.addEventListener('mouseleave', () => {
            if (!this.userControlEnabled) return;
            this.manualControl.rightPressed = false;
            this.updateManualTorque();
        });
        
        // Touch events for mobile
        document.getElementById('control-left')?.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (!this.userControlEnabled) return;
            this.manualControl.leftPressed = true;
            this.updateManualTorque();
        });
        document.getElementById('control-left')?.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (!this.userControlEnabled) return;
            this.manualControl.leftPressed = false;
            this.updateManualTorque();
        });
        
        document.getElementById('control-right')?.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (!this.userControlEnabled) return;
            this.manualControl.rightPressed = true;
            this.updateManualTorque();
        });
        document.getElementById('control-right')?.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (!this.userControlEnabled) return;
            this.manualControl.rightPressed = false;
            this.updateManualTorque();
        });
        
        // Reset button
        document.getElementById('control-reset')?.addEventListener('click', () => {
            this.resetRobotPosition();
        });
    }
    
    updateOnScreenControlsVisibility() {
        const controls = document.getElementById('on-screen-controls');
        if (controls) {
            const shouldShow = this.demoMode === 'freerun';
            controls.style.display = shouldShow ? 'flex' : 'none';
        }
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
    
    updateGamma(gamma) {
        if (this.qlearning) {
            this.qlearning.hyperparams.gamma = gamma;
            console.log(`Gamma updated to ${gamma}`);
        }
    }
    
    updateEpsilonMin(epsilonMin) {
        if (this.qlearning) {
            this.qlearning.hyperparams.epsilonMin = epsilonMin;
            console.log(`Epsilon Min updated to ${epsilonMin}`);
        }
    }
    
    updateEpsilonDecay(epsilonDecay) {
        if (this.qlearning) {
            this.qlearning.hyperparams.epsilonDecay = epsilonDecay;
            console.log(`Epsilon Decay updated to ${epsilonDecay} steps`);
        }
    }
    
    updateBatchSize(batchSize) {
        if (this.qlearning) {
            this.qlearning.hyperparams.batchSize = batchSize;
            console.log(`Batch Size updated to ${batchSize}`);
        }
    }
    
    updateTargetUpdateFreq(targetUpdateFreq) {
        if (this.qlearning) {
            this.qlearning.hyperparams.targetUpdateFreq = targetUpdateFreq;
            console.log(`Target Update Frequency updated to ${targetUpdateFreq}`);
        }
    }
    
    updateMaxEpisodes(maxEpisodes) {
        if (this.qlearning) {
            this.qlearning.hyperparams.maxEpisodes = maxEpisodes;
            console.log(`Max Episodes updated to ${maxEpisodes}`);
        }
    }
    
    updateMaxStepsPerEpisode(maxStepsPerEpisode) {
        if (this.qlearning) {
            this.qlearning.hyperparams.maxStepsPerEpisode = maxStepsPerEpisode;
            console.log(`Max Steps Per Episode updated to ${maxStepsPerEpisode}`);
            
            // If currently training and current step count exceeds new limit, end episode
            if (this.isTraining && !this.isPaused && !this.episodeEnded && 
                this.trainingStep >= maxStepsPerEpisode && 
                (this.demoMode === 'training')) {
                console.log(`Episode ending immediately - current step ${this.trainingStep} >= new limit ${maxStepsPerEpisode}`);
                this.episodeEnded = true;
                // Trigger episode end on next physics update
                setTimeout(() => {
                    if (this.episodeEnded) {
                        this.handleEpisodeEnd({
                            state: this.robot.getState(),
                            reward: this.currentReward,
                            done: true
                        });
                    }
                }, 0);
            }
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
    
    updateMaxAngle(angle) {
        if (this.robot) {
            const config = this.robot.getConfig();
            config.maxAngle = angle;
            this.robot.updateConfig(config);
            console.log(`Max angle updated to ${angle} rad (${(angle * 180 / Math.PI).toFixed(1)} degrees)`);
        }
    }
    
    updateMotorTorqueRange(range) {
        if (this.robot) {
            const config = this.robot.getConfig();
            config.motorTorqueRange = range;
            this.robot.updateConfig(config);
            console.log(`Motor torque range updated to ±${range} Nm`);
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
            this.manualControl.manualTorque = -maxTorque * 0.7;
            this.debugLastAction = 'Left';
        } else if (this.manualControl.rightPressed) {
            // Right pressed - positive torque (move right)  
            this.manualControl.manualTorque = maxTorque * 0.7;
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
     * Toggle PD controller on/off (switches between PD and Model control)
     */
    togglePDController() {
        // Don't allow PD controller during training or evaluation
        if ((this.demoMode === 'training' && this.isTraining) || this.demoMode === 'evaluation') {
            console.log('PD Controller cannot be enabled during training or model testing');
            return;
        }
        
        // If we're in free run mode and trying to disable PD (switch to model), check if model exists
        if (this.demoMode === 'freerun' && this.pdControllerEnabled && !this.qlearning) {
            // User is trying to switch to model control but no model is loaded
            alert('No model loaded!\n\nTo use Model Control, you need to:\n• Train a model (Start Training)\n• Load a saved model (Model Management)\n• Import a pre-trained model\n\nPD Controller will remain active.');
            return; // Don't toggle, keep PD controller enabled
        }
        
        this.pdControllerEnabled = !this.pdControllerEnabled;
        this.updatePDControllerUI();
        
        // Reset robot to clean state when changing control modes
        this.resetRobotPosition();
        
        console.log('PD Controller', this.pdControllerEnabled ? 'enabled' : 'disabled');
    }
    
    /**
     * Toggle User Control Only mode on/off
     */
    toggleUserControlOnly() {
        // Don't allow during training or evaluation
        if ((this.demoMode === 'training' && this.isTraining) || this.demoMode === 'evaluation') {
            console.log('User Control Only cannot be enabled during training or model testing');
            return;
        }
        
        this.userControlOnlyEnabled = !this.userControlOnlyEnabled;
        this.updateUserControlOnlyUI();
        
        // Reset robot to clean state when changing control modes
        this.resetRobotPosition();
        
        console.log('User Control Only', this.userControlOnlyEnabled ? 'enabled' : 'disabled');
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
            if (this.demoMode === 'freerun') {
                // In free run mode, show what control method is active
                if (this.pdControllerEnabled) {
                    text.textContent = 'Switch to Model Control';
                    status.textContent = '(PD Controller Active)';
                    status.style.color = '#00ff88';
                } else {
                    text.textContent = 'Switch to PD Controller';
                    status.textContent = this.qlearning ? '(Model Control Active)' : '(No Control Active)';
                    status.style.color = this.qlearning ? '#2196F3' : '#888';
                }
            } else {
                text.textContent = this.pdControllerEnabled ? 'Disable PD Controller' : 'Enable PD Controller';
                status.textContent = this.pdControllerEnabled ? '(Active)' : '(Disabled)';
                status.style.color = this.pdControllerEnabled ? '#00ff88' : '#888';
            }
            status.style.marginLeft = '0'; // Remove margin since using <br>
        }
    }
    
    /**
     * Update User Control Only button UI
     */
    updateUserControlOnlyUI() {
        const button = document.getElementById('toggle-user-control-only');
        const text = document.getElementById('user-control-text');
        const status = document.getElementById('user-control-status');
        
        if (!button || !text || !status) return;
        
        // Check if User Control Only should be disabled
        const shouldDisable = (this.demoMode === 'training' && this.isTraining) || this.demoMode === 'evaluation';
        
        if (shouldDisable) {
            button.disabled = true;
            text.textContent = 'User Control Only';
            status.textContent = '(Disabled during training/testing)';
            status.style.color = '#666';
            this.userControlOnlyEnabled = false;
        } else {
            button.disabled = false;
            text.textContent = this.userControlOnlyEnabled ? 'Disable User Control Only' : 'Enable User Control Only';
            status.textContent = this.userControlOnlyEnabled ? '(Active)' : '(Disabled)';
            status.style.color = this.userControlOnlyEnabled ? '#ffaa00' : '#888';
            status.style.marginLeft = '0'; // Remove margin since using <br>
        }
    }
    
    /**
     * Set reward function type from dropdown
     */
    setRewardFunction(rewardType) {
        if (!this.robot) {
            console.warn('Robot not initialized');
            return;
        }
        
        this.robot.setRewardType(rewardType);
        
        // Show/hide offset range slider based on reward type
        const offsetRangeContainer = document.getElementById('offset-range-container');
        if (offsetRangeContainer) {
            offsetRangeContainer.style.display = rewardType === 'offset-adaptive' ? 'block' : 'none';
        }
        
        console.log(`Reward function changed to: ${rewardType}`);
    }
    
    
    /**
     * Reset robot position for manual control
     */
    resetRobotPosition() {
        if (this.robot) {
            let resetAngle = 0; // Default: perfectly balanced
            
            // In free run mode, use exact angle from slider
            if (this.demoMode === 'freerun' && Math.abs(this.resetAngleDegrees) > 0) {
                const angleDegrees = this.resetAngleDegrees || 0.0;
                resetAngle = angleDegrees * Math.PI / 180; // Use exact angle from slider
                console.log(`Reset with angle: ${(resetAngle * 180 / Math.PI).toFixed(2)}° (set to ${angleDegrees.toFixed(1)}°)`);
            }
            
            this.robot.reset({
                angle: resetAngle,
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
        
        // Add metadata including timestep configuration
        const saveData = {
            name: modelName,
            timestamp: now.toISOString(),
            type: 'DQN',
            architecture: this.qlearning.getStats(),
            timestepConfig: {
                historyTimesteps: this.robot ? this.robot.historyTimesteps : 1,
                inputSize: this.qlearning.qNetwork ? this.qlearning.qNetwork.getArchitecture().inputSize : 2
            },
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
        
        // Handle timestep configuration if available
        if (saveData.timestepConfig) {
            const savedTimesteps = saveData.timestepConfig.historyTimesteps || 1;
            
            // Update robot timestep configuration
            if (this.robot) {
                this.robot.setHistoryTimesteps(savedTimesteps);
                console.log(`Restored history timesteps: ${savedTimesteps}`);
            }
            
            // Update UI slider
            const historyTimestepsSlider = document.getElementById('history-timesteps');
            const historyTimestepsValue = document.getElementById('history-timesteps-value');
            if (historyTimestepsSlider) {
                historyTimestepsSlider.value = savedTimesteps;
                historyTimestepsValue.textContent = savedTimesteps;
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
        
        // Setup custom sliders after app initialization
        setupCustomSliders();
    } catch (error) {
        console.error('Failed to start application:', error);
    }
});

// Custom slider setup function
function setupCustomSliders() {
    console.log('Setting up custom sliders...');
    
    // Offset angle error slider
    const offsetSlider = document.getElementById('offset-angle-error');
    const offsetValue = document.getElementById('offset-angle-error-value');
    
    if (offsetSlider && offsetValue) {
        console.log('Found offset angle slider');
        offsetSlider.addEventListener('input', (e) => {
            const degrees = parseFloat(e.target.value);
            const radians = degrees * Math.PI / 180;
            offsetValue.textContent = `${degrees.toFixed(1)}°`;
            
            if (app && app.robot) {
                app.robot.angleOffset = radians;
                console.log(`Offset angle set to: ${degrees.toFixed(1)}° (${radians.toFixed(3)} rad)`);
            }
        });
    } else {
        console.warn('Offset angle slider or value element not found');
    }
    
    // Reset angle slider
    const resetSlider = document.getElementById('reset-angle');
    const resetValue = document.getElementById('reset-angle-value');
    
    if (resetSlider && resetValue) {
        console.log('Found reset angle slider');
        resetSlider.addEventListener('input', (e) => {
            const degrees = parseFloat(e.target.value);
            resetValue.textContent = `${degrees.toFixed(1)}°`;
            
            if (app) {
                app.resetAngleDegrees = degrees;
                console.log(`Reset angle set to: ${degrees.toFixed(1)}° (randomly +/-)`);
            }
        });
    } else {
        console.warn('Reset angle slider or value element not found');
    }
    
    // Offset range slider for training
    const offsetRangeSlider = document.getElementById('offset-range');
    const offsetRangeValue = document.getElementById('offset-range-value');
    
    if (offsetRangeSlider && offsetRangeValue) {
        console.log('Found offset range slider');
        offsetRangeSlider.addEventListener('input', (e) => {
            const degrees = parseFloat(e.target.value);
            offsetRangeValue.textContent = `${degrees.toFixed(1)}°`;
            
            if (app) {
                app.maxOffsetRangeDegrees = degrees;
                // Also update robot's training offset range
                if (app.robot) {
                    app.robot.setTrainingOffsetRange(degrees);
                }
                console.log(`Max offset range set to: ±${degrees.toFixed(1)}° for training`);
            }
        });
    } else {
        console.warn('Offset range slider or value element not found');
    }
}

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (app) {
        app.destroy();
    }
});

// Export for potential external access
window.TwoWheelBotRL = TwoWheelBotRL;