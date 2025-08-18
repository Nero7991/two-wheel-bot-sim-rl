/**
 * Example integration of Renderer with BalancingRobot physics and Q-Learning
 * Demonstrates real-time visualization during training
 */

import { createRenderer } from './Renderer.js';
import { createDefaultRobot } from '../physics/BalancingRobot.js';
import { createDefaultQLearning } from '../training/QLearning.js';

/**
 * Visualization Integration Example
 * Shows how to connect the renderer to the physics simulation and training loop
 */
export class VisualizationExample {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            throw new Error(`Canvas element with id '${canvasId}' not found`);
        }
        
        // Initialize components
        this.renderer = createRenderer(this.canvas, {
            showGrid: true,
            showDebugInfo: true,
            showPerformance: true,
            targetFPS: 60
        });
        
        this.robot = createDefaultRobot();
        this.qlearning = null;
        
        // Demo state
        this.isRunning = false;
        this.isTraining = false;
        this.currentEpisode = 0;
        this.currentStep = 0;
        this.demoMode = 'physics'; // 'physics', 'training', 'evaluation'
        
        // Performance tracking
        this.lastPhysicsUpdate = 0;
        this.physicsUpdateInterval = 20; // 50 Hz physics updates
        
        console.log('Visualization example initialized');
    }
    
    /**
     * Initialize Q-learning for training demonstration
     */
    async initializeTraining() {
        this.qlearning = createDefaultQLearning({
            maxEpisodes: 100,
            maxStepsPerEpisode: 1000,
            epsilon: 0.3
        });
        
        await this.qlearning.initialize();
        console.log('Q-learning initialized for training demo');
    }
    
    /**
     * Start the visualization demo
     * @param {string} mode - Demo mode: 'physics', 'training', 'evaluation'
     */
    async start(mode = 'physics') {
        if (this.isRunning) return;
        
        this.demoMode = mode;
        this.isRunning = true;
        
        // Initialize training if needed
        if (mode === 'training' || mode === 'evaluation') {
            if (!this.qlearning) {
                await this.initializeTraining();
            }
        }
        
        // Reset robot state
        this.robot.reset({
            angle: (Math.random() - 0.5) * 0.2, // Small random initial angle
            angularVelocity: 0,
            position: 0,
            velocity: 0
        });
        
        // Update renderer with initial state
        this.updateRenderer();
        
        // Start rendering
        this.renderer.start();
        
        // Start simulation loop
        this.startSimulationLoop();
        
        console.log(`Demo started in ${mode} mode`);
    }
    
    /**
     * Stop the visualization demo
     */
    stop() {
        this.isRunning = false;
        this.isTraining = false;
        this.renderer.stop();
        console.log('Demo stopped');
    }
    
    /**
     * Start the main simulation loop
     */
    startSimulationLoop() {
        const simulate = (timestamp) => {
            if (!this.isRunning) return;
            
            // Update physics at fixed intervals
            if (timestamp - this.lastPhysicsUpdate >= this.physicsUpdateInterval) {
                this.updatePhysics();
                this.updateRenderer();
                this.lastPhysicsUpdate = timestamp;
            }
            
            requestAnimationFrame(simulate);
        };
        
        requestAnimationFrame(simulate);
    }
    
    /**
     * Update physics simulation based on current mode
     */
    updatePhysics() {
        let motorTorque = 0;
        
        switch (this.demoMode) {
            case 'physics':
                motorTorque = this.getPhysicsDemoTorque();
                break;
            case 'training':
                motorTorque = this.getTrainingTorque();
                break;
            case 'evaluation':
                motorTorque = this.getEvaluationTorque();
                break;
        }
        
        // Step physics simulation
        const result = this.robot.step(motorTorque);
        
        // Handle episode completion
        if (result.done && (this.demoMode === 'training' || this.demoMode === 'evaluation')) {
            this.handleEpisodeEnd(result);
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
        
        const torque = -(kp * state.angle + kd * state.angularVelocity);
        
        // Add some noise for more interesting behavior
        const noise = (Math.random() - 0.5) * 0.5;
        
        return torque + noise;
    }
    
    /**
     * Get motor torque from Q-learning during training
     */
    getTrainingTorque() {
        if (!this.qlearning || !this.isTraining) {
            this.startTrainingEpisode();
            return 0;
        }
        
        const state = this.robot.getState();
        const normalizedState = state.getNormalizedInputs();
        
        // Select action using epsilon-greedy policy
        const actionIndex = this.qlearning.selectAction(normalizedState, true);
        const actions = [-1.0, 0.0, 1.0]; // Left, brake, right
        
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
        const actions = [-1.0, 0.0, 1.0];
        
        return actions[actionIndex];
    }
    
    /**
     * Start a new training episode
     */
    startTrainingEpisode() {
        this.isTraining = true;
        this.currentStep = 0;
        
        // Reset robot with small random perturbation
        this.robot.reset({
            angle: (Math.random() - 0.5) * 0.1,
            angularVelocity: (Math.random() - 0.5) * 0.2,
            position: 0,
            velocity: 0
        });
        
        console.log(`Starting training episode ${this.currentEpisode + 1}`);
    }
    
    /**
     * Handle end of training/evaluation episode
     */
    handleEpisodeEnd(result) {
        this.currentEpisode++;
        
        if (this.demoMode === 'training') {
            console.log(`Episode ${this.currentEpisode} completed: Reward=${result.reward.toFixed(2)}, Steps=${this.currentStep}`);
            
            // Start next episode after brief pause
            setTimeout(() => {
                if (this.isRunning && this.currentEpisode < 50) { // Limit for demo
                    this.startTrainingEpisode();
                }
            }, 100);
        } else {
            // Evaluation mode - restart after longer pause
            setTimeout(() => {
                if (this.isRunning) {
                    this.robot.reset({
                        angle: (Math.random() - 0.5) * 0.2,
                        angularVelocity: 0,
                        position: 0,
                        velocity: 0
                    });
                    this.currentStep = 0;
                }
            }, 1000);
        }
    }
    
    /**
     * Update renderer with current simulation state
     */
    updateRenderer() {
        const state = this.robot.getState();
        const config = this.robot.getConfig();
        const stats = this.robot.getStats();
        
        // Update robot visualization
        this.renderer.updateRobot(state, config, stats.currentMotorTorque);
        
        // Update training metrics
        const trainingMetrics = {
            episode: this.currentEpisode,
            step: this.currentStep++,
            reward: this.demoMode === 'physics' ? 0 : stats.totalReward,
            totalReward: stats.totalReward,
            bestReward: this.getBestReward(),
            epsilon: this.qlearning ? this.qlearning.hyperparams.epsilon : 0,
            isTraining: this.isTraining,
            trainingMode: this.demoMode
        };
        
        this.renderer.updateTraining(trainingMetrics);
    }
    
    /**
     * Get best reward achieved so far
     */
    getBestReward() {
        if (!this.qlearning) return 0;
        
        const metrics = this.qlearning.getStats();
        return metrics.bestReward || 0;
    }
    
    /**
     * Switch demo mode
     * @param {string} newMode - New demo mode
     */
    async switchMode(newMode) {
        console.log(`Switching from ${this.demoMode} to ${newMode} mode`);
        
        this.stop();
        await new Promise(resolve => setTimeout(resolve, 100)); // Brief pause
        await this.start(newMode);
    }
    
    /**
     * Handle canvas resize
     */
    handleResize() {
        const container = this.canvas.parentElement;
        const newWidth = container.clientWidth;
        const newHeight = container.clientHeight;
        
        this.renderer.resize(newWidth, newHeight);
    }
    
    /**
     * Get demo statistics
     */
    getStats() {
        return {
            isRunning: this.isRunning,
            mode: this.demoMode,
            episode: this.currentEpisode,
            step: this.currentStep,
            robotStats: this.robot.getStats(),
            qlearningStats: this.qlearning ? this.qlearning.getStats() : null,
            rendererStats: this.renderer.getStats()
        };
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        this.stop();
        this.renderer.destroy();
        this.robot = null;
        this.qlearning = null;
        console.log('Visualization example destroyed');
    }
}

/**
 * Create and setup a complete visualization demo
 * @param {string} canvasId - Canvas element ID
 * @returns {VisualizationExample} Demo instance
 */
export function createVisualizationDemo(canvasId) {
    return new VisualizationExample(canvasId);
}

/**
 * Quick setup function for HTML page integration
 * @param {string} canvasId - Canvas element ID
 * @param {Object} controls - UI control elements
 */
export async function setupVisualizationDemo(canvasId, controls = {}) {
    const demo = new VisualizationExample(canvasId);
    
    // Setup control event listeners
    if (controls.startPhysics) {
        controls.startPhysics.addEventListener('click', () => demo.start('physics'));
    }
    
    if (controls.startTraining) {
        controls.startTraining.addEventListener('click', () => demo.start('training'));
    }
    
    if (controls.startEvaluation) {
        controls.startEvaluation.addEventListener('click', () => demo.start('evaluation'));
    }
    
    if (controls.stop) {
        controls.stop.addEventListener('click', () => demo.stop());
    }
    
    if (controls.reset) {
        controls.reset.addEventListener('click', () => {
            demo.stop();
            setTimeout(() => demo.start(demo.demoMode), 100);
        });
    }
    
    // Handle window resize
    window.addEventListener('resize', () => demo.handleResize());
    
    // Start with physics demo
    await demo.start('physics');
    
    console.log('Visualization demo setup complete');
    return demo;
}