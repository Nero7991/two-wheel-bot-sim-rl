/**
 * 2D Visualization Renderer for Two-Wheel Balancing Robot RL Application
 * 
 * High-performance HTML5 Canvas-based renderer for real-time training visualization
 * Features:
 * - Real-time robot visualization (pendulum + wheels)
 * - Physics state display (angle, velocity, position)
 * - Motor torque indicators
 * - Training metrics overlay
 * - 60 FPS rendering with coordinate transformation
 * - Responsive canvas management
 */

/**
 * Coordinate system transformer for physics <-> screen space conversion
 */
export class CoordinateTransform {
    constructor(canvasWidth, canvasHeight) {
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        
        // Physics coordinate system (meters)
        this.physicsScale = 100; // pixels per meter
        this.centerX = canvasWidth / 2;
        this.centerY = canvasHeight * 0.7; // Ground line at 70% down
        
        this.minX = -canvasWidth / (2 * this.physicsScale);
        this.maxX = canvasWidth / (2 * this.physicsScale);
        this.minY = -canvasHeight / (2 * this.physicsScale);
        this.maxY = canvasHeight / (2 * this.physicsScale);
    }
    
    /**
     * Convert physics coordinates to screen coordinates
     * @param {number} x - Physics X coordinate (meters)
     * @param {number} y - Physics Y coordinate (meters)
     * @returns {Object} {x, y} screen coordinates
     */
    physicsToScreen(x, y) {
        return {
            x: this.centerX + x * this.physicsScale,
            y: this.centerY - y * this.physicsScale // Flip Y axis
        };
    }
    
    /**
     * Convert screen coordinates to physics coordinates
     * @param {number} screenX - Screen X coordinate
     * @param {number} screenY - Screen Y coordinate
     * @returns {Object} {x, y} physics coordinates
     */
    screenToPhysics(screenX, screenY) {
        return {
            x: (screenX - this.centerX) / this.physicsScale,
            y: (this.centerY - screenY) / this.physicsScale
        };
    }
    
    /**
     * Update transform for canvas resize
     * @param {number} canvasWidth - New canvas width
     * @param {number} canvasHeight - New canvas height
     */
    resize(canvasWidth, canvasHeight) {
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.centerX = canvasWidth / 2;
        this.centerY = canvasHeight * 0.7;
        
        this.minX = -canvasWidth / (2 * this.physicsScale);
        this.maxX = canvasWidth / (2 * this.physicsScale);
        this.minY = -canvasHeight / (2 * this.physicsScale);
        this.maxY = canvasHeight / (2 * this.physicsScale);
    }
    
    /**
     * Convert length from physics to screen units
     * @param {number} length - Length in meters
     * @returns {number} Length in pixels
     */
    scaleLength(length) {
        return length * this.physicsScale;
    }
}

/**
 * Performance monitor for tracking rendering performance
 */
export class PerformanceMonitor {
    constructor() {
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 60;
        this.frameTime = 16.67; // milliseconds
        this.avgFrameTime = 16.67;
        this.maxFrameTime = 16.67;
        this.minFrameTime = 16.67;
        
        // Performance history for averaging
        this.frameTimes = [];
        this.historySize = 60; // 1 second of frames at 60fps
    }
    
    /**
     * Update performance metrics
     */
    update() {
        const currentTime = performance.now();
        this.frameTime = currentTime - this.lastTime;
        this.lastTime = currentTime;
        
        // Update frame times history
        this.frameTimes.push(this.frameTime);
        if (this.frameTimes.length > this.historySize) {
            this.frameTimes.shift();
        }
        
        // Calculate metrics
        this.avgFrameTime = this.frameTimes.reduce((sum, t) => sum + t, 0) / this.frameTimes.length;
        this.fps = 1000 / this.avgFrameTime;
        this.maxFrameTime = Math.max(...this.frameTimes);
        this.minFrameTime = Math.min(...this.frameTimes);
    }
    
    /**
     * Get performance summary
     * @returns {Object} Performance metrics
     */
    getMetrics() {
        return {
            fps: Math.round(this.fps),
            avgFrameTime: this.avgFrameTime.toFixed(2),
            maxFrameTime: this.maxFrameTime.toFixed(2),
            minFrameTime: this.minFrameTime.toFixed(2)
        };
    }
    
    /**
     * Check if rendering is running smoothly
     * @returns {boolean} True if performance is good
     */
    isPerformanceGood() {
        return this.fps >= 55 && this.maxFrameTime < 25; // 55+ FPS, max 25ms frame time
    }
}

/**
 * Main 2D renderer for the balancing robot visualization
 */
export class Renderer {
    constructor(canvas, config = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        // Configuration
        this.config = {
            backgroundColor: config.backgroundColor || '#0a0a0a',
            gridColor: config.gridColor || '#1a1a1a',
            robotColor: config.robotColor || '#00d4ff',
            wheelColor: config.wheelColor || '#ffffff',
            torqueColor: config.torqueColor || '#ff6b35',
            textColor: config.textColor || '#ffffff',
            showGrid: config.showGrid !== false,
            showDebugInfo: config.showDebugInfo !== false,
            showPerformance: config.showPerformance !== false,
            targetFPS: config.targetFPS || 60,
            ...config
        };
        
        // Coordinate transformation
        this.transform = new CoordinateTransform(canvas.width, canvas.height);
        
        // Performance monitoring
        this.performance = new PerformanceMonitor();
        
        // Rendering state
        this.isRendering = false;
        this.animationId = null;
        this.lastRenderTime = 0;
        this.frameInterval = 1000 / this.config.targetFPS;
        
        // Robot visualization state
        this.robotState = null;
        this.robotConfig = null;
        this.motorTorque = 0;
        
        // Training visualization state
        this.trainingMetrics = {
            episode: 0,
            step: 0,
            reward: 0,
            totalReward: 0,
            bestReward: 0,
            epsilon: 0,
            isTraining: false,
            trainingMode: 'idle'
        };
        
        // UI elements state
        this.showRobotInfo = true;
        this.showTrainingInfo = true;
        this.showPerformanceInfo = true;
        
        console.log('Renderer initialized:', canvas.width, 'x', canvas.height);
    }
    
    /**
     * Start the rendering loop
     */
    start() {
        if (this.isRendering) return;
        
        this.isRendering = true;
        this.lastRenderTime = performance.now();
        
        const renderLoop = (currentTime) => {
            if (!this.isRendering) return;
            
            // Frame rate limiting
            const deltaTime = currentTime - this.lastRenderTime;
            if (deltaTime >= this.frameInterval) {
                this.render();
                this.performance.update();
                this.lastRenderTime = currentTime - (deltaTime % this.frameInterval);
            }
            
            this.animationId = requestAnimationFrame(renderLoop);
        };
        
        this.animationId = requestAnimationFrame(renderLoop);
        console.log('Renderer started with target FPS:', this.config.targetFPS);
    }
    
    /**
     * Stop the rendering loop
     */
    stop() {
        this.isRendering = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        console.log('Renderer stopped');
    }
    
    /**
     * Handle canvas resize
     * @param {number} width - New canvas width
     * @param {number} height - New canvas height
     */
    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.transform.resize(width, height);
        console.log('Renderer resized to:', width, 'x', height);
    }
    
    /**
     * Update robot state for visualization
     * @param {Object} state - Robot state from physics simulation
     * @param {Object} config - Robot configuration
     * @param {number} motorTorque - Current motor torque
     */
    updateRobot(state, config, motorTorque = 0) {
        this.robotState = state;
        this.robotConfig = config;
        this.motorTorque = motorTorque;
    }
    
    /**
     * Update training metrics for visualization
     * @param {Object} metrics - Training metrics object
     */
    updateTraining(metrics) {
        this.trainingMetrics = { ...this.trainingMetrics, ...metrics };
    }
    
    /**
     * Main render function
     */
    render() {
        // Clear canvas
        this.clear();
        
        // Draw environment
        this.drawEnvironment();
        
        // Draw robot if state is available
        if (this.robotState && this.robotConfig) {
            this.drawRobot();
        }
        
        // Draw UI overlays
        this.drawUI();
    }
    
    /**
     * Clear the canvas
     */
    clear() {
        this.ctx.fillStyle = this.config.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    /**
     * Draw environment (ground, grid, reference lines)
     */
    drawEnvironment() {
        // Draw grid if enabled
        if (this.config.showGrid) {
            this.drawGrid();
        }
        
        // Draw ground line
        this.drawGround();
        
        // Draw coordinate axes for reference
        this.drawAxes();
    }
    
    /**
     * Draw background grid
     */
    drawGrid() {
        this.ctx.strokeStyle = this.config.gridColor;
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.3;
        
        const gridSpacing = this.transform.scaleLength(0.5); // 0.5m grid
        
        // Vertical lines
        for (let x = this.transform.centerX % gridSpacing; x < this.canvas.width; x += gridSpacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = this.transform.centerY % gridSpacing; y < this.canvas.height; y += gridSpacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Draw ground reference line
     */
    drawGround() {
        this.ctx.strokeStyle = this.config.textColor;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.8;
        
        const groundY = this.transform.centerY;
        this.ctx.beginPath();
        this.ctx.moveTo(0, groundY);
        this.ctx.lineTo(this.canvas.width, groundY);
        this.ctx.stroke();
        
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Draw coordinate system axes
     */
    drawAxes() {
        this.ctx.strokeStyle = this.config.textColor;
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.5;
        
        // Center lines
        this.ctx.beginPath();
        this.ctx.moveTo(this.transform.centerX, 0);
        this.ctx.lineTo(this.transform.centerX, this.canvas.height);
        this.ctx.moveTo(0, this.transform.centerY);
        this.ctx.lineTo(this.canvas.width, this.transform.centerY);
        this.ctx.stroke();
        
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Draw the robot (pendulum + wheels + torque indicators)
     */
    drawRobot() {
        if (!this.robotState || !this.robotConfig) return;
        
        const { angle, position } = this.robotState;
        const { centerOfMassHeight } = this.robotConfig;
        
        // Calculate robot position on screen
        const basePos = this.transform.physicsToScreen(position, 0);
        const topPos = this.transform.physicsToScreen(
            position + Math.sin(angle) * centerOfMassHeight,
            Math.cos(angle) * centerOfMassHeight
        );
        
        // Draw robot body (pendulum)
        this.drawRobotBody(basePos, topPos);
        
        // Draw wheels
        this.drawWheels(basePos);
        
        // Draw motor torque indicators
        this.drawTorqueIndicators(basePos);
        
        // Draw angle indicator
        this.drawAngleIndicator(basePos, angle);
    }
    
    /**
     * Draw robot body as a pendulum
     * @param {Object} basePos - Base position {x, y}
     * @param {Object} topPos - Top position {x, y}
     */
    drawRobotBody(basePos, topPos) {
        // Main body line
        this.ctx.strokeStyle = this.config.robotColor;
        this.ctx.lineWidth = 6;
        this.ctx.beginPath();
        this.ctx.moveTo(basePos.x, basePos.y);
        this.ctx.lineTo(topPos.x, topPos.y);
        this.ctx.stroke();
        
        // Robot center of mass
        this.ctx.fillStyle = this.config.robotColor;
        this.ctx.beginPath();
        this.ctx.arc(topPos.x, topPos.y, 8, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Base connection point
        this.ctx.fillStyle = this.config.wheelColor;
        this.ctx.beginPath();
        this.ctx.arc(basePos.x, basePos.y, 6, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    /**
     * Draw robot wheels
     * @param {Object} basePos - Base position {x, y}
     */
    drawWheels(basePos) {
        const wheelRadius = this.transform.scaleLength(0.05); // 5cm wheel radius
        const wheelSpacing = this.transform.scaleLength(0.15); // 15cm between wheels
        
        // Left wheel
        this.ctx.fillStyle = this.config.wheelColor;
        this.ctx.beginPath();
        this.ctx.arc(basePos.x - wheelSpacing/2, basePos.y, wheelRadius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Right wheel
        this.ctx.beginPath();
        this.ctx.arc(basePos.x + wheelSpacing/2, basePos.y, wheelRadius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Wheel spokes for rotation indication
        this.ctx.strokeStyle = this.config.backgroundColor;
        this.ctx.lineWidth = 2;
        
        const time = performance.now() * 0.001;
        for (let i = 0; i < 2; i++) {
            const wheelX = basePos.x + (i === 0 ? -wheelSpacing/2 : wheelSpacing/2);
            const spokeAngle = time * 2 + i * Math.PI;
            
            this.ctx.beginPath();
            this.ctx.moveTo(wheelX, basePos.y);
            this.ctx.lineTo(
                wheelX + Math.cos(spokeAngle) * wheelRadius * 0.7,
                basePos.y + Math.sin(spokeAngle) * wheelRadius * 0.7
            );
            this.ctx.stroke();
        }
    }
    
    /**
     * Draw motor torque indicators
     * @param {Object} basePos - Base position {x, y}
     */
    drawTorqueIndicators(basePos) {
        if (Math.abs(this.motorTorque) < 0.1) return; // Don't show for very small torques
        
        const maxTorque = this.robotConfig?.motorStrength || 5.0;
        const torqueRatio = Math.abs(this.motorTorque) / maxTorque;
        const indicatorLength = torqueRatio * 40;
        
        // Color based on torque direction and magnitude
        const intensity = Math.min(1.0, torqueRatio);
        if (this.motorTorque > 0) {
            this.ctx.strokeStyle = `rgba(255, 107, 53, ${intensity})` // Orange for positive torque
        } else {
            this.ctx.strokeStyle = `rgba(53, 107, 255, ${intensity})` // Blue for negative torque
        }
        
        this.ctx.lineWidth = 4;
        this.ctx.lineCap = 'round';
        
        // Draw torque arrow
        const arrowY = basePos.y - 30;
        const arrowStartX = basePos.x - indicatorLength/2;
        const arrowEndX = basePos.x + indicatorLength/2;
        
        this.ctx.beginPath();
        this.ctx.moveTo(arrowStartX, arrowY);
        this.ctx.lineTo(arrowEndX, arrowY);
        this.ctx.stroke();
        
        // Arrow head indicating direction
        const headSize = 8;
        if (this.motorTorque > 0) {
            // Right arrow
            this.ctx.beginPath();
            this.ctx.moveTo(arrowEndX, arrowY);
            this.ctx.lineTo(arrowEndX - headSize, arrowY - headSize/2);
            this.ctx.moveTo(arrowEndX, arrowY);
            this.ctx.lineTo(arrowEndX - headSize, arrowY + headSize/2);
            this.ctx.stroke();
        } else {
            // Left arrow
            this.ctx.beginPath();
            this.ctx.moveTo(arrowStartX, arrowY);
            this.ctx.lineTo(arrowStartX + headSize, arrowY - headSize/2);
            this.ctx.moveTo(arrowStartX, arrowY);
            this.ctx.lineTo(arrowStartX + headSize, arrowY + headSize/2);
            this.ctx.stroke();
        }
    }
    
    /**
     * Draw angle indicator arc
     * @param {Object} basePos - Base position {x, y}
     * @param {number} angle - Current angle in radians
     */
    drawAngleIndicator(basePos, angle) {
        const arcRadius = 25;
        
        this.ctx.strokeStyle = this.config.robotColor;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.7;
        
        // Draw angle arc
        this.ctx.beginPath();
        this.ctx.arc(basePos.x, basePos.y, arcRadius, -Math.PI/2, -Math.PI/2 + angle, angle < 0);
        this.ctx.stroke();
        
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Draw UI overlays (text, metrics, debug info)
     */
    drawUI() {
        if (this.showRobotInfo) {
            this.drawRobotInfo();
        }
        
        if (this.showTrainingInfo) {
            this.drawTrainingInfo();
        }
        
        if (this.showPerformanceInfo) {
            this.drawPerformanceInfo();
        }
    }
    
    /**
     * Draw robot state information
     */
    drawRobotInfo() {
        if (!this.robotState) return;
        
        const { angle, angularVelocity, position, velocity } = this.robotState;
        
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.font = '14px monospace';
        this.ctx.textAlign = 'left';
        
        const infoX = 20;
        let infoY = 30;
        const lineHeight = 18;
        
        // Background for better readability
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(infoX - 10, infoY - 15, 200, 110);
        
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.fillText('Robot State:', infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Angle: ${(angle * 180 / Math.PI).toFixed(1)}°`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Angular Vel: ${angularVelocity.toFixed(2)} rad/s`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Position: ${position.toFixed(2)} m`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Velocity: ${velocity.toFixed(2)} m/s`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Motor Torque: ${this.motorTorque.toFixed(2)} N⋅m`, infoX, infoY);
    }
    
    /**
     * Draw training information
     */
    drawTrainingInfo() {
        const metrics = this.trainingMetrics;
        
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.font = '14px monospace';
        this.ctx.textAlign = 'right';
        
        const infoX = this.canvas.width - 20;
        let infoY = 30;
        const lineHeight = 18;
        
        // Background for better readability
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(infoX - 190, infoY - 15, 200, 130);
        
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.fillText('Training Status:', infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Mode: ${metrics.trainingMode}`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Episode: ${metrics.episode}`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Step: ${metrics.step}`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Reward: ${metrics.reward.toFixed(1)}`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Best: ${metrics.bestReward.toFixed(1)}`, infoX, infoY);
        infoY += lineHeight;
        
        this.ctx.fillText(`Epsilon: ${metrics.epsilon.toFixed(3)}`, infoX, infoY);
    }
    
    /**
     * Draw performance information
     */
    drawPerformanceInfo() {
        const perfMetrics = this.performance.getMetrics();
        
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.font = '12px monospace';
        this.ctx.textAlign = 'left';
        
        const infoX = 20;
        const infoY = this.canvas.height - 20;
        
        // Background for better readability
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(infoX - 10, infoY - 35, 200, 45);
        
        // Color code FPS based on performance
        const fpsColor = this.performance.isPerformanceGood() ? '#00ff88' : '#ff4444';
        this.ctx.fillStyle = fpsColor;
        this.ctx.fillText(`FPS: ${perfMetrics.fps}`, infoX, infoY - 20);
        
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.fillText(`Frame: ${perfMetrics.avgFrameTime}ms (${perfMetrics.minFrameTime}-${perfMetrics.maxFrameTime})`, infoX, infoY);
    }
    
    /**
     * Toggle UI element visibility
     * @param {string} element - Element to toggle ('robot', 'training', 'performance')
     */
    toggleUI(element) {
        switch (element) {
            case 'robot':
                this.showRobotInfo = !this.showRobotInfo;
                break;
            case 'training':
                this.showTrainingInfo = !this.showTrainingInfo;
                break;
            case 'performance':
                this.showPerformanceInfo = !this.showPerformanceInfo;
                break;
        }
    }
    
    /**
     * Update renderer configuration
     * @param {Object} newConfig - Configuration updates
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }
    
    /**
     * Get renderer statistics
     * @returns {Object} Renderer stats
     */
    getStats() {
        return {
            isRendering: this.isRendering,
            canvasSize: { width: this.canvas.width, height: this.canvas.height },
            performance: this.performance.getMetrics(),
            targetFPS: this.config.targetFPS,
            hasRobotState: !!this.robotState,
            config: this.config
        };
    }
    
    /**
     * Cleanup renderer resources
     */
    destroy() {
        this.stop();
        this.canvas = null;
        this.ctx = null;
        this.transform = null;
        this.performance = null;
        console.log('Renderer destroyed');
    }
}

/**
 * Utility function to create a renderer with default configuration
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Object} config - Configuration overrides
 * @returns {Renderer} New renderer instance
 */
export function createRenderer(canvas, config = {}) {
    return new Renderer(canvas, config);
}

/**
 * Utility function to create a high-performance renderer
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @returns {Renderer} High-performance renderer
 */
export function createHighPerformanceRenderer(canvas) {
    return new Renderer(canvas, {
        targetFPS: 60,
        showGrid: false,
        showDebugInfo: true,
        showPerformance: true
    });
}

/**
 * Utility function to create a debug renderer with all visualizations
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @returns {Renderer} Debug renderer
 */
export function createDebugRenderer(canvas) {
    return new Renderer(canvas, {
        targetFPS: 30,
        showGrid: true,
        showDebugInfo: true,
        showPerformance: true,
        gridColor: '#2a2a2a'
    });
}