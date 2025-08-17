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
        this.physicsScale = 400; // pixels per meter
        this.centerX = canvasWidth / 2;
        this.centerY = canvasHeight * 0.7; // Ground line at 70% down
        this.zoomLevel = 1.0; // Zoom level multiplier
        
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
            x: this.centerX + x * this.physicsScale * this.zoomLevel,
            y: this.centerY - y * this.physicsScale * this.zoomLevel // Flip Y axis
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
            x: (screenX - this.centerX) / (this.physicsScale * this.zoomLevel),
            y: (this.centerY - screenY) / (this.physicsScale * this.zoomLevel)
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
        return length * this.physicsScale * this.zoomLevel;
    }
    
    /**
     * Get physics coordinate bounds
     * @returns {Object} {minX, maxX, minY, maxY} physics bounds in meters
     */
    getPhysicsBounds() {
        return {
            minX: this.minX,
            maxX: this.maxX,
            minY: this.minY,
            maxY: this.maxY
        };
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
     * Get physics coordinate bounds (for position-based episode termination)
     * @returns {Object} {minX, maxX, minY, maxY} physics bounds in meters
     */
    getPhysicsBounds() {
        return this.transform.getPhysicsBounds();
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
        console.log('Robot updated:', state, config);
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
        } else {
            // Debug: Show why robot is not being drawn
            if (!this.robotState) {
                console.log('Robot not drawn: robotState is null');
            }
            if (!this.robotConfig) {
                console.log('Robot not drawn: robotConfig is null');
            }
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
     * Draw realistic ground surface
     */
    drawGround() {
        const groundY = this.transform.centerY;
        
        // Ground surface - more visible with texture pattern
        this.ctx.fillStyle = '#2a2a2a'; // Dark gray surface
        this.ctx.fillRect(0, groundY, this.canvas.width, this.canvas.height - groundY);
        
        // Add a textured pattern to make rolling more visible
        // Draw repeated chevron/arrow pattern on the ground
        this.ctx.strokeStyle = '#404040';
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
        
        const patternSpacing = 60; // Distance between pattern elements
        const patternHeight = 15;  // Height of chevron pattern
        
        // Calculate pattern offset based on camera/view position
        const startX = -patternSpacing;
        const endX = this.canvas.width + patternSpacing;
        
        for (let x = startX; x < endX; x += patternSpacing) {
            // Draw chevron/arrow pattern pointing right
            this.ctx.beginPath();
            this.ctx.moveTo(x, groundY + 5);
            this.ctx.lineTo(x + patternHeight, groundY + 5 + patternHeight/2);
            this.ctx.lineTo(x, groundY + 5 + patternHeight);
            this.ctx.stroke();
            
            // Add small tick marks for finer measurement
            for (let i = 1; i < 4; i++) {
                const tickX = x + (patternSpacing * i / 4);
                this.ctx.beginPath();
                this.ctx.moveTo(tickX, groundY + 2);
                this.ctx.lineTo(tickX, groundY + 8);
                this.ctx.stroke();
            }
        }
        
        // Ground line - top edge (more prominent)
        this.ctx.strokeStyle = '#606060';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(0, groundY);
        this.ctx.lineTo(this.canvas.width, groundY);
        this.ctx.stroke();
        
        // Add subtle grid lines on ground for distance reference
        this.ctx.strokeStyle = 'rgba(64, 64, 64, 0.5)';
        this.ctx.lineWidth = 1;
        const wheelCircumference = 2 * Math.PI * (this.robotConfig?.wheelRadius || 0.12);
        const circumferencePixels = this.transform.scaleLength(wheelCircumference);
        
        // Draw lines at wheel circumference intervals
        const centerX = this.transform.centerX;
        const robotPos = this.robotState?.position || 0;
        const robotScreenX = this.transform.physicsToScreen(robotPos, 0).x;
        
        // Calculate offset to align with wheel position
        const offset = robotScreenX % circumferencePixels;
        
        for (let x = offset; x < this.canvas.width; x += circumferencePixels) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, groundY);
            this.ctx.lineTo(x, groundY + 30);
            this.ctx.stroke();
        }
        
        for (let x = offset - circumferencePixels; x >= 0; x -= circumferencePixels) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, groundY);
            this.ctx.lineTo(x, groundY + 30);
            this.ctx.stroke();
        }
        
        // Draw wheel contact point marker (optional debug visualization)
        if (this.robotState && this.config.showDebugInfo) {
            const contactX = this.transform.physicsToScreen(this.robotState.position, 0).x;
            const contactY = this.transform.centerY;
            
            this.ctx.strokeStyle = 'rgba(255, 255, 0, 0.3)';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(contactX - 10, contactY);
            this.ctx.lineTo(contactX + 10, contactY);
            this.ctx.stroke();
        }
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
        
        // Debug: Log that we're drawing the robot
        console.log('Drawing robot at position:', position, 'angle:', angle);
        
        // FIXED: Wheels sit on ground (center is above ground by radius)
        // Body extends upward from wheel center
        const wheelRadius = this.robotConfig?.wheelRadius || 0.12; // Use actual radius from config
        const wheelPos = this.transform.physicsToScreen(position, wheelRadius);
        const bodyTopPos = this.transform.physicsToScreen(
            position + Math.sin(angle) * centerOfMassHeight,
            wheelRadius + Math.cos(angle) * centerOfMassHeight
        );
        
        // Draw robot body (pendulum extending upward from wheels)
        this.drawRobotBody(wheelPos, bodyTopPos);
        
        // Draw wheels at ground level
        this.drawWheels(wheelPos);
        
        // Draw motor torque indicators
        this.drawTorqueIndicators(wheelPos);
        
        // Draw angle indicator
        this.drawAngleIndicator(wheelPos, angle);
    }
    
    /**
     * Draw robot body (side view - inverted pendulum balancing robot)
     * @param {Object} wheelPos - Wheel position at ground level {x, y}
     * @param {Object} topPos - Top position of pendulum {x, y}
     */
    drawRobotBody(wheelPos, topPos) {
        const bodyWidth = this.transform.scaleLength(0.24); // Robot body width (increased)
        const bodyHeight = Math.sqrt(Math.pow(topPos.x - wheelPos.x, 2) + Math.pow(topPos.y - wheelPos.y, 2));
        const angle = Math.atan2(topPos.x - wheelPos.x, wheelPos.y - topPos.y);
        
        // Main robot body - rectangular chassis extending upward from wheels
        this.ctx.save();
        this.ctx.translate(wheelPos.x, wheelPos.y);
        this.ctx.rotate(angle);
        
        // Robot body - main chassis (extends upward from wheel position)
        this.ctx.fillStyle = '#2a4d3a'; // Dark green
        this.ctx.fillRect(-bodyWidth/2, -bodyHeight, bodyWidth, bodyHeight * 0.9);
        
        // Robot body outline
        this.ctx.strokeStyle = '#1a3d2a';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(-bodyWidth/2, -bodyHeight, bodyWidth, bodyHeight * 0.9);
        
        // Center of mass indicator - bright marker at top of pendulum
        this.ctx.fillStyle = '#ff4444';
        this.ctx.beginPath();
        this.ctx.arc(0, -bodyHeight * 0.9, 15, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.ctx.restore();
        
        // Axle connection - small cylinder connecting to wheel
        this.ctx.fillStyle = '#606060';
        this.ctx.beginPath();
        this.ctx.arc(wheelPos.x, wheelPos.y, 4, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    /**
     * Draw robot wheel (side view - single wheel visible)
     * @param {Object} wheelPos - Wheel position at ground level {x, y}
     */
    drawWheels(wheelPos) {
        // Use actual wheel radius from physics configuration
        const physicsWheelRadius = this.robotConfig?.wheelRadius || 0.12;
        const wheelRadius = this.transform.scaleLength(physicsWheelRadius);
        
        // Main wheel body - dark gray with metallic look
        this.ctx.fillStyle = '#404040';
        this.ctx.beginPath();
        this.ctx.arc(wheelPos.x, wheelPos.y, wheelRadius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Wheel rim - lighter gray
        this.ctx.strokeStyle = '#606060';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(wheelPos.x, wheelPos.y, wheelRadius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Tire tread - black outer ring
        this.ctx.strokeStyle = '#202020';
        this.ctx.lineWidth = 6;
        this.ctx.beginPath();
        this.ctx.arc(wheelPos.x, wheelPos.y, wheelRadius - 2, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Wheel hub - central circle
        this.ctx.fillStyle = '#707070';
        this.ctx.beginPath();
        this.ctx.arc(wheelPos.x, wheelPos.y, wheelRadius * 0.3, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Rotating spokes for motion indication
        // Use actual wheel angle from physics for correct visual rotation
        // When robot moves right (positive velocity), wheel rotates clockwise
        // When robot moves left (negative velocity), wheel rotates counter-clockwise  
        const wheelRotation = (this.robotState.wheelAngle || 0);
        const spokeAngle = wheelRotation;
        
        this.ctx.strokeStyle = '#808080';
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
        
        // Draw 4 spokes
        for (let i = 0; i < 4; i++) {
            const angle = spokeAngle + (i * Math.PI / 2);
            const innerRadius = wheelRadius * 0.4;
            const outerRadius = wheelRadius * 0.8;
            
            this.ctx.beginPath();
            this.ctx.moveTo(
                wheelPos.x + Math.cos(angle) * innerRadius,
                wheelPos.y + Math.sin(angle) * innerRadius
            );
            this.ctx.lineTo(
                wheelPos.x + Math.cos(angle) * outerRadius,
                wheelPos.y + Math.sin(angle) * outerRadius
            );
            this.ctx.stroke();
        }
        
        // Add a reference mark on the wheel to make rotation more obvious
        // Draw a colored dot on the wheel edge
        const markerAngle = spokeAngle;
        const markerRadius = wheelRadius * 0.9;
        this.ctx.fillStyle = '#ff4444'; // Red marker
        this.ctx.beginPath();
        this.ctx.arc(
            wheelPos.x + Math.cos(markerAngle) * markerRadius,
            wheelPos.y + Math.sin(markerAngle) * markerRadius,
            4, 0, Math.PI * 2
        );
        this.ctx.fill();
        
        // Add a secondary marker 180 degrees opposite for better visual feedback
        const oppositeMarkerAngle = spokeAngle + Math.PI;
        this.ctx.fillStyle = '#4444ff'; // Blue marker
        this.ctx.beginPath();
        this.ctx.arc(
            wheelPos.x + Math.cos(oppositeMarkerAngle) * markerRadius,
            wheelPos.y + Math.sin(oppositeMarkerAngle) * markerRadius,
            3, 0, Math.PI * 2
        );
        this.ctx.fill();
    }
    
    /**
     * Draw motor torque indicators
     * @param {Object} wheelPos - Wheel position at ground level {x, y}
     */
    drawTorqueIndicators(wheelPos) {
        if (Math.abs(this.motorTorque) < 0.1) return; // Don't show for very small torques
        
        const maxTorque = this.robotConfig?.motorStrength || 5.0;
        const torqueRatio = Math.abs(this.motorTorque) / maxTorque;
        const indicatorLength = torqueRatio * 80;
        
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
        const arrowY = wheelPos.y - 30;
        const arrowStartX = wheelPos.x - indicatorLength/2;
        const arrowEndX = wheelPos.x + indicatorLength/2;
        
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
     * @param {Object} wheelPos - Wheel position at ground level {x, y}
     * @param {number} angle - Current angle in radians
     */
    drawAngleIndicator(wheelPos, angle) {
        const arcRadius = 50;
        
        this.ctx.strokeStyle = this.config.robotColor;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.7;
        
        // Draw angle arc
        this.ctx.beginPath();
        this.ctx.arc(wheelPos.x, wheelPos.y, arcRadius, -Math.PI/2, -Math.PI/2 + angle, angle < 0);
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
        this.ctx.font = '18px monospace';
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
        this.ctx.font = '18px monospace';
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
        this.ctx.font = '16px monospace';
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
     * Zoom in by 0.1 (clamped to max 3.0)
     */
    zoomIn() {
        this.transform.zoomLevel = Math.min(3.0, this.transform.zoomLevel + 0.1);
    }
    
    /**
     * Zoom out by 0.1 (clamped to min 0.5)
     */
    zoomOut() {
        this.transform.zoomLevel = Math.max(0.5, this.transform.zoomLevel - 0.1);
    }
    
    /**
     * Reset zoom to 1.0
     */
    resetZoom() {
        this.transform.zoomLevel = 1.0;
    }
    
    /**
     * Get current zoom level
     * @returns {number} Current zoom level
     */
    getZoomLevel() {
        return this.transform.zoomLevel;
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