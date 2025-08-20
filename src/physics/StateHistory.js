/**
 * StateHistory class for managing a rolling buffer of past robot states
 * Used for multi-timestep neural network inputs to capture temporal patterns
 */
export class StateHistory {
    
    /**
     * Apply non-linear transformation to angular velocity for better sensitivity
     * Maps small velocities to larger range for better discrimination
     * @param {number} angularVel - Raw angular velocity in rad/s
     * @returns {number} Transformed value in [-1, 1] range
     */
    _transformAngularVelocity(angularVel) {
        // Get absolute value and sign
        const sign = Math.sign(angularVel);
        const absVel = Math.abs(angularVel);
        
        // Define transformation parameters
        const lowRange = 3.0;  // Values 0-3 rad/s map to 0-0.7
        const highRange = 10.0; // Values 3-10 rad/s map to 0.7-1.0
        
        let transformed;
        if (absVel <= lowRange) {
            // Use square root for expansion of low values (0-3 → 0-0.7)
            // sqrt gives more sensitivity to small changes
            transformed = 0.7 * Math.sqrt(absVel / lowRange);
        } else if (absVel <= highRange) {
            // Linear mapping for medium values (3-10 → 0.7-1.0)
            const normalized = (absVel - lowRange) / (highRange - lowRange);
            transformed = 0.7 + 0.3 * normalized;
        } else {
            // Clip high values to 1.0
            transformed = 1.0;
        }
        
        // Apply sign and ensure bounds
        return Math.max(-1, Math.min(1, sign * transformed));
    }
    /**
     * Create a new StateHistory buffer
     * @param {number} maxTimesteps - Maximum number of timesteps to store (1-8)
     */
    constructor(maxTimesteps = 1) {
        this.maxTimesteps = Math.max(1, Math.min(8, maxTimesteps));
        this.history = [];
        this.currentTimesteps = 1; // Start with single timestep
        
        // Initialize with zeros
        this.reset();
    }
    
    /**
     * Update the number of timesteps to use
     * @param {number} timesteps - Number of timesteps (1-8)
     */
    setTimesteps(timesteps) {
        this.currentTimesteps = Math.max(1, Math.min(8, timesteps));
        
        // If we need more history than we have, pad with zeros
        while (this.history.length < this.currentTimesteps) {
            this.history.push({ angle: 0, angularVelocity: 0 });
        }
    }
    
    /**
     * Add a new state to the history buffer
     * @param {number} angle - Current angle (or measured angle with offset)
     * @param {number} angularVelocity - Current angular velocity
     * @param {number} derivative - Optional derivative of transformed angular velocity
     */
    addState(angle, angularVelocity, derivative = null) {
        // Add new state to the front
        this.history.unshift({
            angle: angle,
            angularVelocity: angularVelocity,
            derivative: derivative,
            timestamp: Date.now()
        });
        
        // Keep only the maximum number of timesteps
        if (this.history.length > this.maxTimesteps) {
            this.history.pop();
        }
    }
    
    /**
     * Get normalized inputs for offset-adaptive mode with derivative
     * Replaces oldest action with derivative of transformed angular velocity
     * @returns {Float32Array} Flattened array optimized for offset-adaptive training
     */
    getNormalizedInputsOffsetAdaptive() {
        // For offset-adaptive mode: [action0, angVel0, action1, angVel1, ..., actionN-1, angVelN-1, derivative, angVelN]
        // Replace oldest action (actionN-1) with current derivative
        const inputSize = this.currentTimesteps * 2;
        const inputs = new Float32Array(inputSize);
        
        // Fill most recent timesteps normally
        for (let i = 0; i < this.currentTimesteps - 1; i++) {
            let angle = 0; // This is actually previous action for offset-adaptive
            let angularVelocity = 0;
            
            // Use history if available, otherwise use zeros (padding)
            if (i < this.history.length) {
                angle = this.history[i].angle; // Previous action
                angularVelocity = this.history[i].angularVelocity;
            }
            
            // Normalize: action is already normalized, transform angular velocity
            const normalizedAction = Math.max(-1, Math.min(1, angle));
            const transformedAngularVelocity = this._transformAngularVelocity(angularVelocity);
            
            // Store in flattened array: [action0, angVel0, action1, angVel1, ...]
            inputs[i * 2] = normalizedAction;
            inputs[i * 2 + 1] = transformedAngularVelocity;
        }
        
        // For the oldest timestep, replace action with derivative
        const oldestIndex = this.currentTimesteps - 1;
        if (oldestIndex < this.history.length && this.history[0].derivative !== null) {
            // Use current derivative (from most recent state)
            const normalizedDerivative = Math.max(-1, Math.min(1, this.history[0].derivative / 10)); // Scale derivative
            const oldestAngularVel = oldestIndex < this.history.length ? this.history[oldestIndex].angularVelocity : 0;
            const transformedOldestAngVel = this._transformAngularVelocity(oldestAngularVel);
            
            inputs[oldestIndex * 2] = normalizedDerivative; // Derivative instead of action
            inputs[oldestIndex * 2 + 1] = transformedOldestAngVel;
        } else {
            // Fallback: use zeros for oldest timestep
            inputs[oldestIndex * 2] = 0;
            inputs[oldestIndex * 2 + 1] = 0;
        }
        
        return inputs;
    }
    
    /**
     * Get normalized inputs for neural network
     * @param {number} maxAngle - Maximum angle for normalization
     * @returns {Float32Array} Flattened array of [angle, angVel] for each timestep
     */
    getNormalizedInputs(maxAngle = Math.PI / 3) {
        // Create array with size = currentTimesteps * 2 (angle + angular velocity per timestep)
        const inputSize = this.currentTimesteps * 2;
        const inputs = new Float32Array(inputSize);
        
        // Fill the array with normalized values
        for (let i = 0; i < this.currentTimesteps; i++) {
            let angle = 0;
            let angularVelocity = 0;
            
            // Use history if available, otherwise use zeros (padding)
            if (i < this.history.length) {
                angle = this.history[i].angle;
                angularVelocity = this.history[i].angularVelocity;
            }
            
            // Normalize and clamp values
            const normalizedAngle = Math.max(-1, Math.min(1, angle / maxAngle));
            const transformedAngularVelocity = this._transformAngularVelocity(angularVelocity);
            
            // Store in flattened array: [angle0, angVel0, angle1, angVel1, ...]
            inputs[i * 2] = normalizedAngle;
            inputs[i * 2 + 1] = transformedAngularVelocity;
        }
        
        return inputs;
    }
    
    /**
     * Get raw history for debugging/visualization
     * @returns {Array} Array of state objects
     */
    getHistory() {
        return this.history.slice(0, this.currentTimesteps);
    }
    
    /**
     * Reset the history buffer
     */
    reset() {
        this.history = [];
        // Pre-fill with zeros for the maximum timesteps
        for (let i = 0; i < this.maxTimesteps; i++) {
            this.history.push({
                angle: 0,
                angularVelocity: 0,
                derivative: 0,
                timestamp: Date.now()
            });
        }
    }
    
    /**
     * Get statistics about the history
     * @returns {Object} Statistics object
     */
    getStats() {
        const validHistory = this.history.slice(0, this.currentTimesteps);
        
        // Calculate average angle and angular velocity
        let avgAngle = 0;
        let avgAngularVelocity = 0;
        let angleVariance = 0;
        
        for (const state of validHistory) {
            avgAngle += state.angle;
            avgAngularVelocity += state.angularVelocity;
        }
        
        avgAngle /= validHistory.length;
        avgAngularVelocity /= validHistory.length;
        
        // Calculate variance for angle (useful for detecting oscillations)
        for (const state of validHistory) {
            angleVariance += Math.pow(state.angle - avgAngle, 2);
        }
        angleVariance /= validHistory.length;
        
        return {
            currentTimesteps: this.currentTimesteps,
            historyLength: validHistory.length,
            averageAngle: avgAngle,
            averageAngularVelocity: avgAngularVelocity,
            angleVariance: angleVariance,
            angleStdDev: Math.sqrt(angleVariance)
        };
    }
}