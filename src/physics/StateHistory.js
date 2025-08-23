/**
 * StateHistory class for managing a rolling buffer of past robot states
 * Used for multi-timestep neural network inputs to capture temporal patterns
 */
export class StateHistory {
    /**
     * Create a new StateHistory buffer
     * @param {number} maxTimesteps - Maximum number of timesteps to store (1-8)
     */
    constructor(maxTimesteps = 1) {
        this.maxTimesteps = Math.max(1, Math.min(8, maxTimesteps));
        this.history = [];
        this.currentTimesteps = 1; // Start with single timestep
        
        // Initialize with empty buffer - will be filled by robot on first reset
    }
    
    /**
     * Update the number of timesteps to use
     * @param {number} timesteps - Number of timesteps (1-8)
     */
    setTimesteps(timesteps) {
        this.currentTimesteps = Math.max(1, Math.min(8, timesteps));
        
        // If history is empty or we need more history than we have, pad with the last known state or zeros
        while (this.history.length < this.currentTimesteps) {
            if (this.history.length > 0) {
                // Duplicate the last known state
                const lastState = this.history[this.history.length - 1];
                this.history.push({ 
                    angle: lastState.angle, 
                    angularVelocity: lastState.angularVelocity,
                    timestamp: Date.now()
                });
            } else {
                // No history yet, use zeros as fallback
                this.history.push({ 
                    angle: 0, 
                    angularVelocity: 0,
                    timestamp: Date.now()
                });
            }
        }
    }
    
    /**
     * Add a new state to the history buffer
     * @param {number} angle - Current angle (or measured angle with offset)
     * @param {number} angularVelocity - Current angular velocity
     */
    addState(angle, angularVelocity) {
        // Add new state to the front
        this.history.unshift({
            angle: angle,
            angularVelocity: angularVelocity,
            timestamp: Date.now()
        });
        
        // Keep only the maximum number of timesteps
        if (this.history.length > this.maxTimesteps) {
            this.history.pop();
        }
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
            const normalizedAngularVelocity = Math.max(-1, Math.min(1, angularVelocity / 10));
            
            // Store in flattened array: [angle0, angVel0, angle1, angVel1, ...]
            inputs[i * 2] = normalizedAngle;
            inputs[i * 2 + 1] = normalizedAngularVelocity;
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
        // Clear the buffer - it will be filled with actual states by the caller
        // This allows the robot to pre-fill with appropriate initial states
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