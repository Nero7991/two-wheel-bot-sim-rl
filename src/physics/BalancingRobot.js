/**
 * Physics simulation for a two-wheel balancing robot using inverted pendulum model
 * 
 * This implementation provides:
 * - RobotState class for state representation
 * - BalancingRobot class with configurable physics parameters
 * - Fixed timestep simulation (20ms / 50Hz)
 * - Reward function for reinforcement learning
 * - Parameter validation and boundary conditions
 */

/**
 * Represents the complete state of the balancing robot
 */
export class RobotState {
    /**
     * @param {number} angle - Robot tilt angle in radians (0 = upright, positive = tilting forward)
     * @param {number} angularVelocity - Angular velocity in rad/s
     * @param {number} position - Horizontal position in meters
     * @param {number} velocity - Horizontal velocity in m/s
     */
    constructor(angle = 0, angularVelocity = 0, position = 0, velocity = 0) {
        this.angle = angle;
        this.angularVelocity = angularVelocity;
        this.position = position;
        this.velocity = velocity;
    }

    /**
     * Create a copy of this state
     * @returns {RobotState} New state instance with same values
     */
    clone() {
        return new RobotState(this.angle, this.angularVelocity, this.position, this.velocity);
    }

    /**
     * Get normalized state for neural network input
     * @returns {Float32Array} [normalizedAngle, normalizedAngularVelocity]
     */
    getNormalizedInputs() {
        // Normalize angle to [-1, 1] range (assuming max angle is π/3)
        const normalizedAngle = this.angle / (Math.PI / 3);
        // Normalize angular velocity to [-1, 1] range (assuming max is 10 rad/s)
        const normalizedAngularVelocity = Math.max(-1, Math.min(1, this.angularVelocity / 10));
        
        return new Float32Array([normalizedAngle, normalizedAngularVelocity]);
    }

    /**
     * Check if the robot has failed (fallen over)
     * @returns {boolean} True if robot has fallen
     */
    hasFailed() {
        return Math.abs(this.angle) > Math.PI / 3; // 60 degrees
    }
}

/**
 * Physics simulation for a two-wheel balancing robot
 * Implements inverted pendulum dynamics with motor control
 */
export class BalancingRobot {
    /**
     * @param {Object} config - Robot configuration parameters
     * @param {number} config.mass - Robot mass in kg (0.5 - 3.0, default: 1.0)
     * @param {number} config.centerOfMassHeight - Height of center of mass in meters (0.2 - 1.0, default: 0.5)
     * @param {number} config.motorStrength - Maximum motor torque in N⋅m (2 - 10, default: 5)
     * @param {number} config.friction - Friction coefficient (0 - 1, default: 0.1)
     * @param {number} config.damping - Angular damping coefficient (0 - 1, default: 0.05)
     * @param {number} config.timestep - Physics timestep in seconds (default: 0.02 = 20ms)
     */
    constructor(config = {}) {
        // Validate and set parameters with defaults
        this.mass = this._validateParameter(config.mass, 1.0, 0.5, 3.0, 'mass');
        this.centerOfMassHeight = this._validateParameter(config.centerOfMassHeight, 0.5, 0.2, 1.0, 'centerOfMassHeight');
        this.motorStrength = this._validateParameter(config.motorStrength, 5.0, 2.0, 10.0, 'motorStrength');
        this.friction = this._validateParameter(config.friction, 0.1, 0.0, 1.0, 'friction');
        this.damping = this._validateParameter(config.damping, 0.05, 0.0, 1.0, 'damping');
        this.timestep = this._validateParameter(config.timestep, 0.02, 0.001, 0.1, 'timestep');

        // Physical constants
        this.gravity = 9.81; // m/s²
        this.momentOfInertia = this.mass * this.centerOfMassHeight * this.centerOfMassHeight / 3; // Simplified MOI

        // Initialize state
        this.state = new RobotState();
        this.currentMotorTorque = 0;
        
        // Statistics
        this.stepCount = 0;
        this.totalReward = 0;
    }

    /**
     * Validate a parameter is within acceptable bounds
     * @private
     */
    _validateParameter(value, defaultValue, min, max, name) {
        if (value === undefined || value === null) {
            return defaultValue;
        }
        if (typeof value !== 'number' || isNaN(value)) {
            console.warn(`Invalid ${name}: ${value}, using default: ${defaultValue}`);
            return defaultValue;
        }
        if (value < min || value > max) {
            console.warn(`${name} out of range [${min}, ${max}]: ${value}, clamping`);
            return Math.max(min, Math.min(max, value));
        }
        return value;
    }

    /**
     * Reset the robot to initial state
     * @param {Object} initialState - Optional initial state values
     */
    reset(initialState = {}) {
        this.state = new RobotState(
            this._normalizeAngle(initialState.angle || 0),
            initialState.angularVelocity || 0,
            initialState.position || 0,
            initialState.velocity || 0
        );
        this.currentMotorTorque = 0;
        this.stepCount = 0;
        this.totalReward = 0;
    }

    /**
     * Apply motor torque and update physics simulation
     * @param {number} motorTorque - Motor torque in N⋅m (clamped to ±motorStrength)
     * @returns {Object} {state: RobotState, reward: number, done: boolean}
     */
    step(motorTorque) {
        // Check if robot has already failed before physics update
        if (this.state.hasFailed()) {
            return {
                state: this.state.clone(),
                reward: -100,
                done: true
            };
        }

        // Clamp motor torque to valid range
        this.currentMotorTorque = Math.max(-this.motorStrength, Math.min(this.motorStrength, motorTorque));

        // Store previous state for reward calculation
        const prevState = this.state.clone();

        // Update physics using Euler integration
        this._updatePhysics();

        // Normalize angle to [-π, π] range
        this.state.angle = this._normalizeAngle(this.state.angle);

        // Calculate reward
        const reward = this._calculateReward(prevState);
        this.totalReward += reward;

        // Check if episode is done after physics update
        const done = this.state.hasFailed();

        this.stepCount++;

        return {
            state: this.state.clone(),
            reward: reward,
            done: done
        };
    }

    /**
     * Update physics simulation using Euler integration
     * @private
     */
    _updatePhysics() {
        const dt = this.timestep;
        
        // Calculate forces and torques
        const gravityTorque = this.mass * this.gravity * this.centerOfMassHeight * Math.sin(this.state.angle);
        const dampingTorque = -this.damping * this.state.angularVelocity;
        const totalTorque = this.currentMotorTorque - gravityTorque + dampingTorque;

        // Calculate angular acceleration
        const angularAcceleration = totalTorque / this.momentOfInertia;

        // Update angular motion (Euler integration)
        this.state.angularVelocity += angularAcceleration * dt;
        this.state.angle += this.state.angularVelocity * dt;

        // Calculate horizontal forces
        // Motor force is proportional to torque (simplified wheel dynamics)
        const wheelRadius = 0.05; // 5cm wheel radius (simplified)
        const motorForce = this.currentMotorTorque / wheelRadius;
        const frictionForce = -this.friction * this.state.velocity * this.mass * this.gravity;
        const totalForce = motorForce + frictionForce;

        // Update horizontal motion (Euler integration)
        const acceleration = totalForce / this.mass;
        this.state.velocity += acceleration * dt;
        this.state.position += this.state.velocity * dt;
    }

    /**
     * Calculate reward function for reinforcement learning
     * @private
     * @param {RobotState} prevState - Previous state for comparison
     * @returns {number} Reward value
     */
    _calculateReward(prevState) {
        let reward = 0;

        // Check if robot has failed
        if (this.state.hasFailed()) {
            return -100; // Large penalty for failure
        }

        // Alive bonus for staying upright
        reward += 1.0;

        // Angle penalty - penalize deviation from upright
        reward += -Math.abs(this.state.angle) * 10;

        // Angular velocity penalty - penalize fast spinning
        reward += -Math.abs(this.state.angularVelocity) * 0.5;

        // Motor effort penalty - encourage efficiency
        reward += -Math.abs(this.currentMotorTorque) * 0.1;

        return reward;
    }

    /**
     * Normalize angle to [-π, π] range
     * @private
     */
    _normalizeAngle(angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    /**
     * Get current robot state
     * @returns {RobotState} Current state
     */
    getState() {
        return this.state.clone();
    }

    /**
     * Get robot configuration parameters
     * @returns {Object} Configuration object
     */
    getConfig() {
        return {
            mass: this.mass,
            centerOfMassHeight: this.centerOfMassHeight,
            motorStrength: this.motorStrength,
            friction: this.friction,
            damping: this.damping,
            timestep: this.timestep
        };
    }

    /**
     * Get simulation statistics
     * @returns {Object} Statistics object
     */
    getStats() {
        return {
            stepCount: this.stepCount,
            totalReward: this.totalReward,
            currentMotorTorque: this.currentMotorTorque,
            simulationTime: this.stepCount * this.timestep
        };
    }

    /**
     * Check if simulation is stable (no NaN or infinite values)
     * @returns {boolean} True if simulation is stable
     */
    isStable() {
        return (
            isFinite(this.state.angle) &&
            isFinite(this.state.angularVelocity) &&
            isFinite(this.state.position) &&
            isFinite(this.state.velocity) &&
            isFinite(this.currentMotorTorque)
        );
    }
}

/**
 * Utility function to create a robot with default parameters
 * @param {Object} overrides - Parameter overrides
 * @returns {BalancingRobot} New robot instance
 */
export function createDefaultRobot(overrides = {}) {
    return new BalancingRobot(overrides);
}

/**
 * Utility function to create a lightweight robot for fast training
 * @returns {BalancingRobot} Robot configured for training speed
 */
export function createTrainingRobot() {
    return new BalancingRobot({
        mass: 1.0,
        centerOfMassHeight: 0.3,
        motorStrength: 3.0,
        friction: 0.05,
        damping: 0.02,
        timestep: 0.02
    });
}

/**
 * Utility function to create a realistic robot for demonstration
 * @returns {BalancingRobot} Robot configured for realism
 */
export function createRealisticRobot() {
    return new BalancingRobot({
        mass: 1.5,
        centerOfMassHeight: 0.4,
        motorStrength: 4.0,
        friction: 0.15,
        damping: 0.08,
        timestep: 0.02
    });
}