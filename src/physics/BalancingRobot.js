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
     * @param {number} wheelAngle - Wheel rotation angle in radians
     * @param {number} wheelVelocity - Wheel angular velocity in rad/s
     */
    constructor(angle = 0, angularVelocity = 0, position = 0, velocity = 0, wheelAngle = 0, wheelVelocity = 0) {
        this.angle = angle;
        this.angularVelocity = angularVelocity;
        this.position = position;
        this.velocity = velocity;
        this.wheelAngle = wheelAngle;        // Wheel rotation angle in radians
        this.wheelVelocity = wheelVelocity;     // Wheel angular velocity in rad/s
    }

    /**
     * Create a copy of this state
     * @returns {RobotState} New state instance with same values
     */
    clone() {
        return new RobotState(this.angle, this.angularVelocity, this.position, this.velocity, this.wheelAngle, this.wheelVelocity);
    }

    /**
     * Get normalized state for neural network input
     * @returns {Float32Array} [normalizedAngle, normalizedAngularVelocity]
     */
    getNormalizedInputs() {
        // Normalize angle to [-1, 1] range (assuming max angle is π/3) and clamp
        const normalizedAngle = Math.max(-1, Math.min(1, this.angle / (Math.PI / 3)));
        // Normalize angular velocity to [-1, 1] range (assuming max is 10 rad/s) and clamp
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
        this.centerOfMassHeight = this._validateParameter(config.centerOfMassHeight, 0.4, 0.2, 1.0, 'centerOfMassHeight');
        this.motorStrength = this._validateParameter(config.motorStrength, 5.0, 2.0, 10.0, 'motorStrength');
        this.friction = this._validateParameter(config.friction, 0.02, 0.0, 1.0, 'friction');
        this.damping = this._validateParameter(config.damping, 0.01, 0.0, 1.0, 'damping');
        this.timestep = this._validateParameter(config.timestep, 0.02, 0.001, 0.1, 'timestep');
        this.wheelRadius = this._validateParameter(config.wheelRadius, 0.12, 0.02, 0.20, 'wheelRadius');
        this.wheelMass = this._validateParameter(config.wheelMass, 0.2, 0.1, 1.0, 'wheelMass');
        this.wheelFriction = this._validateParameter(config.wheelFriction, 0.3, 0.0, 1.0, 'wheelFriction');
        
        // Reward function type: 'simple' (CartPole-style) or 'complex' (angle-proportional)
        this.rewardType = config.rewardType || 'simple';

        // Physical constants
        this.gravity = 9.81; // m/s²
        // For a uniform rod of length L rotating about one end: I = (1/3) * m * L^2
        // But for a point mass at height h: I = m * h^2
        // Use point mass approximation for inverted pendulum
        this.momentOfInertia = this.mass * this.centerOfMassHeight * this.centerOfMassHeight;
        
        // Wheel moment of inertia (for solid cylinder: I = 0.5 * m * r^2)
        this.wheelInertia = 0.5 * this.wheelMass * this.wheelRadius * this.wheelRadius;

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
            initialState.velocity || 0,
            initialState.wheelAngle || 0,
            initialState.wheelVelocity || 0
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
                reward: this.rewardType === 'simple' ? 0.0 : -10.0,
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
        const reward = this._calculateReward(prevState, this.currentMotorTorque);
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
        
        // Calculate pendulum torques
        const gravityTorque = this.mass * this.gravity * this.centerOfMassHeight * Math.sin(this.state.angle);
        const dampingTorque = -this.damping * this.state.angularVelocity;
        
        // Motor torque acts on wheels, creating opposite reaction on pendulum (Newton's 3rd law)
        const pendulumTorque = -this.currentMotorTorque + gravityTorque + dampingTorque;
        
        // Calculate angular acceleration of pendulum
        const angularAcceleration = pendulumTorque / this.momentOfInertia;
        
        // Update pendulum angular motion
        this.state.angularVelocity += angularAcceleration * dt;
        this.state.angle += this.state.angularVelocity * dt;
        
        // Simplified wheel dynamics - direct coupling for visual clarity
        // Motor torque creates force through wheels
        const motorForce = this.currentMotorTorque / this.wheelRadius;

        // Ground friction force (opposing motion)
        const normalForce = this.mass * this.gravity;
        const frictionForce = -this.friction * this.state.velocity * normalForce;

        // Total horizontal force
        const totalForce = motorForce + frictionForce;

        // Update horizontal motion
        const horizontalAcceleration = totalForce / this.mass;
        this.state.velocity += horizontalAcceleration * dt;
        this.state.position += this.state.velocity * dt;

        // Update wheel rotation to exactly match movement (no slip)
        // For perfect rolling: distance = radius * angle
        const distanceTraveled = this.state.velocity * dt;
        const wheelRotationChange = distanceTraveled / this.wheelRadius;
        this.state.wheelAngle += wheelRotationChange;

        // Keep wheel angle in reasonable range to prevent overflow
        while (this.state.wheelAngle > Math.PI * 2) this.state.wheelAngle -= Math.PI * 2;
        while (this.state.wheelAngle < -Math.PI * 2) this.state.wheelAngle += Math.PI * 2;

        // Set wheel velocity to match (for consistency)
        this.state.wheelVelocity = this.state.velocity / this.wheelRadius;
    }

    /**
     * Calculate reward function for reinforcement learning
     * @private
     * @param {RobotState} prevState - Previous state for comparison
     * @param {number} motorTorque - Action taken (motor torque applied)
     * @returns {number} Reward value
     */
    _calculateReward(prevState, motorTorque) {
        if (this.rewardType === 'simple') {
            // CARTPOLE-STYLE REWARD: Binary reward
            // Robot is upright = +1, Robot falls = 0
            if (this.state.hasFailed()) {
                return 0.0; // No reward for falling
            }
            return 1.0; // Reward for staying upright
        } else {
            // COMPLEX REWARD: Angle-proportional reward with failure penalty
            if (this.state.hasFailed()) {
                return -10.0; // Penalty for falling in complex mode
            }
            
            const angleError = Math.abs(this.state.angle);
            const maxAngle = Math.PI / 3; // 60 degrees before failure
            
            // Proportional reward: 1.0 when upright, 0.0 when at failure angle
            const uprightReward = 1.0 - (angleError / maxAngle);
            
            return uprightReward;
        }
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
            timestep: this.timestep,
            wheelRadius: this.wheelRadius,
            wheelMass: this.wheelMass,
            wheelFriction: this.wheelFriction,
            rewardType: this.rewardType
        };
    }

    /**
     * Update robot configuration parameters
     * @param {Object} config - New configuration parameters
     */
    updateConfig(config) {
        if (config.mass !== undefined) {
            this.mass = this._validateParameter(config.mass, this.mass, 0.5, 3.0, 'mass');
            // Recalculate moment of inertia when mass or height changes
            this.momentOfInertia = this.mass * this.centerOfMassHeight * this.centerOfMassHeight;
        }
        if (config.centerOfMassHeight !== undefined) {
            this.centerOfMassHeight = this._validateParameter(config.centerOfMassHeight, this.centerOfMassHeight, 0.2, 1.0, 'centerOfMassHeight');
            // Recalculate moment of inertia when mass or height changes
            this.momentOfInertia = this.mass * this.centerOfMassHeight * this.centerOfMassHeight;
        }
        if (config.motorStrength !== undefined) {
            this.motorStrength = this._validateParameter(config.motorStrength, this.motorStrength, 2.0, 10.0, 'motorStrength');
        }
        if (config.friction !== undefined) {
            this.friction = this._validateParameter(config.friction, this.friction, 0.0, 1.0, 'friction');
        }
        if (config.damping !== undefined) {
            this.damping = this._validateParameter(config.damping, this.damping, 0.0, 1.0, 'damping');
        }
        if (config.timestep !== undefined) {
            this.timestep = this._validateParameter(config.timestep, this.timestep, 0.001, 0.1, 'timestep');
        }
        if (config.wheelRadius !== undefined) {
            this.wheelRadius = this._validateParameter(config.wheelRadius, this.wheelRadius, 0.02, 0.20, 'wheelRadius');
            this.wheelInertia = 0.5 * this.wheelMass * this.wheelRadius * this.wheelRadius;
        }
        if (config.wheelMass !== undefined) {
            this.wheelMass = this._validateParameter(config.wheelMass, this.wheelMass, 0.1, 1.0, 'wheelMass');
            this.wheelInertia = 0.5 * this.wheelMass * this.wheelRadius * this.wheelRadius;
        }
        if (config.wheelFriction !== undefined) {
            this.wheelFriction = this._validateParameter(config.wheelFriction, this.wheelFriction, 0.0, 1.0, 'wheelFriction');
        }
        
        console.log('Robot configuration updated:', this.getConfig());
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
     * Set reward function type
     * @param {string} type - 'simple' (CartPole-style) or 'complex' (angle-proportional)
     */
    setRewardType(type) {
        if (type === 'simple' || type === 'complex') {
            this.rewardType = type;
            console.log(`Reward function changed to: ${type}`);
        } else {
            console.warn(`Invalid reward type: ${type}. Use 'simple' or 'complex'`);
        }
    }
    
    /**
     * Get current reward type
     * @returns {string} Current reward type ('simple' or 'complex')
     */
    getRewardType() {
        return this.rewardType;
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
        friction: 0.02,
        damping: 0.01,
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
        friction: 0.1,
        damping: 0.05,
        timestep: 0.02
    });
}