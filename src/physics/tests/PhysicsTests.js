/**
 * Test suite for physics simulation stability and correctness
 */

import { RobotState, BalancingRobot, createDefaultRobot, createTrainingRobot, createRealisticRobot } from '../BalancingRobot.js';

/**
 * Test runner for physics simulation
 */
export class PhysicsTests {
    constructor() {
        this.testResults = [];
    }

    /**
     * Run all physics tests
     * @returns {Object} Test results summary
     */
    runAllTests() {
        console.log('Running Physics Simulation Tests...\n');
        
        this.testRobotStateClass();
        this.testBalancingRobotConstruction();
        this.testParameterValidation();
        this.testPhysicsStability();
        this.testRewardFunction();
        this.testBoundaryConditions();
        this.testResetFunctionality();
        this.testUtilityFunctions();
        this.testNumericalStability();
        this.testLongRunningSimulation();

        return this.summarizeResults();
    }

    /**
     * Test RobotState class functionality
     */
    testRobotStateClass() {
        const testName = 'RobotState Class';
        try {
            // Test constructor
            const state1 = new RobotState();
            this.assert(state1.angle === 0, 'Default constructor sets angle to 0');
            this.assert(state1.angularVelocity === 0, 'Default constructor sets angularVelocity to 0');
            this.assert(state1.position === 0, 'Default constructor sets position to 0');
            this.assert(state1.velocity === 0, 'Default constructor sets velocity to 0');

            const state2 = new RobotState(0.5, 1.0, 2.0, 3.0);
            this.assert(state2.angle === 0.5, 'Constructor with parameters sets angle correctly');
            this.assert(state2.angularVelocity === 1.0, 'Constructor with parameters sets angularVelocity correctly');

            // Test clone
            const cloned = state2.clone();
            this.assert(cloned.angle === state2.angle, 'Clone preserves angle');
            this.assert(cloned !== state2, 'Clone creates new instance');

            // Test normalized inputs
            const inputs = state2.getNormalizedInputs();
            this.assert(inputs instanceof Float32Array, 'getNormalizedInputs returns Float32Array');
            this.assert(inputs.length === 2, 'getNormalizedInputs returns 2 values');

            // Test failure detection
            const failedState = new RobotState(Math.PI / 2, 0, 0, 0); // 90 degrees
            this.assert(failedState.hasFailed(), 'Large angle is detected as failure');
            this.assert(!state1.hasFailed(), 'Small angle is not detected as failure');

            this.addTestResult(testName, true, 'All RobotState tests passed');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test BalancingRobot construction and configuration
     */
    testBalancingRobotConstruction() {
        const testName = 'BalancingRobot Construction';
        try {
            // Test default construction
            const robot1 = new BalancingRobot();
            this.assert(robot1.mass === 1.0, 'Default mass is 1.0');
            this.assert(robot1.centerOfMassHeight === 0.5, 'Default height is 0.5');
            this.assert(robot1.motorStrength === 5.0, 'Default motor strength is 5.0');

            // Test custom construction
            const config = {
                mass: 2.0,
                centerOfMassHeight: 0.8,
                motorStrength: 8.0,
                friction: 0.2,
                damping: 0.1
            };
            const robot2 = new BalancingRobot(config);
            this.assert(robot2.mass === 2.0, 'Custom mass is set correctly');
            this.assert(robot2.centerOfMassHeight === 0.8, 'Custom height is set correctly');

            // Test config retrieval
            const retrievedConfig = robot2.getConfig();
            this.assert(retrievedConfig.mass === 2.0, 'getConfig returns correct mass');
            this.assert(retrievedConfig.friction === 0.2, 'getConfig returns correct friction');

            this.addTestResult(testName, true, 'All construction tests passed');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test parameter validation
     */
    testParameterValidation() {
        const testName = 'Parameter Validation';
        try {
            // Test invalid parameters are clamped/defaulted
            const robot = new BalancingRobot({
                mass: -1,        // Invalid: below minimum
                centerOfMassHeight: 2.0,  // Invalid: above maximum
                motorStrength: 'invalid', // Invalid: not a number
                friction: 1.5,   // Invalid: above maximum
                damping: null    // Invalid: null value
            });

            this.assert(robot.mass >= 0.5 && robot.mass <= 3.0, 'Invalid mass is corrected');
            this.assert(robot.centerOfMassHeight >= 0.2 && robot.centerOfMassHeight <= 1.0, 'Invalid height is corrected');
            this.assert(robot.motorStrength >= 2.0 && robot.motorStrength <= 10.0, 'Invalid motor strength is corrected');
            this.assert(robot.friction >= 0.0 && robot.friction <= 1.0, 'Invalid friction is corrected');
            this.assert(robot.damping >= 0.0 && robot.damping <= 1.0, 'Invalid damping is corrected');

            this.addTestResult(testName, true, 'Parameter validation works correctly');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test physics simulation stability
     */
    testPhysicsStability() {
        const testName = 'Physics Stability';
        try {
            const robot = new BalancingRobot();
            
            // Test stable equilibrium (small angle, no torque)
            robot.reset({ angle: 0.1 });
            for (let i = 0; i < 100; i++) {
                const result = robot.step(0);
                this.assert(robot.isStable(), `Simulation stable at step ${i}`);
                this.assert(isFinite(result.reward), `Reward is finite at step ${i}`);
            }

            // Test with motor input
            robot.reset();
            for (let i = 0; i < 100; i++) {
                const motorTorque = Math.sin(i * 0.1) * 2; // Sinusoidal input
                const result = robot.step(motorTorque);
                this.assert(robot.isStable(), `Simulation stable with motor input at step ${i}`);
            }

            this.addTestResult(testName, true, 'Physics simulation is stable');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test reward function behavior
     */
    testRewardFunction() {
        const testName = 'Reward Function';
        try {
            const robot = new BalancingRobot();

            // Test upright robot gets positive reward
            robot.reset();
            const uprightResult = robot.step(0);
            this.assert(uprightResult.reward > 0, 'Upright robot receives positive reward');

            // Test tilted robot gets lower reward
            robot.reset({ angle: 0.5 });
            const tiltedResult = robot.step(0);
            this.assert(tiltedResult.reward < uprightResult.reward, 'Tilted robot receives lower reward');

            // Test failed robot gets large negative reward
            robot.reset({ angle: Math.PI / 2 });
            const failedResult = robot.step(0);
            this.assert(failedResult.reward === -100, 'Failed robot receives -100 reward');
            this.assert(failedResult.done, 'Failed robot ends episode');

            // Test motor effort penalty
            robot.reset();
            const noMotorResult = robot.step(0);
            const highMotorResult = robot.step(robot.motorStrength);
            this.assert(noMotorResult.reward > highMotorResult.reward, 'High motor effort reduces reward');

            this.addTestResult(testName, true, 'Reward function behaves correctly');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test boundary conditions and edge cases
     */
    testBoundaryConditions() {
        const testName = 'Boundary Conditions';
        try {
            const robot = new BalancingRobot();

            // Test motor torque clamping
            robot.reset();
            robot.step(1000); // Excessive torque
            const stats = robot.getStats();
            this.assert(Math.abs(stats.currentMotorTorque) <= robot.motorStrength, 'Motor torque is clamped to limits');

            // Test angle normalization (use angle that needs normalization but isn't a failure)
            robot.reset({ angle: 2 * Math.PI + 0.2 }); // ~372 degrees, will normalize to ~11.5 degrees
            robot.step(0);
            const state = robot.getState();
            this.assert(state.angle >= -Math.PI && state.angle <= Math.PI, 'Angle is normalized to [-π, π]');

            // Test failure threshold
            const criticalAngle = Math.PI / 3;
            robot.reset({ angle: criticalAngle - 0.01 });
            const almostFailedResult = robot.step(0);
            this.assert(!almostFailedResult.done, 'Robot just below failure threshold continues');

            robot.reset({ angle: criticalAngle + 0.01 });
            const failedResult = robot.step(0);
            this.assert(failedResult.done, 'Robot just above failure threshold fails');

            this.addTestResult(testName, true, 'Boundary conditions handled correctly');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test reset functionality
     */
    testResetFunctionality() {
        const testName = 'Reset Functionality';
        try {
            const robot = new BalancingRobot();

            // Run simulation for a while
            for (let i = 0; i < 50; i++) {
                robot.step(Math.random() * 2 - 1);
            }

            const stateBeforeReset = robot.getState();
            const statsBeforeReset = robot.getStats();

            // Reset and verify
            robot.reset();
            const stateAfterReset = robot.getState();
            const statsAfterReset = robot.getStats();

            this.assert(stateAfterReset.angle === 0, 'Reset sets angle to 0');
            this.assert(stateAfterReset.angularVelocity === 0, 'Reset sets angularVelocity to 0');
            this.assert(stateAfterReset.position === 0, 'Reset sets position to 0');
            this.assert(stateAfterReset.velocity === 0, 'Reset sets velocity to 0');
            this.assert(statsAfterReset.stepCount === 0, 'Reset clears step count');
            this.assert(statsAfterReset.totalReward === 0, 'Reset clears total reward');

            // Test reset with initial state
            robot.reset({ angle: 0.1, position: 1.0 });
            const customState = robot.getState();
            this.assert(customState.angle === 0.1, 'Reset with custom angle works');
            this.assert(customState.position === 1.0, 'Reset with custom position works');

            this.addTestResult(testName, true, 'Reset functionality works correctly');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test utility functions
     */
    testUtilityFunctions() {
        const testName = 'Utility Functions';
        try {
            // Test createDefaultRobot
            const defaultRobot = createDefaultRobot();
            this.assert(defaultRobot instanceof BalancingRobot, 'createDefaultRobot returns BalancingRobot instance');

            const overrideRobot = createDefaultRobot({ mass: 2.5 });
            this.assert(overrideRobot.mass === 2.5, 'createDefaultRobot applies overrides');

            // Test createTrainingRobot
            const trainingRobot = createTrainingRobot();
            this.assert(trainingRobot instanceof BalancingRobot, 'createTrainingRobot returns BalancingRobot instance');
            this.assert(trainingRobot.mass === 1.0, 'createTrainingRobot has expected mass');

            // Test createRealisticRobot
            const realisticRobot = createRealisticRobot();
            this.assert(realisticRobot instanceof BalancingRobot, 'createRealisticRobot returns BalancingRobot instance');
            this.assert(realisticRobot.mass === 1.5, 'createRealisticRobot has expected mass');

            this.addTestResult(testName, true, 'Utility functions work correctly');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test numerical stability with extreme inputs
     */
    testNumericalStability() {
        const testName = 'Numerical Stability';
        try {
            const robot = new BalancingRobot();

            // Test with extreme angles
            robot.reset({ angle: Math.PI - 0.001, angularVelocity: 50 });
            for (let i = 0; i < 10; i++) {
                robot.step(robot.motorStrength);
                this.assert(robot.isStable(), `Stable with extreme angle at step ${i}`);
            }

            // Test with very small timestep
            const fastRobot = new BalancingRobot({ timestep: 0.001 });
            fastRobot.reset({ angle: 0.1 });
            for (let i = 0; i < 100; i++) {
                fastRobot.step(1);
                this.assert(fastRobot.isStable(), `Stable with small timestep at step ${i}`);
            }

            this.addTestResult(testName, true, 'Numerical stability maintained');
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Test long-running simulation for stability
     */
    testLongRunningSimulation() {
        const testName = 'Long Running Simulation';
        try {
            const robot = new BalancingRobot();
            robot.reset();

            let totalReward = 0;
            let successfulSteps = 0;

            // Run for 1000 steps (20 seconds simulation time)
            for (let i = 0; i < 1000; i++) {
                // Simple balance controller
                const state = robot.getState();
                const motorTorque = -state.angle * 20 - state.angularVelocity * 5;
                
                const result = robot.step(motorTorque);
                totalReward += result.reward;

                if (!result.done) {
                    successfulSteps++;
                } else {
                    break;
                }

                this.assert(robot.isStable(), `Long simulation stable at step ${i}`);
            }

            this.assert(successfulSteps > 100, 'Long simulation runs for reasonable duration');
            this.assert(totalReward > -1000, 'Long simulation achieves reasonable total reward');

            this.addTestResult(testName, true, `Long simulation stable for ${successfulSteps} steps`);
        } catch (error) {
            this.addTestResult(testName, false, error.message);
        }
    }

    /**
     * Add a test result
     * @private
     */
    addTestResult(testName, passed, message) {
        const result = { testName, passed, message };
        this.testResults.push(result);
        
        const status = passed ? '✓ PASS' : '✗ FAIL';
        console.log(`${status}: ${testName} - ${message}`);
    }

    /**
     * Assert a condition is true
     * @private
     */
    assert(condition, message) {
        if (!condition) {
            throw new Error(`Assertion failed: ${message}`);
        }
    }

    /**
     * Summarize test results
     * @private
     */
    summarizeResults() {
        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.passed).length;
        const failedTests = totalTests - passedTests;

        console.log(`\n--- Test Summary ---`);
        console.log(`Total Tests: ${totalTests}`);
        console.log(`Passed: ${passedTests}`);
        console.log(`Failed: ${failedTests}`);
        console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);

        if (failedTests > 0) {
            console.log('\nFailed Tests:');
            this.testResults.filter(r => !r.passed).forEach(result => {
                console.log(`- ${result.testName}: ${result.message}`);
            });
        }

        return {
            totalTests,
            passedTests,
            failedTests,
            successRate: (passedTests / totalTests) * 100,
            results: this.testResults
        };
    }
}

/**
 * Run physics tests if this file is executed directly
 */
if (typeof window !== 'undefined') {
    // Browser environment - can be called from console
    window.runPhysicsTests = () => {
        const tests = new PhysicsTests();
        return tests.runAllTests();
    };
} else {
    // Node.js environment - run immediately
    const tests = new PhysicsTests();
    tests.runAllTests();
}