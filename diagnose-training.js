/**
 * Comprehensive Training Diagnostic Script
 * Tests each component of the Q-learning system to identify learning failures
 */

import { BalancingRobot } from './src/physics/BalancingRobot.js';
import { QLearning } from './src/training/QLearning.js';

async function runComprehensiveDiagnostic() {
    console.log('=== COMPREHENSIVE TRAINING DIAGNOSTIC ===\n');
    
    // Test 1: State Representation
    console.log('1. TESTING STATE REPRESENTATION');
    await testStateRepresentation();
    
    // Test 2: Reward Function
    console.log('\n2. TESTING REWARD FUNCTION');
    await testRewardFunction();
    
    // Test 3: Action Selection
    console.log('\n3. TESTING ACTION SELECTION');
    await testActionSelection();
    
    // Test 4: Experience Collection
    console.log('\n4. TESTING EXPERIENCE COLLECTION');
    await testExperienceCollection();
    
    // Test 5: Q-learning Updates
    console.log('\n5. TESTING Q-LEARNING UPDATES');
    await testQLearningUpdates();
    
    // Test 6: Target Network
    console.log('\n6. TESTING TARGET NETWORK');
    await testTargetNetwork();
    
    // Test 7: End-to-End Learning
    console.log('\n7. TESTING END-TO-END LEARNING');
    await testEndToEndLearning();
}

async function testStateRepresentation() {
    const robot = new BalancingRobot();
    
    console.log('Testing state normalization...');
    
    const testStates = [
        { angle: 0, angularVelocity: 0 },           // Balanced
        { angle: 0.1, angularVelocity: 0 },        // Slight tilt
        { angle: 0.5, angularVelocity: 0 },        // Large tilt
        { angle: 0, angularVelocity: 1.0 },        // Angular velocity
        { angle: 0.2, angularVelocity: -0.5 },     // Complex state
    ];
    
    for (const testState of testStates) {
        robot.reset(testState);
        const state = robot.getState();
        const normalized = state.getNormalizedInputs();
        
        console.log(`  Angle: ${testState.angle.toFixed(3)} -> Normalized: ${normalized[0].toFixed(3)}`);
        console.log(`  AngVel: ${testState.angularVelocity.toFixed(3)} -> Normalized: ${normalized[1].toFixed(3)}`);
        
        // Check if normalization is in expected range
        if (Math.abs(normalized[0]) > 1.0 || Math.abs(normalized[1]) > 1.0) {
            console.log('  ❌ WARNING: Normalized values outside [-1,1] range!');
        }
        console.log('');
    }
}

async function testRewardFunction() {
    const robot = new BalancingRobot();
    
    console.log('Testing reward function signals...');
    
    const scenarios = [
        { angle: 0, angularVelocity: 0, motor: 0, desc: 'Perfect balance' },
        { angle: 0.1, angularVelocity: 0, motor: 0.5, desc: 'Slight tilt with correction' },
        { angle: 0.3, angularVelocity: 0, motor: 0, desc: 'Large tilt' },
        { angle: 0, angularVelocity: 2.0, motor: 0, desc: 'High angular velocity' },
        { angle: 1.1, angularVelocity: 0, motor: 0, desc: 'Failed (should be negative)' },
    ];
    
    for (const scenario of scenarios) {
        robot.reset({ angle: scenario.angle, angularVelocity: scenario.angularVelocity });
        const result = robot.step(scenario.motor);
        
        console.log(`  ${scenario.desc}:`);
        console.log(`    Angle: ${scenario.angle.toFixed(3)}, AngVel: ${scenario.angularVelocity.toFixed(3)}, Motor: ${scenario.motor.toFixed(1)}`);
        console.log(`    Reward: ${result.reward.toFixed(3)}, Done: ${result.done}`);
        
        // Check reward sanity
        if (scenario.desc.includes('Perfect') && result.reward < 0) {
            console.log('    ❌ WARNING: Perfect balance gives negative reward!');
        }
        if (scenario.desc.includes('Failed') && result.reward > 0) {
            console.log('    ❌ WARNING: Failed state gives positive reward!');
        }
        console.log('');
    }
}

async function testActionSelection() {
    const qlearning = new QLearning({ hiddenSize: 8 });
    await qlearning.initialize();
    
    console.log('Testing action selection...');
    
    const testState = new Float32Array([0.1, -0.05]); // Slight tilt left
    
    // Test deterministic selection (epsilon = 0)
    qlearning.hyperparams.epsilon = 0;
    const deterministicActions = [];
    for (let i = 0; i < 10; i++) {
        deterministicActions.push(qlearning.selectAction(testState, true));
    }
    console.log('  Deterministic actions (ε=0):', deterministicActions);
    
    if (new Set(deterministicActions).size !== 1) {
        console.log('  ❌ WARNING: Deterministic selection not consistent!');
    }
    
    // Test exploration (epsilon = 1)
    qlearning.hyperparams.epsilon = 1.0;
    const exploratoryActions = [];
    for (let i = 0; i < 50; i++) {
        exploratoryActions.push(qlearning.selectAction(testState, true));
    }
    
    const actionCounts = [0, 0, 0];
    exploratoryActions.forEach(a => actionCounts[a]++);
    console.log('  Exploratory action distribution (ε=1):', actionCounts);
    
    if (Math.max(...actionCounts) / Math.min(...actionCounts) > 3) {
        console.log('  ❌ WARNING: Action distribution heavily skewed!');
    }
    
    // Test Q-values
    const qValues = qlearning.getAllQValues(testState);
    console.log('  Q-values for test state:', Array.from(qValues).map(v => v.toFixed(4)));
}

async function testExperienceCollection() {
    const qlearning = new QLearning({ batchSize: 4 });
    await qlearning.initialize();
    
    console.log('Testing experience collection and replay buffer...');
    
    // Add some test experiences
    const experiences = [
        { state: [0.0, 0.0], action: 1, reward: 1.0, nextState: [0.1, 0.0], done: false },
        { state: [0.1, 0.0], action: 0, reward: 0.5, nextState: [0.05, 0.0], done: false },
        { state: [0.05, 0.0], action: 2, reward: 0.8, nextState: [0.2, 0.0], done: false },
        { state: [0.2, 0.0], action: 1, reward: -1.0, nextState: [0.0, 0.0], done: true },
    ];
    
    console.log('  Adding experiences to replay buffer...');
    for (const exp of experiences) {
        qlearning.replayBuffer.add(
            new Float32Array(exp.state),
            exp.action,
            exp.reward,
            new Float32Array(exp.nextState),
            exp.done
        );
    }
    
    console.log(`  Buffer size: ${qlearning.replayBuffer.size()}`);
    
    // Test sampling
    const sample = qlearning.replayBuffer.sample(3);
    console.log('  Sample experiences:');
    sample.forEach((exp, i) => {
        console.log(`    ${i}: Action=${exp.action}, Reward=${exp.reward.toFixed(2)}, Done=${exp.done}`);
    });
    
    if (sample.length === 0) {
        console.log('  ❌ WARNING: Replay buffer sampling failed!');
    }
}

async function testQLearningUpdates() {
    const qlearning = new QLearning({ 
        batchSize: 2, 
        learningRate: 0.1,  // Higher LR for visible changes
        targetUpdateFreq: 1 
    });
    await qlearning.initialize();
    
    console.log('Testing Q-learning weight updates...');
    
    // Get initial Q-values
    const testState = new Float32Array([0.1, -0.05]);
    const initialQValues = qlearning.getAllQValues(testState);
    console.log('  Initial Q-values:', Array.from(initialQValues).map(v => v.toFixed(4)));
    
    // Get initial weights
    const initialWeights = qlearning.qNetwork.getWeights();
    const initialOutputWeightSum = initialWeights.weightsHiddenOutput.reduce((sum, w) => sum + Math.abs(w), 0);
    console.log('  Initial output weight sum (abs):', initialOutputWeightSum.toFixed(6));
    
    // Add training experiences with clear learning signal
    const trainingExperiences = [
        { state: [0.1, 0.0], action: 0, reward: 2.0, nextState: [0.05, 0.0], done: false },  // Action 0 good
        { state: [0.1, 0.0], action: 2, reward: -1.0, nextState: [0.15, 0.0], done: false }, // Action 2 bad
    ];
    
    for (const exp of trainingExperiences) {
        qlearning.train(
            new Float32Array(exp.state),
            exp.action,
            exp.reward,
            new Float32Array(exp.nextState),
            exp.done
        );
    }
    
    // Check if weights changed
    const finalWeights = qlearning.qNetwork.getWeights();
    const finalOutputWeightSum = finalWeights.weightsHiddenOutput.reduce((sum, w) => sum + Math.abs(w), 0);
    console.log('  Final output weight sum (abs):', finalOutputWeightSum.toFixed(6));
    
    const weightChange = Math.abs(finalOutputWeightSum - initialOutputWeightSum);
    console.log('  Weight change magnitude:', weightChange.toFixed(6));
    
    if (weightChange < 1e-6) {
        console.log('  ❌ WARNING: Weights didn\'t change after training!');
    }
    
    // Check if Q-values changed
    const finalQValues = qlearning.getAllQValues(testState);
    console.log('  Final Q-values:', Array.from(finalQValues).map(v => v.toFixed(4)));
    
    const qValueChange = Math.max(...Array.from(finalQValues).map((v, i) => Math.abs(v - initialQValues[i])));
    console.log('  Max Q-value change:', qValueChange.toFixed(6));
    
    if (qValueChange < 1e-6) {
        console.log('  ❌ WARNING: Q-values didn\'t change after training!');
    }
}

async function testTargetNetwork() {
    const qlearning = new QLearning({ targetUpdateFreq: 1 });
    await qlearning.initialize();
    
    console.log('Testing target network updates...');
    
    const testState = new Float32Array([0.1, -0.05]);
    
    // Get initial target Q-values
    const initialTargetQValues = qlearning.targetNetwork.forward(testState);
    console.log('  Initial target Q-values:', Array.from(initialTargetQValues).map(v => v.toFixed(4)));
    
    // Modify main network weights
    const weights = qlearning.qNetwork.getWeights();
    for (let i = 0; i < weights.weightsHiddenOutput.length; i++) {
        weights.weightsHiddenOutput[i] += 0.1; // Small change
    }
    qlearning.qNetwork.setWeights(weights);
    
    const modifiedMainQValues = qlearning.qNetwork.forward(testState);
    console.log('  Modified main Q-values:', Array.from(modifiedMainQValues).map(v => v.toFixed(4)));
    
    // Force target network update
    qlearning._updateTargetNetwork();
    
    const updatedTargetQValues = qlearning.targetNetwork.forward(testState);
    console.log('  Updated target Q-values:', Array.from(updatedTargetQValues).map(v => v.toFixed(4)));
    
    // Check if target network updated
    const targetChange = Math.max(...Array.from(updatedTargetQValues).map((v, i) => Math.abs(v - initialTargetQValues[i])));
    console.log('  Target network change:', targetChange.toFixed(6));
    
    if (targetChange < 1e-6) {
        console.log('  ❌ WARNING: Target network didn\'t update!');
    }
}

async function testEndToEndLearning() {
    console.log('Testing end-to-end learning on simple task...');
    
    const robot = new BalancingRobot({ 
        mass: 1.0,
        centerOfMassHeight: 0.3,
        motorStrength: 3.0 
    });
    
    const qlearning = new QLearning({
        learningRate: 0.01,  // Higher learning rate for test
        epsilon: 0.3,
        batchSize: 4,
        targetUpdateFreq: 10,
        hiddenSize: 8
    });
    await qlearning.initialize();
    
    const episodeRewards = [];
    const qValueEvolution = [];
    
    for (let episode = 0; episode < 50; episode++) {
        robot.reset({ angle: (Math.random() - 0.5) * 0.2 }); // Small random initial angle
        let totalReward = 0;
        let step = 0;
        const maxSteps = 50;
        
        while (step < maxSteps && !robot.getState().hasFailed()) {
            const state = robot.getState().getNormalizedInputs();
            const actionIndex = qlearning.selectAction(state, true);
            const actions = [-1.0, 0.0, 1.0];
            
            const result = robot.step(actions[actionIndex]);
            totalReward += result.reward;
            
            if (step > 0) {
                // Train on experience
                qlearning.train(
                    lastState,
                    lastActionIndex, 
                    result.reward,
                    state,
                    result.done
                );
            }
            
            var lastState = state;
            var lastActionIndex = actionIndex;
            step++;
        }
        
        episodeRewards.push(totalReward);
        
        // Record Q-value for balanced state
        const balancedState = new Float32Array([0.0, 0.0]);
        const qValues = qlearning.getAllQValues(balancedState);
        qValueEvolution.push(Math.max(...Array.from(qValues)));
        
        if (episode % 10 === 0) {
            console.log(`  Episode ${episode}: Reward=${totalReward.toFixed(2)}, Steps=${step}, MaxQ=${qValueEvolution[episode].toFixed(4)}`);
        }
    }
    
    // Analyze learning trend
    const firstHalf = episodeRewards.slice(0, 25);
    const secondHalf = episodeRewards.slice(25);
    const firstAvg = firstHalf.reduce((a, b) => a + b) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b) / secondHalf.length;
    
    console.log(`  First 25 episodes avg reward: ${firstAvg.toFixed(3)}`);
    console.log(`  Last 25 episodes avg reward: ${secondAvg.toFixed(3)}`);
    console.log(`  Improvement: ${(secondAvg - firstAvg).toFixed(3)}`);
    
    if (secondAvg <= firstAvg) {
        console.log('  ❌ WARNING: No learning improvement detected!');
        
        // Additional diagnostics
        console.log('  Final Q-values for balanced state:', Array.from(qlearning.getAllQValues(new Float32Array([0.0, 0.0]))).map(v => v.toFixed(4)));
        console.log('  Replay buffer size:', qlearning.replayBuffer.size());
        console.log('  Current epsilon:', qlearning.hyperparams.epsilon.toFixed(4));
    } else {
        console.log('  ✅ Learning improvement detected!');
    }
}

// Run the diagnostic
runComprehensiveDiagnostic().catch(console.error);