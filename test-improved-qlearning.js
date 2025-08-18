/**
 * Test the improved Q-learning implementation with PyTorch DQN standards
 */

import { BalancingRobot } from './src/physics/BalancingRobot.js';
import { QLearning } from './src/training/QLearning.js';

async function testImprovedQLearning() {
    console.log('=== TESTING IMPROVED Q-LEARNING IMPLEMENTATION ===\n');
    
    // Test with PyTorch DQN standard hyperparameters
    const qlearning = new QLearning({
        learningRate: 3e-4,    // PyTorch standard
        epsilon: 0.9,          // PyTorch standard start
        epsilonMin: 0.01,      // PyTorch standard end
        epsilonDecay: 2500,    // Linear decay over 2500 steps
        gamma: 0.99,           // PyTorch standard
        batchSize: 128,        // PyTorch standard
        targetUpdateFreq: 100, // Standard update frequency
        hiddenSize: 128        // PyTorch standard (much larger)
    });
    
    await qlearning.initialize();
    
    const robot = new BalancingRobot({
        mass: 1.0,
        centerOfMassHeight: 0.3,
        motorStrength: 3.0
    });
    
    console.log('Hyperparameters:', qlearning.hyperparams);
    console.log(`Network architecture: 2 → ${qlearning.hyperparams.hiddenSize} → 3`);
    console.log(`Total parameters: ~${(2 * 128 + 128) + (128 * 3 + 3)} = ${2 * 128 + 128 + 128 * 3 + 3}`);
    
    const episodeRewards = [];
    const qValueHistory = [];
    
    console.log('\n=== TRAINING WITH IMPROVED IMPLEMENTATION ===');
    
    // Run learning test with improved implementation
    for (let episode = 0; episode < 50; episode++) {
        robot.reset({ 
            angle: (Math.random() - 0.5) * 0.2,  // Small random start
            angularVelocity: 0 
        });
        
        let totalReward = 0;
        let steps = 0;
        const maxSteps = 200;
        
        let previousState = null;
        let previousAction = null;
        let previousReward = 0;
        
        while (steps < maxSteps && !robot.getState().hasFailed()) {
            const state = robot.getState().getNormalizedInputs();
            const actionIndex = qlearning.selectAction(state, true);
            const actions = [-1.0, 0.0, 1.0];
            
            const result = robot.step(actions[actionIndex]);
            totalReward += result.reward;
            
            // Train on previous experience
            if (previousState && previousAction !== null) {
                const loss = qlearning.train(
                    previousState,
                    previousAction,
                    previousReward,
                    state,
                    false
                );
            }
            
            if (!result.done) {
                previousState = state.slice();
                previousAction = actionIndex;
                previousReward = result.reward;
            } else {
                // Final training step
                qlearning.train(
                    state,
                    actionIndex,
                    result.reward,
                    state,
                    true
                );
                break;
            }
            
            steps++;
        }
        
        episodeRewards.push(totalReward);
        
        // Track Q-values for balanced state
        const balancedState = new Float32Array([0.0, 0.0]);
        const qValues = qlearning.getAllQValues(balancedState);
        const maxQ = Math.max(...Array.from(qValues));
        qValueHistory.push(maxQ);
        
        if (episode % 10 === 0) {
            const recent10 = episodeRewards.slice(-10);
            const avgRecent = recent10.reduce((a, b) => a + b, 0) / recent10.length;
            const currentEpsilon = qlearning.hyperparams.epsilon;
            
            console.log(`Episode ${episode}: Reward=${totalReward.toFixed(1)}, Steps=${steps}, Avg(10)=${avgRecent.toFixed(1)}, MaxQ=${maxQ.toFixed(3)}, Eps=${currentEpsilon.toFixed(3)}`);
        }
    }
    
    // Final analysis
    const first20 = episodeRewards.slice(0, 20);
    const last20 = episodeRewards.slice(-20);
    const firstAvg = first20.reduce((a, b) => a + b, 0) / first20.length;
    const lastAvg = last20.reduce((a, b) => a + b, 0) / last20.length;
    
    console.log(`\n=== RESULTS ===`);
    console.log(`First 20 episodes avg: ${firstAvg.toFixed(2)}`);
    console.log(`Last 20 episodes avg: ${lastAvg.toFixed(2)}`);
    console.log(`Improvement: ${(lastAvg - firstAvg).toFixed(2)}`);
    console.log(`Final epsilon: ${qlearning.hyperparams.epsilon.toFixed(4)}`);
    
    // Check Q-value stability
    const firstQValues = qValueHistory.slice(0, 20);
    const lastQValues = qValueHistory.slice(-20);
    const firstQAvg = firstQValues.reduce((a, b) => a + b, 0) / firstQValues.length;
    const lastQAvg = lastQValues.reduce((a, b) => a + b, 0) / lastQValues.length;
    
    console.log(`Q-values - First: ${firstQAvg.toFixed(3)}, Last: ${lastQAvg.toFixed(3)}`);
    
    if (lastAvg > firstAvg + 0.1) {
        console.log('✅ IMPROVED Q-LEARNING IS WORKING! Clear improvement detected.');
    } else if (Math.abs(lastAvg - firstAvg) < 0.05) {
        console.log('⚠️  No clear improvement - may need more episodes or tuning');
    } else {
        console.log('❌ Performance degraded - check implementation');
    }
    
    return {
        improvement: lastAvg - firstAvg,
        finalEpsilon: qlearning.hyperparams.epsilon,
        qValueChange: lastQAvg - firstQAvg
    };
}

// Run the test
testImprovedQLearning().catch(console.error);