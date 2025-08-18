/**
 * Test the simplified reward function
 */

import { BalancingRobot } from './src/physics/BalancingRobot.js';
import { QLearning } from './src/training/QLearning.js';

async function testSimpleReward() {
    console.log('=== SIMPLE REWARD FUNCTION TEST ===\n');
    
    // Test reward function values
    const robot = new BalancingRobot();
    
    console.log('Testing reward function values:');
    const testAngles = [0, 0.1, 0.3, 0.5, 0.8, 1.1]; // Last one should fail
    
    for (const angle of testAngles) {
        robot.reset({ angle: angle, angularVelocity: 0 });
        const result = robot.step(0);
        console.log(`  Angle: ${angle.toFixed(2)} rad (${(angle * 180 / Math.PI).toFixed(1)}°) -> Reward: ${result.reward.toFixed(3)}, Failed: ${result.done}`);
    }
    
    console.log('\n=== QUICK LEARNING TEST ===');
    
    // Test with simplified Q-learning
    const qlearning = new QLearning({
        learningRate: 0.01,   // Higher learning rate for simple reward
        epsilon: 0.2,         // Moderate exploration
        epsilonDecay: 0.999,  // Slow decay
        gamma: 0.99,          // High discount
        batchSize: 8,         // Small batch
        targetUpdateFreq: 50, // Frequent updates for simple task
        hiddenSize: 6         // Smaller network
    });
    
    await qlearning.initialize();
    
    const episodeRewards = [];
    
    // Run learning test
    for (let episode = 0; episode < 100; episode++) {
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
            const actions = [-0.5, 0.0, 0.5]; // Smaller actions for stability
            
            const result = robot.step(actions[actionIndex]);
            totalReward += result.reward;
            
            // Train on previous experience
            if (previousState && previousAction !== null) {
                qlearning.train(
                    previousState,
                    previousAction,
                    previousReward,
                    state,
                    false // Not done yet
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
                    state, // Dummy next state
                    true   // Done
                );
                break;
            }
            
            steps++;
        }
        
        episodeRewards.push(totalReward);
        
        if (episode % 20 === 0) {
            const recent20 = episodeRewards.slice(-20);
            const avgRecent = recent20.reduce((a, b) => a + b, 0) / recent20.length;
            
            // Test policy (no exploration)
            const testState = new Float32Array([0.0, 0.0]); // Balanced
            const qValues = qlearning.getAllQValues(testState);
            const maxQ = Math.max(...Array.from(qValues));
            
            console.log(`Episode ${episode}: Reward=${totalReward.toFixed(1)}, Steps=${steps}, Avg(20)=${avgRecent.toFixed(1)}, MaxQ=${maxQ.toFixed(3)}`);
        }
    }
    
    // Final analysis
    const first25 = episodeRewards.slice(0, 25);
    const last25 = episodeRewards.slice(-25);
    const firstAvg = first25.reduce((a, b) => a + b, 0) / first25.length;
    const lastAvg = last25.reduce((a, b) => a + b, 0) / last25.length;
    
    console.log(`\nFirst 25 episodes avg: ${firstAvg.toFixed(2)}`);
    console.log(`Last 25 episodes avg: ${lastAvg.toFixed(2)}`);
    console.log(`Improvement: ${(lastAvg - firstAvg).toFixed(2)}`);
    
    if (lastAvg > firstAvg) {
        console.log('✅ LEARNING DETECTED with simple reward function!');
    } else {
        console.log('❌ Still no learning - deeper issue exists');
    }
    
    return lastAvg > firstAvg;
}

// Run the test
testSimpleReward().catch(console.error);