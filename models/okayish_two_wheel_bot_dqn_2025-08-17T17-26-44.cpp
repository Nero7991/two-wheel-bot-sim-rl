/**
 * Two-Wheel Balancing Robot DQN Model
 * Generated: 2025-08-17T17-26-44
 * Architecture: 2-64-3
 * 
 * This file contains the trained neural network weights for deployment
 * on embedded systems (Arduino, ESP32, STM32, etc.)
 */

#include <math.h>

class TwoWheelBotDQN {
private:
    static const int INPUT_SIZE = 2;
    static const int HIDDEN_SIZE = 64;
    static const int OUTPUT_SIZE = 3;
    
    // Network weights (stored in program memory to save RAM)
    const float weightsInputHidden[INPUT_SIZE * HIDDEN_SIZE] = {
                7.023232f, -1.946496f, -3.660843f, 0.602734f, -5.156502f, -2.125623f, -2.262152f, 3.537727f,
        -1.847513f, 5.594237f, 5.157835f, -2.229493f, -2.576713f, -4.074543f, 3.075437f, 1.582539f,
        -2.657667f, 6.860506f, -4.326180f, -3.920493f, 1.961780f, -2.236488f, 8.866314f, 8.776635f,
        -6.881963f, 7.039897f, 6.978577f, 3.682001f, 8.947897f, -5.578982f, 5.578310f, -0.396936f,
        1.448075f, 2.259323f, -3.596382f, 5.706481f, 3.215534f, 9.003716f, 2.990098f, 5.682712f,
        5.422194f, -2.412235f, 8.792321f, 4.318284f, -2.177967f, 8.304332f, 0.656789f, -5.843938f,
        -6.586642f, -4.976937f, 9.008966f, 5.064896f, 1.330268f, -2.393737f, -3.755072f, 5.433015f,
        -7.779222f, 5.952105f, 6.251626f, 8.590270f, 1.916585f, 3.361299f, 4.988776f, 8.524696f,
        7.040170f, 6.249614f, -5.528822f, -1.290155f, -2.341282f, 6.861738f, 7.467060f, 9.133445f,
        6.195847f, -0.543552f, 1.495834f, 7.389530f, 3.538911f, -2.879457f, 5.602779f, 9.999001f,
        7.287007f, 7.165514f, -5.238105f, -5.412622f, 6.280796f, 5.985641f, 9.998694f, 6.671149f,
        -9.272775f, 9.999001f, 6.411644f, 9.999001f, 2.362712f, -0.143927f, 9.987533f, -1.337093f,
        9.999001f, -3.802574f, -5.505105f, 5.309973f, 9.999001f, 1.243797f, 0.625026f, 9.999001f,
        5.444474f, 7.289879f, 6.816528f, 4.470291f, 6.592383f, 0.702628f, -7.943575f, -8.206820f,
        -9.986835f, -3.605889f, 0.986598f, 9.999001f, -6.502712f, 7.262049f, -9.582799f, 9.456708f,
        -8.293652f, 6.828340f, 6.330016f, 5.177098f, 4.480667f, 6.679293f, 6.803697f, 5.973219f
    };
    
    const float biasHidden[HIDDEN_SIZE] = {
                -4.608638f, -1.025601f, -6.501798f, -0.179922f, 9.999697f, -1.505780f, -1.288382f, -0.063906f,
        -1.098465f, -5.406548f, -5.626944f, -1.283827f, -1.419015f, 7.609716f, -0.019569f, -0.267095f,
        -0.812000f, 8.893893f, -7.017145f, -6.701336f, -0.298858f, -0.930873f, -0.756151f, 9.998854f,
        -0.740462f, -0.782925f, 9.059819f, -0.454074f, -2.636194f, -5.653224f, 0.432449f, -0.904040f,
        -0.257348f, -0.390401f, -6.424275f, -5.094511f, -0.413040f, -2.548503f, 9.998443f, -0.652023f,
        -6.177557f, -1.055423f, -1.111208f, 7.945953f, -0.957842f, 9.960999f, -2.262724f, -0.004519f,
        -0.576031f, 9.999621f, -2.522706f, -0.591737f, -0.043445f, -1.060988f, -0.601453f, -0.395452f,
        0.041157f, 9.997232f, 8.007367f, 8.574869f, 8.618369f, -0.475475f, -3.885806f, 9.999194f
    };
    
    const float weightsHiddenOutput[HIDDEN_SIZE * OUTPUT_SIZE] = {
                -5.587954f, -1.338409f, 7.894243f, -4.962707f, -3.182529f, -2.710292f, 1.461152f, -9.709034f,
        -5.648414f, -2.212940f, -0.139113f, 5.160562f, -0.084272f, 0.671394f, 2.650271f, -6.373718f,
        -1.190761f, 0.246599f, -6.727904f, -1.919167f, -2.161101f, -9.998485f, -9.104272f, -6.751316f,
        -4.544266f, -4.427130f, -3.654868f, -2.031543f, -2.058283f, 9.277313f, -4.725622f, -2.927679f,
        2.245592f, -7.851984f, -1.791459f, -2.572003f, 0.655623f, 1.198598f, 1.451485f, 1.544230f,
        -1.286419f, 0.873134f, -4.064549f, -5.273642f, -7.706302f, -9.997818f, -3.833695f, -5.459013f,
        -5.847614f, -3.030030f, -1.662569f, 1.970976f, 0.252092f, -0.554032f, 1.141466f, -10.000000f,
        -5.519770f, 1.364865f, -10.000000f, -5.648401f, 1.014188f, 0.426735f, -8.632522f, -4.884659f,
        -3.638630f, 2.209774f, -9.998488f, -9.236064f, -0.818472f, 0.186232f, 6.200472f, 1.062291f,
        -9.567947f, -7.802730f, -9.969671f, -9.998501f, -8.178892f, -2.274161f, 1.750523f, 0.317791f,
        -0.568474f, -9.998501f, -9.998934f, -8.020748f, 9.990403f, 7.111800f, 5.202513f, 5.953860f,
        9.845404f, -2.366477f, -5.184835f, -5.301702f, 9.998660f, 9.742731f, -2.212579f, 9.827650f,
        -9.712143f, -4.027054f, -6.541830f, 7.397160f, 8.965155f, 9.583761f, 1.531406f, -7.872448f,
        -5.429743f, -7.620375f, -6.061645f, 1.329547f, -9.998339f, -9.999308f, -8.597130f, 9.978527f,
        7.311152f, 6.552685f, -3.781931f, 0.233861f, 1.429024f, -7.103989f, -9.731123f, -2.981201f,
        -6.539904f, -0.878213f, 7.422890f, -7.699227f, -2.127329f, -1.782914f, 8.523526f, 7.226751f,
        -5.621420f, 2.145602f, 0.017955f, -0.266393f, -5.200203f, -2.964766f, -2.204410f, 0.989793f,
        -1.440522f, 3.423784f, -3.725695f, 5.854098f, -4.168884f, -5.795030f, -7.439397f, -5.605638f,
        9.576832f, -2.339823f, -5.102937f, -0.833721f, 1.272744f, 2.634202f, 9.973733f, 8.245879f,
        5.849717f, -5.402317f, -6.336952f, -4.632036f, -7.499793f, -7.818459f, -8.956123f, -7.782010f,
        -2.361067f, -1.846055f, -9.937052f, 4.737638f, -8.084605f, -3.914376f, -5.222555f, -6.069197f,
        9.083991f, 9.500952f, 9.353291f, 1.528680f, 1.264627f, -1.153892f, 1.587600f, 0.252650f,
        -0.293399f, 0.698731f, -1.105670f, 0.858467f, 2.815850f, -0.328581f, -0.599793f, 7.979664f,
        2.225731f, -10.000000f, -0.752829f, 0.822771f, -6.823290f, 0.721177f, 2.456748f, -0.943365f
    };
    
    const float biasOutput[OUTPUT_SIZE] = {
                9.509856f, 9.999767f, 9.999404f
    };
    
    // Activation function (ReLU)
    float relu(float x) {
        return x > 0 ? x : 0;
    }
    
public:
    /**
     * Get action from current state
     * @param angle Robot angle in radians
     * @param angularVelocity Angular velocity in rad/s
     * @return Action index (0=left, 1=brake, 2=right)
     */
    int getAction(float angle, float angularVelocity) {
        // Normalize inputs
        float input[INPUT_SIZE];
        input[0] = constrain(angle / (M_PI / 3), -1.0, 1.0);
        input[1] = constrain(angularVelocity / 10.0, -1.0, 1.0);
        
        // Hidden layer computation
        float hidden[HIDDEN_SIZE];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            hidden[h] = biasHidden[h];
            for (int i = 0; i < INPUT_SIZE; i++) {
                hidden[h] += input[i] * weightsInputHidden[i * HIDDEN_SIZE + h];
            }
            hidden[h] = relu(hidden[h]);
        }
        
        // Output layer computation
        float output[OUTPUT_SIZE];
        float maxValue = -1e10;
        int bestAction = 0;
        
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            output[o] = biasOutput[o];
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                output[o] += hidden[h] * weightsHiddenOutput[h * OUTPUT_SIZE + o];
            }
            
            // Track best action
            if (output[o] > maxValue) {
                maxValue = output[o];
                bestAction = o;
            }
        }
        
        return bestAction;
    }
    
    /**
     * Get motor torque for action
     * @param action Action index
     * @return Motor torque (-1.0 to 1.0)
     */
    float getMotorTorque(int action) {
        const float actions[3] = {-1.0, 0.0, 1.0};
        return actions[action];
    }
    
private:
    float constrain(float value, float min, float max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
};

// Usage example:
// TwoWheelBotDQN bot;
// int action = bot.getAction(angle, angularVelocity);
// float torque = bot.getMotorTorque(action);
