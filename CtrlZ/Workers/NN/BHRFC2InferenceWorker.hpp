#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "Utils/ZenBuffer.hpp"
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace z {

template<typename SchedulerType, typename InferencePrecision, size_t INPUT_STUCK_LENGTH, size_t JOINT_NUMBER>
class BHRFC2InferenceWorker : public CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER> {
public:
    using Base = CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>;
    using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
    using ValVec3 = math::Vector<InferencePrecision, 3>;
    using ClockVec = math::Vector<InferencePrecision, 2>; // sin, cos

private:
    // Define the size of a single observation frame based on bhr_fc2_env.py
    // command_input (5) + q (12) + dq (12) + last_action (12) + ang_vel (3) + projected_gravity (3) = 47
    static constexpr size_t SINGLE_FRAME_LENGTH = 5 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER + 3 + 3;
    static constexpr size_t TOTAL_INPUT_LENGTH = SINGLE_FRAME_LENGTH * INPUT_STUCK_LENGTH;
    static constexpr size_t OUTPUT_LENGTH = JOINT_NUMBER;

    // Input tensor for ONNX model
    z::math::Tensor<InferencePrecision, 1, TOTAL_INPUT_LENGTH> InputTensor;
    // Output tensor from ONNX model
    z::math::Tensor<InferencePrecision, 1, OUTPUT_LENGTH> OutputTensor;

    // History buffer for single frames
    z::RingBuffer<z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH>> HistoryInputBuffer;

    // Scales for each part of the single frame observation vector
    // Order must match the concatenation order in PreProcess!
    z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH> InputScaleVec;

    // Scale for the output action
    z::math::Vector<InferencePrecision, OUTPUT_LENGTH> OutputScaleVec;

    // const ValVec3 GravityVector = {0.0, 0.0, -9.81}; // Or {0,0,-1} depending on normalization goal
    const ValVec3 GravityVector; // = {0.0, 0.0, -1.0}; // Or {0,0,-1} depending on normalization goal
    InferencePrecision cycle_time;
    InferencePrecision dt;
    std::chrono::steady_clock::time_point start_time, end_time;

public:
    BHRFC2InferenceWorker(SchedulerType* scheduler, const nlohmann::json& Net_cfg, const nlohmann::json& Motor_cfg)
        : Base(scheduler, Net_cfg, Motor_cfg),
          GravityVector({ 0.0,0.0,-1.0 }),
          HistoryInputBuffer(INPUT_STUCK_LENGTH) // Initialize RingBuffer size
    {
        // --- Read Config specific to BHRFC2 if any (e.g., cycle time) ---
        nlohmann::json InferenceCfg = Net_cfg["Inference"];
        nlohmann::json NetworkCfg = Net_cfg["Network"];
        this->cycle_time = NetworkCfg["Cycle_time"].get<InferencePrecision>(); // Ensure this key exists
        this->dt = scheduler->getSpinOnceTime();
        // this->dt = cfg["Scheduler"]["dt"].get<InferencePrecision>(); // Ensure this key exists

        // --- Construct the scaling vector based on CommonLocoInferenceWorker loaded scales ---
        // IMPORTANT: Order MUST match the concatenation order in PreProcess
        auto clock_scales = math::Vector<InferencePrecision, 2>::ones(); // Clock usually not scaled or scale=1
        this->InputScaleVec = math::cat(
            clock_scales,             // 2 (sin, cos)
            this->Scales_command3,    // 3 (cmd_x, cmd_y, cmd_yaw) - Uses lin_vel, ang_vel scales
            this->Scales_dof_pos,     // JOINT_NUMBER (pos - default_pos)
            this->Scales_dof_vel,     // JOINT_NUMBER (vel)
            this->Scales_last_action, // JOINT_NUMBER (last action) - Usually scale=1
            this->Scales_ang_vel,     // 3 (base ang vel)
            this->Scales_project_gravity // 3 (projected gravity) - Usually scale=1 (quat scale)
        );

        // Verify size
        // static_assert(InputScaleVec.size() == SINGLE_FRAME_LENGTH, "InputScaleVec size mismatch with SINGLE_FRAME_LENGTH");
        std::cout << "InputScaleVec size: " << this->InputScaleVec.size() << std::endl;
        std::cout << "SINGLE_FRAME_LENGTH: " << SINGLE_FRAME_LENGTH << std::endl;

        // Output scale is directly from CommonLocoInferenceWorker
        this->OutputScaleVec = this->ActionScale;

        // --- Warp Tensors for ONNX Runtime IO Binding ---
        // Ensure InputOrtTensors__ and OutputOrtTensors__ are cleared or managed correctly by base class
        this->InputOrtTensors__.clear();
        this->OutputOrtTensors__.clear();
        this->InputOrtTensors__.push_back(this->WarpOrtTensor(InputTensor));
        this->OutputOrtTensors__.push_back(this->WarpOrtTensor(OutputTensor));

        // --- Print Worker Info ---
        this->PrintSplitLine();
        std::cout << "BHRFC2InferenceWorker Initialized" << std::endl;
        std::cout << "JOINT_NUMBER: " << JOINT_NUMBER << std::endl;
        std::cout << "SINGLE_FRAME_LENGTH: " << SINGLE_FRAME_LENGTH << std::endl;
        std::cout << "INPUT_STUCK_LENGTH: " << INPUT_STUCK_LENGTH << std::endl;
        std::cout << "TOTAL_INPUT_LENGTH: " << TOTAL_INPUT_LENGTH << std::endl;
        std::cout << "OUTPUT_LENGTH: " << OUTPUT_LENGTH << std::endl;
        std::cout << "Cycle Time: " << this->cycle_time << std::endl;
        std::cout << "DT: " << this->dt << std::endl;

        // print the HistoryInputBuffer initial values
        std::cout << "HistoryInputBuffer Size: " << this->HistoryInputBuffer.size() << std::endl;
        // std::cout << "HistoryInputBuffer Capacity: " << this->HistoryInputBuffer.capacity() << std::endl;
        for (size_t i = 0; i < this->HistoryInputBuffer.size(); ++i) {
            std::cout << "HistoryInputBuffer[" << i << "]: " << this->HistoryInputBuffer[i] << std::endl;
        }

        // std::cout << "Input Scales: " << this->InputScaleVec << std::endl; // Optional: Print if needed
        this->PrintSplitLine();
    }

    virtual ~BHRFC2InferenceWorker() {}

    void PreProcess() override {
        this->start_time = std::chrono::steady_clock::now();

        // --- 1. Get data from Scheduler (DataCenter) ---
        MotorValVec CurrentMotorPos;
        this->Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);
        MotorValVec CurrentMotorVel;
        this->Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);
        MotorValVec LastAction;
        this->Scheduler->template GetData<"NetLastAction">(LastAction); // Ensure this is set correctly in PostProcess
        ValVec3 UserCmd3;
        this->Scheduler->template GetData<"NetUserCommand3">(UserCmd3);
        ValVec3 AngVel;
        this->Scheduler->template GetData<"AngleVelocityValue">(AngVel); // Assuming this is base angular velocity
        ValVec3 Ang; // Assuming this contains orientation (e.g., Euler angles)
        this->Scheduler->template GetData<"AngleValue">(Ang);

        // --- 2. Compute derived values ---
        // Projected Gravity
        // TODO: Replace with your actual ComputeProjectedGravity implementation or ensure it exists
        ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
        this->Scheduler->template SetData<"NetProjectedGravity">(ProjectedGravity); // Optional: Store if needed elsewhere

        ValVec3 BaseAngVel = ComputeBaseAngVelocity(Ang, AngVel); 
        this->Scheduler->template SetData<"NetBaseAngVelocityValue">(BaseAngVel); // Optional: Store if needed elsewhere

        // Clock signals
        size_t t = this->Scheduler->getTimeStamp();
        InferencePrecision phase = fmod(this->dt * static_cast<InferencePrecision>(t), this->cycle_time) / this->cycle_time; // Phase [0, 1)
        InferencePrecision clock_sin = std::sin(phase * 2 * M_PI);
        InferencePrecision clock_cos = std::cos(phase * 2 * M_PI);
        // ClockVec ClockVector = {std::sin(2 * M_PI * phase), std::cos(2 * M_PI * phase)};
        ClockVec ClockVector = { clock_sin, clock_cos };
        this->Scheduler->template SetData<"NetClockVector">(ClockVector); // Optional: Store if needed elsewhere

        // Relative DOF position
        MotorValVec RelativeMotorPos = CurrentMotorPos - this->JointDefaultPos;

        // --- 3. Construct single frame observation vector (Order MUST match Python!) ---
        auto SingleInputVec = math::cat(
            ClockVector,         // 2
            UserCmd3,            // 3
            RelativeMotorPos,    // JOINT_NUMBER
            CurrentMotorVel,     // JOINT_NUMBER
            LastAction,          // JOINT_NUMBER
            AngVel,              // 3
            // BaseAngVel,          // 3
            ProjectedGravity     // 3
        );

        // Verify size
        static_assert(SingleInputVec.size() == SINGLE_FRAME_LENGTH, "Constructed SingleInputVec size mismatch");

        // --- 4. Scale the single frame vector ---
        auto SingleInputVecScaled = SingleInputVec * this->InputScaleVec;

        // --- 5. Push scaled frame into history buffer ---
        this->HistoryInputBuffer.push(SingleInputVecScaled);

        // --- 6. Fill the final input tensor from history buffer ---
        // Assumes InputTensor is flat [1, TOTAL_INPUT_LENGTH]
        math::Vector<InferencePrecision, TOTAL_INPUT_LENGTH> InputVec;
        for (size_t i = 0; i < INPUT_STUCK_LENGTH; ++i) {
            // HistoryInputBuffer[0] is the oldest, HistoryInputBuffer[INPUT_STUCK_LENGTH-1] is the newest
            // The python implementation likely stacks newest first? Check legged_gym implementation if unsure.
            // Assuming newest first (like standard frame stacking):

            // size_t history_idx = INPUT_STUCK_LENGTH - 1 - i; // Newest first
            size_t history_idx = i; // Oldest first
            size_t tensor_offset = i * SINGLE_FRAME_LENGTH;
            std::copy(this->HistoryInputBuffer[history_idx].begin(),
                      this->HistoryInputBuffer[history_idx].end(),
                      InputVec.begin() + tensor_offset);
                    //   this->InputTensor.Array().begin() + tensor_offset);
        }

        // --- 7. Clip the final input tensor ---
        this->InputTensor.Array() = decltype(InputVec)::clamp(
            InputVec, -this->ClipObservation, this->ClipObservation
        );
    }

    void PostProcess() override {
        // --- 1. Get raw action output from ONNX model ---
        auto RawAction = this->OutputTensor.toVector(); // Shape [JOINT_NUMBER]

        // --- 2. Clip raw action (if needed, based on training settings) ---
        auto ClippedRawAction = MotorValVec::clamp(RawAction, -this->ClipAction, this->ClipAction);

        // --- 3. Store clipped raw action as "NetLastAction" for next step's observation ---
        this->Scheduler->template SetData<"NetLastAction">(ClippedRawAction);

        // --- 4. Apply action scale and add default joint position ---
        // target_pos = clipped_raw_action * action_scale + default_pos
        auto ScaledAction = ClippedRawAction * this->OutputScaleVec + this->JointDefaultPos;
        this->Scheduler->template SetData<"NetScaledAction">(ScaledAction); // Optional: Store if needed

        // --- 5. Apply joint limits ---
        auto FinalTargetPosition = MotorValVec::clamp(ScaledAction, this->JointClipLower, this->JointClipUpper);

        // --- 6. Set the final target position in the DataCenter ---
        // This will be read by MotorControlWorker or MotorPDWorker
        this->Scheduler->template SetData<"TargetMotorPosition">(FinalTargetPosition);

        // --- 7. Record inference time ---
        this->end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
        InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count()) / 1000.0; // Time in milliseconds
        this->Scheduler->template SetData<"InferenceTime">(inference_time);
    }
};

} // namespace z