/**
 * @file DWAQInferenceWorker.hpp
 * @author Gemini AI (guided by Junhang Lai)
 * @brief An inference worker for the DreamWaQ policy architecture.
 * @date 2025-07-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "CommonLocoInferenceWorker.hpp"
#include "Utils/ZenBuffer.hpp"
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace z {

/**
 * @brief DWAQInferenceWorker 实现了 DreamWaQ [https://arxiv.org/abs/2301.05268] 论文中策略网络的推理逻辑。
 * @details 该Worker专门处理DWAQ模型的双输入特性：
 *          1. `obs`: 当前时刻的本体感知观测。
 *          2. `obs_history`: 过去N个时刻的本体感知观测构成的历史序列。
 *          它继承自 CommonLocoInferenceWorker 以复用通用的参数加载逻辑。
 *
 * @tparam SchedulerType 调度器类型
 * @tparam InferencePrecision 推理精度 (float or double)
 * @tparam HISTORY_LENGTH 历史观测的帧数 (对应 leggedlab 中 actor_obs_history_length)
 * @tparam JOINT_NUMBER 关节数量
 */
template<typename SchedulerType, typename InferencePrecision, size_t HISTORY_LENGTH, size_t JOINT_NUMBER>
class DWAQInferenceWorker : public CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER> {
public:
    using Base = CommonLocoInferenceWorker<SchedulerType, InferencePrecision, JOINT_NUMBER>;
    using MotorValVec = math::Vector<InferencePrecision, JOINT_NUMBER>;
    using ValVec3 = math::Vector<InferencePrecision, 3>;
    using ClockVec = math::Vector<InferencePrecision, 2>;
    using GaitVec = math::Vector<InferencePrecision, 2>;

private:
    // --- 定义网络输入输出的维度 ---
    // 根据 leggedlab/envs/bhrfc2_dwaq_mm/bhr_fc2_dwaq_mm_env.py 中 compute_current_observations 的定义
    // ang_vel(3) + projected_gravity(3) + command(3) + phase_obs(2) + one_hot_gait(2) + joint_pos(12) + joint_vel(12) + action(12) = 49
    static constexpr size_t SINGLE_FRAME_LENGTH = 3 + 3 + 3 + 2 + 2 + JOINT_NUMBER + JOINT_NUMBER + JOINT_NUMBER;
    
    // ONNX模型的两个输入张量
    // 1. 当前观测
    z::math::Tensor<InferencePrecision, 1, SINGLE_FRAME_LENGTH> InputObsTensor;
    // 2. 历史观测
    static constexpr size_t HISTORY_TENSOR_LENGTH = SINGLE_FRAME_LENGTH * HISTORY_LENGTH;
    z::math::Tensor<InferencePrecision, 1, HISTORY_TENSOR_LENGTH> InputHistoryTensor;

    // ONNX模型的输出张量
    static constexpr size_t OUTPUT_LENGTH = JOINT_NUMBER;
    z::math::Tensor<InferencePrecision, 1, OUTPUT_LENGTH> OutputTensor;

    // 用于管理历史观测帧的环形缓冲区
    z::RingBuffer<z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH>> HistoryObsBuffer;

    // 用于构建和缩放单帧观测的向量
    z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH> InputScaleVec;
    z::math::Vector<InferencePrecision, OUTPUT_LENGTH> OutputScaleVec;

    const ValVec3 GravityVector;
    InferencePrecision cycle_time;
    InferencePrecision dt;
    std::chrono::steady_clock::time_point start_time, end_time;

public:
    DWAQInferenceWorker(SchedulerType* scheduler, const nlohmann::json& Net_cfg, const nlohmann::json& Motor_cfg)
        : Base(scheduler, Net_cfg, Motor_cfg),
          GravityVector({0.0, 0.0, -1.0}),
          HistoryObsBuffer(HISTORY_LENGTH, z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH>::zeros()) // 用0初始化
    {
        // --- 1. 读取DWAQ特定配置 ---
        nlohmann::json NetworkCfg = Net_cfg["Network"];
        this->cycle_time = NetworkCfg["Cycle_time"].get<InferencePrecision>();
        this->dt = scheduler->getSpinOnceTime();
        
        // --- 2. 精确构建观测缩放向量 (顺序必须与PreProcess中拼接的顺序完全一致) ---
        this->InputScaleVec = math::cat(
            this->Base::Scales_ang_vel,         // 3
            this->Base::Scales_project_gravity, // 3
            this->Base::Scales_command3,        // 3
            ValVec3::ones() * this->Base::Scales_command3[0], // 2 for phase (use command scale)
            ValVec3::ones() * this->Base::Scales_command3[0], // 2 for gait_mode (use command scale)
            this->Base::Scales_dof_pos,         // JOINT_NUMBER
            this->Base::Scales_dof_vel,         // JOINT_NUMBER
            this->Base::Scales_last_action      // JOINT_NUMBER
        );
        // 验证维度
        if (this->InputScaleVec.size() != SINGLE_FRAME_LENGTH) {
            throw std::runtime_error("DWAQInferenceWorker: InputScaleVec size mismatch with SINGLE_FRAME_LENGTH");
        }

        this->OutputScaleVec = this->Base::ActionScale;

        // --- 3. 绑定ONNX模型的输入输出张量 ---
        this->Base::InputOrtTensors__.clear();
        this->Base::OutputOrtTensors__.clear();
        // 顺序必须与 "InputNodeNames" 在JSON中的顺序一致: ["obs", "obs_history"]
        this->Base::InputOrtTensors__.push_back(this->Base::WarpOrtTensor(InputObsTensor));
        this->Base::InputOrtTensors__.push_back(this->Base::WarpOrtTensor(InputHistoryTensor));
        // 绑定输出
        this->Base::OutputOrtTensors__.push_back(this->Base::WarpOrtTensor(OutputTensor));

        // --- 4. 打印初始化信息 ---
        this->Base::PrintSplitLine();
        std::cout << "DWAQInferenceWorker Initialized" << std::endl;
        std::cout << "  - Single Obs Frame Size: " << SINGLE_FRAME_LENGTH << std::endl;
        std::cout << "  - History Length: " << HISTORY_LENGTH << std::endl;
        std::cout << "  - ONNX Input 'obs' Shape: (1, " << SINGLE_FRAME_LENGTH << ")" << std::endl;
        std::cout << "  - ONNX Input 'obs_history' Shape: (1, " << HISTORY_TENSOR_LENGTH << ")" << std::endl;
        std::cout << "  - ONNX Output 'actions' Shape: (1, " << OUTPUT_LENGTH << ")" << std::endl;
        this->Base::PrintSplitLine();
    }

    virtual ~DWAQInferenceWorker() {}

    void PreProcess() override {
        this->start_time = std::chrono::steady_clock::now();

        // --- 1. 从DataCenter获取所有必需的原始数据 ---
        MotorValVec CurrentMotorPos;
        this->Base::Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos);
        MotorValVec CurrentMotorVel;
        this->Base::Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel);
        MotorValVec LastAction;
        this->Base::Scheduler->template GetData<"NetLastAction">(LastAction);
        ValVec3 UserCmd3;
        this->Base::Scheduler->template GetData<"NetUserCommand3">(UserCmd3);
        ValVec3 AngVel;
        this->Base::Scheduler->template GetData<"AngleVelocityValue">(AngVel);
        ValVec3 Ang;
        this->Base::Scheduler->template GetData<"AngleValue">(Ang);

        // --- 2. 计算派生观测量 ---
        ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
        
        // 步态相位 (Gait Phase)
        size_t t = this->Base::Scheduler->getTimeStamp();
        InferencePrecision phase = fmod(this->dt * static_cast<InferencePrecision>(t), this->cycle_time) / this->cycle_time;
        ClockVec ClockVector = {std::sin(phase * 2.0 * M_PI), std::cos(phase * 2.0 * M_PI)};

        // 步态模式 (Gait Mode, standing vs walking) - one hot编码
        bool is_standing = (UserCmd3.toVector().abs().max() <= 0.1); // 简化的站立判断
        GaitVec GaitModeVector = {is_standing ? 1.0f : 0.0f, is_standing ? 0.0f : 1.0f};

        // --- 3. 构建单帧观测向量 (顺序严格对齐Python实现) ---
        auto SingleFrameObs = math::cat(
            AngVel,
            ProjectedGravity,
            UserCmd3,
            ClockVector,
            GaitModeVector,
            CurrentMotorPos - this->Base::JointDefaultPos, // 相对关节位置
            CurrentMotorVel,
            LastAction
        );

        // --- 4. 缩放单帧观测并推入历史缓冲区 ---
        auto ScaledSingleFrameObs = SingleFrameObs * this->InputScaleVec;
        this->HistoryObsBuffer.push(ScaledSingleFrameObs);

        // --- 5. 构建最终送入ONNX模型的两个输入张量 ---
        // `obs` input: 当前最新的、已缩放的观测帧
        this->InputObsTensor.Array() = ScaledSingleFrameObs;

        // `obs_history` input: 拼接历史缓冲区中的所有帧
        for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
            // HistoryObsBuffer[0] 是最旧的, [-1] 是最新的
            // 我们需要按照Python中扁平化的顺序填充 (通常是[oldest, ..., newest])
            size_t tensor_offset = i * SINGLE_FRAME_LENGTH;
            std::copy(this->HistoryObsBuffer[i].begin(),
                      this->HistoryObsBuffer[i].end(),
                      this->InputHistoryTensor.Array().begin() + tensor_offset);
        }

        // --- 6. 裁剪最终的输入张量 ---
        this->InputObsTensor.Array() = decltype(this->InputObsTensor.Array())::clamp(
            this->InputObsTensor.Array(), -this->Base::ClipObservation, this->Base::ClipObservation
        );
        this->InputHistoryTensor.Array() = decltype(this->InputHistoryTensor.Array())::clamp(
            this->InputHistoryTensor.Array(), -this->Base::ClipObservation, this->Base::ClipObservation
        );
    }

    void PostProcess() override {
        // --- 这部分与标准推理Worker完全相同 ---
        auto RawAction = this->OutputTensor.toVector();
        auto ClippedRawAction = MotorValVec::clamp(RawAction, -this->Base::ClipAction, this->Base::ClipAction);
        
        this->Base::Scheduler->template SetData<"NetLastAction">(ClippedRawAction);

        auto ScaledAction = ClippedRawAction * this->OutputScaleVec + this->Base::JointDefaultPos;
        this->Base::Scheduler->template SetData<"NetScaledAction">(ScaledAction);

        auto FinalTargetPosition = MotorValVec::clamp(ScaledAction, this->Base::JointClipLower, this->Base::JointClipUpper);
        this->Base::Scheduler->template SetData<"TargetMotorPosition">(FinalTargetPosition);

        // 记录推理时间
        this->end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
        InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count()) / 1000.0;
        this->Base::Scheduler->template SetData<"InferenceTime">(inference_time);
    }
};

} // namespace z