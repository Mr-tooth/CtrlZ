/**
 * @file DWAQInferenceWorker.hpp
 * @author Gemini AI (guided by Junhang Lai)
 * @brief An inference worker for the DreamWaQ policy architecture.
 * @version 2.0 (Optimized for data logging consistency and logic alignment)
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
#include <vector>
#include <string>
#include <algorithm> // for std::find
#include <stdexcept> // for std::runtime_error

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace z {

/**
 * @brief DWAQInferenceWorker 实现了 DreamWaQ [https://arxiv.org/abs/2301.05268] 论文中策略网络的推理逻辑。
 * @details 该Worker专门处理DWAQ模型的双输入特性，并内置了关节顺序重映射功能，以解决训练和部署框架间的差异。
 *          V2.0 版本修复了日志数据顺序不一致的问题，并与leggedlab中的逻辑严格对齐。
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
    static constexpr size_t HISTORY_TENSOR_LENGTH = SINGLE_FRAME_LENGTH * HISTORY_LENGTH;
    static constexpr size_t OUTPUT_LENGTH = JOINT_NUMBER;

    // ONNX模型的输入输出张量
    z::math::Tensor<InferencePrecision, 1, SINGLE_FRAME_LENGTH> InputObsTensor;
    z::math::Tensor<InferencePrecision, 1, HISTORY_TENSOR_LENGTH> InputHistoryTensor;
    z::math::Tensor<InferencePrecision, 1, OUTPUT_LENGTH> OutputTensor;

    z::RingBuffer<z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH>> HistoryObsBuffer;
    z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH> InputScaleVec;
    z::math::Vector<InferencePrecision, OUTPUT_LENGTH> OutputScaleVec;
    const ValVec3 GravityVector;
    InferencePrecision cycle_time;
    InferencePrecision dt;
    std::chrono::steady_clock::time_point start_time, end_time;

    // --- 关节重映射索引向量 ---
    // LeggedLab 关节顺序 -> CtrlZ 关节顺序
    z::math::Vector<int, JOINT_NUMBER> LeggedLabToCtrlZ_RemapIdx;

public:
    DWAQInferenceWorker(
        SchedulerType* scheduler, 
        const nlohmann::json& Net_cfg, 
        const nlohmann::json& Motor_cfg,
        const std::vector<std::string>& leggedlab_joint_order,
        const std::vector<std::string>& ctrlz_joint_order)
        : Base(scheduler, Net_cfg, Motor_cfg),
          GravityVector({0.0, 0.0, -1.0}),
          HistoryObsBuffer(HISTORY_LENGTH, z::math::Vector<InferencePrecision, SINGLE_FRAME_LENGTH>::zeros())
    {
        nlohmann::json NetworkCfg = Net_cfg["Network"];
        this->cycle_time = NetworkCfg["Cycle_time"].get<InferencePrecision>();
        this->dt = scheduler->getSpinOnceTime();
        
        // --- 1. 计算关节重映射索引 ---
        if (leggedlab_joint_order.size() != JOINT_NUMBER || ctrlz_joint_order.size() != JOINT_NUMBER) {
            throw std::runtime_error("Joint order vectors size mismatch with JOINT_NUMBER.");
        }
        for (size_t i = 0; i < JOINT_NUMBER; ++i) {
            const auto& leggedlab_name = leggedlab_joint_order[i];
            auto it = std::find(ctrlz_joint_order.begin(), ctrlz_joint_order.end(), leggedlab_name);
            if (it == ctrlz_joint_order.end()) {
                throw std::runtime_error("Joint '" + leggedlab_name + "' from leggedlab order not found in ctrlz order.");
            }
            this->LeggedLabToCtrlZ_RemapIdx[i] = std::distance(ctrlz_joint_order.begin(), it);
        }
        
        // --- 2. 精确构建观测缩放向量 (顺序严格对齐Python实现) ---
        auto clock_scales = math::Vector<InferencePrecision, 2>::ones(); // Per request, scale is 1
        auto gait_scales = math::Vector<InferencePrecision, 2>::ones();  // Per request, scale is 1
        
        this->InputScaleVec = math::cat(
            this->Base::Scales_ang_vel,
            this->Base::Scales_project_gravity,
            this->Base::Scales_command3,
            clock_scales,
            gait_scales,
            this->Base::Scales_dof_pos,
            this->Base::Scales_dof_vel,
            this->Base::Scales_last_action
        );
        if (this->InputScaleVec.size() != SINGLE_FRAME_LENGTH) {
            throw std::runtime_error("DWAQInferenceWorker: InputScaleVec size mismatch with SINGLE_FRAME_LENGTH");
        }
        this->OutputScaleVec = this->Base::ActionScale;

        // --- 3. 绑定ONNX模型的输入输出张量 ---
        this->Base::InputOrtTensors__.clear();
        this->Base::OutputOrtTensors__.clear();
        this->Base::InputOrtTensors__.push_back(this->Base::WarpOrtTensor(InputObsTensor));
        this->Base::InputOrtTensors__.push_back(this->Base::WarpOrtTensor(InputHistoryTensor));
        this->Base::OutputOrtTensors__.push_back(this->Base::WarpOrtTensor(OutputTensor));

        // --- 4. 打印初始化信息 ---
        this->Base::PrintSplitLine();
        std::cout << "DWAQInferenceWorker Initialized (V2.0 - Data-Consistent Logging)" << std::endl;
        std::cout << "  - Single Obs Frame Size: " << SINGLE_FRAME_LENGTH << std::endl;
        std::cout << "  - History Length: " << HISTORY_LENGTH << std::endl;
        std::cout << "\n[*] Joint Remapping (LeggedLab Index -> CtrlZ Index):" << std::endl;
        for (size_t i = 0; i < JOINT_NUMBER; ++i) {
            std::cout << "    '" << leggedlab_joint_order[i] << "' (" << i << ") -> " << this->LeggedLabToCtrlZ_RemapIdx[i] << std::endl;
        }
        this->Base::PrintSplitLine();
    }

    virtual ~DWAQInferenceWorker() {}

    void PreProcess() override {
        this->start_time = std::chrono::steady_clock::now();

        // --- 1. 获取原始数据 (CtrlZ顺序) ---
        MotorValVec CurrentMotorPos_ctrlz;
        this->Base::Scheduler->template GetData<"CurrentMotorPosition">(CurrentMotorPos_ctrlz);
        MotorValVec CurrentMotorVel_ctrlz;
        this->Base::Scheduler->template GetData<"CurrentMotorVelocity">(CurrentMotorVel_ctrlz);
        MotorValVec LastAction_ctrlz; // **V2.0改动**: 从总线获取CtrlZ顺序的上一步动作
        this->Base::Scheduler->template GetData<"NetLastAction">(LastAction_ctrlz);
        ValVec3 UserCmd3;
        this->Base::Scheduler->template GetData<"NetUserCommand3">(UserCmd3);
        ValVec3 AngVel;
        this->Base::Scheduler->template GetData<"AngleVelocityValue">(AngVel);
        ValVec3 Ang;
        this->Base::Scheduler->template GetData<"AngleValue">(Ang);

        // --- 2. 派生观测量 ---
        ValVec3 ProjectedGravity = ComputeProjectedGravity(Ang, this->GravityVector);
        
        // **V2.0改动**: is_standing逻辑与leggedlab严格对齐
        InferencePrecision cmd_speed_linear = std::sqrt(UserCmd3[0] * UserCmd3[0] + UserCmd3[1] * UserCmd3[1]);
        InferencePrecision cmd_speed_angular = std::abs(UserCmd3[2]);
        bool is_standing = (cmd_speed_linear <= 0.1) && (cmd_speed_angular <= 0.05);

        InferencePrecision phase {0.0};
        if (!is_standing) {
            size_t t = this->Base::Scheduler->getTimeStamp();
            phase = fmod(this->dt * static_cast<InferencePrecision>(t), this->cycle_time) / this->cycle_time;
        }
        ClockVec ClockVector = {std::sin(phase * 2.0 * M_PI), std::cos(phase * 2.0 * M_PI)};
        GaitVec GaitModeVector = {is_standing ? 1.0f : 0.0f, is_standing ? 0.0f : 1.0f};

        // --- 3. 构建单帧观测向量 (legggedlab顺序) ---
        MotorValVec RelativeMotorPos_leggedlab;
        MotorValVec CurrentMotorVel_leggedlab;
        MotorValVec LastAction_leggedlab;
        
        for (size_t i = 0; i < JOINT_NUMBER; ++i) {
            int ctrlz_idx = this->LeggedLabToCtrlZ_RemapIdx[i];
            RelativeMotorPos_leggedlab[i] = CurrentMotorPos_ctrlz[ctrlz_idx] - this->Base::JointDefaultPos[ctrlz_idx];
            CurrentMotorVel_leggedlab[i] = CurrentMotorVel_ctrlz[ctrlz_idx];
            LastAction_leggedlab[i] = LastAction_ctrlz[ctrlz_idx]; // **V2.0改动**
        }

        auto SingleFrameObs = math::cat(
            AngVel, ProjectedGravity, UserCmd3, ClockVector, GaitModeVector,
            RelativeMotorPos_leggedlab, CurrentMotorVel_leggedlab, LastAction_leggedlab
        );

        // --- 4. 缩放、推入历史、填充输入张量 ---
        auto ScaledSingleFrameObs = SingleFrameObs * this->InputScaleVec;
        this->HistoryObsBuffer.push(ScaledSingleFrameObs);
        
        this->InputObsTensor.Array() = ScaledSingleFrameObs;
        for (size_t i = 0; i < HISTORY_LENGTH; ++i) {
            size_t tensor_offset = i * SINGLE_FRAME_LENGTH;
            std::copy(this->HistoryObsBuffer[i].begin(),
                      this->HistoryObsBuffer[i].end(),
                      this->InputHistoryTensor.Array().begin() + tensor_offset);
        }
        
        this->InputObsTensor.Array() = decltype(this->InputObsTensor.Array())::clamp(
            this->InputObsTensor.Array(), -this->Base::ClipObservation, this->Base::ClipObservation);
        this->InputHistoryTensor.Array() = decltype(this->InputHistoryTensor.Array())::clamp(
            this->InputHistoryTensor.Array(), -this->Base::ClipObservation, this->Base::ClipObservation);
    }

    void PostProcess() override {
        // --- 1. 获取网络输出 (leggedlab顺序) ---
        auto RawAction_leggedlab = this->OutputTensor.toVector();

        // --- 2. 裁剪和缩放 (leggedlab顺序) ---
        auto ClippedRawAction_leggedlab = MotorValVec::clamp(RawAction_leggedlab, -this->Base::ClipAction, this->Base::ClipAction);
        auto ScaledAction_leggedlab = ClippedRawAction_leggedlab * this->OutputScaleVec; // 注意：先缩放再加默认位置
        
        // --- 3. 重映射回CtrlZ顺序 ---
        MotorValVec ClippedRawAction_ctrlz;
        MotorValVec ScaledAction_with_default_ctrlz;
        for (size_t i = 0; i < JOINT_NUMBER; ++i) {
            int leggedlab_idx = i;
            int ctrlz_idx = this->LeggedLabToCtrlZ_RemapIdx[leggedlab_idx];
            
            // 将leggedlab顺序的第i个关节的数据，放到CtrlZ顺序的第ctrlz_idx个位置
            ClippedRawAction_ctrlz[ctrlz_idx] = ClippedRawAction_leggedlab[leggedlab_idx];
            // 在CtrlZ顺序下加上对应的默认关节位置
            ScaledAction_with_default_ctrlz[ctrlz_idx] = ScaledAction_leggedlab[leggedlab_idx] + this->Base::JointDefaultPos[ctrlz_idx];
        }

        // --- 4. 写入数据总线 (全部使用CtrlZ顺序) ---
        // **V2.0改动**: 写入CtrlZ顺序的last_action，保证日志和下一帧输入的一致性
        this->Base::Scheduler->template SetData<"NetLastAction">(ClippedRawAction_ctrlz);
        
        // **V2.0改动**: NetScaledAction现在是CtrlZ顺序，且包含了default_pos，更具物理意义
        this->Base::Scheduler->template SetData<"NetScaledAction">(ScaledAction_with_default_ctrlz);

        // --- 5. 最终目标位置限幅并写入总线 (CtrlZ顺序) ---
        auto FinalTargetPosition = MotorValVec::clamp(ScaledAction_with_default_ctrlz, this->Base::JointClipLower, this->Base::JointClipUpper);
        this->Base::Scheduler->template SetData<"TargetMotorPosition">(FinalTargetPosition);

        // --- 6. 记录推理时间 ---
        this->end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time);
        InferencePrecision inference_time = static_cast<InferencePrecision>(duration.count()) / 1000.0; // ms
        this->Base::Scheduler->template SetData<"InferenceTime">(inference_time);
    }
};

} // namespace z