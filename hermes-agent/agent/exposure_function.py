"""
双向暴露函数 (Exposure Function)

基于"活着"数学框架的暴露函数E

核心功能：
1. 计算两个agent之间的暴露程度
2. 实现双向信息耦合
3. 支持多agent协作

数学定义：
- E(Sᵢ, Sⱼ) = (1/T)∫₀ᵀ I(Oᵢ(ωⱼ(t)); Oⱼ(ωᵢ(t))) dt
- 对称性：E(Sᵢ,Sⱼ) = E(Sⱼ,Sᵢ)
- 独立性：Sᵢ⊥Sⱼ ⟹ E = 0
- 耦合性：E > 0 ⟹ 两系统有信息耦合
"""

import time
import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExposureInfo:
    """暴露信息结构"""
    timestamp: float
    agent_i_id: str
    agent_j_id: str
    exposure_value: float  # E(Sᵢ, Sⱼ)
    mutual_information: float  # I(Oᵢ(ωⱼ); Oⱼ(ωᵢ))
    coupling_strength: str  # 'none', 'weak', 'medium', 'strong'
    observation_quality_i: float  # Oᵢ观测质量
    observation_quality_j: float  # Oⱼ观测质量
    is_symmetric: bool  # 是否对称


class ExposureFunction:
    """
    双向暴露函数 - 让多agent协作真正"双向"
    
    数学定义：
    - E(Sᵢ, Sⱼ) = (1/T)∫₀ᵀ I(Oᵢ(ωⱼ(t)); Oⱼ(ωᵢ(t))) dt
    - 对称性：E(Sᵢ,Sⱼ) = E(Sⱼ,Sᵢ)
    - 独立性：Sᵢ⊥Sⱼ ⟹ E = 0
    - 耦合性：E > 0 ⟹ 两系统有信息耦合
    
    实现：
    - 在delegate_task时计算E(Sᵢ,Sⱼ)
    - 根据E值调整协作策略
    - 支持多agent之间的双向信息耦合
    """
    
    def __init__(self):
        """初始化暴露函数"""
        self.exposure_history: List[ExposureInfo] = []
        self.agent_registry: Dict[str, Any] = {}  # agent_id -> agent_info
        self.observation_window: float = 60.0  # 观测时间窗口（秒）
        
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """
        注册agent到暴露函数
        
        Args:
            agent_id: agent唯一标识
            agent_info: agent信息
        """
        self.agent_registry[agent_id] = {
            'info': agent_info,
            'registered_at': time.time(),
            'last_observed': None,
            'observation_count': 0
        }
        logger.debug(f"注册agent到暴露函数: {agent_id}")
    
    def compute_exposure(self, agent_i_id: str, agent_j_id: str) -> Optional[ExposureInfo]:
        """
        计算两个agent之间的暴露程度
        
        Args:
            agent_i_id: 第一个agent的ID
            agent_j_id: 第二个agent的ID
            
        Returns:
            ExposureInfo: 暴露信息，如果agent未注册则返回None
        """
        try:
            # 检查agent是否注册
            if agent_i_id not in self.agent_registry:
                logger.warning(f"Agent {agent_i_id} 未注册")
                return None
            if agent_j_id not in self.agent_registry:
                logger.warning(f"Agent {agent_j_id} 未注册")
                return None
            
            # 获取agent信息
            agent_i_info = self.agent_registry[agent_i_id]['info']
            agent_j_info = self.agent_registry[agent_j_id]['info']
            
            # 计算观测质量
            observation_quality_i = self._compute_observation_quality(agent_i_info)
            observation_quality_j = self._compute_observation_quality(agent_j_info)
            
            # 计算互信息
            mutual_information = self._compute_mutual_information(
                agent_i_info, agent_j_info
            )
            
            # 计算暴露值
            exposure_value = self._compute_exposure_value(
                mutual_information, observation_quality_i, observation_quality_j
            )
            
            # 判断耦合强度
            coupling_strength = self._classify_coupling_strength(exposure_value)
            
            # 检查对称性
            is_symmetric = self._check_symmetry(
                agent_i_id, agent_j_id, exposure_value
            )
            
            # 创建暴露信息
            exposure_info = ExposureInfo(
                timestamp=time.time(),
                agent_i_id=agent_i_id,
                agent_j_id=agent_j_id,
                exposure_value=exposure_value,
                mutual_information=mutual_information,
                coupling_strength=coupling_strength,
                observation_quality_i=observation_quality_i,
                observation_quality_j=observation_quality_j,
                is_symmetric=is_symmetric
            )
            
            # 更新历史
            self.exposure_history.append(exposure_info)
            
            # 更新agent注册信息
            self.agent_registry[agent_i_id]['last_observed'] = time.time()
            self.agent_registry[agent_i_id]['observation_count'] += 1
            self.agent_registry[agent_j_id]['last_observed'] = time.time()
            self.agent_registry[agent_j_id]['observation_count'] += 1
            
            logger.debug(f"计算暴露: E({agent_i_id}, {agent_j_id}) = {exposure_value:.3f}, "
                        f"耦合强度: {coupling_strength}")
            
            return exposure_info
            
        except Exception as e:
            logger.error(f"计算暴露失败: {e}")
            return None
    
    def _compute_observation_quality(self, agent_info: Dict[str, Any]) -> float:
        """计算agent的观测质量"""
        try:
            # 基于agent信息计算观测质量
            # 这里简化处理，实际应该基于agent的能力和状态
            
            quality_factors = []
            
            # 1. 工具数量
            tool_count = len(agent_info.get('tools', []))
            if tool_count > 0:
                quality_factors.append(min(1.0, tool_count / 20.0))
            
            # 2. Memory大小
            memory_size = agent_info.get('memory_size', 0)
            if memory_size > 0:
                quality_factors.append(min(1.0, memory_size / 1000.0))
            
            # 3. 迭代次数
            iteration_count = agent_info.get('iteration_count', 0)
            if iteration_count > 0:
                quality_factors.append(min(1.0, iteration_count / 100.0))
            
            # 计算平均质量
            if quality_factors:
                return sum(quality_factors) / len(quality_factors)
            else:
                return 0.5  # 默认质量
                
        except Exception as e:
            logger.error(f"计算观测质量失败: {e}")
            return 0.0
    
    def _compute_mutual_information(self, agent_i_info: Dict[str, Any], 
                                  agent_j_info: Dict[str, Any]) -> float:
        """计算互信息 I(Oᵢ(ωⱼ); Oⱼ(ωᵢ))"""
        try:
            # 简化计算：基于agent信息的相似度
            # 实际应该基于真正的互信息计算
            
            # 1. 工具重叠度
            tools_i = set(agent_i_info.get('tools', []))
            tools_j = set(agent_j_info.get('tools', []))
            
            if tools_i and tools_j:
                tool_overlap = len(tools_i.intersection(tools_j)) / len(tools_i.union(tools_j))
            else:
                tool_overlap = 0.0
            
            # 2. Memory相似度
            memory_i = agent_i_info.get('memory_content', '')
            memory_j = agent_j_info.get('memory_content', '')
            
            if memory_i and memory_j:
                # 简单的字符串相似度
                memory_similarity = self._compute_string_similarity(memory_i, memory_j)
            else:
                memory_similarity = 0.0
            
            # 3. 目标相似度
            goal_i = agent_i_info.get('goal', '')
            goal_j = agent_j_info.get('goal', '')
            
            if goal_i and goal_j:
                goal_similarity = self._compute_string_similarity(goal_i, goal_j)
            else:
                goal_similarity = 0.0
            
            # 计算加权互信息
            mutual_information = (
                tool_overlap * 0.4 +
                memory_similarity * 0.3 +
                goal_similarity * 0.3
            )
            
            return mutual_information
            
        except Exception as e:
            logger.error(f"计算互信息失败: {e}")
            return 0.0
    
    def _compute_string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度（简化版本）"""
        try:
            if not str1 or not str2:
                return 0.0
            
            # 使用简单的字符重叠度
            set1 = set(str1.lower())
            set2 = set(str2.lower())
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算字符串相似度失败: {e}")
            return 0.0
    
    def _compute_exposure_value(self, mutual_information: float,
                              observation_quality_i: float,
                              observation_quality_j: float) -> float:
        """计算暴露值 E(Sᵢ, Sⱼ)"""
        try:
            # E = I * min(Qᵢ, Qⱼ)
            # 其中I是互信息，Q是观测质量
            
            exposure = mutual_information * min(observation_quality_i, observation_quality_j)
            
            # 归一化到[0, 1]
            return min(1.0, max(0.0, exposure))
            
        except Exception as e:
            logger.error(f"计算暴露值失败: {e}")
            return 0.0
    
    def _classify_coupling_strength(self, exposure_value: float) -> str:
        """分类耦合强度"""
        if exposure_value < 0.1:
            return 'none'
        elif exposure_value < 0.3:
            return 'weak'
        elif exposure_value < 0.7:
            return 'medium'
        else:
            return 'strong'
    
    def _check_symmetry(self, agent_i_id: str, agent_j_id: str,
                       current_exposure: float) -> bool:
        """检查对称性 E(Sᵢ,Sⱼ) = E(Sⱼ,Sᵢ)"""
        try:
            # 查找历史中反向的暴露值
            for exposure in reversed(self.exposure_history):
                if (exposure.agent_i_id == agent_j_id and 
                    exposure.agent_j_id == agent_i_id):
                    # 检查是否对称（允许小误差）
                    return abs(exposure.exposure_value - current_exposure) < 0.01
            
            # 没有历史记录，假设对称
            return True
            
        except Exception as e:
            logger.error(f"检查对称性失败: {e}")
            return False
    
    def get_coupling_recommendation(self, exposure_info: ExposureInfo) -> Dict[str, Any]:
        """获取协作建议"""
        try:
            coupling_strength = exposure_info.coupling_strength
            exposure_value = exposure_info.exposure_value
            
            if coupling_strength == 'none':
                return {
                    'recommendation': '独立工作',
                    'reason': '无信息耦合',
                    'action': '各自独立执行任务'
                }
            elif coupling_strength == 'weak':
                return {
                    'recommendation': '有限协作',
                    'reason': '弱信息耦合',
                    'action': '共享基本信息，减少重复工作'
                }
            elif coupling_strength == 'medium':
                return {
                    'recommendation': '中度协作',
                    'reason': '中等信息耦合',
                    'action': '共享上下文，协调任务分配'
                }
            else:  # strong
                return {
                    'recommendation': '紧密协作',
                    'reason': '强信息耦合',
                    'action': '深度共享状态，联合决策'
                }
                
        except Exception as e:
            logger.error(f"获取协作建议失败: {e}")
            return {'recommendation': '默认协作', 'reason': str(e)}
    
    def inject_exposure_info(self, messages: List[Dict[str, Any]], 
                           exposure_info: ExposureInfo) -> List[Dict[str, Any]]:
        """将暴露信息注入到messages中"""
        try:
            # 创建暴露信息message
            exposure_message = {
                "role": "system",
                "content": f"[暴露函数] E({exposure_info.agent_i_id}, {exposure_info.agent_j_id}) = "
                          f"{exposure_info.exposure_value:.3f}\n"
                          f"[耦合强度] {exposure_info.coupling_strength}\n"
                          f"[互信息] {exposure_info.mutual_information:.3f}\n"
                          f"[对称性] {'是' if exposure_info.is_symmetric else '否'}"
            }
            
            # 注入到messages中
            messages.append(exposure_message)
            
            return messages
            
        except Exception as e:
            logger.error(f"注入暴露信息失败: {e}")
            return messages
    
    def get_exposure_summary(self) -> Dict[str, Any]:
        """获取暴露总结"""
        try:
            if not self.exposure_history:
                return {'status': 'no_exposures'}
            
            # 统计耦合强度
            coupling_counts = {}
            for exposure in self.exposure_history:
                coupling_counts[exposure.coupling_strength] = \
                    coupling_counts.get(exposure.coupling_strength, 0) + 1
            
            # 计算平均暴露值
            avg_exposure = sum(e.exposure_value for e in self.exposure_history) / \
                          len(self.exposure_history)
            
            return {
                'total_exposures': len(self.exposure_history),
                'average_exposure': avg_exposure,
                'coupling_distribution': coupling_counts,
                'registered_agents': len(self.agent_registry),
                'latest_exposure': self.exposure_history[-1].exposure_value
            }
            
        except Exception as e:
            logger.error(f"获取暴露总结失败: {e}")
            return {'error': str(e)}
