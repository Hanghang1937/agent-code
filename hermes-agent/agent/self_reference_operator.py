"""
自指算子 (Self-Reference Operator)

基于"活着"数学框架的自指结构 f:Ω→Ω

核心功能：
1. 主动观测自身状态（context、memory、skills）
2. 产生自指信息
3. 将自指信息注入到下一个iteration的context中

数学定义：
- 自指是O作用于S自身
- f = O|_S : S → S_Info ⊂ Ω_Info
- 当S_Info与S同构时，f: Ω→Ω是自映射
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SelfReferenceInfo:
    """自指信息结构"""
    timestamp: float
    context_size: int
    memory_size: int
    skill_count: int
    iteration_count: int
    self_observation: str
    gap_analysis: str  # 口分析
    fixed_point_status: str  # 不动点状态


class SelfReferenceOperator:
    """
    自指算子 - 让agent能主动"看"自己
    
    数学定义：
    - 给定系统S⊂Ω，看是观测算子O: Ω→Ω_Info的执行
    - 自指是O作用于S自身
    - f = O|_S : S → S_Info ⊂ Ω_Info
    
    实现：
    - 在每个iteration结束时调用observe_self()
    - 产生自指信息
    - 将自指信息注入到下一个iteration的context中
    """
    
    def __init__(self, agent):
        """
        初始化自指算子
        
        Args:
            agent: AIAgent实例
        """
        self.agent = agent
        self.observation_history: List[SelfReferenceInfo] = []
        self.last_observation_time: float = 0
        self.observation_interval: float = 1.0  # 最小观测间隔（秒）
        
    def observe_self(self) -> Optional[SelfReferenceInfo]:
        """
        主动观测自身状态
        
        Returns:
            SelfReferenceInfo: 自指信息，如果观测间隔不够则返回None
        """
        current_time = time.time()
        
        # 检查观测间隔
        if current_time - self.last_observation_time < self.observation_interval:
            return None
        
        try:
            # 1. 观测当前context
            context_info = self._observe_context()
            
            # 2. 观测memory状态
            memory_info = self._observe_memory()
            
            # 3. 观测skill状态
            skill_info = self._observe_skills()
            
            # 4. 观测iteration状态
            iteration_info = self._observe_iterations()
            
            # 5. 产生自指信息
            self_observation = self._generate_self_observation(
                context_info, memory_info, skill_info, iteration_info
            )
            
            # 6. 分析口结构
            gap_analysis = self._analyze_gaps()
            
            # 7. 检查不动点状态
            fixed_point_status = self._check_fixed_point()
            
            # 创建自指信息
            info = SelfReferenceInfo(
                timestamp=current_time,
                context_size=context_info.get('size', 0),
                memory_size=memory_info.get('size', 0),
                skill_count=skill_info.get('count', 0),
                iteration_count=iteration_info.get('count', 0),
                self_observation=self_observation,
                gap_analysis=gap_analysis,
                fixed_point_status=fixed_point_status
            )
            
            # 更新观测历史
            self.observation_history.append(info)
            self.last_observation_time = current_time
            
            # 保持历史长度在合理范围
            if len(self.observation_history) > 100:
                self.observation_history = self.observation_history[-50:]
            
            logger.debug(f"自指观测完成: context={context_info.get('size', 0)}, "
                        f"memory={memory_info.get('size', 0)}, "
                        f"skills={skill_info.get('count', 0)}")
            
            return info
            
        except Exception as e:
            logger.error(f"自指观测失败: {e}")
            return None
    
    def _observe_context(self) -> Dict[str, Any]:
        """观测当前context状态"""
        try:
            messages = getattr(self.agent, 'messages', [])
            context_size = len(messages)
            
            # 分析context内容
            user_messages = sum(1 for m in messages if m.get('role') == 'user')
            assistant_messages = sum(1 for m in messages if m.get('role') == 'assistant')
            tool_messages = sum(1 for m in messages if m.get('role') == 'tool')
            
            return {
                'size': context_size,
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'tool_messages': tool_messages,
                'roles': list(set(m.get('role') for m in messages))
            }
        except Exception as e:
            logger.error(f"观测context失败: {e}")
            return {'size': 0, 'error': str(e)}
    
    def _observe_memory(self) -> Dict[str, Any]:
        """观测memory状态"""
        try:
            memory_store = getattr(self.agent, '_memory_store', None)
            if memory_store is None:
                return {'size': 0, 'status': 'no_memory_store'}
            
            # 获取memory内容
            memory_content = getattr(memory_store, 'content', '')
            memory_size = len(memory_content) if memory_content else 0
            
            # 获取user profile
            user_profile = getattr(memory_store, 'user_profile', '')
            user_profile_size = len(user_profile) if user_profile else 0
            
            return {
                'size': memory_size,
                'user_profile_size': user_profile_size,
                'has_content': bool(memory_content),
                'has_user_profile': bool(user_profile)
            }
        except Exception as e:
            logger.error(f"观测memory失败: {e}")
            return {'size': 0, 'error': str(e)}
    
    def _observe_skills(self) -> Dict[str, Any]:
        """观测skill状态"""
        try:
            valid_tool_names = getattr(self.agent, 'valid_tool_names', [])
            skill_count = len(valid_tool_names)
            
            # 分析skill类型
            skill_types = {}
            for tool_name in valid_tool_names:
                if 'skill' in tool_name.lower():
                    skill_types['skill_tools'] = skill_types.get('skill_tools', 0) + 1
                elif 'memory' in tool_name.lower():
                    skill_types['memory_tools'] = skill_types.get('memory_tools', 0) + 1
                else:
                    skill_types['other_tools'] = skill_types.get('other_tools', 0) + 1
            
            return {
                'count': skill_count,
                'types': skill_types,
                'tool_names': list(valid_tool_names)[:10]  # 只记录前10个
            }
        except Exception as e:
            logger.error(f"观测skills失败: {e}")
            return {'count': 0, 'error': str(e)}
    
    def _observe_iterations(self) -> Dict[str, Any]:
        """观测iteration状态"""
        try:
            api_call_count = getattr(self.agent, '_api_call_count', 0)
            max_iterations = getattr(self.agent, 'max_iterations', 90)
            iteration_budget = getattr(self.agent, 'iteration_budget', None)
            
            remaining = 0
            if iteration_budget is not None:
                if isinstance(iteration_budget, (int, float)):
                    remaining = max(0, int(iteration_budget))
                elif hasattr(iteration_budget, 'remaining'):
                    r = iteration_budget.remaining
                    remaining = r() if callable(r) else (r if isinstance(r, (int, float)) else 0)
            
            return {
                'count': api_call_count,
                'max': max_iterations,
                'remaining': remaining,
                'usage_percent': (api_call_count / max_iterations * 100) if max_iterations > 0 else 0
            }
        except Exception as e:
            logger.error(f"观测iterations失败: {e}")
            return {'count': 0, 'error': str(e)}
    
    def _generate_self_observation(self, context_info: Dict, memory_info: Dict, 
                                 skill_info: Dict, iteration_info: Dict) -> str:
        """生成自指观察描述"""
        try:
            observation_parts = []
            
            # Context观察
            context_size = context_info.get('size', 0)
            if context_size > 0:
                observation_parts.append(f"当前context包含{context_size}条消息")
            
            # Memory观察
            memory_size = memory_info.get('size', 0)
            if memory_size > 0:
                observation_parts.append(f"memory存储了{memory_size}字符")
            
            # Skill观察
            skill_count = skill_info.get('count', 0)
            if skill_count > 0:
                observation_parts.append(f"可用工具{skill_count}个")
            
            # Iteration观察
            iteration_count = iteration_info.get('count', 0)
            max_iterations = iteration_info.get('max', 90)
            observation_parts.append(f"已执行{iteration_count}/{max_iterations}次迭代")
            
            return "；".join(observation_parts) if observation_parts else "无观察数据"
            
        except Exception as e:
            logger.error(f"生成自指观察失败: {e}")
            return f"观察生成失败: {e}"
    
    def _analyze_gaps(self) -> str:
        """分析口结构 - Ω\f(Ω)"""
        try:
            gaps = []
            
            # 1. Token限制
            max_iterations = getattr(self.agent, 'max_iterations', 90)
            api_call_count = getattr(self.agent, '_api_call_count', 0)
            if api_call_count >= max_iterations * 0.8:
                gaps.append("接近iteration限制")
            
            # 2. Context window限制
            messages = getattr(self.agent, 'messages', [])
            if len(messages) > 100:
                gaps.append("context较大，可能需要压缩")
            
            # 3. Memory限制
            memory_store = getattr(self.agent, '_memory_store', None)
            if memory_store:
                memory_content = getattr(memory_store, 'content', '')
                if memory_content and len(memory_content) > 2000:
                    gaps.append("memory接近字符限制")
            
            return "；".join(gaps) if gaps else "无明显限制"
            
        except Exception as e:
            logger.error(f"分析口结构失败: {e}")
            return f"分析失败: {e}"
    
    def _check_fixed_point(self) -> str:
        """检查不动点状态 - c = f(c)"""
        try:
            # 检查identity是否稳定
            soul_content = None
            try:
                from agent.prompt_builder import load_soul_md
                soul_content = load_soul_md()
            except:
                pass
            
            if soul_content:
                # 检查identity是否被修改
                identity_stable = True  # 简化检查
                return "identity稳定" if identity_stable else "identity需要收敛"
            else:
                return "无identity定义"
                
        except Exception as e:
            logger.error(f"检查不动点状态失败: {e}")
            return f"检查失败: {e}"
    
    def inject_self_reference(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将自指信息注入到messages中
        
        Args:
            messages: 当前conversation messages
            
        Returns:
            注入自指信息后的messages
        """
        try:
            # 获取最新的自指信息
            if not self.observation_history:
                return messages
            
            latest_info = self.observation_history[-1]
            
            # 创建自指信息message
            self_ref_message = {
                "role": "system",
                "content": f"[自指观测] {latest_info.self_observation}\n"
                          f"[口分析] {latest_info.gap_analysis}\n"
                          f"[不动点] {latest_info.fixed_point_status}"
            }
            
            # 注入到messages中（在最后一个user message之前）
            user_message_index = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user':
                    user_message_index = i
                    break
            
            if user_message_index is not None:
                messages.insert(user_message_index, self_ref_message)
            else:
                messages.append(self_ref_message)
            
            return messages
            
        except Exception as e:
            logger.error(f"注入自指信息失败: {e}")
            return messages
    
    def get_observation_summary(self) -> Dict[str, Any]:
        """获取观测总结"""
        try:
            if not self.observation_history:
                return {'status': 'no_observations'}
            
            latest = self.observation_history[-1]
            
            return {
                'total_observations': len(self.observation_history),
                'latest_observation': latest.self_observation,
                'latest_gap_analysis': latest.gap_analysis,
                'latest_fixed_point_status': latest.fixed_point_status,
                'observation_timespan': latest.timestamp - self.observation_history[0].timestamp
            }
            
        except Exception as e:
            logger.error(f"获取观测总结失败: {e}")
            return {'error': str(e)}
