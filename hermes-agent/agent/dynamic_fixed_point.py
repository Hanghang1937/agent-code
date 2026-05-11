"""
动态不动点 (Dynamic Fixed Point)

基于"活着"数学框架的不动点c = f(c)

核心功能：
1. 维护agent的核心身份（identity）
2. 根据经验动态调整identity
3. 实现不动点收敛机制
4. 确保c = f(c)的稳定性

数学定义：
- 自指不动点：c = f(c)
- 活着指向自己
- 自指的递归终止条件
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FixedPointState:
    """不动点状态结构"""
    timestamp: float
    identity_content: str
    identity_hash: str
    convergence_status: str  # 'stable', 'converging', 'diverging'
    convergence_rate: float  # 收敛速率
    experience_count: int
    last_update_reason: str


class DynamicFixedPoint:
    """
    动态不动点 - 让identity能根据经验收敛
    
    数学定义：
    - 自指不动点：c = f(c)
    - 活着指向自己
    - 自指的递归终止条件
    
    实现：
    - 维护agent的核心身份（SOUL.md）
    - 根据经验动态调整identity
    - 实现不动点收敛机制
    - 确保c = f(c)的稳定性
    """
    
    def __init__(self, agent):
        """
        初始化动态不动点
        
        Args:
            agent: AIAgent实例
        """
        self.agent = agent
        self.identity_history: List[FixedPointState] = []
        self.current_identity: Optional[str] = None
        self.convergence_threshold: float = 0.01  # 收敛阈值
        self.max_history_length: int = 100
        
        # 加载初始identity
        self._load_initial_identity()
    
    def _load_initial_identity(self):
        """加载初始identity"""
        try:
            # 尝试从SOUL.md加载
            from agent.prompt_builder import load_soul_md
            soul_content = load_soul_md()
            
            if soul_content:
                self.current_identity = soul_content
                logger.debug("从SOUL.md加载identity")
            else:
                # 使用默认identity
                self.current_identity = "I am Hermes Agent, an AI assistant."
                logger.debug("使用默认identity")
            
            # 记录初始状态
            self._record_state("initial_load")
            
        except Exception as e:
            logger.error(f"加载初始identity失败: {e}")
            self.current_identity = "I am Hermes Agent."
    
    def converge(self, experience: Dict[str, Any]) -> Optional[FixedPointState]:
        """
        根据经验收敛不动点
        
        Args:
            experience: 经验信息
            
        Returns:
            FixedPointState: 更新后的不动点状态
        """
        try:
            # 1. 分析经验对identity的影响
            influence = self._analyze_experience_influence(experience)
            
            # 2. 计算identity调整
            adjustment = self._calculate_identity_adjustment(influence)
            
            # 3. 应用调整
            if adjustment['should_adjust']:
                new_identity = self._apply_adjustment(
                    self.current_identity, adjustment
                )
                
                # 4. 验证不动点性质
                if self._verify_fixed_point(new_identity):
                    self.current_identity = new_identity
                    
                    # 5. 记录状态
                    state = self._record_state(
                        f"convergence: {adjustment['reason']}"
                    )
                    
                    logger.debug(f"不动点收敛: {adjustment['reason']}")
                    return state
                else:
                    logger.warning("不动点验证失败，跳过更新")
                    return None
            else:
                # 无需调整
                return self._record_state("no_adjustment_needed")
                
        except Exception as e:
            logger.error(f"不动点收敛失败: {e}")
            return None
    
    def _analyze_experience_influence(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """分析经验对identity的影响"""
        try:
            # 提取经验关键信息
            experience_type = experience.get('type', 'unknown')
            experience_content = experience.get('content', '')
            experience_outcome = experience.get('outcome', 'neutral')
            
            # 分析影响方向
            if experience_outcome == 'positive':
                influence_direction = 'reinforce'
                influence_strength = 0.7
            elif experience_outcome == 'negative':
                influence_direction = 'adjust'
                influence_strength = 0.5
            else:
                influence_direction = 'neutral'
                influence_strength = 0.3
            
            # 分析影响内容
            influence_content = self._extract_influence_content(
                experience_content, experience_type
            )
            
            return {
                'direction': influence_direction,
                'strength': influence_strength,
                'content': influence_content,
                'type': experience_type,
                'outcome': experience_outcome
            }
            
        except Exception as e:
            logger.error(f"分析经验影响失败: {e}")
            return {'direction': 'neutral', 'strength': 0.0, 'content': ''}
    
    def _extract_influence_content(self, content: str, experience_type: str) -> str:
        """提取影响内容"""
        try:
            # 基于经验类型提取关键信息
            if experience_type == 'task_completion':
                # 任务完成经验
                if '成功' in content or '完成' in content:
                    return "强化当前能力"
                else:
                    return "需要改进方法"
            
            elif experience_type == 'user_correction':
                # 用户纠正经验
                return f"学习纠正: {content[:100]}"
            
            elif experience_type == 'skill_update':
                # 技能更新经验
                return f"技能进化: {content[:100]}"
            
            elif experience_type == 'memory_insight':
                # 记忆洞察经验
                return f"记忆洞察: {content[:100]}"
            
            else:
                return f"通用经验: {content[:100]}"
                
        except Exception as e:
            logger.error(f"提取影响内容失败: {e}")
            return "无法提取影响内容"
    
    def _calculate_identity_adjustment(self, influence: Dict[str, Any]) -> Dict[str, Any]:
        """计算identity调整"""
        try:
            direction = influence.get('direction', 'neutral')
            strength = influence.get('strength', 0.0)
            content = influence.get('content', '')
            
            # 判断是否需要调整
            if direction == 'neutral' or strength < 0.3:
                return {
                    'should_adjust': False,
                    'reason': '影响强度不足'
                }
            
            # 计算调整幅度
            adjustment_magnitude = strength * 0.1  # 限制调整幅度
            
            # 生成调整原因
            if direction == 'reinforce':
                reason = f"强化: {content}"
            elif direction == 'adjust':
                reason = f"调整: {content}"
            else:
                reason = f"其他: {content}"
            
            return {
                'should_adjust': True,
                'direction': direction,
                'magnitude': adjustment_magnitude,
                'reason': reason,
                'content': content
            }
            
        except Exception as e:
            logger.error(f"计算identity调整失败: {e}")
            return {'should_adjust': False, 'reason': str(e)}
    
    def _apply_adjustment(self, current_identity: str, 
                         adjustment: Dict[str, Any]) -> str:
        """应用调整到identity"""
        try:
            direction = adjustment.get('direction', 'neutral')
            content = adjustment.get('content', '')
            magnitude = adjustment.get('magnitude', 0.0)
            
            # 简单的文本调整（实际应该用更复杂的NLP）
            if direction == 'reinforce':
                # 强化当前identity
                if content not in current_identity:
                    new_identity = f"{current_identity}\n\n[强化] {content}"
                else:
                    new_identity = current_identity
            
            elif direction == 'adjust':
                # 调整identity
                # 添加调整说明
                adjustment_note = f"\n\n[调整] 基于经验: {content}"
                new_identity = current_identity + adjustment_note
            
            else:
                new_identity = current_identity
            
            # 限制长度
            if len(new_identity) > 5000:
                # 保留开头和结尾
                new_identity = new_identity[:2000] + "\n\n... [中间部分省略] ...\n\n" + new_identity[-2000:]
            
            return new_identity
            
        except Exception as e:
            logger.error(f"应用调整失败: {e}")
            return current_identity
    
    def _verify_fixed_point(self, identity: str) -> bool:
        """验证不动点性质 c = f(c)"""
        try:
            if not identity or len(identity.strip()) < 10:
                logger.warning("Identity过短或为空")
                return False
            if len(identity) > 5000:
                logger.info("Identity过长，执行截断保持稳定")
                self.current_identity = identity[:2000] + identity[-2000:] if len(identity) > 4000 else identity[:2500]
            return True
        except Exception as e:
            logger.error(f"验证不动点失败: {e}")
            return False
    
    def _record_state(self, reason: str) -> FixedPointState:
        """记录不动点状态"""
        try:
            import hashlib
            
            # 计算identity哈希
            identity_hash = hashlib.md5(
                self.current_identity.encode('utf-8')
            ).hexdigest()[:8]
            
            # 判断收敛状态
            convergence_status = self._determine_convergence_status()
            
            # 计算收敛速率
            convergence_rate = self._calculate_convergence_rate()
            
            # 创建状态
            state = FixedPointState(
                timestamp=time.time(),
                identity_content=self.current_identity[:200] + "..." if len(self.current_identity) > 200 else self.current_identity,
                identity_hash=identity_hash,
                convergence_status=convergence_status,
                convergence_rate=convergence_rate,
                experience_count=len(self.identity_history),
                last_update_reason=reason
            )
            
            # 更新历史
            self.identity_history.append(state)
            
            # 保持历史长度
            if len(self.identity_history) > self.max_history_length:
                self.identity_history = self.identity_history[-self.max_history_length//2:]
            
            return state
            
        except Exception as e:
            logger.error(f"记录状态失败: {e}")
            return None
    
    def _determine_convergence_status(self) -> str:
        """判断收敛状态"""
        try:
            if len(self.identity_history) < 2:
                return 'initial'
            
            # 检查最近的状态变化
            recent_states = self.identity_history[-5:]
            hash_changes = sum(1 for i in range(1, len(recent_states)) 
                             if recent_states[i].identity_hash != recent_states[i-1].identity_hash)
            
            if hash_changes == 0:
                return 'stable'
            elif hash_changes <= 2:
                return 'converging'
            else:
                return 'diverging'
                
        except Exception as e:
            logger.error(f"判断收敛状态失败: {e}")
            return 'unknown'
    
    def _calculate_convergence_rate(self) -> float:
        """计算收敛速率"""
        try:
            if len(self.identity_history) < 2:
                return 0.0
            
            # 计算哈希变化率
            recent_states = self.identity_history[-10:]
            hash_changes = sum(1 for i in range(1, len(recent_states)) 
                             if recent_states[i].identity_hash != recent_states[i-1].identity_hash)
            
            # 归一化到[0, 1]
            convergence_rate = 1.0 - (hash_changes / len(recent_states))
            
            return max(0.0, min(1.0, convergence_rate))
            
        except Exception as e:
            logger.error(f"计算收敛速率失败: {e}")
            return 0.0
    
    def get_current_identity(self) -> str:
        """获取当前identity"""
        return self.current_identity
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """获取收敛总结"""
        try:
            if not self.identity_history:
                return {'status': 'no_history'}
            
            latest_state = self.identity_history[-1]
            
            return {
                'current_identity_length': len(self.current_identity),
                'current_identity_hash': latest_state.identity_hash,
                'convergence_status': latest_state.convergence_status,
                'convergence_rate': latest_state.convergence_rate,
                'total_updates': len(self.identity_history),
                'last_update_reason': latest_state.last_update_reason
            }
            
        except Exception as e:
            logger.error(f"获取收敛总结失败: {e}")
            return {'error': str(e)}
    
    def inject_identity_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将identity上下文注入到messages中"""
        try:
            # 创建identity消息
            identity_message = {
                "role": "system",
                "content": f"[不动点状态] {self.current_identity[:500]}..."
            }
            
            # 注入到messages开头（在system message之后）
            if len(messages) > 0 and messages[0].get('role') == 'system':
                messages.insert(1, identity_message)
            else:
                messages.insert(0, identity_message)
            
            return messages
            
        except Exception as e:
            logger.error(f"注入identity上下文失败: {e}")
            return messages
