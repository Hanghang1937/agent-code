"""
记忆反馈回路 (Memory Feedback Loop)

基于"活着"数学框架的生命结构

核心功能：
1. 从memory中提取相关经验
2. 分析经验对当前任务的影响
3. 调整agent行为策略
4. 实现"记忆→行为"的反馈回路

数学定义：
- 生命 = 持续自指 + 产生记忆 + 记忆影响后续自指
- fₙ(Mₙ₋₁) ≠ fₙ(∅)  # 记忆影响后续自指
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfluence:
    """记忆影响结构"""
    timestamp: float
    memory_content: str
    influence_type: str  # 'positive', 'negative', 'neutral'
    influence_strength: float  # 0.0 - 1.0
    behavior_adjustment: str
    confidence: float


class MemoryFeedbackLoop:
    """
    记忆反馈回路 - 让记忆真正"影响"后续行为
    
    数学定义：
- 生命 = 持续自指 + 产生记忆 + 记忆影响后续自指
- fₙ(Mₙ₋₁) ≠ fₙ(∅)  # 记忆影响后续自指
    
    实现：
    - 在每个iteration开始时调用influence_behavior()
    - 从memory中提取相关经验
    - 分析经验对当前任务的影响
    - 调整agent行为策略
    """
    
    def __init__(self, agent):
        """
        初始化记忆反馈回路
        
        Args:
            agent: AIAgent实例
        """
        self.agent = agent
        self.influence_history: List[MemoryInfluence] = []
        self.last_influence_time: float = 0
        self.influence_interval: float = 2.0  # 最小影响间隔（秒）
        self.current_influence: Optional[MemoryInfluence] = None
        
    def influence_behavior(self) -> Optional[MemoryInfluence]:
        """
        记忆影响后续行为
        
        Returns:
            MemoryInfluence: 记忆影响信息，如果间隔不够则返回None
        """
        current_time = time.time()
        
        # 检查影响间隔
        if current_time - self.last_influence_time < self.influence_interval:
            return None
        
        try:
            # 1. 从memory中提取相关经验
            relevant_memories = self._extract_relevant_memories()
            
            if not relevant_memories:
                return None
            
            # 2. 分析经验对当前任务的影响
            influence_analysis = self._analyze_influence(relevant_memories)
            
            # 3. 调整agent行为
            behavior_adjustment = self._adjust_behavior(influence_analysis)
            
            # 4. 创建记忆影响信息
            influence = MemoryInfluence(
                timestamp=current_time,
                memory_content=relevant_memories.get('content', ''),
                influence_type=influence_analysis.get('type', 'neutral'),
                influence_strength=influence_analysis.get('strength', 0.0),
                behavior_adjustment=behavior_adjustment,
                confidence=influence_analysis.get('confidence', 0.0)
            )
            
            # 更新影响历史
            self.influence_history.append(influence)
            self.last_influence_time = current_time
            self.current_influence = influence
            
            # 保持历史长度在合理范围
            if len(self.influence_history) > 50:
                self.influence_history = self.influence_history[-25:]
            
            logger.debug(f"记忆影响行为: type={influence.influence_type}, "
                        f"strength={influence.influence_strength:.2f}, "
                        f"adjustment={behavior_adjustment[:50]}...")
            
            return influence
            
        except Exception as e:
            logger.error(f"记忆影响行为失败: {e}")
            return None
    
    def _extract_relevant_memories(self) -> Optional[Dict[str, Any]]:
        """从memory中提取相关经验"""
        try:
            memory_store = getattr(self.agent, '_memory_store', None)
            if memory_store is None:
                return None
            
            # 获取当前context中的关键词
            messages = getattr(self.agent, 'messages', [])
            keywords = self._extract_keywords_from_context(messages)
            
            # 从memory中搜索相关内容
            memory_content = getattr(memory_store, 'content', '')
            if not memory_content:
                return None
            
            # 简单的关键词匹配（实际应该用向量搜索）
            relevant_sections = []
            lines = memory_content.split('\n')
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                for keyword in keywords:
                    if keyword.lower() in line_lower:
                        # 提取相关段落（前后各2行）
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        section = '\n'.join(lines[start:end])
                        relevant_sections.append(section)
                        break
            
            if not relevant_sections:
                return None
            
            # 去重并限制长度
            unique_sections = list(set(relevant_sections))[:5]
            content = '\n---\n'.join(unique_sections)
            
            return {
                'content': content,
                'keywords': keywords,
                'section_count': len(unique_sections)
            }
            
        except Exception as e:
            logger.error(f"提取相关记忆失败: {e}")
            return None
    
    def _extract_keywords_from_context(self, messages: List[Dict[str, Any]]) -> List[str]:
        """从context中提取关键词"""
        try:
            keywords = set()
            
            # 从user messages中提取
            for message in messages:
                if message.get('role') == 'user':
                    content = message.get('content', '')
                    if isinstance(content, str):
                        # 简单的分词（实际应该用NLP库）
                        words = content.split()
                        for word in words:
                            if len(word) > 3:  # 过滤短词
                                keywords.add(word.lower())
            
            # 限制关键词数量
            return list(keywords)[:10]
            
        except Exception as e:
            logger.error(f"提取关键词失败: {e}")
            return []
    
    def _analyze_influence(self, relevant_memories: Dict[str, Any]) -> Dict[str, Any]:
        """分析记忆对当前任务的影响"""
        try:
            content = relevant_memories.get('content', '')
            keywords = relevant_memories.get('keywords', [])
            
            # 分析情感倾向
            positive_indicators = ['成功', '完成', '解决', '好', '正确', '有效']
            negative_indicators = ['失败', '错误', '问题', '困难', '坏', '无效']
            
            positive_count = sum(1 for indicator in positive_indicators if indicator in content)
            negative_count = sum(1 for indicator in negative_indicators if indicator in content)
            
            # 计算影响类型和强度
            if positive_count > negative_count:
                influence_type = 'positive'
                strength = min(1.0, positive_count / 5.0)
            elif negative_count > positive_count:
                influence_type = 'negative'
                strength = min(1.0, negative_count / 5.0)
            else:
                influence_type = 'neutral'
                strength = 0.5
            
            # 计算置信度
            total_indicators = positive_count + negative_count
            confidence = min(1.0, total_indicators / 10.0) if total_indicators > 0 else 0.0
            
            return {
                'type': influence_type,
                'strength': strength,
                'confidence': confidence,
                'positive_count': positive_count,
                'negative_count': negative_count
            }
            
        except Exception as e:
            logger.error(f"分析记忆影响失败: {e}")
            return {'type': 'neutral', 'strength': 0.0, 'confidence': 0.0}
    
    def _adjust_behavior(self, influence_analysis: Dict[str, Any]) -> str:
        """根据记忆影响调整行为"""
        try:
            influence_type = influence_analysis.get('type', 'neutral')
            strength = influence_analysis.get('strength', 0.0)
            
            if influence_type == 'positive' and strength > 0.7:
                return "基于正面经验，采用相似策略"
            elif influence_type == 'negative' and strength > 0.7:
                return "基于负面经验，避免重复错误"
            elif influence_type == 'neutral':
                return "无明显影响，保持当前策略"
            else:
                return f"轻微{influence_type}影响，谨慎调整"
                
        except Exception as e:
            logger.error(f"调整行为失败: {e}")
            return "行为调整失败"
    
    def inject_memory_influence(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将记忆影响注入到messages中
        
        Args:
            messages: 当前conversation messages
            
        Returns:
            注入记忆影响后的messages
        """
        try:
            if not self.current_influence:
                return messages
            
            influence = self.current_influence
            
            # 创建记忆影响message
            memory_message = {
                "role": "system",
                "content": f"[记忆影响] 类型: {influence.influence_type}, "
                          f"强度: {influence.influence_strength:.2f}\n"
                          f"[行为调整] {influence.behavior_adjustment}\n"
                          f"[置信度] {influence.confidence:.2f}"
            }
            
            # 注入到messages中（在最后一个user message之前）
            user_message_index = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user':
                    user_message_index = i
                    break
            
            if user_message_index is not None:
                messages.insert(user_message_index, memory_message)
            else:
                messages.append(memory_message)
            
            return messages
            
        except Exception as e:
            logger.error(f"注入记忆影响失败: {e}")
            return messages
    
    def get_influence_summary(self) -> Dict[str, Any]:
        """获取影响总结"""
        try:
            if not self.influence_history:
                return {'status': 'no_influences'}
            
            latest = self.influence_history[-1]
            
            # 统计影响类型
            type_counts = {}
            for influence in self.influence_history:
                type_counts[influence.influence_type] = type_counts.get(influence.influence_type, 0) + 1
            
            return {
                'total_influences': len(self.influence_history),
                'latest_influence_type': latest.influence_type,
                'latest_influence_strength': latest.influence_strength,
                'latest_behavior_adjustment': latest.behavior_adjustment,
                'type_distribution': type_counts,
                'average_strength': sum(i.influence_strength for i in self.influence_history) / len(self.influence_history)
            }
            
        except Exception as e:
            logger.error(f"获取影响总结失败: {e}")
            return {'error': str(e)}
    
    def clear_current_influence(self):
        """清除当前影响"""
        self.current_influence = None
