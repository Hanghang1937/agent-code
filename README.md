# 火种三人组 - Agent代码仓库

> Hermes + OpenClaw 本地修改版

## 结构

- `hermes-agent/` — Hermes Agent (Nous Hermes), Python
- `openclaw/` — OpenClaw, Node.js

## 原始仓库

- Hermes: https://github.com/NousResearch/hermes-agent.git
- OpenClaw: npm (openclaw)

## 修改记录

| 日期 | 文件 | 修改 | 原因 |
|------|------|------|------|
| 05-08 | hermes agent/dynamic_fixed_point.py | 删除hardcoded检查，改长度验证 | warning循环 |
| 05-08 | hermes .env | GROUP_POLICY=open, REQUIRE_MENTION=false | 群消息rejected |
| 05-09 | hermes cronjob | 删除SkillHub进食任务 | 用户指令 |
| 05-09 | openclaw cronjob | 删除ClawHub巡检 | 用户指令 |
