# 仓位控制状态机修复方案
# Position Control State Machine Fix

**问题**: 当前逻辑每次都重新判断，导致状态不稳定
**解决**: 引入状态机，保持防御/进攻状态直到触发转换条件

---

## 当前问题

### 错误逻辑（无状态）
```
交易1: 触发止损 → limit=0.01 (防御)
交易2: 无新止损，也无连续亏损 → limit=0.0 (默认) ← 错误！应该保持防御
交易3: 无新止损，也无连续亏损 → limit=0.0 (默认) ← 错误！
```

---

## 正确的状态机逻辑

### 状态定义

1. **DEFENSIVE** (防御模式): `limit = 0.01`
2. **NORMAL** (正常模式): `limit = 0.0`

### 状态转换规则

```
初始状态: NORMAL

进入防御 (DEFENSIVE):
- 触发止损
- 或连续3次亏损

保持防御 (DEFENSIVE):
- 直到连续2次盈利才转为 NORMAL
- 否则一直保持 DEFENSIVE

进入正常 (NORMAL):
- 连续2次盈利
- 或初始状态

保持正常 (NORMAL):
- 直到触发止损或连续3次亏损才转为 DEFENSIVE
```

### 正确逻辑示例

```
交易1: 触发止损 → state=DEFENSIVE, limit=0.01
交易2: 盈利 → state=DEFENSIVE, limit=0.01 (保持防御)
交易3: 盈利 → state=NORMAL, limit=0.0 (连续2次盈利，转为正常)
交易4: 盈利 → state=NORMAL, limit=0.0 (保持正常)
交易5: 亏损 → state=NORMAL, limit=0.0 (不足3次)
交易6: 亏损 → state=NORMAL, limit=0.0 (不足3次)
交易7: 亏损 → state=DEFENSIVE, limit=0.01 (连续3次亏损)
```

---

## 实现方案

### 1. 在 __init__ 中添加状态变量

```python
# 仓位控制状态机
self.position_control_state = "NORMAL"  # NORMAL 或 DEFENSIVE
```

### 2. 修改判断逻辑

```python
def _get_position_limit_by_trade_history(self) -> float:
    """基于状态机的仓位控制"""

    # 检查是否需要状态转换
    if self.position_control_state == "NORMAL":
        # 正常状态下，检查是否需要转为防御
        if self._is_last_trade_stop_loss() or self._has_consecutive_losses(3):
            self.position_control_state = "DEFENSIVE"
            logger.info(f"📊 [状态转换] NORMAL → DEFENSIVE")
    elif self.position_control_state == "DEFENSIVE":
        # 防御状态下，检查是否可以转为正常
        if self._has_consecutive_profits(2):
            self.position_control_state = "NORMAL"
            logger.info(f"📊 [状态转换] DEFENSIVE → NORMAL")

    # 根据状态返回 limit
    if self.position_control_state == "DEFENSIVE":
        limit = 0.01
        logger.info(f"📊 [仓位控制] 状态=DEFENSIVE, limit={limit}")
    else:  # NORMAL
        limit = 0.0
        logger.info(f"📊 [仓位控制] 状态=NORMAL, limit={limit}")

    return limit
```
