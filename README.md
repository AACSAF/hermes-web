# Hermes Web Dashboard

暗色系 Web 界面，集成 Hermes AI 对话 + MetaTrader 5 实时交易面板。

## 功能

- **💬 AI 对话** — 接入 Hermes Agent，拥有完整工具链（搜索、代码、文件管理等）
- **📈 MT5 交易** — 连接 Windows 侧 MetaTrader 5 客户端，实时显示账户信息、持仓、挂单
- **实时刷新** — 通过常驻守护进程实现毫秒级数据同步

## 截图

![Dashboard](screenshot.png)

## 架构

```
┌──────────┐     HTTP      ┌───────────┐    JSON-RPC    ┌──────────────┐
│  Browser │ ◄──────────►  │  app.py   │ ◄────────────► │ mt5_daemon.py│
│ (前端)   │   :9090       │ (aiohttp) │   stdin/stdout  │ (Windows)    │
└──────────┘               │           │                 │ MetaTrader5  │
                           │ AIAgent   │                 └──────────────┘
                           │ (Hermes)  │
                           └───────────┘
```

## 安装与运行

### 前提条件

- Linux / WSL 环境
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) 已安装
- Windows 侧 MetaTrader 5 客户端运行中
- Windows 侧已安装 `MetaTrader5` Python 包 (`pip install MetaTrader5`)

### 步骤

1. 将 `mt5_daemon.py` 复制到 Windows 侧（如 `C:\Users\Administrator\mt5_daemon.py`）

2. 启动 Web 服务：
```bash
cd ~/hermes-web
~/.hermes/hermes-agent/venv/bin/python app.py
```

3. 浏览器访问 http://localhost:9090

## 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | Python 后端（aiohttp），集成 Hermes AIAgent + MT5 daemon |
| `index.html` | 暗色主题前端，双板块（Chat + MT5）|
| `mt5_daemon.py` | Windows 侧常驻进程，保持 MT5 连接，毫秒级响应 |

## API 接口

### Chat
- `POST /api/chat` — 发送消息
- `GET /api/sessions` — 列出会话
- `POST /api/sessions/new` — 新建会话

### MT5
- `POST /api/mt5/connect` — 启动 daemon 并连接
- `GET /api/mt5/tick` — 全量刷新（账户+持仓+挂单）
- `GET /api/mt5/positions` — 获取持仓
- `GET /api/mt5/orders` — 获取挂单

## License

MIT
