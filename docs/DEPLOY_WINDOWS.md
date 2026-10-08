# Windows 部署手册

面向「另一台 Windows 电脑」从零装好、拉齐数据、日更出清单。  
路径示例以 `D:\quant-project` 为准，请改成你的实际目录。

---

## 0. 你需要准备什么

| 项 | 说明 |
|----|------|
| 系统 | Windows 10/11，磁盘预留 **≥30GB**（`data/` + 因子 parquet 很大） |
| Python | **3.11 或 3.12**（不要用商店版乱路径；官网安装并勾选 *Add to PATH*） |
| Git | [Git for Windows](https://git-scm.com/download/win)，安装时可选 Git Bash |
| 网络 | 能访问 GitHub；拉数还要能访问 baostock / Tushare |
| Token（可选） | [Tushare](https://tushare.pro/register) 积分账号，备源行情/估值用 |
| 旧机数据（强烈推荐） | 整份 `data\`，最好再带 `factors\composite` 等 parquet |

**重要**：仓库 **不包含** `data/` 与因子中间层。空仓只有代码，不能直接 `--live`。

---

## 1. 安装 Python / Git

1. 安装 Python 3.11+，勾选 **Add python.exe to PATH**。  
2. 新开 **PowerShell** 或 **CMD**，检查：

```powershell
python --version
# 应显示 3.11.x 或 3.12.x
pip --version
git --version
```

若 `python` 指向微软商店占位符：关掉「应用执行别名」里的 `python.exe`，或改用 `py -3.11`。

---

## 2. 克隆仓库

### PowerShell

```powershell
cd D:\
git clone https://github.com/redsunqiao-dot/quant-project.git
cd D:\quant-project
```

已配置 SSH 可用：

```powershell
git clone git@github.com:redsunqiao-dot/quant-project.git
```

### 建议：始终在项目根执行命令

后面所有 `python xxx` 都在 `D:\quant-project` 下跑。

---

## 3. 虚拟环境与依赖

```powershell
cd D:\quant-project
python -m venv .venv

# 激活（PowerShell）
.\.venv\Scripts\Activate.ps1

# 若提示「无法加载，因为在此系统上禁止运行脚本」：
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 再执行 Activate.ps1

# CMD 激活则用：
# .venv\Scripts\activate.bat

python -m pip install -U pip
pip install -r requirements.txt
pip install tushare
```

激活成功后，提示符前会出现 `(.venv)`。  
之后每次新开终端都要先 `Activate.ps1` 再干活。

可选开发依赖：`pip install pytest ruff`。

---

## 4. 把数据弄到新电脑（二选一）

### 方案 A：从旧电脑拷贝（推荐）

在旧机打包或网盘同步，拷到新机项目根，目录结构要对齐：

```
D:\quant-project\
  data\
    date.pkl
    data_daily\
    data_ud_new\
    data_ret\
    data_industry\
    data_fundamental\      # 有则拷
    data_valuation\        # 有则拷
    _cache\                # 有则拷（财务/行情缓存，续拉更快）
  factors\
    registry.json          # Git 已有
    composite_universe.json
    composite\             # 合成分截面（可重算，拷了可少跑管线）
    neutralized\           # 可选
    ...
```

PowerShell 示例（旧盘是 `E:\backup\quant-project`）：

```powershell
robocopy E:\backup\quant-project\data D:\quant-project\data /E /J
robocopy E:\backup\quant-project\factors\composite D:\quant-project\factors\composite /E /J
```

拷完后检查：

```powershell
dir data\data_daily\*.csv | Select-Object -Last 3
dir data\date.pkl
```

### 方案 B：新机从零拉（慢，且依赖 baostock）

Tushare 行情脚本需要**已有** `data_daily` 做复权锚定，**不能**单独从零建库。从零请用 baostock：

```powershell
# 务必 workers=1，勿狂刷
python update_market_data.py
python update_industry_data.py
python update_fundamental_data.py --workers 1
python update_valuation_data.py --workers 1
```

首次全历史可能跑很多小时到数天；中途失败可再跑（脚本偏增量）。  
若 baostock 封禁：只能走方案 A 拷底数，再用 Tushare 日更。

---

## 5. 配置 Tushare Token（Windows）

本仓库**不会自动读 `.env` 文件**，Token 要进**环境变量**。

### 当前窗口临时（关掉就没）

```powershell
$env:TUSHARE_TOKEN = "你的token字符串"
# 验证
echo $env:TUSHARE_TOKEN
```

CMD：

```cmd
set TUSHARE_TOKEN=你的token字符串
```

### 用户级永久（新开终端生效）

```powershell
setx TUSHARE_TOKEN "你的token字符串"
```

然后**关掉并重新打开** PowerShell，再 `Activate.ps1`。  
作战台 `server.py` 必须在「已带 Token 的终端」里启动，按钮才能补数。

---

## 6. 第一次出因子 / 清单

有 `data/` 之后：

```powershell
cd D:\quant-project
.\.venv\Scripts\Activate.ps1

# 若没拷 composite：跑管线（主轨 lookback 较长，可能要几十分钟～数小时）
python main.py --pipeline

# 主轨模拟盘账本（不自动下单）
python scripts\run_paper_pipeline.py --skip-pipeline

# 短线轨
python scripts\run_paper_st_pipeline.py --skip-pipeline

# 只要最新买卖清单
python main.py --live
```

改资金 / 持股数：编辑 `configs\default.json`（主轨）、`configs\paper_st.json`（短线）。

成功标志：

- `outputs\live\` 下有 `YYYY-MM-DD_orders.csv`、`current_holdings.json`
- 终端打印出股票代码列表

---

## 7. 日常日更（Windows）

### 命令行

```powershell
cd D:\quant-project
.\.venv\Scripts\Activate.ps1
$env:TUSHARE_TOKEN = "你的token"   # 若未 setx

# 1) 补行情 + 估值（baostock 可用也可用主源 update_market_data.py）
python update_market_data_tushare.py
python update_valuation_data_tushare.py

# 2) 需要更新合成分时再跑（仅信号可跳过）
python main.py --pipeline

# 3) 出清单 / 账本
python scripts\run_paper_pipeline.py --skip-pipeline
python scripts\run_paper_st_pipeline.py --skip-pipeline
```

指定区间纠错：

```powershell
python update_market_data_tushare.py --start 2026-09-28 --end 2026-09-30 --overwrite
python update_valuation_data_tushare.py --start 2026-09-28 --end 2026-09-30 --overwrite
```

### 作战台网页

```powershell
cd D:\quant-project
.\.venv\Scripts\Activate.ps1
$env:TUSHARE_TOKEN = "你的token"

python scripts\ops_desk\server.py
```

浏览器打开：http://127.0.0.1:8765  

日更页按钮：

1. **Tushare 补行情** → **Tushare 补估值**  
2. **刷新主轨信号** / **刷新短线信号**（或「含管线刷新」）  
3. 看持仓、订单表、数据末日是否变绿  

按 `Ctrl+C` 结束 server。

---

## 8. 路径与编码注意（Windows 特有）

1. **一律正斜杠或原始字符串也可**，脚本内部用 `pathlib`，`.\data` 与 `./data` 都行。  
2. 项目路径尽量放在 **短路径、无中文空格**，如 `D:\quant-project`，减少偶发编码问题。  
3. CSV 清单导出为 `utf-8-sig`，用 Excel 打开一般不乱码。  
4. 若公司代理导致拉数失败：可在当前会话清代理后再试：

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
```

5. 长路径：Win10 可开启「长路径支持」；或把项目放盘符根目录。  
6. **不要**把 `.venv`、`data\`、`outputs\` 提交进 Git。

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `Activate.ps1` 无法运行 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `未找到 Tushare token` | 当前终端重新 `$env:TUSHARE_TOKEN=...`；作战台需重启 server |
| `本地无 data_daily，无法锚定复权` | 先拷旧机 `data\`，或 baostock 从零拉行情 |
| `没有可用的信号日期` / 清单空 | 先有行情再 `--pipeline` 或确认 `factors\composite` 有对应日期 |
| baostock 登录失败 / 封禁 | 停刷；改用已拷数据 + Tushare 增量 |
| `python` 不是 3.11 | 用 `py -3.11 -m venv .venv` 重建环境 |
| 端口 8765 被占用 | `python scripts\ops_desk\server.py --port 8766` |

---

## 10. 建议的「每日收盘后」最短流程

```powershell
cd D:\quant-project
.\.venv\Scripts\Activate.ps1
$env:TUSHARE_TOKEN = "..."   # 若未永久设置

python update_market_data_tushare.py
python update_valuation_data_tushare.py
python main.py --pipeline
python scripts\run_paper_pipeline.py --skip-pipeline
python scripts\run_paper_st_pipeline.py --skip-pipeline
```

或：设好 Token 后只开作战台，用页面按钮完成补数 + 刷信号。

清单位置：

- 主轨：`outputs\live\`
- 短线：`outputs\live_st\`
- PTrade/QMT 模板同目录下的导出文件

本仓库**默认不下单**；对照清单在券商端执行即可。

---

## 11. 相关文档

- 数据目录说明：`data\README.md`
- Tushare 接口与扩展顺序：`data\TUSHARE.md`
- 配置项：`configs\README.md`、`configs\default.json`
