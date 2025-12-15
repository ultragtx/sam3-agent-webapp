# SAM3 Agent WebApp - Quick Start Guide

这个指南帮助你快速启动和使用 SAM3 Agent Web 应用。

## 架构概览

```
前端 (React + Vite) <-> 后端 (Flask) <-> SAM3 模型 + Qwen3-VL (MLLM)
```

## 快速开始

### 1. 环境准备

```bash
# 确保已安装
- Python 3.10+
- Node.js 18+
- CUDA GPU (用于 SAM3)
```

### 2. 配置环境变量

```bash
cd /nfshome/xinrong/pubNAS3/projects/sam3-agent-webapp
cp .env.example .env

# 编辑 .env 文件,配置以下关键参数:
# MLLM_API_BASE=http://your-qwen3-vl-endpoint/v1
# MLLM_API_KEY=your-api-key
# MLLM_MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
# SAM3_MODEL_PATH=/path/to/sam3/checkpoint
```

### 3. 安装后端依赖

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 如果需要,安装 SAM3
# pip install git+https://github.com/facebookresearch/sam3.git
```

### 4. 安装前端依赖

```bash
cd ../frontend
npm install
```

### 5. 启动服务

**启动后端 (终端1):**
```bash
cd backend
source venv/bin/activate
python app.py

# 后端将在 http://localhost:5000 启动
# SAM3 模型会自动加载到 GPU 显存
```

**启动前端 (终端2):**
```bash
cd frontend
npm run dev

# 前端将在 http://localhost:5173 启动
```

## 使用方法

### 基本流程

1. **打开浏览器访问** `http://localhost:5173`

2. **上传图片**
   - 点击或拖拽图片到上传区域
   - 支持 JPG, PNG, WebP 格式

3. **输入查询**
   - 在文本框输入你想分割的对象描述
   - 例如: "person holding a phone"
   - 例如: "红色的杯子"

4. **配置 MLLM (可选)**
   - API Base URL: Qwen3-VL 服务地址
   - API Key: 认证密钥
   - Model Name: 模型名称
   - Max Tokens: 最大生成长度

5. **启动 Agent**
   - 点击 "Start Agent" 按钮
   - Agent 将开始迭代推理
   - 右侧会实时显示每一轮的结果

### 查看结果

- **状态指示**: 显示后端连接状态和 SAM3 加载状态
- **最终结果**: 显示最终分割的masks和可视化图像
- **迭代历史**: 展开查看每一轮的:
  - MLLM 的输入消息
  - MLLM 的输出响应
  - 工具调用 (segment_phrase, examine_each_mask, etc.)
  - 中间结果图像

## 示例查询

### 简单对象
```
"person"
"car"
"dog"
"laptop"
```

### 描述性查询
```
"person holding a phone"
"red car on the left"
"dog sitting on the grass"
"laptop with a blue screen"
```

### 复杂推理
```
"the person wearing glasses"
"the second car from the right"
"the cat that is sleeping"
"objects on the table that can hold water"
```

## 工作原理

### Agent 迭代流程

1. **初始化**: MLLM 接收图像和用户查询
2. **分析**: MLLM 分析图像,决定调用哪个工具
3. **工具调用**:
   - `segment_phrase`: 用简单名词短语分割对象
   - `examine_each_mask`: 逐个检查每个 mask 是否正确
   - `select_masks_and_return`: 选择最终的 masks
   - `report_no_mask`: 报告没有匹配的对象
4. **迭代**: 根据结果继续推理,直到找到正确的 masks

### 可视化功能

- ✅ 所有 MLLM 输入/输出都显示在界面上
- ✅ SAM3 生成的每个中间结果都可视化
- ✅ 每一轮的 mask 都用不同颜色标注编号
- ✅ 最终结果高亮显示

## 故障排除

### SAM3 加载失败

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查模型路径
ls -la /path/to/sam3/checkpoint

# 检查 GPU 内存
nvidia-smi
```

### MLLM 连接失败

```bash
# 测试 API 端点
curl http://your-mllm-endpoint/v1/models

# 检查 API key
echo $MLLM_API_KEY

# 查看后端日志
# 在后端终端查看错误信息
```

### 前端无法连接后端

```bash
# 检查后端是否运行
curl http://localhost:5000/api/health

# 检查端口是否被占用
lsof -i :5000
lsof -i :5173
```

## API 端点

### 后端 API

- `GET /api/health` - 健康检查
- `POST /api/models/load` - 加载 SAM3 模型
- `POST /api/upload` - 上传图片
- `POST /api/agent/run` - 运行 Agent
- `POST /api/sam3/segment` - 直接 SAM3 分割
- `GET /api/config` - 获取配置
- `GET /api/outputs/<filename>` - 获取输出文件

## 高级配置

### 调整 MLLM 参数

在前端界面或 `.env` 文件中调整:

- `temperature`: 控制生成多样性 (0.1-1.0)
- `max_tokens`: 限制生成长度
- `top_p`: nucleus sampling 参数

### SAM3 模型大小

在 `.env` 中设置:
```
SAM3_MODEL_SIZE=large  # large, base, small
```

### 内存管理

SAM3 会常驻显存以加快推理速度。如果需要释放:

```python
# 在后端调用
POST /api/models/unload  # (需要实现此端点)
```

## 性能优化

### 后端优化
- SAM3 模型缓存在 GPU 显存中
- 使用 Flask 的生产环境部署 (gunicorn/uwsgi)
- 配置合理的超时时间

### 前端优化
- 生产构建: `npm run build`
- 使用 nginx 反向代理
- 启用 gzip 压缩

## 部署到生产环境

### Docker 部署 (推荐)

创建 `docker-compose.yml`:
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - MLLM_API_BASE=${MLLM_API_BASE}
      - SAM3_MODEL_PATH=/models/sam3
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

### 使用 gunicorn 运行后端

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app --timeout 300
```

## 常见问题

**Q: Agent 一直在运行但没有结果?**
A: 检查 max_generations 参数,可能陷入了死循环。查看后端日志了解详情。

**Q: 内存不足?**
A: 尝试使用更小的 SAM3 模型 (base 或 small),或增加系统交换空间。

**Q: MLLM 响应太慢?**
A: 调整 max_tokens 参数,或使用更快的 MLLM 模型。

**Q: 可视化图片不显示?**
A: 检查文件路径权限,确保 outputs 文件夹可访问。

## 开发指南

### 后端开发

```bash
# 安装开发依赖
pip install pytest black flake8

# 运行测试
pytest tests/

# 代码格式化
black backend/
```

### 前端开发

```bash
# 开发服务器
npm run dev

# 类型检查
npm run type-check

# Lint
npm run lint

# 构建
npm run build
```

## 贡献

欢迎提交 Pull Request!

## License

基于 Meta 的 SAM3 项目,遵循其开源协议。

## 联系方式

如有问题,请提交 Issue 或联系开发者。
