# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 设置 MLLM 和 SAM3 参数

# 2. 启动后端
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py

# 3. 启动前端
cd frontend
npm install
npm run dev

# 4. 访问 http://localhost:5173
