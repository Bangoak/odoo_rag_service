# Deployment Guide

## 🚀 GitHub Deployment

### Step 1: Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Odoo Intelligent RAG Search Server"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/odoo-rag-service.git

# Push to GitHub
git push -u origin main
```

### Step 2: Set Up GitHub Secrets (Optional)

For CI/CD, you can set up GitHub Actions by creating `.github/workflows/deploy.yml`:

```yaml
name: Deploy Odoo RAG Service

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -c "import app; print('App imports successfully')"
```

### Step 3: Environment Setup

1. **Create `.env` file** (not committed to git):
   ```env
   ODOO_URL=https://your-company.odoo.com
   ODOO_DB=your_database_name
   ODOO_LOGIN=your_email@company.com
   ODOO_API_KEY=your_api_key_here
   USE_RAG=true
   EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
   HOST=0.0.0.0
   PORT=8001
   ```

2. **Get Odoo API Key**:
   - Log into your Odoo instance
   - Go to Settings → Users & Companies → Users
   - Click on your user account
   - Go to "Access Rights" tab
   - Scroll to "API Keys" section
   - Click "Create API Key"
   - Copy the generated key

### Step 4: Local Development

```bash
# Install dependencies
./deploy.sh install

# Start locally
./deploy.sh start-local

# Or use Docker
./deploy.sh deploy
```

### Step 5: Production Deployment

#### Option A: Docker Compose (Recommended)

```bash
# Full deployment
./deploy.sh deploy

# Or step by step
./deploy.sh build
./deploy.sh start
```

#### Option B: Docker

```bash
# Build image
docker build -t odoo-rag-service .

# Run container
docker run -p 8001:8001 --env-file .env odoo-rag-service
```

#### Option C: Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

## 🔒 Security Checklist

- [ ] `.env` file is in `.gitignore`
- [ ] No hardcoded credentials in code
- [ ] CORS settings configured for production
- [ ] HTTPS enabled in production
- [ ] Rate limiting implemented
- [ ] Input validation enabled
- [ ] Logging configured
- [ ] Health checks implemented

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8001/health

# Test connection
curl -X POST http://localhost:8001/test_connection

# Test search
curl -X POST http://localhost:8001/search_products \
  -H "Content-Type: application/json" \
  -d '{"term": "CABLE PARA SPO2", "limit": 5}'

# Test intelligent search
curl -X POST http://localhost:8001/recommend_products \
  -H "Content-Type: application/json" \
  -d '{"query": "I need SPO2 monitoring cable", "max_results": 5}'
```

## 📊 Monitoring

### Health Checks
- Endpoint: `GET /health`
- Docker health check configured
- Returns service status and version

### Logs
```bash
# View logs
./deploy.sh logs

# Or with Docker Compose
docker-compose logs -f
```

### Metrics
- Request/response times
- Error rates
- Model loading status
- Odoo connection status

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ODOO_URL` | Required | Your Odoo instance URL |
| `ODOO_DB` | `odoo` | Database name |
| `ODOO_LOGIN` | Required | User email |
| `ODOO_API_KEY` | Required | API key |
| `USE_RAG` | `true` | Enable RAG features |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | AI model |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8001` | Server port |

### Production Settings

```env
# Production environment
ODOO_URL=https://your-company.odoo.com
ODOO_DB=production_db
ODOO_LOGIN=api_user@company.com
ODOO_API_KEY=your_production_api_key
USE_RAG=true
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
HOST=0.0.0.0
PORT=8001

# Optional: Add logging
LOG_LEVEL=INFO
```

## 🚀 Scaling

### Horizontal Scaling
- Use load balancer
- Multiple container instances
- Shared Redis cache for model caching

### Vertical Scaling
- Increase container resources
- Use larger AI models
- Optimize database queries

## 🔄 Updates

### Code Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
./deploy.sh deploy
```

### Model Updates
```bash
# Update model in .env
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Restart service
./deploy.sh restart
```

## 🆘 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change port in .env
   PORT=8002
   ```

2. **Odoo connection failed**
   - Check credentials in `.env`
   - Verify API key permissions
   - Test Odoo connection manually

3. **Model loading slow**
   - First request may be slow
   - Consider pre-loading model
   - Use smaller model if needed

4. **Memory issues**
   - Increase container memory
   - Use smaller AI model
   - Optimize product search limits

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python app.py
```

## 📞 Support

For issues:
1. Check logs: `./deploy.sh logs`
2. Test health: `curl http://localhost:8001/health`
3. Verify environment: `./deploy.sh test`
4. Check API docs: `http://localhost:8001/docs`
