# Odoo Intelligent RAG Search Server

A FastAPI-based intelligent product search and recommendation server that connects to Odoo and uses RAG (Retrieval-Augmented Generation) technology to provide smart product recommendations.

## 🚀 Features

- **Intelligent Search**: Understands natural language queries and maps them to relevant products
- **RAG Technology**: Uses sentence transformers for semantic product matching
- **Smart Recommendations**: Provides premium and budget alternatives
- **Medical Product Support**: Specialized categories for medical equipment and monitoring devices
- **Voiceflow Integration**: Ready for chatbot integration
- **Production Ready**: Includes logging, error handling, and security features

## 📋 Prerequisites

- Python 3.8+
- Odoo instance with API access
- Odoo API key

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd odoo-rag-service
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the project root:
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

## 🚀 Quick Start

1. **Start the server**
   ```bash
   python app.py
   ```

2. **Test the connection**
   ```bash
   curl http://localhost:8001/health
   ```

3. **View API documentation**
   Open http://localhost:8001/docs in your browser

## 📚 API Endpoints

### Health Check
```http
GET /health
```

### Basic Product Search
```http
POST /search_products
Content-Type: application/json

{
  "term": "CABLE PARA SPO2",
  "limit": 10
}
```

### Intelligent Search
```http
POST /intelligent_search
Content-Type: application/json

{
  "query": "I need a cable for SPO2 monitoring",
  "max_results": 10,
  "include_alternatives": true
}
```

### Product Recommendations (Voiceflow Integration)
```http
POST /recommend_products
Content-Type: application/json

{
  "query": "Hospital monitoring equipment",
  "max_results": 5,
  "include_alternatives": true
}
```

### Get Categories
```http
GET /categories
```

### Test Connection
```http
POST /test_connection
```

## 🏥 Medical Product Support

The server includes specialized categories for medical equipment:

- **Medical Monitoring**: SPO2 cables, vital signs monitoring, pulse oximeters
- **Medical Equipment**: Hospital equipment, clinical devices, medical instruments

### Example Queries
- "I need a cable for SPO2 monitoring" → Finds CABLE PARA SPO2 - MINDRAY
- "Hospital monitoring equipment" → Finds monitors, accessories, replacement parts
- "Vital signs monitoring accessories" → Finds monitoring equipment, sensors, cables

## 🔗 Voiceflow Integration

For Voiceflow chatbot integration, use the `/recommend_products` endpoint:

1. **Create HTTP Request node in Voiceflow**
2. **URL**: `http://localhost:8001/recommend_products`
3. **Method**: `POST`
4. **Headers**: `Content-Type: application/json`
5. **Body**:
   ```json
   {
     "query": "{{user_input}}",
     "max_results": 5,
     "include_alternatives": true
   }
   ```

## 🏗️ Architecture

The server uses a two-stage approach:

1. **Query Analysis**: Categorizes natural language queries into product categories
2. **Odoo Search**: Searches Odoo products using relevant terms
3. **RAG Analysis**: Uses sentence transformers to rank products by relevance
4. **Value Ranking**: Ranks products by features, quality, and value proposition
5. **Alternative Generation**: Provides premium and budget alternatives

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ODOO_URL` | Your Odoo instance URL | Yes |
| `ODOO_DB` | Database name | Yes |
| `ODOO_LOGIN` | User email | Yes |
| `ODOO_API_KEY` | API key | Yes |
| `USE_RAG` | Enable RAG features | No (default: true) |
| `EMBED_MODEL` | Sentence transformer model | No (default: all-MiniLM-L6-v2) |
| `HOST` | Server host | No (default: 0.0.0.0) |
| `PORT` | Server port | No (default: 8001) |

### Product Categories

The server includes predefined categories for:
- Back pain products
- Joint pain products
- Headache relief
- General pain relief
- Medical monitoring equipment
- Medical equipment

## 🚀 Deployment

### Docker (Recommended)

1. **Build the image**
   ```bash
   docker build -t odoo-rag-service .
   ```

2. **Run the container**
   ```bash
   docker run -p 8001:8001 --env-file .env odoo-rag-service
   ```

### Production Considerations

- Use HTTPS in production
- Configure proper CORS settings
- Set up rate limiting
- Use environment-specific configurations
- Monitor server health and performance

## 🧪 Testing

Test the server endpoints:

```bash
# Health check
curl http://localhost:8001/health

# Test connection
curl -X POST http://localhost:8001/test_connection

# Search products
curl -X POST http://localhost:8001/search_products \
  -H "Content-Type: application/json" \
  -d '{"term": "CABLE PARA SPO2", "limit": 5}'

# Intelligent search
curl -X POST http://localhost:8001/recommend_products \
  -H "Content-Type: application/json" \
  -d '{"query": "I need SPO2 monitoring cable", "max_results": 5}'
```

## 📝 Response Format

### Intelligent Search Response
```json
{
  "query": "I need a cable for SPO2 monitoring",
  "search_terms": ["SPO2", "cable", "monitoring", "mindray"],
  "categories": {"medical_monitoring": 1.0},
  "recommendations": [
    {
      "name": "CABLE PARA SPO2 - MINDRAY",
      "list_price": 299.99,
      "description_sale": "Professional SPO2 monitoring cable",
      "qty_available": 5,
      "_value_score": 8.5
    }
  ],
  "alternatives": {
    "premium": [...],
    "budget": [...]
  },
  "search_metadata": {
    "total_products_found": 15,
    "ranked_products": 10,
    "final_recommendations": 5,
    "search_terms_used": ["SPO2", "cable", "monitoring"]
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify your Odoo configuration
3. Test with the health endpoint first
4. Review the API documentation at `/docs`

## 🔒 Security

- Never commit `.env` files
- Use environment variables for secrets
- Configure CORS appropriately for production
- Implement rate limiting for public endpoints
- Use HTTPS in production environments
