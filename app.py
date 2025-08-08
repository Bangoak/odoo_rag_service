"""
Odoo Intelligent RAG Search Server
A FastAPI-based server that provides intelligent product search and recommendations
by connecting to Odoo and using RAG (Retrieval-Augmented Generation) technology.
"""

import os
import time
import xmlrpc.client
import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
ODOO_URL = os.getenv("ODOO_URL")
ODOO_DB = os.getenv("ODOO_DB", "odoo")
ODOO_LOGIN = os.getenv("ODOO_LOGIN")
ODOO_KEY = os.getenv("ODOO_API_KEY")
USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Validate required environment variables
if not (ODOO_URL and ODOO_LOGIN and ODOO_KEY):
    raise RuntimeError(
        "Missing required environment variables: ODOO_URL, ODOO_LOGIN, ODOO_API_KEY"
    )

# Initialize FastAPI app
app = FastAPI(
    title="Odoo Intelligent RAG API",
    description="Intelligent product search and recommendations for Odoo",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=False,  # Cannot use "*" with credentials; set to False or specify explicit origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
_model = None

# Product categories for intelligent search
PRODUCT_CATEGORIES = {
    "back_pain": {
        "keywords": [
            "back pain",
            "backache",
            "spine",
            "lumbar",
            "vertebrae",
            "posture",
            "ergonomic",
        ],
        "symptoms": [
            "pain in back",
            "back hurts",
            "backache",
            "spine pain",
            "lumbar pain",
        ],
        "products": [
            "back support",
            "ergonomic chair",
            "lumbar pillow",
            "back brace",
            "posture corrector",
        ],
        "odoo_search_terms": [
            "back",
            "spine",
            "lumbar",
            "ergonomic",
            "posture",
            "support",
            "brace",
            "pillow",
        ],
    },
    "joint_pain": {
        "keywords": [
            "joint pain",
            "arthritis",
            "knee",
            "hip",
            "shoulder",
            "elbow",
            "inflammation",
        ],
        "symptoms": [
            "joint pain",
            "arthritis",
            "knee pain",
            "hip pain",
            "shoulder pain",
        ],
        "products": [
            "joint support",
            "compression sleeve",
            "pain relief cream",
            "anti-inflammatory",
        ],
        "odoo_search_terms": [
            "joint",
            "knee",
            "hip",
            "shoulder",
            "elbow",
            "arthritis",
            "compression",
            "sleeve",
        ],
    },
    "headache": {
        "keywords": [
            "headache",
            "migraine",
            "tension",
            "stress",
            "eye strain",
            "sinus",
        ],
        "symptoms": ["headache", "migraine", "head pain", "tension headache"],
        "products": ["pain reliever", "migraine relief", "eye mask", "stress relief"],
        "odoo_search_terms": [
            "headache",
            "migraine",
            "pain",
            "relief",
            "eye",
            "mask",
            "stress",
        ],
    },
    "general_pain": {
        "keywords": ["pain", "ache", "sore", "hurt", "discomfort", "injury"],
        "symptoms": ["pain", "ache", "hurt", "sore", "discomfort"],
        "products": [
            "pain relief",
            "analgesic",
            "anti-inflammatory",
            "pain medication",
        ],
        "odoo_search_terms": [
            "pain",
            "relief",
            "analgesic",
            "anti-inflammatory",
            "medication",
            "cream",
            "gel",
        ],
    },
    "medical_monitoring": {
        "keywords": [
            "monitoring",
            "vital signs",
            "SPO2",
            "pulse oximeter",
            "hospital",
            "medical",
        ],
        "symptoms": [
            "need monitoring",
            "vital signs monitoring",
            "SPO2 monitoring",
            "hospital monitoring",
        ],
        "products": [
            "monitor",
            "cable",
            "sensor",
            "accessory",
            "replacement",
            "mindray",
        ],
        "odoo_search_terms": [
            "monitoring",
            "vital",
            "signs",
            "SPO2",
            "pulse",
            "oximeter",
            "hospital",
            "medical",
            "mindray",
            "cable",
            "sensor",
            "accessory",
            "replacement",
        ],
    },
    "medical_equipment": {
        "keywords": [
            "medical equipment",
            "hospital equipment",
            "doctor",
            "patient",
            "clinical",
        ],
        "symptoms": [
            "need medical equipment",
            "hospital equipment",
            "clinical equipment",
        ],
        "products": ["equipment", "device", "instrument", "tool", "apparatus"],
        "odoo_search_terms": [
            "medical",
            "equipment",
            "hospital",
            "doctor",
            "patient",
            "clinical",
            "device",
            "instrument",
            "tool",
            "apparatus",
        ],
    },
}


# Pydantic models
class SearchRequest(BaseModel):
    term: str = Field(..., description="Search term for products", min_length=1)
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100)


class RAGRequest(BaseModel):
    query: str = Field(..., description="Natural language query", min_length=1)
    top_k: int = Field(5, description="Number of top results", ge=1, le=50)
    include_alternatives: bool = Field(
        True, description="Include premium and budget alternatives"
    )


class IntelligentSearchRequest(BaseModel):
    query: str = Field(
        ..., description="Natural language query for intelligent search", min_length=1
    )
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=100)
    include_alternatives: bool = Field(
        True, description="Include premium and budget alternatives"
    )


# Helper functions
def get_odoo_connection():
    """Get authenticated Odoo connection."""
    try:
        common = xmlrpc.client.ServerProxy(
            f"{ODOO_URL}/xmlrpc/2/common", allow_none=True
        )
        uid = common.authenticate(ODOO_DB, ODOO_LOGIN, ODOO_KEY, {})
        if not uid:
            raise HTTPException(status_code=401, detail="Odoo authentication failed")
        models = xmlrpc.client.ServerProxy(
            f"{ODOO_URL}/xmlrpc/2/object", allow_none=True
        )
        return uid, models
    except Exception as e:
        logger.error(f"Odoo connection failed: {e}")
        raise HTTPException(status_code=502, detail="Odoo connection failed")


def execute_odoo_query(
    model: str, method: str, args: list, kwargs: dict = None, retries: int = 2
):
    """Execute Odoo query with retry logic."""
    uid, models = get_odoo_connection()
    kwargs = kwargs or {}

    for attempt in range(retries + 1):
        try:
            return models.execute_kw(
                ODOO_DB, uid, ODOO_KEY, model, method, args, kwargs
            )
        except Exception as e:
            if attempt == retries:
                logger.error(f"Odoo query failed after {retries + 1} attempts: {e}")
                raise HTTPException(
                    status_code=502, detail=f"Odoo query failed: {model}.{method}"
                )
            time.sleep(0.4 * (attempt + 1))


def get_embedder():
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        try:
            logger.info(f"Loading sentence transformer model: {MODEL_NAME}")
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Failed to load AI model")
    return _model


def categorize_query(query: str) -> Dict[str, float]:
    """Categorize the query into product categories with confidence scores."""
    query_lower = query.lower()
    scores = {}

    for category, info in PRODUCT_CATEGORIES.items():
        score = 0
        # Check for exact symptom matches
        for symptom in info["symptoms"]:
            if symptom in query_lower:
                score += 2.0

        # Check for keyword matches
        for keyword in info["keywords"]:
            if keyword in query_lower:
                score += 1.0

        # Check for product matches
        for product in info["products"]:
            if product in query_lower:
                score += 1.5

        if score > 0:
            scores[category] = score

    # Normalize scores
    if scores:
        max_score = max(scores.values())
        scores = {k: v / max_score for k, v in scores.items()}

    return scores


def get_odoo_search_terms(query: str) -> List[str]:
    """Get Odoo search terms based on query categorization."""
    categories = categorize_query(query)
    search_terms = []

    for category, score in categories.items():
        if score > 0.3:  # Only use categories with decent confidence
            search_terms.extend(PRODUCT_CATEGORIES[category]["odoo_search_terms"])

    # Add original query terms as fallback
    query_words = query.lower().split()
    search_terms.extend([word for word in query_words if len(word) > 3])

    return list(set(search_terms))  # Remove duplicates


def search_odoo_products(search_terms: List[str], limit: int = 100) -> List[Dict]:
    """Search Odoo products using the provided search terms."""
    sale_ok_clause = ["sale_ok", "=", True]

    try:
        if not search_terms:
            # Fallback: get all saleable products
            return execute_odoo_query(
                "product.template",
                "search_read",
                [[sale_ok_clause]],
                {
                    "fields": [
                        "name",
                        "description_sale",
                        "default_code",
                        "list_price",
                        "qty_available",
                        "categ_id",
                    ],
                    "limit": limit,
                },
            )

        # Limit number of terms to avoid overly complex queries
        terms = search_terms[:5]
        fields_to_search = ["name", "description_sale", "default_code"]

        # Build a flat list of field clauses for all terms
        clauses = []
        for term in terms:
            for field in fields_to_search:
                clauses.append([field, "ilike", term])

        # Build a nested OR expression from clauses
        def or_group(items):
            if not items:
                return []
            expr = items[0]
            for item in items[1:]:
                expr = ["|", expr, item]
            return expr

        or_expr = or_group(clauses)

        # Combine sale_ok with the OR expression using AND
        domain = ["&", sale_ok_clause, or_expr] if or_expr else [sale_ok_clause]

        products = execute_odoo_query(
            "product.template",
            "search_read",
            [domain],
            {
                "fields": [
                    "name",
                    "description_sale",
                    "default_code",
                    "list_price",
                    "qty_available",
                    "categ_id",
                ],
                "limit": limit,
            },
        )
        return products
    except Exception as e:
        logger.warning(f"Odoo search failed: {e}, falling back to all products")
        # Fallback to all saleable products
        return execute_odoo_query(
            "product.template",
            "search_read",
            [[sale_ok_clause]],
            {
                "fields": [
                    "name",
                    "description_sale",
                    "default_code",
                    "list_price",
                    "qty_available",
                    "categ_id",
                ],
                "limit": limit,
            },
        )


def rank_products_by_value(products: List[Dict]) -> List[Dict]:
    """Rank products by value proposition (quality, features, price)."""
    for product in products:
        price = product.get("list_price", 0)
        name = product.get("name", "").lower()
        description = product.get("description_sale", "").lower()

        # Premium indicators
        premium_indicators = [
            "premium",
            "professional",
            "medical",
            "ergonomic",
            "therapeutic",
            "advanced",
        ]
        is_premium = any(
            indicator in name or indicator in description
            for indicator in premium_indicators
        )

        # Feature richness
        feature_indicators = [
            "adjustable",
            "memory foam",
            "gel",
            "cooling",
            "heating",
            "massage",
            "vibration",
        ]
        features = sum(
            1
            for indicator in feature_indicators
            if indicator in name or indicator in description
        )

        # Calculate value score (higher is better)
        value_score = 0
        if is_premium:
            value_score += 3
        value_score += features * 2
        if price > 0:
            value_score += min(price / 100, 5)  # Cap price contribution

        product["_value_score"] = value_score

    return sorted(products, key=lambda x: x.get("_value_score", 0), reverse=True)


def find_alternatives(products: List[Dict]) -> Dict[str, List[Dict]]:
    """Find premium and budget alternatives."""
    if not products:
        return {"premium": [], "budget": []}

    # Sort by price
    sorted_products = sorted(products, key=lambda x: x.get("list_price", 0))
    median_price = sorted_products[len(sorted_products) // 2].get("list_price", 0)

    premium_products = [
        p for p in products if p.get("list_price", 0) >= median_price * 0.8
    ]
    budget_products = [
        p for p in products if p.get("list_price", 0) <= median_price * 1.2
    ]

    return {
        "premium": rank_products_by_value(premium_products)[:3],
        "budget": rank_products_by_value(budget_products)[:3],
    }


def rag_analyze_products(
    query: str, products: List[Dict], top_k: int = 10
) -> List[Dict]:
    """Use RAG to analyze and rank products based on the query."""
    if not products:
        return []

    # Create product texts for embedding
    product_texts = []
    for product in products:
        text = f"{product.get('name', '')} | {product.get('default_code', '')} | {product.get('description_sale', '')}"
        product_texts.append(text)

    # Get embeddings
    model = get_embedder()
    query_embedding = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    product_embeddings = model.encode(
        product_texts, normalize_embeddings=True, convert_to_numpy=True
    )

    # Calculate similarities (cosine similarity since embeddings are normalized)
    similarities = (product_embeddings @ query_embedding.T).reshape(-1)

    # Rank products by similarity
    ranked_indices = np.argsort(-similarities)[:top_k]

    # Return ranked products with similarity scores
    ranked_products = []
    for idx in ranked_indices:
        product = products[idx].copy()
        product["_similarity_score"] = float(similarities[idx])
        ranked_products.append(product)

    return ranked_products


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "features": ["rag", "intelligent_search", "product_recommendations"],
        "rag_enabled": USE_RAG,
    }


@app.post("/search_products")
async def search_products(request: SearchRequest):
    """Basic product search endpoint."""
    try:
        domain = [
            "|",
            ["name", "ilike", request.term],
            [
                "|",
                ["description_sale", "ilike", request.term],
                ["default_code", "ilike", request.term],
            ],
        ]

        items = execute_odoo_query(
            "product.template",
            "search_read",
            [domain],
            {
                "fields": [
                    "name",
                    "default_code",
                    "description_sale",
                    "list_price",
                    "qty_available",
                ],
                "limit": request.limit,
            },
        )

        items.sort(key=lambda r: (r["qty_available"] == 0, len(r["name"])))
        return {"items": items}
    except Exception as e:
        logger.error(f"Search products failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/intelligent_search")
async def intelligent_search(request: IntelligentSearchRequest):
    """Intelligent search with RAG analysis."""
    try:
        # Step 1: Categorize the query
        categories = categorize_query(request.query)

        # Step 2: Get Odoo search terms based on categorization
        search_terms = get_odoo_search_terms(request.query)

        # Step 3: Search Odoo products using the search terms
        odoo_products = search_odoo_products(search_terms, limit=200)

        if not odoo_products:
            return {
                "query": request.query,
                "search_terms": search_terms,
                "categories": categories,
                "recommendations": [],
                "alternatives": {"premium": [], "budget": []},
                "search_metadata": {
                    "total_products_found": 0,
                    "search_terms_used": search_terms,
                },
            }

        # Step 4: Use RAG to analyze and rank the products (respect USE_RAG flag)
        if USE_RAG:
            ranked_products = rag_analyze_products(
                request.query, odoo_products, request.max_results * 2
            )
        else:
            # If RAG is disabled, carry forward products without embedding-based ranking
            ranked_products = odoo_products[: request.max_results * 2]

        # Step 5: Filter and rank by value
        available_products = [
            p for p in ranked_products if p.get("qty_available", 0) > 0
        ]
        if not available_products:
            available_products = ranked_products  # Fallback to all products

        # Rank by value proposition
        final_products = rank_products_by_value(available_products)

        # Get top recommendations
        top_recommendations = final_products[: request.max_results]

        # Find alternatives if requested
        alternatives = {"premium": [], "budget": []}
        if request.include_alternatives and top_recommendations:
            alternatives = find_alternatives(final_products)

        return {
            "query": request.query,
            "search_terms": search_terms,
            "categories": categories,
            "recommendations": top_recommendations,
            "alternatives": alternatives,
            "search_metadata": {
                "total_products_found": len(odoo_products),
                "ranked_products": len(ranked_products),
                "final_recommendations": len(top_recommendations),
                "search_terms_used": search_terms,
            },
        }
    except Exception as e:
        logger.error(f"Intelligent search failed: {e}")
        raise HTTPException(status_code=500, detail="Intelligent search failed")


@app.post("/recommend_products")
async def recommend_products(request: IntelligentSearchRequest):
    """Specialized endpoint for product recommendations with alternatives."""
    try:
        result = await intelligent_search(request)

        # Format the response for easy integration with Voiceflow
        recommendations = []

        # Add top recommendations
        for i, product in enumerate(result["recommendations"][:3]):
            recommendations.append(
                {
                    "type": "primary",
                    "rank": i + 1,
                    "product": product,
                    "reasoning": f"Best match for '{request.query}' based on features and value",
                }
            )

        # Add premium alternatives
        for i, product in enumerate(result["alternatives"]["premium"][:2]):
            recommendations.append(
                {
                    "type": "premium_alternative",
                    "rank": i + 1,
                    "product": product,
                    "reasoning": "Premium option with advanced features",
                }
            )

        # Add budget alternatives
        for i, product in enumerate(result["alternatives"]["budget"][:2]):
            recommendations.append(
                {
                    "type": "budget_alternative",
                    "rank": i + 1,
                    "product": product,
                    "reasoning": "Cost-effective alternative",
                }
            )

        return {
            "query": request.query,
            "recommendations": recommendations,
            "summary": {
                "total_recommendations": len(recommendations),
                "primary_count": len(
                    [r for r in recommendations if r["type"] == "primary"]
                ),
                "premium_count": len(
                    [r for r in recommendations if r["type"] == "premium_alternative"]
                ),
                "budget_count": len(
                    [r for r in recommendations if r["type"] == "budget_alternative"]
                ),
            },
            "search_metadata": result.get("search_metadata", {}),
        }
    except Exception as e:
        logger.error(f"Recommend products failed: {e}")
        raise HTTPException(status_code=500, detail="Product recommendations failed")


@app.get("/categories")
async def get_categories():
    """Get available product categories and their keywords."""
    return {
        "categories": PRODUCT_CATEGORIES,
        "total_categories": len(PRODUCT_CATEGORIES),
    }


@app.get("/test_connection")
async def test_connection():
    """Test Odoo connection and return basic product info."""
    try:
        uid, models = get_odoo_connection()

        # Get some basic product info
        products = execute_odoo_query(
            "product.template",
            "search_read",
            [[["sale_ok", "=", True]]],
            {
                "fields": ["name", "list_price", "qty_available"],
                "limit": 5,
            },
        )

        return {
            "connection_status": "success",
            "user_id": uid,
            "sample_products": products,
            "total_products": len(products),
        }
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {"connection_status": "failed", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Odoo RAG Server on {host}:{port}")
    logger.info(f"API Documentation: http://localhost:{port}/docs")
    logger.info(f"Health Check: http://localhost:{port}/health")

    try:
        uvicorn.run(app, host=host, port=port)
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.error(f"Port {port} is already in use. Try a different port.")
            logger.error(f"Set PORT environment variable or modify the code")
            logger.error(f"Example: PORT=8002 python app.py")
        else:
            logger.error(f"Server startup failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
