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
            logger.info("No search terms provided, getting all saleable products")
            return execute_odoo_query(
                "product.product",
                "search_read",
                [[sale_ok_clause]],
                {
                    "fields": [
                        "name",
                        "description_sale",
                        "website_description",  # Add website_description
                        "default_code",
                        "list_price",
                        "qty_available",
                        "categ_id",
                    ],
                    "limit": limit,
                    "context":{"lang":'es_ES'}
                },
            )

        # Strategy: Search with the most relevant terms, then combine results
        all_products = []
        seen_ids = set()
        
        # Limit to top 3 search terms to avoid overly complex queries
        terms_to_use = search_terms[:3]
        logger.info(f"Searching with terms: {terms_to_use}")
        
        for i, term in enumerate(terms_to_use):
            try:
                logger.info(f"Searching for term: '{term}'")
                # Updated domain to include website_description
                domain = [
                    "&",  # AND operator
                    sale_ok_clause,  # Must be saleable
                    "|",  # OR operator for the search fields
                    "|",  # Another OR to chain 4 conditions
                    "|",  # Another OR to chain 4 conditions
                    ["name", "ilike", term],
                    ["description_sale", "ilike", term], 
                    ["default_code", "ilike", term],
                    ["website_description", "ilike", term]  # Add website_description
                ]

                # Calculate limit per term (avoid division by zero)
                term_limit = max(20, limit // len(terms_to_use)) if len(terms_to_use) > 0 else limit
                
                products = execute_odoo_query(
                    "product.product",
                    "search_read",
                    [domain],
                    {
                        "fields": [
                            "name",
                            "description_sale",
                            "website_description",  # Add website_description
                            "default_code",
                            "list_price",
                            "qty_available",
                            "categ_id",
                        ],
                        "limit": term_limit,
                        "context": {"lang": 'es_ES'}

                    },
                )
                
                logger.info(f"Found {len(products)} products for term '{term}'")
                
                # Add unique products only
                for product in products:
                    if product["id"] not in seen_ids:
                        all_products.append(product)
                        seen_ids.add(product["id"])
                        
                        # Stop if we have enough products
                        if len(all_products) >= limit:
                            break
                            
                if len(all_products) >= limit:
                    logger.info(f"Reached limit of {limit} products")
                    break
                    
            except Exception as term_error:
                logger.warning(f"Search failed for term '{term}': {term_error}")
                continue

        # If no products found with any term, fallback to all saleable products
        if not all_products:
            logger.info("No products found with search terms, falling back to all products")
            all_products = execute_odoo_query(
                "product.product",
                "search_read",
                [[sale_ok_clause]],
                {
                    "fields": [
                        "name",
                        "description_sale",
                        "website_description",  # Add website_description
                        "default_code",
                        "list_price",
                        "qty_available",
                        "categ_id",
                    ],
                    "limit": limit,
                    "context": {"lang": 'es_ES'}

                },
            )

        logger.info(f"Returning {len(all_products[:limit])} products total")
        return all_products[:limit]
        
    except Exception as e:
        logger.error(f"Odoo search failed: {e}", exc_info=True)
        # Fallback to all saleable products
        try:
            logger.info("Attempting fallback to all products")
            return execute_odoo_query(
                "product.product",
                "search_read",
                [[sale_ok_clause]],
                {
                    "fields": [
                        "name",
                        "description_sale",
                        "website_description",  # Add website_description
                        "default_code",
                        "list_price",
                        "qty_available",
                        "categ_id",
                    ],
                    "limit": limit,
                    "context": {"lang": 'es_ES'}

                },
            )
        except Exception as fallback_error:
            logger.error(f"Even fallback failed: {fallback_error}")
            return []  # Return empty list if everything fails


def rank_products_by_value(products: List[Dict]) -> List[Dict]:
    """Rank products by value proposition (quality, features, price)."""
    for product in products:
        price = product.get("list_price", 0)
        name = product.get("name", "").lower() if product.get("name") else ""
        
        # Handle description_sale that might be False, None, or a string
        description_raw = product.get("description_sale", "")
        if isinstance(description_raw, str):
            description = description_raw.lower()
        elif description_raw:  # Could be True/False or other truthy value
            description = str(description_raw).lower()
        else:  # None, False, empty string, etc.
            description = ""
            
        # Handle website_description that might be False, None, or a string
        website_desc_raw = product.get("website_description", "")
        if isinstance(website_desc_raw, str):
            website_desc = website_desc_raw.lower()
        elif website_desc_raw:  # Could be True/False or other truthy value
            website_desc = str(website_desc_raw).lower()
        else:  # None, False, empty string, etc.
            website_desc = ""
        
        # Combine all text for analysis
        all_text = f"{name} {description} {website_desc}"

        # Premium indicators
        premium_indicators = [
            "premium",
            "professional",
            "medical",
            "ergonomic",
            "therapeutic",
            "advanced",
            "high-quality",
            "precision",
            "clinical",
            "hospital-grade"
        ]
        is_premium = any(
            indicator in all_text for indicator in premium_indicators
        )

        # Feature richness - expanded list for medical equipment
        feature_indicators = [
            "adjustable",
            "memory foam",
            "gel",
            "cooling",
            "heating",
            "massage",
            "vibration",
            "digital",
            "wireless",
            "bluetooth",
            "rechargeable",
            "waterproof",
            "sterilizable",
            "autoclavable",
            "disposable",
            "reusable"
        ]
        features = sum(
            1 for indicator in feature_indicators if indicator in all_text
        )

        # Calculate value score (higher is better)
        value_score = 0
        if is_premium:
            value_score += 3
        value_score += features * 2
        
        # Improved price scoring: normalize by product category/type
        if price > 0:
            # Use price more strategically - higher prices aren't always better value
            # Focus on feature-to-price ratio instead
            if features > 0:
                value_score += min(features / max(price / 100, 1), 3)  # Feature density
            else:
                value_score += min(price / 200, 2)  # Basic price contribution for products without clear features
            
        # Bonus for having detailed website description
        if len(website_desc) > 50:  # Substantial description
            value_score += 1
            
        # Availability bonus (prefer in-stock items)
        qty_available = product.get("qty_available", 0)
        if qty_available > 0:
            value_score += 1
        if qty_available > 10:  # Well-stocked items
            value_score += 0.5

        product["_value_score"] = value_score

    return sorted(products, key=lambda x: x.get("_value_score", 0), reverse=True)

def find_alternatives(products: List[Dict], main_product_id: int = None) -> Dict[str, List[Dict]]:
    """
    Find premium and budget alternatives, prioritizing curated alternatives when available.
    
    Args:
        products: List of products to analyze
        main_product_id: ID of the main product to find alternatives for (optional)
    """
    if not products:
        return {"premium": [], "budget": []}

    # If we have a main product ID, try to get its curated alternatives first
    curated_alternatives = []
    if main_product_id:
        try:
            # Fetch the main product to get its alternative_related_product_ids
            main_product_data = execute_odoo_query(
                "product.product",
                "search_read",
                [[["id", "=", main_product_id]]],
                {
                    "fields": ["alternative_related_product_ids"],
                    "limit": 1,
                    "context": {"lang": 'es_ES'}
                },
            )
            
            if main_product_data and main_product_data[0].get("alternative_related_product_ids"):
                alternative_ids = main_product_data[0]["alternative_related_product_ids"]
                
                # Fetch the curated alternative products
                curated_alternatives = execute_odoo_query(
                    "product.product",
                    "search_read",
                    [[["id", "in", alternative_ids], ["sale_ok", "=", True]]],
                    {
                        "fields": [
                            "name",
                            "description_sale",
                            "website_description",
                            "default_code",
                            "list_price",
                            "qty_available",
                            "categ_id",
                        ],
                        "context": {"lang": 'es_ES'}
                    },
                )
                
                logger.info(f"Found {len(curated_alternatives)} curated alternatives for product {main_product_id}")
                
        except Exception as e:
            logger.warning(f"Failed to fetch curated alternatives for product {main_product_id}: {e}")

    # Combine curated alternatives with the original products list
    all_products_for_analysis = products.copy()
    
    # Add curated alternatives that aren't already in the products list
    existing_ids = {p.get("id") for p in products if p.get("id")}
    for alt in curated_alternatives:
        if alt.get("id") not in existing_ids:
            all_products_for_analysis.append(alt)

    # Sort by price (primary indicator of premium vs budget)
    sorted_products = sorted(all_products_for_analysis, key=lambda x: x.get("list_price", 0))
    
    if not sorted_products:
        return {"premium": [], "budget": []}
    
    # Calculate price tiers based on the distribution
    prices = [p.get("list_price", 0) for p in sorted_products if p.get("list_price", 0) > 0]
    
    if not prices:
        # If no products have prices, fall back to feature-based classification
        return {
            "premium": rank_products_by_value(all_products_for_analysis)[:3],
            "budget": rank_products_by_value(all_products_for_analysis)[-3:],
        }
    
    # Use quartiles for better price segmentation
    q1 = np.percentile(prices, 25) if len(prices) > 4 else min(prices)
    q3 = np.percentile(prices, 75) if len(prices) > 4 else max(prices)
    median_price = np.median(prices)
    
    # Define price-based categories (primary criteria)
    premium_products = [
        p for p in all_products_for_analysis 
        if p.get("list_price", 0) >= q3
    ]
    
    budget_products = [
        p for p in all_products_for_analysis 
        if p.get("list_price", 0) <= q1 and p.get("list_price", 0) > 0
    ]
    
    # Mid-range products (can be classified as either based on features)
    mid_range_products = [
        p for p in all_products_for_analysis 
        if q1 < p.get("list_price", 0) < q3
    ]
    
    # Prioritize curated alternatives in the results
    def prioritize_curated(product_list):
        """Sort products with curated alternatives first, then by value score"""
        curated_ids = {alt.get("id") for alt in curated_alternatives}
        
        curated = [p for p in product_list if p.get("id") in curated_ids]
        non_curated = [p for p in product_list if p.get("id") not in curated_ids]
        
        # Rank both groups by value
        curated_ranked = rank_products_by_value(curated)
        non_curated_ranked = rank_products_by_value(non_curated)
        
        # Combine: curated first, then non-curated
        return curated_ranked + non_curated_ranked
    
    # Apply value ranking within each price category
    premium_ranked = prioritize_curated(premium_products)
    budget_ranked = prioritize_curated(budget_products)
    
    # If we don't have enough in premium/budget, supplement from mid-range based on features
    mid_range_ranked = rank_products_by_value(mid_range_products)
    
    # Add high-value mid-range products to premium if needed
    if len(premium_ranked) < 3:
        high_value_mid = [p for p in mid_range_ranked if p.get("_value_score", 0) > 5]
        premium_ranked.extend(high_value_mid[:3 - len(premium_ranked)])
    
    # Add low-value mid-range products to budget if needed
    if len(budget_ranked) < 3:
        low_value_mid = [p for p in mid_range_ranked if p.get("_value_score", 0) <= 5]
        budget_ranked.extend(low_value_mid[:3 - len(budget_ranked)])

    return {
        "premium": premium_ranked[:3],
        "budget": budget_ranked[:3],
    }

def rag_analyze_products(
    query: str, products: List[Dict], top_k: int = 10
) -> List[Dict]:
    """Use RAG to analyze and rank products based on the query."""
    if not products:
        return []

    # Create enhanced product texts for embedding using all available text fields
    product_texts = []
    for product in products:
        # Combine all text fields for richer context
        name = product.get('name', '')
        code = product.get('default_code', '')
        
        # Handle description_sale (might be False/None)
        desc_sale = product.get('description_sale', '')
        if isinstance(desc_sale, str):
            desc_sale = desc_sale
        elif desc_sale:
            desc_sale = str(desc_sale)
        else:
            desc_sale = ''
            
        # Handle website_description (might be False/None)
        website_desc = product.get('website_description', '')
        if isinstance(website_desc, str):
            website_desc = website_desc
        elif website_desc:
            website_desc = str(website_desc)
        else:
            website_desc = ''
        
        # Create comprehensive text combining all fields
        # Give more weight to name and website_description as they're usually more descriptive
        text_parts = [
            f"Product: {name}",
            f"Code: {code}",
            f"Description: {desc_sale}",
            f"Details: {website_desc}"
        ]
        
        # Filter out empty parts and join
        text = " | ".join([part for part in text_parts if part.split(": ", 1)[1].strip()])
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
            "&",  # AND operator
            ["sale_ok", "=", True],  # Must be saleable
            "|",  # OR operator for the search fields
            "|",  # Another OR to chain 4 conditions
            "|",  # Another OR to chain 4 conditions
            ["name", "ilike", request.term],
            ["description_sale", "ilike", request.term], 
            ["default_code", "ilike", request.term],
            ["website_description", "ilike", request.term]  # Add website_description
        ]

        items = execute_odoo_query(
            "product.product",
            "search_read",
            [domain],
            {
                "fields": [
                    "name",
                    "default_code",
                    "description_sale",
                    "website_description",  # Add website_description
                    "list_price",
                    "qty_available",
                ],
                "limit": request.limit,
                "context": {"lang": 'es_ES'}

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
            main_product_id = top_recommendations[0].get("id") if top_recommendations else None
            alternatives = find_alternatives(final_products, main_product_id)

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
            "product.product",
            "search_read",
            [[["sale_ok", "=", True]]],
            {
                "fields": ["name", "list_price", "qty_available"],
                "limit": 5,
                "context": {"lang": 'es_ES'}
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
