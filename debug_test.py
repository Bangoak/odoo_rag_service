# debug_test.py
import os
import xmlrpc.client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_domain_structures():
    """Test different domain structures to find what works"""
    try:
        # Get connection
        url = os.getenv("ODOO_URL")
        db = os.getenv("ODOO_DB")
        login = os.getenv("ODOO_LOGIN") 
        api_key = os.getenv("ODOO_API_KEY")
        
        common = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/common', allow_none=True)
        uid = common.authenticate(db, login, api_key, {})
        models = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/object', allow_none=True)
        
        print("=== Testing Domain Structures ===")
        
        # Test 1: Simple single condition
        print("\n1. Testing simple single condition...")
        try:
            domain = [["sale_ok", "=", True]]
            results = models.execute_kw(
                db, uid, api_key,
                'product.template', 'search_read',
                [domain],
                {'fields': ['name'], 'limit': 3}
            )
            print(f"✅ Simple domain works: {len(results)} products found")
        except Exception as e:
            print(f"❌ Simple domain failed: {e}")
        
        # Test 2: Simple name search
        print("\n2. Testing simple name search...")
        try:
            domain = [
                ["sale_ok", "=", True],
                ["name", "ilike", "cable"]
            ]
            results = models.execute_kw(
                db, uid, api_key,
                'product.template', 'search_read',
                [domain],
                {'fields': ['name'], 'limit': 3}
            )
            print(f"✅ Name search works: {len(results)} products found")
            for r in results:
                print(f"  - {r['name']}")
        except Exception as e:
            print(f"❌ Name search failed: {e}")
        
        # Test 3: Check if detailed_type field exists
        print("\n3. Testing detailed_type field...")
        try:
            domain = [["detailed_type", "=", "product"]]
            results = models.execute_kw(
                db, uid, api_key,
                'product.template', 'search_read',
                [domain],
                {'fields': ['name'], 'limit': 3}
            )
            print(f"✅ detailed_type field works: {len(results)} products found")
        except Exception as e:
            print(f"❌ detailed_type field failed: {e}")
            print("   This field might not exist in your Odoo version")
        
        # Test 4: Simple OR structure
        print("\n4. Testing simple OR structure...")
        try:
            domain = [
                "|",
                ["name", "ilike", "cable"],
                ["default_code", "ilike", "cable"]
            ]
            results = models.execute_kw(
                db, uid, api_key,
                'product.template', 'search_read',
                [domain],
                {'fields': ['name', 'default_code'], 'limit': 3}
            )
            print(f"✅ Simple OR works: {len(results)} products found")
            for r in results:
                print(f"  - {r['name']} (Code: {r.get('default_code', 'N/A')})")
        except Exception as e:
            print(f"❌ Simple OR failed: {e}")
        
        # Test 5: AND + OR structure (your failing one)
        print("\n5. Testing your failing domain structure...")
        try:
            domain = [
                "&",  # AND operator
                ["sale_ok", "=", True],  # Must be saleable
                "|",  # OR operator for the search fields
                "|",  # Another OR to chain 3 conditions
                ["name", "ilike", "cable"],
                ["description_sale", "ilike", "cable"], 
                ["default_code", "ilike", "cable"]
            ]
            results = models.execute_kw(
                db, uid, api_key,
                'product.template', 'search_read',
                [domain],
                {'fields': ['name'], 'limit': 3}
            )
            print(f"✅ Complex AND+OR works: {len(results)} products found")
        except Exception as e:
            print(f"❌ Complex AND+OR failed: {e}")
        
        # Test 6: Alternative OR structure
        print("\n6. Testing alternative OR structure...")
        try:
            domain = [
                "&",
                ["sale_ok", "=", True],
                "|",
                ["name", "ilike", "cable"],
                "|",
                ["description_sale", "ilike", "cable"],
                ["default_code", "ilike", "cable"]
            ]
            results = models.execute_kw(
                db, uid, api_key,
                'product.template', 'search_read',
                [domain],
                {'fields': ['name'], 'limit': 3}
            )
            print(f"✅ Alternative OR works: {len(results)} products found")
        except Exception as e:
            print(f"❌ Alternative OR failed: {e}")
        
        # Test 7: Multiple separate searches (fallback approach)
        print("\n7. Testing multiple separate searches...")
        try:
            all_results = []
            search_fields = ["name", "description_sale", "default_code"]
            
            for field in search_fields:
                domain = [
                    ["sale_ok", "=", True],
                    [field, "ilike", "cable"]
                ]
                results = models.execute_kw(
                    db, uid, api_key,
                    'product.template', 'search_read',
                    [domain],
                    {'fields': ['name', field], 'limit': 2}
                )
                all_results.extend(results)
            
            # Remove duplicates
            unique_results = []
            seen_ids = set()
            for r in all_results:
                if r['id'] not in seen_ids:
                    unique_results.append(r)
                    seen_ids.add(r['id'])
            
            print(f"✅ Multiple searches works: {len(unique_results)} unique products found")
            
        except Exception as e:
            print(f"❌ Multiple searches failed: {e}")
        
        # Test 8: Check available fields
        print("\n8. Checking available fields...")
        try:
            fields_info = models.execute_kw(
                db, uid, api_key,
                'product.template', 'fields_get',
                [],
                {'attributes': ['string', 'type']}
            )
            
            relevant_fields = {
                k: v for k, v in fields_info.items() 
                if k in ['sale_ok', 'detailed_type', 'type', 'name', 'default_code', 'description_sale']
            }
            
            print("✅ Available relevant fields:")
            for field, info in relevant_fields.items():
                print(f"  - {field}: {info.get('string', 'N/A')} ({info.get('type', 'unknown')})")
                
        except Exception as e:
            print(f"❌ Fields check failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Domain structure test failed: {e}")
        return False

def test_odoo_connection():
    """Test Odoo connection step by step"""
    try:
        # Get environment variables
        url = os.getenv("ODOO_URL")
        db = os.getenv("ODOO_DB")
        login = os.getenv("ODOO_LOGIN") 
        api_key = os.getenv("ODOO_API_KEY")
        
        print("=== Odoo Connection Test ===")
        print(f"URL: {url}")
        print(f"Database: {db}")
        print(f"Login: {login}")
        print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'Not set'}")
        print()
        
        if not all([url, db, login, api_key]):
            print("❌ Missing required environment variables!")
            return False
            
        # Test authentication
        print("Step 1: Testing authentication...")
        common = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/common', allow_none=True)
        uid = common.authenticate(db, login, api_key, {})
        
        if not uid:
            print("❌ Authentication failed!")
            return False
            
        print(f"✅ Authentication successful! User ID: {uid}")
        
        # Test basic query
        print("\nStep 2: Testing basic product query...")
        models = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/object', allow_none=True)
        
        # Count total products
        total_count = models.execute_kw(
            db, uid, api_key,
            'product.template', 'search_count',
            [[]]
        )
        print(f"✅ Total products in database: {total_count}")
        
        # Test saleable products count
        sale_count = models.execute_kw(
            db, uid, api_key,
            'product.template', 'search_count',
            [[['sale_ok', '=', True]]]
        )
        print(f"✅ Saleable products: {sale_count}")
        
        # Get sample products
        print("\nStep 3: Getting sample products...")
        products = models.execute_kw(
            db, uid, api_key,
            'product.template', 'search_read',
            [[['sale_ok', '=', True]]],
            {
                'fields': ['name', 'default_code', 'list_price', 'qty_available'],
                'limit': 3
            }
        )
        
        print(f"✅ Sample products:")
        for product in products:
            print(f"  - {product['name']} (Code: {product.get('default_code', 'N/A')}, Price: {product.get('list_price', 0)})")
            
        # Test search query similar to your API
        print("\nStep 4: Testing search query...")
        search_domain = [
            '|', 
            ['name', 'ilike', 'cable'],
            ['description_sale', 'ilike', 'cable']
        ]
        
        search_results = models.execute_kw(
            db, uid, api_key,
            'product.template', 'search_read',
            [search_domain],
            {
                'fields': ['name', 'default_code', 'description_sale', 'list_price'],
                'limit': 5
            }
        )
        
        print(f"✅ Search for 'cable' found {len(search_results)} products")
        for product in search_results[:2]:  # Show first 2
            print(f"  - {product['name']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test AI model loading"""
    try:
        print("\n=== AI Model Test ===")
        USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"
        MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        print(f"RAG Enabled: {USE_RAG}")
        print(f"Model: {MODEL_NAME}")
        
        if USE_RAG:
            print("Loading sentence transformer model...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL_NAME)
            
            # Test encoding
            test_text = "test product search"
            embedding = model.encode([test_text])
            print(f"✅ Model loaded successfully! Embedding shape: {embedding.shape}")
            return True
        else:
            print("✅ RAG disabled, skipping model test")
            return True
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def suggest_working_function():
    """Suggest a working search function based on test results"""
    print("\n=== Suggested Working Function ===")
    print("""
Based on the tests above, here's a working search_products function:

```python
@app.post("/search_products")
async def search_products(request: SearchRequest):
    try:
        # Method 1: Try simple approach first
        domain = [
            ["sale_ok", "=", True],
            ["name", "ilike", request.term]
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
        
        # If we don't have enough results, search other fields
        if len(items) < request.limit:
            # Search by code
            code_domain = [
                ["sale_ok", "=", True],
                ["default_code", "ilike", request.term]
            ]
            
            code_items = execute_odoo_query(
                "product.template",
                "search_read", 
                [code_domain],
                {
                    "fields": [
                        "name",
                        "default_code",
                        "description_sale",
                        "list_price",
                        "qty_available",
                    ],
                    "limit": request.limit - len(items),
                },
            )
            
            # Add items that aren't already in results
            existing_ids = {item['id'] for item in items}
            for item in code_items:
                if item['id'] not in existing_ids:
                    items.append(item)
        
        items.sort(key=lambda r: (r["qty_available"] == 0, len(r["name"])))
        return {"items": items}
        
    except Exception as e:
        logger.error(f"Search products failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")
```
""")

if __name__ == "__main__":
    print("Starting Enhanced Odoo RAG Debug Tests...\n")
    
    # Test domain structures first
    domain_ok = test_domain_structures()
    
    # Test basic connection
    odoo_ok = test_odoo_connection()
    
    # Test model loading
    model_ok = test_model_loading()
    
    print(f"\n=== Summary ===")
    print(f"Domain Structures: {'✅ OK' if domain_ok else '❌ FAILED'}")
    print(f"Odoo Connection: {'✅ OK' if odoo_ok else '❌ FAILED'}")
    print(f"AI Model: {'✅ OK' if model_ok else '❌ FAILED'}")
    
    if domain_ok and odoo_ok and model_ok:
        suggest_working_function()
        print("\n🎉 All tests passed! Check the suggested function above.")
    else:
        print("\n🚨 Some tests failed. Fix the issues before proceeding.")