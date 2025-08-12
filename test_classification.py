# ==============================================================================
# DATA-DRIVEN PRODUCT CLASSIFICATION SYSTEM
# Based on your actual product summary data
# ==============================================================================

import json
import csv
from typing import Dict, List, Set

# Load and process your product classification data
def load_product_classification_data():
    """Load and process the product summary CSV data."""
    
    # This will be loaded from your products_summary.json file
    # For now, I'll create the structure based on your top categories
    
    PRODUCT_HIERARCHY = {
        # Based on your top medical classes
        "rehabilitacion": {
            "product_classes": ["Rehabilitacion"],
            "product_subclasses": [
                "Material blando", "Fisioterapia", "Movilidad", "Cuidado en casa",
                "Terapia respiratoria", "Alivio del dolor", "Bienestar y relajacion"
            ],
            "product_families": [
                "Inmovilizadores", "Soportes", "Fajas", "Bastones", "Sillas de ruedas",
                "Andadores", "Muletas", "Estimuladores", "Colchones", "Almohadas y cojines"
            ],
            "keywords": [
                "rehabilitacion", "fisioterapia", "silla", "ruedas", "baston", "muleta",
                "andador", "faja", "soporte", "inmovilizador", "estimulador", "movilidad"
            ],
            "spanish_terms": [
                "rehabilitacion", "fisioterapia", "silla de ruedas", "baston", "muleta",
                "andador", "faja", "soporte", "movilidad", "terapia"
            ]
        },
        
        "material_medico": {
            "product_classes": ["Material medico"],
            "product_subclasses": [
                "Quirurgico", "Instrumental", "Descartable", "Protesis", "Cirugia vascular"
            ],
            "product_families": [
                "Cateter", "Pinzas", "Canulas", "Sondas", "Agujas", "Jeringas",
                "Guantes", "Mascarillas", "Bisturi", "Tijeras", "Hilos de sutura"
            ],
            "keywords": [
                "quirurgico", "instrumental", "cateter", "pinza", "canula", "sonda",
                "aguja", "jeringa", "guante", "mascarilla", "bisturi", "cirugia"
            ],
            "spanish_terms": [
                "material medico", "quirurgico", "instrumental", "cateter", "pinza",
                "canula", "sonda", "aguja", "jeringa", "cirugia"
            ]
        },
        
        "medico_hospitalario": {
            "product_classes": ["Medico hospitalario"],
            "product_subclasses": [
                "Ultrasonido", "Monitoreo de signos vitales", "Quirofano", "Mobiliario",
                "Anestesia", "Ventilacion mecanica", "Cuidado materno infantil"
            ],
            "product_families": [
                "Ultrasonido", "Monitores", "Maquina de anestesia", "Lamparas",
                "Mesas", "Ventiladores mecanicos", "Transductores", "Cables", "Sensores"
            ],
            "keywords": [
                "hospital", "ultrasonido", "monitor", "anestesia", "ventilador",
                "lampara", "mesa", "quirofano", "transductor", "cable", "sensor"
            ],
            "spanish_terms": [
                "medico hospitalario", "hospital", "ultrasonido", "monitor", "anestesia",
                "ventilador", "quirofano", "mesa quirurgica", "lampara"
            ]
        },
        
        "cardiologia_intervencionista": {
            "product_classes": ["Cardiologia intervencionista"],
            "product_subclasses": [
                "Coronario", "Periferico vascular", "TAVI", "Denervacion renal"
            ],
            "product_families": [
                "Stents", "Balones", "Cateter", "Guias", "Introductores", "Valvulas"
            ],
            "keywords": [
                "cardiologia", "stent", "balon", "cateter", "guia", "introductor",
                "valvula", "coronario", "vascular", "TAVI", "cardiaco"
            ],
            "spanish_terms": [
                "cardiologia", "stent", "balon", "cateter", "guia", "introductor",
                "valvula", "coronario", "vascular", "corazon"
            ]
        },
        
        "medico_clinico": {
            "product_classes": ["Medico clinico"],
            "product_subclasses": [
                "Medicion y diagnostico", "Fisioterapia", "Mobiliario", "Cirugia menor"
            ],
            "product_families": [
                "Estetoscopios", "Esfigmomanometros", "Electrocardiografos", "Balanzas",
                "Termometros", "Otoscopios", "Oximetros", "Estimuladores"
            ],
            "keywords": [
                "clinico", "estetoscopio", "esfigmomanometro", "electrocardiografo",
                "balanza", "termometro", "otoscopio", "oximetro", "diagnostico"
            ],
            "spanish_terms": [
                "medico clinico", "estetoscopio", "tension", "electrocardiografo",
                "balanza", "termometro", "diagnostico", "clinica"
            ]
        },
        
        "atencion_medica": {
            "product_classes": ["Atencion medica"],
            "product_subclasses": [
                "Cuidado de heridas", "Ostomia", "Incontinencia", "Alimentacion",
                "Terapia para varices y linfedema", "Monitoreo"
            ],
            "product_families": [
                "Medias y calcetines", "Apositos", "Bolsas", "Mascaras", "Gel",
                "Cintas y vendas", "Sondas", "Bombas"
            ],
            "keywords": [
                "atencion", "cuidado", "heridas", "ostomia", "incontinencia",
                "varices", "medias", "aposito", "vendas", "gel"
            ],
            "spanish_terms": [
                "atencion medica", "cuidado", "heridas", "ostomia", "medias",
                "apositos", "vendas", "gel", "compresion"
            ]
        }
    }
    
    return PRODUCT_HIERARCHY

# Enhanced classification function using your actual data structure
def classify_with_product_data(query: str, product_hierarchy: Dict) -> Dict[str, float]:
    """Classify query using your actual product hierarchy data."""
    query_lower = query.lower()
    scores = {}
    
    for category, info in product_hierarchy.items():
        score = 0
        
        # Check product families (highest weight - most specific)
        for family in info["product_families"]:
            if family.lower() in query_lower:
                score += 4.0
        
        # Check product subclasses (high weight)
        for subclass in info["product_subclasses"]:
            if subclass.lower() in query_lower:
                score += 3.5
        
        # Check Spanish terms (optimized for your market)
        for term in info["spanish_terms"]:
            if term.lower() in query_lower:
                score += 3.0
        
        # Check keywords
        for keyword in info["keywords"]:
            if keyword.lower() in query_lower:
                score += 2.0
        
        # Check product classes
        for prod_class in info["product_classes"]:
            if prod_class.lower() in query_lower:
                score += 2.5
        
        if score > 0:
            scores[category] = score
    
    # Normalize scores
    if scores:
        max_score = max(scores.values())
        scores = {k: v / max_score for k, v in scores.items()}
    
    return scores

def get_search_terms_from_classification(query: str, classification: Dict[str, float], product_hierarchy: Dict) -> List[str]:
    """Generate search terms based on classification and your product data."""
    search_terms = []
    
    # Add original query words
    query_words = [word for word in query.lower().split() if len(word) > 2]
    search_terms.extend(query_words)
    
    # Add terms from top-scoring categories
    for category, score in sorted(classification.items(), key=lambda x: x[1], reverse=True):
        if score > 0.3:  # Only high-confidence matches
            category_info = product_hierarchy[category]
            
            # Add Spanish terms (best for Odoo search)
            search_terms.extend(category_info["spanish_terms"][:4])
            
            # Add top product families
            search_terms.extend(category_info["product_families"][:3])
    
    # Remove duplicates
    seen = set()
    unique_terms = []
    for term in search_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)
    
    return unique_terms[:15]

# Test function
def test_data_driven_classification():
    """Test the data-driven classification system."""
    
    product_hierarchy = load_product_classification_data()
    
    test_queries = [
        "cable para SPO2 mindray",
        "silla de ruedas electrica",
        "estetoscopio cardiologia",
        "stent coronario medicado",
        "ventilador mecanico UCI",
        "medias compresion varices",
        "ultrasonido transductor",
        "cateter doble J",
        "faja lumbar soporte"
    ]
    
    print("=== DATA-DRIVEN CLASSIFICATION TEST ===\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 50)
        
        classification = classify_with_product_data(query, product_hierarchy)
        search_terms = get_search_terms_from_classification(query, classification, product_hierarchy)
        
        print(f"Categories: {len(classification)}")
        for category, score in sorted(classification.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {category}: {score:.2f}")
        
        print(f"Search terms: {search_terms[:10]}")
        print(f"Top category: {max(classification.items(), key=lambda x: x[1])[0] if classification else 'None'}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_data_driven_classification()