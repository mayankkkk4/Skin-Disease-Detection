"""
Database of skin disease information based on the HAM10000 dataset classifications.
Provides clinical descriptions, risk categories, recommendations, and metrics.
"""

DISEASES = {
    "akiec": {
        "id": 0,
        "code": "akiec",
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "short_name": "Actinic Keratoses",
        "type": "Pre-cancerous / Intraepithelial",
        "risk_level": "Moderate Risk",
        "risk_color": "#f39c12",  # Amber / Warning
        "risk_description": "Pre-cancerous skin lesion caused by long-term sun exposure. Potential to develop into Squamous Cell Carcinoma if untreated.",
        "microscopic_features": [
            "Atypical keratinocytes in basal epidermal layers",
            "Parakeratosis alternating with orthokeratosis",
            "Solar elastosis in upper dermis",
            "Loss of normal epidermal maturation hierarchy"
        ],
        "symptoms": [
            "Rough, scaly, or crusty patch on sun-exposed skin",
            "Slightly raised or flat bump",
            "Pink, red, brown, or flesh-colored lesion",
            "May feel tender, itchy, or burning"
        ],
        "risk_factors": [
            "Fair skin, light hair, blue/green eyes",
            "History of severe sun exposure or sunburns",
            "Age over 50 years",
            "Weakened immune system"
        ],
        "common_locations": ["Face", "Lips", "Ears", "Scalp", "Back of hands", "Forearms"],
        "recommendations": [
            "Consult a dermatologist for clinical examination or biopsy.",
            "Avoid direct sun exposure during peak hours (10 AM - 4 PM).",
            "Apply broad-spectrum SPF 30+ sunscreen daily.",
            "Cryotherapy, photodynamic therapy, or topical 5-FU may be prescribed."
        ],
        "precision": 0.71,
        "recall": 0.36,
        "f1_score": 0.48,
        "support": 28
    },
    "bcc": {
        "id": 1,
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "short_name": "Basal Cell Carcinoma",
        "type": "Malignant Skin Cancer",
        "risk_level": "High Risk",
        "risk_color": "#e74c3c",  # Red / Danger
        "risk_description": "The most common form of skin cancer. Highly treatable when detected early, but requires prompt medical excision to prevent tissue invasion.",
        "microscopic_features": [
            "Nests of basaloid cells with peripheral palisading",
            "Stromal retraction artifact (clefting)",
            "Hyperchromatic nuclei with scanty cytoplasm",
            "Mitotic figures and apoptotic bodies"
        ],
        "symptoms": [
            "Pearly, waxy, or translucent bump with visible tiny blood vessels",
            "Flat, firm, red or brown scar-like lesion",
            "Bleeding or oozing sore that heals and returns",
            "Slightly elevated rolled border with a central indentation"
        ],
        "risk_factors": [
            "Cumulative ultraviolet (UV) radiation exposure",
            "Tanning bed usage",
            "Personal or family history of skin cancer",
            "Radiation therapy exposure"
        ],
        "common_locations": ["Face", "Nose", "Neck", "Ears", "Shoulders", "Chest"],
        "recommendations": [
            "Immediate dermatological evaluation and biopsy recommended.",
            "Surgical excision, Mohs micrographic surgery, or topical therapy required.",
            "Perform regular monthly skin self-examinations.",
            "Protect skin with sun-protective clothing and high SPF sunscreen."
        ],
        "precision": 0.91,
        "recall": 0.72,
        "f1_score": 0.80,
        "support": 57
    },
    "bkl": {
        "id": 2,
        "code": "bkl",
        "name": "Benign Keratosis-like Lesions",
        "short_name": "Benign Keratosis",
        "type": "Benign / Non-Cancerous",
        "risk_level": "Low Risk",
        "risk_color": "#2ecc71",  # Green / Safe
        "risk_description": "Harmless skin growth common in older adults. Includes seborrheic keratosis, solar lentigines, and lichen-planus like keratosis.",
        "microscopic_features": [
            "Acanthosis with horn pseudocysts",
            "Basaloid cell proliferation with hyperkeratosis",
            "Lichenoid lymphocytic infiltrate (in lichenoid variant)",
            "Intact basement membrane"
        ],
        "symptoms": [
            "Waxy, pasted-on appearance, like a drop of candle wax",
            "Round or oval shape ranging from tan, brown, to dark black",
            "Slightly raised or flat rough surface",
            "Non-painful, though may occasionally itch"
        ],
        "risk_factors": [
            "Advancing age (most prevalent > 50 years)",
            "Genetic predisposition",
            "Sun exposure (solar lentigines)",
            "Friction from tight clothing"
        ],
        "common_locations": ["Chest", "Back", "Shoulders", "Face", "Neck"],
        "recommendations": [
            "Generally harmless and requires no active treatment.",
            "Monitor for changes in size, shape, color, or sudden bleeding.",
            "Can be removed for cosmetic reasons or if irritated by clothing.",
            "Consult a medical professional if uncertain."
        ],
        "precision": 0.59,
        "recall": 0.76,
        "f1_score": 0.66,
        "support": 108
    },
    "df": {
        "id": 3,
        "code": "df",
        "name": "Dermatofibroma",
        "short_name": "Dermatofibroma",
        "type": "Benign Nodule",
        "risk_level": "Low Risk",
        "risk_color": "#2ecc71",  # Green / Safe
        "risk_description": "Common, harmless fibrous skin growth often resulting from minor skin injuries like bug bites or splinter pricks.",
        "microscopic_features": [
            "Dermal proliferation of fibroblasts and histiocytes",
            "Induction of overlying epidermal hyperplasia",
            "Collagen trapping at peripheral margins",
            "Hemosiderin pigment accumulation in macrophages"
        ],
        "symptoms": [
            "Small, firm button-like red, brown, or pink nodule",
            "Pinch sign: dimples inward when pinched from sides",
            "May feel slightly tender or itchy when touched",
            "Remains stationary in size over years"
        ],
        "risk_factors": [
            "Minor trauma or arthropod bite",
            "Female gender (more common in women)",
            "Young to middle adult age group",
            "Immunosuppression (multiple lesions)"
        ],
        "common_locations": ["Lower legs", "Arms", "Upper back"],
        "recommendations": [
            "Benign nature means treatment is usually unnecessary.",
            "Avoid picking or cutting the lesion.",
            "Surgical removal available if lesion causes pain or discomfort.",
            "Consult a clinician if rapid growth or color alteration occurs."
        ],
        "precision": 0.86,
        "recall": 0.40,
        "f1_score": 0.55,
        "support": 15
    },
    "nv": {
        "id": 4,
        "code": "nv",
        "name": "Melanocytic Nevi",
        "short_name": "Melanocytic Nevi (Mole)",
        "type": "Benign Melanocytic Lesion",
        "risk_level": "Low Risk",
        "risk_color": "#2ecc71",  # Green / Safe
        "risk_description": "Common benign moles formed by clusters of melanocytes. Most adults have 10-40 harmless nevi across their body.",
        "microscopic_features": [
            "Nests of uniform nevus cells at dermo-epidermal junction or dermis",
            "Maturation of nevus cells with depth in dermis",
            "Absence of cytological atypia or atypical mitoses",
            "Symmetrical architecture with sharp lateral borders"
        ],
        "symptoms": [
            "Uniform brown, black, or tan pigmentation",
            "Distinct, smooth, well-defined borders",
            "Flat or slightly raised dome shape",
            "Consistent size usually less than 6mm (pencil eraser size)"
        ],
        "risk_factors": [
            "Sun exposure during childhood and adolescence",
            "Genetic inheritance",
            "Fair skin type",
            "Hormonal changes (pregnancy/puberty)"
        ],
        "common_locations": ["Sun-exposed and non-exposed skin throughout body"],
        "recommendations": [
            "Normal nevus requires standard routine skin monitoring.",
            "Track using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolving).",
            "Wear sunscreen and protect skin from sunburns.",
            "Have a professional checkup if a mole bleeds, itches, or changes shape."
        ],
        "precision": 0.91,
        "recall": 0.96,
        "f1_score": 0.94,
        "support": 678
    },
    "vasc": {
        "id": 5,
        "code": "vasc",
        "name": "Vascular Lesions",
        "short_name": "Vascular Lesions",
        "type": "Benign Vascular Growth",
        "risk_level": "Moderate Risk",
        "risk_color": "#f39c12",  # Amber / Warning
        "risk_description": "Skin conditions involving blood vessels, such as cherry angiomas, pyogenic granulomas, and cutaneous hemorrhages.",
        "microscopic_features": [
            "Proliferation of capillary-sized blood vessels lined by flattened endothelium",
            "Extravasated erythrocytes and hemosiderin deposition",
            "Lobular vascular arrangement in pyogenic granuloma variant",
            "Epidermal collarette formation"
        ],
        "symptoms": [
            "Bright red, purple, or dark red spot or bump",
            "May bleed easily after minor friction or injury",
            "Smooth or lobulated texture",
            "Size varies from pinpoint dots to several millimeters"
        ],
        "risk_factors": [
            "Minor skin trauma or irritation",
            "Pregnancy or hormonal fluctuations",
            "Increasing age (cherry angiomas)",
            "Certain medications (e.g., retinoids)"
        ],
        "common_locations": ["Trunk", "Face", "Hands", "Lips", "Oral cavity"],
        "recommendations": [
            "Have a doctor evaluate to confirm benign vascular etiology.",
            "Avoid friction to prevent recurrent bleeding.",
            "Laser therapy, electrocautery, or excision can remove bothersome lesions.",
            "Seek medical care if rapid enlargement or persistent bleeding happens."
        ],
        "precision": 0.72,
        "recall": 0.46,
        "f1_score": 0.56,
        "support": 102
    },
    "mel": {
        "id": 6,
        "code": "mel",
        "name": "Melanoma",
        "short_name": "Melanoma",
        "type": "Malignant Melanocytic Cancer",
        "risk_level": "High Risk",
        "risk_color": "#9b59b6",  # Critical Purple / High Danger
        "risk_description": "The most serious form of skin cancer originating in pigment-producing melanocytes. Early diagnosis and prompt surgical intervention are critical.",
        "microscopic_features": [
            "Asymmetrical intraepidermal melanocytic proliferation (pagetoid spread)",
            "Cytological atypia: enlarged, pleomorphic, hyperchromatic nuclei",
            "Atypical mitotic figures at all dermal levels",
            "Lack of cellular maturation with dermal descent"
        ],
        "symptoms": [
            "Asymmetrical shape with irregular, notched, or blurred borders",
            "Uneven color distribution (shades of brown, black, pink, red, white, blue)",
            "Diameter larger than 6mm (though can be smaller)",
            "Evolving: changes in size, shape, elevation, color, or symptoms (bleeding, itching)"
        ],
        "risk_factors": [
            "Severe blistering sunburns, especially in youth",
            "High mole count (> 50 typical moles or dysplastic nevi)",
            "Family or personal history of melanoma",
            "Immunosuppression or organ transplant history"
        ],
        "common_locations": ["Back and legs (most common), but can occur anywhere"],
        "recommendations": [
            "URGENT: Immediate comprehensive evaluation by a dermatologist or oncologist.",
            "Excisional biopsy and staging diagnostic assessment required.",
            "Do NOT attempt to treat or scratch the lesion.",
            "Inform family members, as genetic predisposition can play a role."
        ],
        "precision": 0.93,
        "recall": 0.93,
        "f1_score": 0.93,
        "support": 14
    }
}

CONFUSION_MATRIX = [
    [10,  3, 11,  0,  1,  3,  0],  # akiec
    [ 1, 41,  8,  0,  6,  1,  0],  # bcc
    [ 1,  1, 82,  0, 18,  6,  0],  # bkl
    [ 2,  0,  3,  6,  4,  0,  0],  # df
    [ 0,  0, 16,  0, 653, 8,  1],  # nv
    [ 0,  0, 20,  0, 35, 47,  0],  # vasc
    [ 0,  0,  0,  1,  0,  0, 13]   # mel
]

MODEL_METRICS = {
    "Ensemble": {
        "name": "Ensemble (VGG16 + InceptionV3 + DenseNet201)",
        "accuracy": 0.8503,
        "loss": 0.4350,
        "val_accuracy": 0.8470,
        "val_loss": 0.3974,
        "precision_avg": 0.80,
        "recall_avg": 0.66,
        "f1_avg": 0.70,
        "input_resolution": "256x192 / 224x224",
        "parameters": "45.8M (Combined)",
        "description": "Combines weighted predictions of VGG16, InceptionV3, and DenseNet201 to maximize overall classification accuracy."
    },
    "DenseNet201": {
        "name": "DenseNet201 Architecture",
        "accuracy": 0.8250,
        "loss": 0.4910,
        "val_accuracy": 0.8210,
        "val_loss": 0.4780,
        "precision_avg": 0.78,
        "recall_avg": 0.64,
        "f1_avg": 0.68,
        "input_resolution": "224x224",
        "parameters": "20.0M",
        "description": "201-layer Dense Convolutional Network with direct feature reuse across dense blocks."
    },
    "InceptionV3": {
        "name": "InceptionV3 Architecture",
        "accuracy": 0.8120,
        "loss": 0.5210,
        "val_accuracy": 0.8090,
        "val_loss": 0.5100,
        "precision_avg": 0.76,
        "recall_avg": 0.63,
        "f1_avg": 0.66,
        "input_resolution": "299x299",
        "parameters": "23.8M",
        "description": "Inception module architecture with factorized convolutions and asymmetric filtering."
    },
    "VGG16": {
        "name": "VGG16 Architecture",
        "accuracy": 0.8034,
        "loss": 0.5621,
        "val_accuracy": 0.7980,
        "val_loss": 0.5430,
        "precision_avg": 0.74,
        "recall_avg": 0.61,
        "f1_avg": 0.64,
        "input_resolution": "224x224",
        "parameters": "138.4M",
        "description": "Deep 16-layer Convolutional Neural Network with sequential 3x3 filter blocks."
    },
    "Basic_CNN": {
        "name": "Custom CNN (3-Layer)",
        "accuracy": 0.7350,
        "loss": 0.7810,
        "val_accuracy": 0.7240,
        "val_loss": 0.7920,
        "precision_avg": 0.65,
        "recall_avg": 0.52,
        "f1_avg": 0.55,
        "input_resolution": "28x28",
        "parameters": "1.2M",
        "description": "Baseline Custom Convolutional Neural Network trained from scratch on 28x28 resized lesions."
    }
}
