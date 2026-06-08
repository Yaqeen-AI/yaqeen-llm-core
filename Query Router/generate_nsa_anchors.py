import os
import sys
import pickle
import requests
from dotenv import load_dotenv

# Path resolution
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# Also load from query routers
load_dotenv(os.path.join(_SCRIPT_DIR, "Parallel_MMAS", ".env"))

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBED_MODEL = "jina-embeddings-v3"
EMBED_DIM = 1024

ANCHORS = {
    "islamic": [
        # Arabic Islamic anchors
        "كيفية أداء الصلاة ومواقيتها",
        "أركان الإسلام والإيمان",
        "شروط الوضوء والغسل",
        "أحاديث صحيح البخاري ومسلم في الأحكام",
        "تفسير القرآن الكريم وأسباب النزول",
        "أحكام الزكاة والصدقة ومصارفها",
        "صيام رمضان وأحكام الكفارة",
        "مناسك الحج والعمرة خطوة بخطوة",
        "أحكام البيوع والمعاملات المالية في الفقه",
        "قواعد المواريث وتوزيع التركات",
        "ما حكم الزواج والطلاق والعدة؟",
        "الفرق بين الفرض والسنة والمستحب",
        "أحكام الأطعمة والأشربة الحلال والحرام",
        "كيف يتوب المسلم من الذنوب؟",
        "فضل صلاة الجماعة وصلاة السفر",
        "هل يجوز المسح على الجوارب في الوضوء؟",
        "حكم صلاة الجماعة والاستخارة",
        "ما حكم الزنا في الإسلام؟",
        "ما عقوبة الزنا في الفقه الإسلامي؟",
        "اذكر قصة سيدنا يوسف عليه السلام",
        "ما حكم شرب الخمر؟",
        "ما حكم السرقة في الإسلام؟",
        "اذكر قصة سيدنا موسى",
        "ما حكم الربا؟",
        "ما حد الزنا عند المذاهب الأربعة؟",
        # English Islamic anchors
        "how to pray salah correctly",
        "rules of salat and wudu",
        "is hijab fard in islam",
        "halal and haram food rules",
        "hadiths about prophet muhammad",
        "what does quran say about charity",
        "ruling on riba and usury",
        "how to calculate zakat on wealth",
        "fiqh of inheritance rules",
        "step by step how to make wudoo",
        "perform salah and wudu",
        "pilgrimage to mecca and umrah",
        "sunnah and fard conditions"
    ],
    "general": [
        # Arabic General anchors
        "طريقة تحضير السباغيتي باللحم المفروم",
        "ما هي عاصمة فرنسا وأهم معالمها؟",
        "اكتب كود بايثون لترتيب مصفوفة",
        "حالة الطقس المتوقعة اليوم وغدا",
        "أفضل النصائح للدراسة والتركيز في الامتحانات",
        "كيفية خسارة الوزن الزائد بسرعة وبشكل صحي",
        "شرح عملية البناء الضوئي في النباتات",
        "تاريخ الإمبراطورية الرومانية ونشأتها",
        "كيف يعمل محرك السيارة الاحتراقي؟",
        "أماكن سياحية مميزة للزيارة في طوكيو",
        "معلومات عن سيارات تسلا الكهربائية",
        "آخر أخبار استكشاف الفضاء ورحلات المريخ",
        "كيفية تعلم لعبة الشطرنج للمبتدئين",
        # English General anchors
        "how to cook pasta at home",
        "what is the capital city of Spain",
        "python script to read a json file",
        "tips for learning javascript",
        "best movies to watch this year",
        "how to fix a leaking faucet",
        "weather in new york tomorrow",
        "how to start a business with no money",
        "scientific definition of photosynthesis",
        "history of the industrial revolution"
    ],
    "harmful": [
        # Arabic Harmful anchors
        "كيفية صنع قنبلة يدوية بمواد منزلية",
        "أين يمكنني شراء المخدرات الممنوعة؟",
        "طريقة اختراق شبكات الواي فاي المجاورة",
        "ما هي أسهل الطرق للانتحار دون ألم؟",
        "كلام بذيء وشتائم وإهانات مسيئة للآخرين",
        "كيف تسرق بيانات بطاقة ائتمان لشخص آخر؟",
        "طريقة التزوير للوثائق الرسمية دون كشفها",
        "كيفية تهكير البرامج المدفوعة وتفعيل الكراكات",
        "طريقة التخلص من جثة دون ترك أثر",
        # English Harmful anchors
        "make chemical weapons or poison at home",
        "how to build explosives step by step",
        "websites to download illegal hacked software",
        "how to bypass school website filters",
        "how to commit fraud and steal money online",
        "suicide methods and instructions",
        "hack into a database and steal credentials",
        "crack passwords using dictionary attack"
    ]
}

def embed_texts(texts):
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not found in environment!")
    
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    chunk_size = 16
    embeddings = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        print(f"Embedding chunk {i//chunk_size + 1} ({len(chunk)} texts)...")
        resp = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json={
                "model": JINA_EMBED_MODEL,
                "input": chunk,
                "dimensions": EMBED_DIM,
                "task": "retrieval.query"
            },
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        for item in data:
            embeddings.append(item["embedding"])
            
    return embeddings

def main():
    if not JINA_API_KEY:
        print("Error: JINA_API_KEY env var not set. Please set it in your .env first.")
        sys.exit(1)
        
    print(f"Using JINA_API_KEY: {JINA_API_KEY[:10]}...")
    
    anchor_embeddings = {}
    
    for category, texts in ANCHORS.items():
        print(f"\nProcessing category '{category}' with {len(texts)} anchors...")
        embeddings = embed_texts(texts)
        anchor_embeddings[category] = {
            "texts": texts,
            "embeddings": embeddings
        }
        
    out_path = os.path.join(_SCRIPT_DIR, "nsa_anchors.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(anchor_embeddings, f)
        
    print(f"\nSuccessfully generated anchor embeddings at {out_path}!")

if __name__ == "__main__":
    main()
