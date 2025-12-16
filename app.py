from flask import Flask, request, jsonify, render_template
import os
import re
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__, static_folder='static', template_folder='templates')
ARTIFACT = os.path.join('model', 'artifacts.pkl')
RECIPES_JSON = os.path.join('model', 'recipes.json')


# --- Utilities to (re)train when recipes change ---
def train_if_needed():
    needs_train = False
    if not os.path.exists(ARTIFACT):
        needs_train = True
    elif os.path.exists(RECIPES_JSON):
        try:
            if os.path.getmtime(RECIPES_JSON) > os.path.getmtime(ARTIFACT):
                needs_train = True
        except Exception:
            pass
    if needs_train:
        from model.train import train_and_save
        train_and_save()


def load_artifacts():
    train_if_needed()
    return joblib.load(ARTIFACT)


# --- Load artifacts and normalize recipes ---
artifacts = load_artifacts()
vectorizer = artifacts['vectorizer']
matrix = artifacts['matrix']
recipes = artifacts['recipes']

for r in recipes:
    ing = r.get('ingredients', '') or ''
    parts = [p.strip().lower() for p in ing.split(',') if p.strip()]
    r['ingredients_set'] = set(parts)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json() or {}
    query = data.get('ingredients', '')
    if not query:
        return jsonify({'error': 'No ingredients provided'}), 400

    q_parts = [p.strip().lower() for p in query.split(',') if p.strip()]
    q_set = set(q_parts)

    # TF-IDF similarity for ranking
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]

    # Step 1: overlap >= 2
    def build_candidates(min_overlap: int):
        cands = []
        for i, r in enumerate(recipes):
            r_set = r.get('ingredients_set') or set()
            overlap = len(r_set & q_set)
            if overlap >= min_overlap:
                cands.append((i, overlap, float(sims[i] if sims[i] is not None else 0.0)))
        return cands

    candidates = build_candidates(2)
    fallback_used = None

    # Step 2: relax to overlap >= 1
    if not candidates:
        candidates = build_candidates(1)
        if candidates:
            fallback_used = 'relaxed_overlap_1'

    # Step 3: if still none, take pure similarity top-N (overlap can be 0)
    if not candidates:
        idx_sorted = sorted(range(len(recipes)), key=lambda i: float(sims[i] if sims[i] is not None else 0.0), reverse=True)
        candidates = [(i, 0, float(sims[i] if sims[i] is not None else 0.0)) for i in idx_sorted[:24]]
        fallback_used = 'similarity_only'

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

    results = []
    for i, overlap, score in candidates[:24]:
        r = recipes[i]
        results.append({
            'name': r.get('name'),
            'ingredients': r.get('ingredients'),
            'instructions': r.get('instructions'),
            'overlap': overlap,
            'score': score,
            'category': r.get('category')
        })

    payload = {'query': query, 'results': results}
    if fallback_used:
        payload['fallback'] = fallback_used
    return jsonify(payload)


# --- Indonesian -> English translation maps for better search ---
ID_PHRASE_MAP = {
    'bawang putih': 'garlic',
    'bawang merah': 'shallot',
    'bawang bombay': 'onion',
    'daging sapi': 'beef',
}

ID_WORD_MAP = {
    'ayam': 'chicken',
    'sapi': 'beef',
    'daging': 'meat',
    'ikan': 'fish',
    'tuna': 'tuna',
    'udang': 'shrimp',
    'cumi': 'squid',
    'kerang': 'clam',
    'kambing': 'lamb',
    'telur': 'egg',
    'mie': 'noodle', 'mi': 'noodle', 'bakmi': 'noodle',
    'nasi': 'rice',
    'bubur': 'porridge',
    'sayur': 'vegetable', 'sayuran': 'vegetable',
    'tahu': 'tofu',
    'tempe': 'tempeh',
    'jamur': 'mushroom',
    'wortel': 'carrot',
    'kentang': 'potato',
    'kol': 'cabbage', 'kubis': 'cabbage',
    'tomat': 'tomato',
    'cabe': 'chili', 'cabai': 'chili',
    'kecap': 'soy',
    'santan': 'coconut',
    'kari': 'curry',
    'sup': 'soup', 'soto': 'soup',
    'goreng': 'fried',
    'bakar': 'grilled', 'panggang': 'baked',
    'tumis': 'stir-fry',
    'kuah': 'soup',
}


def id_to_en_query(q: str) -> str:
    text = (q or '').lower()
    for idp, enp in ID_PHRASE_MAP.items():
        text = text.replace(idp, enp)
    tokens = re.findall(r"[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]+", text)
    mapped = [ID_WORD_MAP.get(t, t) for t in tokens]
    uniq = []
    for m in mapped:
        if m and m not in uniq:
            uniq.append(m)
    return ' '.join(uniq).strip()


# --- Google Custom Search (Programmable Search Engine) ---
GOOGLE_CSE_KEY = os.getenv('GOOGLE_CSE_KEY')
GOOGLE_CSE_CX = os.getenv('GOOGLE_CSE_CX')


def google_cse_search(query: str):
    if not (GOOGLE_CSE_KEY and GOOGLE_CSE_CX):
        return None
    params = {
        'key': GOOGLE_CSE_KEY,
        'cx': GOOGLE_CSE_CX,
        'q': query,
        'num': 10,
        'safe': 'active',
        'gl': 'id',
        'lr': 'lang_id',
    }
    resp = requests.get('https://www.googleapis.com/customsearch/v1', params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    items = data.get('items') or []
    results = []
    for it in items:
        results.append({
            'name': it.get('title'),
            'ingredients': '',  # unknown without scraping
            'instructions': it.get('snippet') or '',
            'image': (it.get('pagemap', {}).get('cse_image', [{}])[0].get('src') if it.get('pagemap') else None),
            'source': it.get('link')
        })
    return results


# --- TheMealDB helpers (fallback) ---
API_BASE = 'https://www.themealdb.com/api/json/v1/1'


def tmdb_search(q: str):
    resp = requests.get(f'{API_BASE}/search.php', params={'s': q}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get('meals') or []


def tmdb_filter_by_ingredient(ing: str):
    resp = requests.get(f'{API_BASE}/filter.php', params={'i': ing}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get('meals') or []


def tmdb_lookup_full(meal_id: str):
    resp = requests.get(f'{API_BASE}/lookup.php', params={'i': meal_id}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    meals = data.get('meals') or []
    return meals[0] if meals else None


def to_result_item(m):
    ings = []
    for i in range(1, 21):
        ing = (m.get(f'strIngredient{i}') or '').strip()
        meas = (m.get(f'strMeasure{i}') or '').strip()
        if ing:
            ings.append((meas + ' ' + ing).strip())
    ingredients = ', '.join([x for x in ings if x])
    instr = (m.get('strInstructions') or '').replace('\r', ' ').replace('\n', ' ').strip()
    return {
        'name': m.get('strMeal'),
        'ingredients': ingredients,
        'instructions': instr,
        'image': m.get('strMealThumb'),
        'source': m.get('strSource') or m.get('strYoutube')
    }


@app.route('/api/search', methods=['GET'])
def search_online():
    original_q = (request.args.get('query') or '').strip()
    if not original_q:
        return jsonify({'error': 'Query is required'}), 400

    # Build candidate queries (ID and EN), extend with recipe keyword and site focus
    translated = id_to_en_query(original_q)
    base_queries = []
    base_queries.append(original_q)
    if translated and translated != original_q:
        base_queries.append(translated)

    candidates = []
    recipe_keywords = ['resep', 'recipe']
    site_focus = [
        '',
        'site:allrecipes.com',
        'site:foodnetwork.com',
        'site:bbcgoodfood.com',
        'site:cookpad.com',
        'site:delish.com',
        'site:epicurious.com'
    ]
    for bq in base_queries:
        for rk in recipe_keywords:
            for sf in site_focus:
                q = f"{bq} {rk} {sf}".strip()
                candidates.append(q)

    # Try Google CSE if configured
    if GOOGLE_CSE_KEY and GOOGLE_CSE_CX:
        for q in candidates[:8]:  # limit calls
            try:
                results = google_cse_search(q)
                if results:
                    return jsonify({'query': original_q, 'used_query': q, 'mode': 'google', 'results': results})
            except requests.RequestException:
                continue

    # Fallback to TheMealDB: search, then ingredient filter+lookup
    # Try search.php for various tokens
    tokens = (translated or original_q).split()
    search_list = [original_q, translated] + tokens
    for q in [x for x in search_list if x]:
        try:
            meals = tmdb_search(q)
            if meals:
                results = [to_result_item(m) for m in meals[:30]]
                return jsonify({'query': original_q, 'used_query': q, 'mode': 'tmdb:search', 'results': results})
        except requests.RequestException:
            continue

    # Fallback filter by ingredient
    for t in tokens[:5]:
        try:
            meals = tmdb_filter_by_ingredient(t)
            if not meals:
                continue
            detailed = []
            for m in meals[:10]:
                mid = m.get('idMeal')
                full = tmdb_lookup_full(mid) if mid else None
                if full:
                    detailed.append(to_result_item(full))
            if detailed:
                return jsonify({'query': original_q, 'used_query': t, 'mode': 'tmdb:filter+lookup', 'results': detailed})
        except requests.RequestException:
            continue

    return jsonify({'query': original_q, 'results': []})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
