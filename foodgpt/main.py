from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import re
import ast

app = Flask(__name__)

# Load model and tokenizer

model_load_path = r"models\foodgptfinalmodel_v2"
tokenizer = AutoTokenizer.from_pretrained(model_load_path)
model = AutoModelForCausalLM.from_pretrained(model_load_path)
tokenizer.pad_token = tokenizer.eos_token
# Load the dataset
df = pd.read_csv(r'data\cuisine_v10.csv')
df['prep_time'] = pd.to_numeric(df['prep_time'], errors='coerce')
# Fill NaN values with a default value (e.g., 0)
df['prep_time'] = df['prep_time'].fillna(0)
synonym_to_tag = {
    "veg": "vegetarian",
    "vegetarian": "vegetarian",
    "non-veg": "non-vegetarian",
    "fast": "easy",
    "instant": "quick",
    "rapid": "quick",
    "spice": "spicy",
    "hot": "hot",
    "mild": "mild",
    "creamy": "creamy",
    "sauce": "saucy",
    "fried": "fried",
    "grilled": "grilled",
    "baked": "baked",
    "roasted": "roasted",
    "desserts": "dessert",
    "sweet": "dessert",
    "sweets": "dessert",
    "savory": "savory",
    "quickly": "quick",
    "snack": "snack",
    "appetizer": "snack",
    "main": "main course",
    "entree": "main course",
    "breakfast": "breakfast",
    "appetizers":"appetizer",
    "brunch": "breakfast",
    "lunch": "lunch",
    "quick": "quick",
    "dinner": "dinner",
    "healthy": "healthy",
    "low-fat": "healthy",
    "prawn": "shrimp",
    "prawn": "prawns",
    "prawns": "prawn",
    "shrimp": "prawn",
    "shrimps": "shrimp",
    "shrimp": "shrimps",
    "gluten-free": "gluten free",
    "dairy-free": "dairy free",
    "dairy": "milk",
    "slow-cooked": "slow",
    "time-consuming": "slow",
    "nut-free": "nut free",
    "vegan": "vegan",
    "plant-based": "vegan",
    "protein-rich": "high-protein",
    "high-protein": "high-protein",
    "hard": "complex",
    "complex": "complicated",
    "complex": "complex",
    "simple": "easy",
    "easy": "easy",
    "beginner": "easy",
    "beginners": "easy",
    "beginner-friendly": "easy",
    "time-consuming": "difficult",
    "difficult": "difficult",
    "challenging": "difficult",
    "effortless": "simple",
    "straightforward": "simple",
    "basic": "simple",
}


# Function to generate recipe suggestions
def extract_relevant_keywords(text):
    stopwords = {
        "suggest", "me", "some", "dishes", "dish", "are", "that", "have", "is", "with",
        "I", "i", "you", "your", "use", "using", "used", "uses", "a", "an", "the", "and",
        "for", "to", "in", "on", "at", "by", "of", "as", "be", "this", "which", "but", "prepare",
        "or", "from", "about", "if", "can", "you", "your", "we", "my", "it's", "its", "it",
        "where", "when", "who", "how", "than", "there", "what", "these", "those", "both", "options",
        "generate", "something", "either", "neither", "each", "many", "much", "more",
        "most", "few", "less", "least", "any", "all", "every", "such", "no", "not", "food", "foods",
        "now", "then", "so", "just", "like", "want", "need", "say", "said", "go",
        "get", "make", "do", "come", "see", "think", "know", "take", "find", "give", "should",
        "show", "use", "tell", "feel", "try", "ask", "work", "play", "seem", "seems", "leftover", "craving", "am",
        "rather", "really", "also", "very", "too", "still", "then", "again", "could", "do"
        "should", "would", "might", "may", "must", "can", "will", "shall", "contain", "cooked", "cook", "cooks",
        "recipe", "recipes", "generate", "something", "uses", "at", "home", "house",
        "dish", "food", "meal", "ingredient", "suggestion", "best", "great", "delicious", "try", "looking",
        "help", "find", "favorite", "suggest", "recommend", "make", "create", "prepare", "idea", "ideas"
    }

    keywords = [word for word in re.findall(
        r'\b\w+\b', text.lower()) if word not in stopwords]
    # Replace synonyms with the exact tag terms
    replaced_keywords = [synonym_to_tag.get(word, word) for word in keywords]
    return list(set(replaced_keywords))


def generate_food_suggestions(user_input, model, tokenizer, df):
    # Tokenize input and generate response (same as your original code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(user_input, return_tensors="pt",
                       padding=True, truncation=True, max_length=50).to(device)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=50,
                             do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    keywords = extract_relevant_keywords(user_input)
    print(keywords)

    def parse_tags(tags):
        try:
            if isinstance(tags, str):
                return ast.literal_eval(tags) if tags.startswith('[') else []
            elif isinstance(tags, list):
                return tags
            else:
                return []
        except (ValueError, SyntaxError):
            return []

    # Filter recipes based on the keywords
    df['parsed_tags'] = df['tags'].apply(parse_tags)
    df['flattened_tags'] = df['parsed_tags'].apply(
        lambda tags: ' '.join(tag for tag in tags))
    filtered_df = df[df['flattened_tags'].apply(
        lambda tags: all(keyword in tags.lower() for keyword in keywords))]

    complexity_keywords = ['complex', 'hard', 'advanced', 'complicated']
    simple_keywords = ['simple', 'quick', 'easy', 'beginner', 'beginner-level']

    if any(keyword in complexity_keywords for keyword in keywords):
        rank = pd.DataFrame(sorted(filtered_df.to_dict(
            orient='records'), key=lambda x: x['ICS'], reverse=True))
        return rank
    elif any(keyword in simple_keywords for keyword in keywords):
        rank = pd.DataFrame(sorted(filtered_df.to_dict(
            orient='records'), key=lambda x: x['ICS']))
        return rank
    else:
        return filtered_df.to_dict(orient='records')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/second', methods=['GET', 'POST'])
def second():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        sort_by = request.form.get('sortBy')
        filter_value = request.form.get('filterValue')
        prep_time = request.form.get('prepTime')

        # Get min and max prep times from the entire dataset
        all_prep_times = df['prep_time'].dropna().tolist()
        min_prep_time = int(min(all_prep_times))
        max_prep_time = int(max(all_prep_times))

        # Generate initial suggestions
        suggestions = generate_food_suggestions(
            user_input, model, tokenizer, df)

        # Convert suggestions to list if it's a DataFrame
        if isinstance(suggestions, pd.DataFrame):
            suggestions = suggestions.to_dict('records')

        # Filter by prep time if specified
        if prep_time and prep_time.isdigit():
            prep_time_int = int(prep_time)
            suggestions = [s for s in suggestions if isinstance(
                s.get('prep_time'), (int, float)) and s.get('prep_time', 0) <= prep_time_int]

        if sort_by:
            # Get unique values for the selected category
            if sort_by == 'cuisine':
                unique_values = sorted(set(df['cuisine'].dropna().unique()))
                # List of cuisines to remove
                excluded_cuisines = ['Afghan', 'African', 'Asian', 'Arab', 'Middle Eastern',
                                     'Nepalese', 'Pakistani', 'Sichuan', 'Thai','Sri Lankan','Parsi Recipes']
                # Filter out excluded cuisines
                unique_values = [
                    cuisine for cuisine in unique_values if cuisine not in excluded_cuisines]
            elif sort_by == 'course':
                unique_values = sorted(set(df['course'].dropna().unique()))
                excluded_course = ['World Breakfast']
                # Filter out excluded course
                unique_values = [
                    course for course in unique_values if course not in excluded_course]
            elif sort_by == 'diet':
                unique_values = sorted(set(df['diet'].dropna().unique()))
                excluded_diet = ['Not Specified']
                # Filter out excluded diet
                unique_values = [
                    diet for diet in unique_values if diet not in excluded_diet]
            else:
                unique_values = []

            # Filter by category if specified
            if filter_value:
                suggestions = [s for s in suggestions if s.get(
                    sort_by) == filter_value]

            return render_template(
                'second.html',
                suggestions=suggestions,
                search_performed=True,
                sort_by=sort_by,
                unique_values=unique_values,
                selected_value=filter_value,
                user_input=user_input,
                min_prep_time=min_prep_time,
                max_prep_time=max_prep_time,
                selected_prep_time=prep_time if prep_time else max_prep_time
            )

        return render_template(
            'second.html',
            suggestions=suggestions,
            search_performed=True,
            user_input=user_input,
            min_prep_time=min_prep_time,
            max_prep_time=max_prep_time,
            selected_prep_time=prep_time if prep_time else max_prep_time
        )

    return render_template('second.html', suggestions=[], search_performed=False)


@app.route('/filter', methods=['POST'])
def filter_results():
    try:
        sort_by = request.form.get('sortBy')
        filter_value = request.form.get('filterValue')
        user_input = request.form.get('user_input', '')
        prep_time = request.form.get('prepTime')

        # Generate fresh suggestions
        suggestions = generate_food_suggestions(
            user_input, model, tokenizer, df)

        # Convert to list if it's a DataFrame
        if isinstance(suggestions, pd.DataFrame):
            suggestions = suggestions.to_dict('records')

        # Get min and max prep times
        all_prep_times = df['prep_time'].dropna().tolist()
        min_prep_time = int(min(all_prep_times))
        max_prep_time = int(max(all_prep_times))

        # Filter by prep time if specified
        if prep_time and prep_time.isdigit():
            prep_time_int = int(prep_time)
            suggestions = [s for s in suggestions if isinstance(
                s.get('prep_time'), (int, float)) and s.get('prep_time', 0) <= prep_time_int]

        # Get unique values for the selected category
        if sort_by == 'cuisine':
            unique_values = sorted(set(df['cuisine'].dropna().unique()))
            # List of cuisines to remove
            excluded_cuisines = ['Afghan', 'African', 'Asian', 'Arab', 'Middle Eastern',
                                 'Nepalese', 'Pakistani', 'Sichuan', 'Thai']
            # Filter out excluded cuisines
            unique_values = [
                cuisine for cuisine in unique_values if cuisine not in excluded_cuisines]
        elif sort_by == 'course':
            unique_values = sorted(set(df['course'].dropna().unique()))
            excluded_course = ['World Breakfast', 'Any Course']
            # Filter out excluded cuisines
            unique_values = [
                course for course in unique_values if course not in excluded_course]
        elif sort_by == 'diet':
            unique_values = sorted(set(df['diet'].dropna().unique()))
        else:
            unique_values = []

        # Filter suggestions if a value is selected
        if filter_value:
            suggestions = [s for s in suggestions if s.get(
                sort_by) == filter_value]

        return render_template(
            'second.html',
            suggestions=suggestions,
            search_performed=True,
            sort_by=sort_by,
            unique_values=unique_values,
            selected_value=filter_value,
            user_input=user_input,
            min_prep_time=min_prep_time,
            max_prep_time=max_prep_time,
            selected_prep_time=prep_time
        )

    except Exception as e:
        print(f"Error in filter_results: {str(e)}")
        return render_template(
            'second.html',
            suggestions=[],
            search_performed=True,
            error="An error occurred while filtering results."
        )


@app.route('/details/<string:recipe_name>')
def details(recipe_name):
    # Fetch recipe details by name (case insensitive search)
    recipe = df[df['name'] == recipe_name].iloc[0]  # Case insensitive search
    skill_level_mapping = {
        1: 'Beginner',
        2: 'Intermediate',
        3: 'Advanced'
    }
    skill_level = skill_level_mapping.get(recipe['skill_level'], 'Unknown')
    recipe_details = {
        'name': recipe['name'],
        'ingredients': recipe['ingredients'],
        'description': recipe['description'],
        'instructions': recipe['instructions'],
        'preptime': recipe['prep_time'],
        'skill_level': skill_level,
        'cuisine': recipe['cuisine'],
        'course': recipe['course'],
        'diet': recipe['diet'],
        'image': recipe['image']
    }
    return render_template('details.html', recipe=recipe_details)


if __name__ == '__main__':
    app.run(debug=True)
