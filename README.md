# 🌶️🥘 JugaadKitchen.AI: Personalised Recipe Recommender System

**JugaadKitchen.AI** is a generative model-based recipe recommender system inspired by the Indian concept of *jugaad* — making the best out of what you have.  
It helps users decide what to cook based on the ingredients they already have, their skill level, and available cooking time. The system uses a fine-tuned GPT-2 model alongside a custom **Ingredient Complexity Score (ICS)** algorithm for personalized, feasible recipe suggestions.

---
### 🔗 Demo Video: 
https://drive.google.com/file/d/1swrIw3AvQF9FKgkn2zeqqQU_qkyAWcEv/view?usp=sharing

---
## 🎯 Project Goals

- Solve the everyday dilemma: “What do I cook?”
- Reduce food waste by using available ingredients efficiently
- Promote sustainable, budget-friendly cooking
- Deliver realistic recipe suggestions — not speculative AI creations

---
## 💡 Key Features

🧠 **GPT-2 Powered Recipe Matching**  
Fine-tuned GPT-2 model generates recipes using structured prompts based on ingredients, tags, time, and cooking level.

📊 **Ingredient Complexity Score (ICS)**  
A custom scoring system that ranks recipes by:
- Skill level (Beginner/Intermediate/Advanced)
- Prep time buckets
- Ingredient rarity
- Ingredient count

🔍 **NLP-Powered Prompt Processing**  
Uses SpaCy for keyword extraction and synonym handling to interpret prompts like  
_“easy vegetarian curry with rice”_.

⚖️ **Tailored Suggestions**  
Ranks recipes dynamically: simple ones for beginners, more complex ones for advanced users.

🌐 **Web-Based Interface**  
Frontend built with HTML/CSS and Flask, allowing users to input prompts and browse recipe cards.

---
## 📦 Model Download
Due to GitHub file size limits, the fine-tuned GPT-2 model isn't included here.

👉 Download the model from Google Drive: https://drive.google.com/file/d/1-2cnCIjMAC7DiBouVtUEwW_CD4Rz8ym6/view?usp=sharing

After downloading, place it in:
foodgpt/models/foodgptfinalmodel_v2/model.safetensors
For details, see trainedModel.txt in that folder.

---
## 🔍 How It Works
- User inputs prompt (e.g., “quick vegetarian dinner with paneer”).
- NLP extracts keywords like “quick”, “vegetarian”, “paneer”.
- GPT-2 and ICS rank relevant recipes from the dataset. The ICS works such that if the user requests simple, beginner-friendly dishes, recipes are ranked in ascending ICS order. For users requesting more complex meals, the results are sorted in descending ICS order.
- User receives tailored suggestions based on their prompts.
- User can further filter the recommended recipes based on cuisine(e.g., Bengali, South Indian, etc.), course(e.g., breakfast, lunch, dessert, etc.) and diet(e.g., Vegetarian, Diabetic-friendly, Gluten-free, etc.).
- User can also filter recipes based on preparation times.
- On clicking on one of the suggested recipes, user is taken to the next page with full recipe details including name, ingredients, prep steps, cuisine, course, diet, cooking time and image.

---
## 🧪 Technologies Used
- 🐍 Python 3.9
- 🤗 Transformers (Hugging Face)
- 🔍 SpaCy for NLP
- ⚗️ Flask for backend
- 🎨 HTML + CSS for frontend

---
## 🪔 Inspired By
- The spirit of Jugaad — turning kitchen scraps into soulful meals.
- For everyone who ever stood in front of a half-empty fridge thinking, “What can I make with this?”
