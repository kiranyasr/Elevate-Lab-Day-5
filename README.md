# Heart Tree Models

## Project Objective
Learn **Decision Trees** and **Random Forests** for classification and regression.

## Workflow
1. Load `data/heart.csv`
2. Train a Decision Tree (baseline)
3. Tune tree depth to avoid overfitting
4. Train Random Forest & compare accuracy
5. Interpret feature importances
6. Save plots & models in `outputs/`

## How to Run
```bash
# 1. Create virtual env
python -m venv venv
# 2. Activate it
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
# 3. Install requirements
pip install -r requirements.txt
# 4. Run the workflow
python src/tree_workflow.py

Outputs:

outputs/decision_tree.png (visualization)

outputs/rf_importances.png (feature importances)

outputs/dt_tuned.joblib & outputs/rf_model.joblib (models)

1. Open `Heart-Tree-Models/` in **VS Code** (`File → Open Folder`).  
2. Create a **Python virtual environment**:
python -m venv venv

css
Copy code
3. Activate the venv inside VS Code terminal:
venv\Scripts\activate # Windows
source venv/bin/activate # Mac/Linux

markdown
Copy code
4. Install dependencies:
pip install -r requirements.txt

markdown
Copy code
5. Run your script:
python src/tree_workflow.py

yaml
Copy code

---

# 🏆 What You’ll Get in `/outputs`
- `decision_tree.png` → visualization of tuned decision tree  
- `rf_importances.png` → bar plot of top features  
- `dt_tuned.joblib` → tuned decision tree model  
- `rf_model.joblib` → random forest model  

---
