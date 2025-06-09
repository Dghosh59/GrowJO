from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
from preprocessor import preproccess
import math
from keyword_similarity import similarity

app = Flask(__name__)

model = joblib.load("lead_score_model.pkl")

model2=joblib.load("simscore_model.pkl")

similarity_engine=similarity(model2)
PAGINATION_SIZE = 20

results_df = pd.DataFrame()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global results_df

    file = request.files['file']
    tags = request.form.get('tags',None).strip()
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file)
    df1 = preproccess(df.copy())

    scores = model.predict_proba(df1)[:, 1].round(3)
    df = df[df["valuation"] != df["valuation"].max()]
    df["Lead_Score(%)"] = (scores * 100)[:len(df)]
    df = df.drop(columns=["previous_ranking", "temp_ranking", "Contact Data"])
    df = df.sort_values(by="Lead_Score(%)", ascending=False).reset_index(drop=True)
    if tags and tags.strip():
        df = similarity_engine.compute(tags,df)
        df = df.drop(columns=["Vectorised"])

    if 'similarity_score' in df.columns:
        df = df.sort_values(by='similarity_score', ascending=False)

    results_df = df


    return redirect(url_for('results', page=1))

@app.route('/results')
def results():
    global results_df

    page = int(request.args.get('page', 1))

    # Filter query parameters (defaults to 'All')
    country = request.args.get('country', 'All')
    city = request.args.get('city', 'All')
    industry = request.args.get('industry', 'All')

    df = results_df.copy()

    # Apply filters if selected
    if country != 'All':
        df = df[df['country'] == country]
    if city != 'All':
        df = df[df['city'] == city]
    if industry != 'All':
        df = df[df['Industry'] == industry]

    # Pagination
    total_pages = max(math.ceil(len(df) / PAGINATION_SIZE), 1)
    start = (page - 1) * PAGINATION_SIZE
    end = start + PAGINATION_SIZE
    table_data = df.iloc[start:end].to_html(classes="table table-bordered table-striped", index=False)

    # Get unique values for filters
    countries = ['All'] + sorted(results_df['country'].dropna().unique().tolist())
    cities = ['All'] + sorted(results_df['city'].dropna().unique().tolist())
    industries = ['All'] + sorted(results_df['Industry'].dropna().unique().tolist())

    return render_template(
        'results.html',
        table=table_data,
        page=page,
        total_pages=total_pages,
        countries=countries,
        cities=cities,
        industries=industries,
        selected_country=country,
        selected_city=city,
        selected_industry=industry
    )

if __name__ == '__main__':
    app.run(debug=True)