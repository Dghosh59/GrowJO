import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ast

class similarity:

    def __init__(self,model):
        self.model=model

    def compute(self,tags,df):

        desc=pd.read_csv("Description.csv")
        desc["Vectorised"] = desc["Vectorised"].apply(ast.literal_eval)
        desc=desc[["company_name","Vectorised"]]
        desc = desc.drop_duplicates(subset=["company_name"])


        df=pd.merge(df,desc,how="inner",on="company_name")
        scores = []

        tag_vector = self.model.transform([tags]).toarray() 

        for i in range(df.shape[0]):
            vector = df.iloc[i, -1] 
            vector = np.array(vector).reshape(1, -1)

            # Compute cosine similarity
            score = cosine_similarity(vector, tag_vector)[0][0]
            scores.append(score)

        df["similarity_score"] = scores        
        sc = MinMaxScaler()
        df["similarity_score"] = np.round(sc.fit_transform(df[["similarity_score"]]),3)

        return df



if __name__ == "__main__":
    model = joblib.load("simscore_model.pkl")
    sim = similarity(model)
    df = pd.read_csv("export.csv")
    from preprocessor import preproccess
    df1 = preproccess(df.copy())
    lead_model = joblib.load("lead_score_model.pkl")
    scores = lead_model.predict_proba(df1)[:, 1].round(3)
    df = df[df["valuation"] != df["valuation"].max()]
    df["Lead_Score(%)"] = (scores * 100)[:len(df)]
    drop_cols = ["previous_ranking", "temp_ranking", "Contact Data"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.sort_values(by="Lead_Score(%)", ascending=False)

    d = sim.compute("ai machine learning", df)
    print(d.sort_values(by="similarity_score", ascending=False).head(20))
