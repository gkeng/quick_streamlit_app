import shap
import streamlit as st

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


model_choices = ["Boosting", "Réseau de Neurones",
                 "Random Forest", "Régression Logistique", ]
dic_model = {"Régression Logistique": LogisticRegression(),
             "Réseau de Neurones": MLPClassifier(hidden_layer_sizes=[10], solver='lbfgs'),
             "Random Forest": RandomForestClassifier(),
             "Boosting": LGBMClassifier(), }


@st.cache(suppress_st_warning=True)
def get_shap_values(model_name, model, data):
    if model_name == "Boosting":
        explainer = shap.TreeExplainer(
            model=model, data=data, model_output="probability")
        shap_values = explainer(data)
    return shap_values


def get_dataframe():
    uploaded_file = st.file_uploader(" Choisis ton CSV", type="csv")

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file, sep=';')
            st.write(data)
            return data
        except Exception:
            st.write("Choisis un csv stp...")
            st.stop()


def get_target(data):
    target = st.text_input(label='Nom de la cible', )
    if not target or target not in data.columns:
        st.warning('Mets une cible valide stp')
        st.stop()
    st.success('Merci !')

    return target


def get_model_name():
    model_name = st.selectbox(label="Maintenant, choisis ton modèle",
                              options=model_choices)

    if not model_name:
        st.warning('Choisis un modèle stp')
        st.stop()
    st.success('Merci !')
    return model_name


def model_fit(model_name, data, target):

    model = dic_model[model_name]

    predictors = [col for col in data.columns if col != target]
    y = data[target]
    X = data[predictors]
    model.fit(X, y)

    return model


def get_row_number():

    row = st.number_input('Numéro de la ligne à expliquer', format='%i')

    if not row:
        st.warning('Choisis une ligne stp')
        st.stop()
    st.success('Merci !')
    return row
