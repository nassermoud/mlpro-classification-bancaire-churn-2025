import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

def get_clean_data():
    
    data = pd.read_csv("data/train_data.csv")
    data = data.drop(['Surname', 'CustomerId'], axis=1)
    data['Gender'] = data['Gender'].map({ 'Male': 0, 'Female': 1 })
    data['Geography'] = data['Geography'].map({ 'France': 0, 'Germany': 1, 'Spain': 2 })
    
    return data

def add_sidebar():
    st.sidebar.header("Navigation")
    
    data = get_clean_data()
    
    slider_labels = [
    ("Score de crédit", "CreditScore"),
    ("Pays", "Geography"),
    ("Genre", "Gender"),
    ("Âge", "Age"),
    ("Ancienneté (années)", "Tenure"),
    ("Solde du compte", "Balance"),
    ("Nombre de produits", "NumOfProducts"),
    ("Carte de crédit (1=Oui,0=Non)", "HasCrCard"),
    ("Membre actif (1=Oui,0=Non)", "IsActiveMember"),
    ("Salaire estimé", "EstimatedSalary")
    ]
    
    # Create a dictionary to hold the input values
    input_dict = {}
    
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value = float(data[key].mean()))
    return input_dict

def get_radar_chart(input_data):
    
    categories = ['CreditScore',  'Geography',  'Gender',   'Age',  'Tenure',    'Balance',  
                'NumOfProducts',  'HasCrCard',  'IsActiveMember',  'EstimatedSalary']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['CreditScore'], input_data['Geography'], input_data['Gender'], input_data['Age'], 
            input_data['Tenure'], input_data['Balance'], input_data['NumOfProducts'], input_data['HasCrCard'], 
            input_data['IsActiveMember'], input_data['EstimatedSalary']
            ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['CreditScore'], input_data['Geography'], input_data['Gender'], input_data['Age'], 
            input_data['Tenure'], input_data['Balance'], input_data['NumOfProducts'], input_data['HasCrCard'], 
            input_data['IsActiveMember'], input_data['EstimatedSalary']
            ],
        theta=categories,
        fill='toself',
        name='Standard Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 5]
        )),
    showlegend=False
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Modèle de Classification Bancaire", 
        page_icon= "classification",
        layout="wide",
        initial_sidebar_state="expanded")
    
    
    
    input_data = add_sidebar()
    
    

    with st.container():
        st.title("Modèle de Classification Bancaire")
        st.write("Ce modèle prédit si un client va continuer à utiliser les services de la banque ou s'il va clôturer son compte (churn).")
        st.write("Pour ce faire, nous disposons d'un ensemble de données clients contenant plusieurs caractéristiques démographiques, financières et comportementales.")
        st.write("Nous devons développer un modèle de prédiction capable de déterminer, pour chaque client, s'il va résilier ou non car cela permet aux banques de mettre en place des stratégies de rétention efficaces.")
        
        
    col1, col2 = st.columns([4,1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        st.write("c'est la colonne 2")
    

if __name__ == "__main__":
    main()
