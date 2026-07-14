# includes the new design with Table instead of custom bars
import dash
from dash import dcc, html, ctx
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import MATCH, ALL, Patch
from dash import dash_table
import plotly.graph_objects as go
import plotly.express as px
import pickle
from sklearn.tree import DecisionTreeClassifier
# from sklearn.dummy import DummyClassifier
import math
from sklearn.model_selection import cross_val_score, cross_validate, TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import norm, fisher_exact, ks_2samp, ttest_ind, mannwhitneyu
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from statsmodels.stats.contingency_tables import Table
from datetime import date, datetime
# from sklearn.metrics import roc_auc_score
# import pyodbc
# from sqlalchemy import create_engine, text
# from urllib import parse
# import time
# import tracemalloc
# import ipg_functions as ipg
# from xgboost import XGBRegressor
import xgboost as xgb
import base64
import io
import dash_ag_grid as dag
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


# app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, "style.css"])
# app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
# app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


# ############# Input Data and Model #######################################
# with open("fake_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "Salary"

# with open("fake_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False



# ############# Input Data and Model #######################################
# with open("bob_data2.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# target = "is_good"
# selected_features = list(df.columns)

# with open("bob_model2.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = True


# ############# Input Data and Model #######################################
# df = pd.read_csv("QC_Shear_data.csv")
# print(df.shape)
# target = "Shear Values"
# selected_features = list(df.columns)

# with open("QC_shear_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# df = pd.read_csv("QC_Shear_data_w_random_no_corr.csv")
# print(df.shape)
# target = "Shear Values"
# selected_features = list(df.columns)

# with open("QC_shear_model_w_random_no_corr.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# df = pd.read_csv("QC_Shear_data_w_random_seed1_no_corr.csv")
# print(df.shape)
# target = "Shear Values"
# selected_features = list(df.columns)

# with open("QC_shear_model_w_random_seed1_no_corr.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# df = pd.read_csv("QC_Shear_data_w_random_no_corr_w_shear_curves.csv")
# print(df.shape)
# target = "Shear Values"
# selected_features = list(df.columns)

# with open("QC_shear_model_w_random_no_corr_w_shear_curves.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False

# ############# Input Data and Model #######################################
# df = pd.read_csv("QC_Shear_data_no_corr_w_shear_curves.csv")
# print(df.shape)
# target = "Shear Values"
# selected_features = list(df.columns)

# with open("QC_shear_model_no_corr_w_shear_curves.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


############# Input Data and Model #######################################
df = pd.read_csv("QC_Shear_data_ALL_TAGS.csv")
print(df.shape)
target = "Shear Values"
selected_features = list(df.columns)

with open("QC_shear_model_ALL_TAGS.pickle", "rb") as file:
    model = pickle.load(file)
is_classification = False




# ############# Input Data and Model #######################################
# with open("yield_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# selected_features = list(df.columns)
# target = "Slitter Yield (%)"

# with open("yield_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


# ############# Input Data and Model #######################################
# with open("die_pressure_data2.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# selected_features = list(df.columns)
# target = "Die Pressure Act  (PSI)"

# with open("die_pressure_model2.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False


############# Input Data and Model #######################################
# with open("HMC_Jan2026_data.pickle", "rb") as file:
#     df = pickle.load(file)
# print(df.shape)
# selected_features = list(df.columns)
# target = "Die Pressure Act  (PSI)"

# with open("HMC_Jan2026_model.pickle", "rb") as file:
#     model = pickle.load(file)
# is_classification = False



def sort_features_on_importance(model, X):
    # Encode df
    encoded_df = pd.get_dummies(X, prefix_sep="?")

    # Get prediction contributions
    contribs = model.get_booster().predict(xgb.DMatrix(encoded_df), pred_contribs=True)
    # contribs = model.predict(xgb.DMatrix(encoded_df), pred_contribs=True)[:, :-1]

    # separate bias from feature contributions
    bias = contribs[:, -1]
    contrib_df = pd.DataFrame(contribs[:, :-1], columns=encoded_df.columns)

    # Aggregate SHAP values for categorical features
    contrib_df = contrib_df.groupby(lambda x: x.split('?')[0], axis=1).sum()
    
    # Place contributions in a dataframe, calculate avg positive contribution score, and sort
    feature_importance_df = pd.DataFrame({"features":contrib_df.columns, "std_contribs":contrib_df.std(axis=0)}).sort_values("std_contribs", ascending=False)
    
    # Add bias to contributions df
    contrib_df["_bias_"] = bias

    return contrib_df, list(feature_importance_df["features"])


def update_model_metadata(model, df_clean_target):
    global max_contrib
    global sig_figs# Remove this global variable because it is no longer being used!
    global contrib_decimals
    global predict_decimals
    global categorical_features
    global model_features
    global contributions_df
    global ordered_features
    global model_features_dtypes

    ############# Get Max Contribution ######################################
    if is_classification:
        y = np.where(df_clean_target[target] == target_map[0], 0, 1)
        max_contrib = (y.std() / 2) * 100# max_contrib = 0.5
    else:
        max_contrib = df_clean_target[target].std()# * 2
    print(max_contrib)

    ############# Get Predicted value precision ##############################
    if is_classification:
        # y = np.where(df_clean_target[target] == target_map[0], 0, 1)
        # prediction_std = y.std()
        contrib_decimals = 2
        predict_decimals = 1
    else:
        prediction_std = df_clean_target[target].std()
        if prediction_std < 1:
            contrib_decimals = 3
            predict_decimals = 2
            sig_figs = 3
        elif prediction_std < 10:
            contrib_decimals = 2
            predict_decimals = 1
            sig_figs = 2
        else:
            contrib_decimals = 1
            predict_decimals = 0
            sig_figs = 1
    
    ############## Get Categorical Features and values ###################################
    X = df_clean_target.drop(columns=target)
    categorical_features = {}
    categorical_df = X.select_dtypes(include=["object"])
    for categorical_feature in categorical_df.columns:
        categorical_features[categorical_feature] = list(np.unique(categorical_df[categorical_feature].dropna()))


    ############## Get Encoded X #############################################
    encoded_X = pd.get_dummies(X, prefix_sep="?")# ONLY used when model requires dummy encodings

    ############## Get model features #########################################
    model_features = list(encoded_X.columns)# ONLY used when model REQUIRES dummy encodings
    # model_features = list(X.columns)# ONLY used when model does NOT REQUIRE dummy encodings

    ############## Get feature data types #####################################
    model_features_dtypes = encoded_X.dtypes# ONLY used when model REQUIRES dummy encodings
    # model_features_dtypes = X.dtypes# ONLY used when model does NOT REQUIRE dummy encodings

    ############### Get Features list sorted by contribution ###############
    contributions_df, ordered_features = sort_features_on_importance(model, X)
    

def quick_updates():
    global balancing_ratio
    global target_map
    if is_classification:
        unique_vals = np.unique(df[target].dropna())
        target_map = {0:unique_vals[0], 1:unique_vals[1]}
        # pos_count = df[target].sum()
        # neg_count = len(df[target]) - pos_count
        # balancing_ratio = neg_count / pos_count
        balancing_ratio = 1
    update_model_metadata(model, df)

quick_updates()




def get_indicator_KPI(curr_value, prev_value, title):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        value=curr_value,
        number={'font': {'size': 40}},
        delta = {'reference': prev_value, "font":{"size":14}},
        title={"text":title, "font":{"size":20}},
        gauge={
            'axis': {'range': [0, 100], 'visible': True}},
        # domain = {'row': 0, 'column': 0},
        # number = {'suffix': "%"}
    ))
    fig.update_layout(
        template = {'data' : {'indicator': [{
            'mode' : "number+delta+gauge"}]
                            }},
        height=180,
        width=280,
        margin=dict(l=30, r=35, t=50, b=0)
    )
    return fig


############################################# Tag_Selection_UI ########################################
right_panel = html.Div([
    html.Div([
        html.Div(f"{len(df.columns[0:10])} Selected", id="total-selected-tags-right-panel", style={"display":"inline-block","border":"0px solid",
                                                                                 "height":"30px","marginLeft":"0vw"}),
    ], style={"borderBottom":"1px solid #0000002e"}),
    html.Div(style={"height":"8px"}),

    html.Div([
        dcc.Checklist(
            id="final-features-checklist",
            options=df.columns,#[],
            value=df.columns[0:10],#[],
            inputStyle={"marginRight":"8px"}
        )

    ], style={"border":"0px solid","width":"100%","height":"365px","overflowY":"scroll","marginLeft":"0vw"}),

], id="right-panel", style={"border":"0px solid","width":"50%","height":"auto","position":"absolute",#"width":"25vw"#"height":"413px"
                            "padding":"10px 0vw 0vh 1vw","display":"inline-block","backgroundColor":"transparent",
                            "borderLeft":"1px solid","verticalAlign":"top","left":"50%","top":"0px","zIndex":"99"})#"left":"27vw"


left_panel = html.Div([
    html.Div([
        html.Div(f"{len(df.columns)} Total Variables", id="total-tags", style={"display":"inline-block","border":"0px solid","height":"30px"}),
    ], style={"borderBottom":"1px solid #0000002e"}),

    html.Div(style={"height":"8px"}),
    html.Div([
        dcc.Input(id="search", type="text", placeholder="Search...", style={"display":"inline-block", "width":"80%", "border":"1px solid", "border":"none","backgroundColor":"rgb(231,231,231)"}),
        html.Button("X", id="clear-search", style={"display":"inline-block", "border":"none","backgroundColor":"rgb(231,231,231)"}),
    ]),
    html.Div(style={"height":"8px"}),

    html.Div([
        dcc.Checklist(
            id="select-all-checkbox",
            options=["(ALL)"],
            value=["(ALL)"],
            inputStyle={"marginRight":"8px"}
        )
    ], style={"border":"0px solid"}),
    html.Div(style={"height":"8px"}),

    html.Div([
        dcc.Checklist(
            id="all-features-checklist",
            options=df.columns,
            value=df.columns[0:10],
            inputStyle={"marginRight":"8px"}
        )
    ], style={"border":"0px solid","width":"100%","height":"300px","overflowY":"scroll"}),
    
], id="left-panel", style={"border":"0px solid","width":"50%","height":"auto","position":"absolute",#"width":"25vw"
                           "padding":"10px 1vw 0vh 1vw","display":"inline-block","verticalAlign":"top",
                           "zIndex":"100","backgroundColor":"transparent","left":"0","top":"0px"})

tags_selection_UI = html.Div([
    left_panel,
    right_panel
], id="outer-panel", 
style={"border":"0px solid","width":"100%","height":"420px","left":"0vw","position":"relative","top":"0px",#"width":"53vw"
       "padding":"0px 0vw 0px 0vw","borderRadius":"5px","borderTopLeftRadius":"0px","zIndex":"98",
       "backgroundColor":"white","color":"black"})#"boxShadow":"0 2px 6px 0 gray",


@app.callback(
    dash.dependencies.Output(component_id='all-features-checklist', component_property='value', allow_duplicate=True),
    dash.dependencies.Input(component_id='select-all-checkbox', component_property='value'),
    dash.dependencies.State(component_id='all-features-checklist', component_property='value'),
    dash.dependencies.State(component_id='all-features-checklist', component_property='options'),
    prevent_initial_call=True,
)
def clicked_ALL_checkbox(select_all_checkbox, all_features_selected, all_features_options):
    if len(select_all_checkbox) == 1:
        additional_features_selected = []
        for feature in all_features_options:
            if feature not in all_features_selected:
                additional_features_selected.append(feature)
        all_features_selected.extend(additional_features_selected)
    else:
        for feature in all_features_options:
            if feature in all_features_selected:
                all_features_selected.remove(feature)
    return all_features_selected

@app.callback(
    dash.dependencies.Output(component_id='final-features-checklist', component_property='value', allow_duplicate=True),
    dash.dependencies.Input(component_id='all-features-checklist', component_property='value'),
    dash.dependencies.State(component_id='all-features-checklist', component_property='options'),
    dash.dependencies.State(component_id='final-features-checklist', component_property='value'),
    prevent_initial_call=True,
)
def clicked_some_all_features_checkbox(all_features_selected, all_features_options, final_features):
    return all_features_selected

@app.callback(
    dash.dependencies.Output(component_id='final-features-checklist', component_property='options', allow_duplicate=True),
    dash.dependencies.Output(component_id='all-features-checklist', component_property='value', allow_duplicate=True),
    dash.dependencies.Output(component_id='total-selected-tags-right-panel', component_property='children', allow_duplicate=True),
    dash.dependencies.Input(component_id='final-features-checklist', component_property='value'),
    dash.dependencies.State(component_id='all-features-checklist', component_property='value'),
    prevent_initial_call=True,
)
def unselected_final_features_checkbox(final_features, all_features_selected):
    result = final_features
    return final_features, result, f"{len(final_features)} Selected"

@app.callback(
    dash.dependencies.Output(component_id='search', component_property='value', allow_duplicate=True),
    dash.dependencies.Input(component_id='clear-search', component_property='n_clicks'),
    prevent_initial_call=True,
)
def clear_search_field(n_click):
    return ""

@app.callback(
    dash.dependencies.Output(component_id='all-features-checklist', component_property='options', allow_duplicate=True),
    dash.dependencies.Input(component_id='search', component_property='value'),
    dash.dependencies.State(component_id="all-features-checklist", component_property='options'),
    prevent_initial_call=True,
)
def updating_search_field(search_text, all_features):
    # print("updating_search_field")
    if search_text == None or search_text == "":
        return df.columns
    if search_text[0:6] == "**top " and len(search_text) >= 7:
        return ordered_features[0:int(search_text[6:])]
    filtered_tags = df.columns[df.columns.str.contains(search_text, case=False, regex=False)]
    # all_features = pd.Series(all_features)
    # filtered_tags = all_features[all_features.str.contains(search_text, case=False)]
    return filtered_tags
###################################### End Tag Selection UI ###############################################


fake_df = pd.DataFrame({"feature":["Feature1\n20", "Feature2\n75.3", "Feature3\n250", "Feature4\n450.8"], "pos":[20.738913287, 50.184762376836, 100.152527637, 80.8983808318028], "neg":[30, 65, 10, 45]})
# "headerComponent":"ClickableHeader",
# columnDefs = [{"field": fake_df.columns[i], "maxWidth":60, "minWidth":40, 'cellStyle': {
#                "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + params.value + '%, transparent ' + params.value + '%)', 'font-size':'14px'} : {'backgroundColor':'white'}",
# }} for i in range(0, len(fake_df.columns))]

# columnDefs = [{"field": fake_df.columns[i], 'cellStyle': {
#                "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + params.value + '%, transparent ' + params.value + '%)', 'font-size':'14px'} : {'backgroundColor':'white'}",
# }} for i in range(0, len(fake_df.columns))]

# pos = {"field": "pos", 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + params.value + '%, transparent ' + params.value + '%)', 'font-size':'14px'} : {'backgroundColor':'white'}",
# }}



# pos = {"field": "pos", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + (params.value/0.5*100) + '%, transparent ' + params.value + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'1px solid white', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'white', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'white'}",
# }}

# pos = {"field": "pos", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + (params.value/0.5*100) + '%, transparent ' + (params.value/0.5*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'1px solid white', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'white', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'white'}",
# }}

# pos = {"field": "pos", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + (params.value/0.5*100) + '%, transparent ' + (params.value/0.5*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'transparent'}",
# }}

# pos = {"field": "pos", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + (params.value/5000*100) + '%, transparent ' + (params.value/5000*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'transparent'}",
# }}


# pos = {"field": "pos", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + (params.value/5000*100) + '%, transparent ' + (params.value/5000*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'rgb(100, 100, 100)', 'border-top':'5px solid #FDFBF7', 'border-bottom':'5px solid #FDFBF7'} : {'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'transparent'}",
# }}



# pos = {"field": "pos", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + (params.value/5000*100) + '%, transparent ' + (params.value/5000*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'rgb(100, 100, 100)', 'border-top':'5px solid #FDFBF7', 'border-bottom':'5px solid #FDFBF7'} : {'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'transparent'}",
# }}

# pos has the dynamic max_contribution
pos = {"field": "pos", "headerName":"", "flex":1, 'cellStyle': {
       "function": f"params.value > 0 ? {{'background': 'linear-gradient(90deg, #599ad358 0%, #599ad358 ' + (params.value/{max_contrib}*100) + '%, transparent ' + (params.value/{max_contrib}*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'rgb(100, 100, 100)', 'border-top':'5px solid #FDFBF7', 'border-bottom':'5px solid #FDFBF7'}} : {{'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-start', 'color':'transparent'}}",
}}


# feature_def = {"field": "feature", 'cellStyle': {'font-size':'14px'},#,'backgroundColor':'white'
# }

# feature_def = {"field": "feature", 'cellStyle': {'font-size':'14px', 'whiteSpace':'pre', 'lineHeight':'unset', 'display':'flex', 'alignItems':'center', 'justifyContent':'center', 'textAlign':'center'}, "wrapText":True, "autoHeight":True,#,'backgroundColor':'white'
# }

# feature_def = {
#     "field": "feature",
#     "headerName":"CONTRIBUTIONS",
#     "headerClass":"center-header",
#     'cellStyle': {
#         'font-size':'14px', 'whiteSpace':'pre', 'lineHeight':'unset', 
#         'display':'flex', 'alignItems':'center', 'justifyContent':'center', 'textAlign':'center',
#         'border':'1px solid black',
#     }, 
#     "wrapText":True, 
#     "autoHeight":True,#,'backgroundColor':'white'
#     'width':'500px',
# }


# 'font-size':'14px'
feature_def = {
    "field": "feature",
    "headerName":"isDefect (prob)\n28.5%",
    "headerClass":"center-header",
    'cellStyle': {
        'fontSize':'14px', 'whiteSpace':'pre-line', 'lineHeight':'1.2', 
        'display':'inline-flex', 'alignItems':'center', 'justifyContent':'center', 'textAlign':'center',
        'border':'1px solid #d3d3d3', 'flexDirection':'column','height':'100%', 'cursor':'pointer', 'backgroundColor':'#f1f1f1',
        'borderRadius':'5px',
    }, 
    "wrapText":True, 
    "autoHeight":True,#,'backgroundColor':'white'
    'width':'250px',
    "suppressSizeToFit":True,
    "wrapHeaderText": True,
    "autoHeaderHeight": True,
    # "cellRenderer": {"function": "styleSecondLine"},
    # "cellRenderer": {"function": "dashAgGridFunctions.styleSecondLine"},
    # "cellRenderer": "styleSecondLine",
    "cellRendererParams": {"dangerously_allow_code": True},
}# Last think I added was the borderRadius


# neg = {"field": "neg", 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(270deg, #599ad358 0%, #599ad358 ' + params.value + '%, transparent ' + params.value + '%)', 'font-size':'14px', 'textAlign':'center'} : {'backgroundColor':'white'}",
# }}

# neg = {"field": "neg", "headerName":"", 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(270deg, #599ad358 0%, #599ad358 ' + params.value + '%, transparent ' + params.value + '%)', 'font-size':'14px', 'textAlign':'center', 'border':'1px solid white'} : {'backgroundColor':'white'}",
# }}

# neg = {"field": "neg", "headerName":"", 'cellStyle': {
#        "function": "params.value > 0 ? {'background': 'linear-gradient(270deg, #599ad358 0%, #599ad358 ' + params.value + '%, transparent ' + params.value + '%)', 'font-size':'14px', 'textAlign':'center', 'border':'1px solid white', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'center'} : {'backgroundColor':'white'}",
# }}

# neg = {"field": "neg", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value < 0 ? {'background': 'linear-gradient(270deg, #f9a75a59 0%, #f9a75a59 ' + (Math.abs(params.value)/0.5*100) + '%, transparent ' + Math.abs(params.value) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'1px solid white', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'white', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'white'}",
# }}

# neg = {"field": "neg", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value < 0 ? {'background': 'linear-gradient(270deg, #f9a75a59 0%, #f9a75a59 ' + (Math.abs(params.value)/0.5*100) + '%, transparent ' + (Math.abs(params.value)/0.5*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'1px solid white', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'white', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'white'}",
# }}

# neg = {"field": "neg", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value < 0 ? {'background': 'linear-gradient(270deg, #f9a75a59 0%, #f9a75a59 ' + (Math.abs(params.value)/0.5*100) + '%, transparent ' + (Math.abs(params.value)/0.5*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'1px solid white', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'white', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'white'}",
# }}

# neg = {"field": "neg", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value < 0 ? {'background': 'linear-gradient(270deg, #f9a75a59 0%, #f9a75a59 ' + (Math.abs(params.value)/0.5*100) + '%, transparent ' + (Math.abs(params.value)/0.5*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'transparent'}",
# }}

# neg = {"field": "neg", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value < 0 ? {'background': 'linear-gradient(270deg, #f9a75a59 0%, #f9a75a59 ' + (Math.abs(params.value)/5000*100) + '%, transparent ' + (Math.abs(params.value)/5000*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'rgb(100, 100, 100)'} : {'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'transparent'}",
# }}

# neg = {"field": "neg", "headerName":"", "flex":1, 'cellStyle': {
#        "function": "params.value < 0 ? {'background': 'linear-gradient(270deg, #f9a75a59 0%, #f9a75a59 ' + (Math.abs(params.value)/5000*100) + '%, transparent ' + (Math.abs(params.value)/5000*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'rgb(100, 100, 100)', 'border-top':'5px solid #FDFBF7', 'border-bottom':'5px solid #FDFBF7'} : {'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'transparent'}",
# }}# I recently added a 5px border at the top and bottom of the cell


# neg has the dynamic max_contribution
neg = {"field": "neg", "headerName":"", "flex":1, 'cellStyle': {
       "function": f"params.value < 0 ? {{'background': 'linear-gradient(270deg, #f9a75a59 0%, #f9a75a59 ' + (Math.abs(params.value)/{max_contrib}*100) + '%, transparent ' + (Math.abs(params.value)/{max_contrib}*100) + '%)', 'fontSize':'14px', 'textAlign':'center', 'border':'none', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'rgb(100, 100, 100)', 'border-top':'5px solid #FDFBF7', 'border-bottom':'5px solid #FDFBF7'}} : {{'backgroundColor':'transparent', 'textAlign':'center', 'display':'inline-flex', 'alignItems':'center', 'justifyContent':'flex-end', 'color':'transparent'}}",
}}# I recently added a 5px border at the top and bottom of the cell

columnDefs = [neg, feature_def, pos]



app.layout = html.Div([

    html.Button("click me", id="update-page-btn", n_clicks=0, style={"border":"none","color":"transparent"}),
    html.Div(
        # target.upper() + " - SIMULATION",
        id = "page-title",
        style={"position":"absolute","height":"8vh","top":"2vh","left":"5vw","fontSize":"4vh","fontWeight":"600", "display":"none"}),


    html.Div([
        
        html.Div(
            id="feature-slider-container",
            style={"position":"absolute","height":"70vh","width":"15vw","left":"15vw","top":"20px","border":"0px solid"}),#"height":"80vh"

        html.Div(
            id="positive-contribution-container",
            style={"position":"absolute","height":"70vh","width":"15vw","left":"30vw","top":"20px","border":"0px solid"}),#"height":"80vh"

        html.Div(
            id="negative-contribution-container",
            style={"position":"absolute","height":"70vh","width":"15vw","left":"0vw","top":"20px","border":"0px solid"}),#"height":"80vh"


    ], style={"position":"absolute","height":"80vh","width":"45vw","left":"5vw","top":"104vh","border":"0px solid",#"top":"15vh"
              "overflowY":"auto","overflowX":"hidden","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)",
              "backgroundColor":"#ffffff",  "display":"none"}, className="split-background"),#"backgroundColor":"#fff"
    
    html.Div([
        "CONTRIBUTIONS BY TAG",
        html.Div([
            dcc.Checklist(
                id="synced-movement-checkbox",
                options=["Synced Movement"],
                value=[],
                inputStyle={"marginRight":"5px"}
            )
        ], style={"border":"0px solid","width":"auto","height":"auto","marginLeft":"3vw","fontSize":"2vh","visibility":"hidden"}),



        html.Button("Optimization", id="show-optimization-modal-btn-DONTUSE",
                    style={"float":"right","visibility":"visible","backgroundColor":"transparent","border":"none",
                           "fontSize":"2vh","color":"#599ad3","borderBottom":"1px solid","padding":"2px 0px 0px 0px","marginRight":"5px","marginLeft":"3vw"}),
    ], style={"position":"absolute","height":"4vh","width":"45vw","left":"5vw","top":"100vh","fontSize":"2.5vh","border":"0px solid",
              "borderTopLeftRadius":"5px","borderTopRightRadius":"5px","backgroundColor":"#dfdfdf","paddingLeft":"1vw",
              "color":"#599ad3","fontWeight":"600","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)",
              "clipPath":"inset(0px -5px 0px -5px)","display":"none"}),#"display":"flex"
    


    html.Div([
        html.Img(id="left-arrow", src='/assets/left-arrow3.png', style={"display":"inline-block", "width":"32px", "height":"auto", "marginLeft":"0px", "cursor":"pointer"}),
        html.Img(id="right-arrow", src='/assets/right-arrow3.png', style={"display":"inline-block", "width":"32px", "height":"auto", "marginLeft":"15px", "cursor":"pointer"}),

        dcc.RadioItems(
            options=['Graph', 'Edit', 'Synced Edit'],
            value='Graph', inline=True, inputStyle={"marginRight":"1px"}, labelStyle={"marginRight":"20px"},
            id="mode-selection-radio", style={"display":"inline-block", "marginLeft":"20px", "cursor":"pointer"}
        ),

        html.Div("Optimization", id="show-optimization-modal-btn", style={"display":"inline-block","marginLeft":"20px","border":"0px solid","color":"#599ad3","cursor":"pointer"}),

        html.Div("PCA", id="pca-button", style={"display":"none","marginLeft":"20px","border":"0px solid","color":"#599ad3","cursor":"pointer"}),
    ], style={"position":"absolute","left":"4vw","top":"5vh","border":"0px solid"}),

    # html.I(className="bi bi-arrow-right fs-4 me-2", style={"position":"absolute","left":"4vw","top":"11vh"}),

    dag.AgGrid(
        id="contributions-table",
        defaultColDef={
            "cellStyle": {"styleConditions": [], "defaultStyle": {"textAlign": "center"}},
            "cellRenderer":"markdown",
            "wrapText":True, "autoHeight":True,
        },
        columnDefs=columnDefs,
        rowData=fake_df.to_dict('records'),
        columnSize="sizeToFit",#"sizeToFit" #autoSize
        style={"position":"absolute", "height":"88vh", "width":"550px", "left":"4vw", "top":"10vh",#"top":"110vh", #"width":"450px"
               "boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)", "borderRadius":"7px","border":"0px solid"},#"width":"40vw" #"width":"450px"
        dashGridOptions={
            "suppressHorizontalScroll":True, "suppressRowHoverHighlight": True,# "suppressScrollOnNewData": True,
        },


        # getRowId="params.data.id", # Required if targeting scroll via rowId
        # scrollTo={"rowIndex":20, "rowPosition":"middle"},
    ),


    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Edit Value")),
        dbc.ModalBody([
            html.Div(id="selected-feature-numerical-modal", style={"fontWeight":"bold"}),
            html.Div([
                dcc.Slider(
                    id="numerical-slider",
                    min=60, max=75, step=0.01, marks=None,
                    value=60,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},#,"lineHeight":"1"
                    # className="slider-class",#
                )
            ], style={"width":"90%","marginTop":"10px"}),
            
        ]),
    ], id="edit-numerical-value-modal", is_open=False, zIndex=99999, autoFocus=False, centered=True, keyboard=True),#backdrop="static",

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Edit Value")),
        dbc.ModalBody([
            html.Div(id="selected-feature-categorical-modal", style={"fontWeight":"bold"}),
            dcc.Dropdown(
                id="categorical-dropdown",
                multi=False,
                clearable=False,
                placeholder="None",
                optionHeight=40,
                disabled=False,
                # options=df.columns,
                style={"fontSize":"14px", "cursor":"pointer"},
                # className="regular-dropdown"
            ),
        ]),
    ], id="edit-categorical-value-modal", is_open=False, zIndex=99999, autoFocus=False, centered=True, keyboard=True),#backdrop="static",


    dcc.Store(id="selected-feature-data", data={}),


    # html.Div(style={"position":"absolute", "height":"10vh", "width":"45vw", "left":"5vw", "top":"155vh"}),


    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Variation Analysis")),
        dbc.ModalBody([
            # html.Div(id="selected-feature-categorical-modal", style={"fontWeight":"bold"}),
            dcc.Dropdown(
                id="source-pca-animation-dropdown",
                options=[f"Source {i+1}" for i in range(0, 10)],
                value="Source 1",
                multi=False,
                clearable=False,
                # placeholder="None",
                optionHeight=40,
                disabled=False,
                style={"fontSize":"14px", "cursor":"pointer","width":"200px"},
                # className="regular-dropdown"
            ),

            html.Div(
                dcc.Graph(
                    id="pca-animation",
                    config={'displayModeBar': False},
                    style={"position":"absolute","height":"100%","width":"100%","border":"0px solid", 
                        "padding":"5px", "boxSizing":"border-box", "borderRadius":"0px"}
                ),
                style={"position":"absolute","left":"0%","top":"10%","height":"88%","width":"600px","overflowY":"scroll",
                    "border":"0px solid","borderRadius":"10px"}
            ),


            html.Div([
                dcc.Graph(
                    id="explained-var-chart",
                    config={'displayModeBar': False},
                    style={"border":"0px solid","padding":"5px", "boxSizing":"border-box", "height":"100%", "width":"100%"},
                ),
            ], style={"position":"absolute","left":"48vw","top":"2vh","height":"30vh","width":"35vw","border":"0px solid", 
                        "padding":"5px", "boxSizing":"border-box", "boxShadow":"0 2px 6px 0 gray","borderRadius":"10px"}),



        ]),
    ], id="pca-modal", is_open=False, fullscreen=True, zIndex=99999, autoFocus=False, centered=True, keyboard=True),#backdrop="static",

    
    dcc.Store(id="pca-data"),
    
    dcc.Download(id="download-optimization-data"),


    # html.Div([
    #             dcc.Dropdown(
    #                 id="source-pca-animation-dropdown",
    #                 options=[f"Source {i+1}" for i in range(0, 10)],
    #                 value="Source 1",
    #                 multi=False,
    #                 clearable=False,
    #                 # className="blue-dropdown",
    #                 # className="multiline-dropdown white-dropdown",
    #                 # optionHeight=60,
    #                 style={"fontSize":"14px"},#"left":"50%"
    #                 # style={"fontSize":"14px","textAlign":"left","border":"none"}#,"textOrientation":"upright","writingMode":"vertical-lr"
    #             )
    #         ], style={"position":"absolute","top":"1.5%","right":"25%","width":"7vw","textAlign":"center"}),
    


    html.Div([
        dcc.Dropdown(
            id="additional-feature-dropdown",
            multi=False,
            clearable=True,
            placeholder="Add Another Variable",
            optionHeight=40,
            disabled=False,
            options=df.columns,
            style={"fontSize":"14px", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
            # className="regular-dropdown"
        )
    ], style={"position":"absolute", "left":"calc(5vw + 550px)", "top":"5vh", "width":"20vw"}),


    dcc.Graph(
        id="feature-contrib-graph",
        style={"position":"absolute","height":"43vh","left":"calc(5vw + 550px)","width":"calc(91vw - 550px)","top":"calc(6vh + 38px)","border":"0px solid","padding":"0px", 
               "boxSizing":"border-box","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)"},# "left":"51vw" #"width":"45vw"
        config={'displayModeBar': False}, className="rounded-graph",
    ),

    dcc.Graph(
        id="feature-prediction-graph",
        style={"position":"absolute","height":"43vh","left":"calc(5vw + 550px)","width":"calc(91vw - 550px)","top":"calc(50vh + 38px)","border":"0px solid","padding":"0px", 
               "boxSizing":"border-box","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)"},#"height":"50vh","left":"0%","width":"60vw" #"top":"calc(12vh + 38px)"
        config={'displayModeBar': False}, className="rounded-graph",
    ),

    # html.Div([
    #     dcc.Graph(
    #         id="feature-prediction-graph",
    #         style={"height":"100%","width":"100%","borderRadius":"10px"},
    #         config={'displayModeBar': False}
    #     ),
    # ], style={"position":"absolute","height":"43vh","left":"calc(5vw + 550px)","width":"calc(91vw - 550px)","top":"calc(50vh + 38px)","border":"0px solid","padding":"5px", 
    #           "boxSizing":"border-box","boxShadow":"0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)", "borderRadius":"10px","overFlow":"hidden"}),

    



    dbc.Modal([
        # dbc.ModalHeader(dbc.ModalTitle("", id="modal-title")),
        dbc.ModalBody([
            html.Div([
                html.Div("Prediction  vs  ", id="chart-title-target",
                         style={"display":"inline-block","border":"0px solid","whiteSpace":"pre","marginLeft":"2vw","fontSize":"20px"}),
                html.Div("Load cell 23 ACTUAL TENSION", id="chart-title-feature",
                         style={"display":"inline-block","border":"0px solid","whiteSpace":"pre","fontSize":"20px"}),
                html.Div(" ,", style={"display":"inline-block","border":"0px solid","whiteSpace":"pre","fontSize":"20px"}),
                html.Div(
                    dcc.Dropdown(
                        id="add-another-feature-dropdown",
                        multi=False,
                        clearable=True,
                        placeholder="Add Another Variable",
                        optionHeight=40,
                        disabled=False,
                        options=df.columns,
                        style={"fontSize":"14px", "border":"none", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                        # className="regular-dropdown"
                    )
                , style={"display":"inline-block","width":"20vw","verticalAlign":"middle","marginTop":"2px"}),
                

            ]),#, style={"display":"flex","justifyContent":"space-between","gap":"0px"}
            
            dcc.Graph(
                id="contrib-graph",
                style={"position":"relative","height":"75vh","left":"0%","width":"100%","top":"0vh","border":"0px solid","padding":"5px", 
                    "boxSizing":"border-box"},#"height":"50vh","left":"0%","width":"60vw"
                config={'displayModeBar': False}
            ),

        ], id="crazy"),
    ], id="modal", is_open=False, size="xl", zIndex=99999, autoFocus=False, centered=True),



    ##################### Optimization ##########################
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Optimization")),
        dbc.ModalBody([
            
            html.Div("Excluded Variables..."),
            dcc.Dropdown(
                id="exluded-optmization-features-dropdown",
                multi=True,
                clearable=False,
                placeholder="None",
                optionHeight=40,
                disabled=False,
                options=df.columns,
                style={"fontSize":"14px", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                # className="regular-dropdown"
            ),

            html.Div(f"Are Higher Values of {target} Better?", id="optimization-direction", style={"marginTop":"2vh"}),
            dcc.Dropdown(
                id="higher-is-better-dropdown",
                multi=False,
                clearable=False,
                placeholder="None",
                optionHeight=40,
                disabled=False,
                # options=["Higher values of Die Pressure", "Lower values of Die Pressure"],
                options=[{'label': 'Yes', 'value': 'Yes'},
                         {'label': 'No', 'value': 'No'}],
                value="Yes",
                style={"fontSize":"14px", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                # className="regular-dropdown"
            ),

            html.Div("Optimization Method", style={"marginTop":"2vh"}),
            dcc.RadioItems(
                options=['Historical', 'Interpolated'],
                value='Historical', inline=True, inputStyle={"marginRight":"1px"}, labelStyle={"marginRight":"20px"},
                id="optimization-method-radio", style={"marginLeft":"15px"}
            ),

            html.Div([
                html.Div("Variable Percentile Range", style={"marginTop":"2vh", "marginBottom":"10px"}),
                dcc.RangeSlider(
                    min=1, max=99, step=1, marks=None,
                    value=[1, 99],
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="percentile-slider",
                    className="slider-class",#
                ),
            ], id="interpolated-options-div", style={"display":"none"}),

            html.Div([
                # html.Div("Top Percent of Samples to Use", style={"marginTop":"2vh", "marginBottom":"10px"}),
                html.Div("# of Configurations to Return", style={"marginTop":"2vh", "marginBottom":"10px"}),
                dcc.Input(id="top-n", type="text", value="100", style={"width":"50px","marginLeft":"15px"}),
                # html.Span("%"),
            ], id="historical-options-div", style={"display":"block"}),

            html.Div(dbc.Button("Optimize", id="optimize-btn", class_name="me-1", style={"height":"auto","marginTop":"2vh","width":"100%"})),
        ]),
        # dbc.ModalFooter(dbc.Button("Optimize", id="optimize-btn", class_name="me-1"))
    ], id="optimization-modal", is_open=False, zIndex=99999, autoFocus=False, centered=True),#size="lg",


    ###################### Optimization Low vs High ############################
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Optimization Comparison: Best vs Worst Regions", id="optimization-comparison-modal-title")),
        dbc.ModalBody([
            dag.AgGrid(
                id="optimization-comparison-table",
                # rowData=df_head.to_dict('records'),
                columnDefs=[
                    {
                        "headerName":"Target =>",
                        "headerClass": "BOB-center-header",# "headerClass": "ag-header-cell-label-center",
                        "suppressStickyLabel": True,
                        "children":[
                            {"field":"Variable","wrapText":True, "autoHeight":True, "width":"200px","suppressSizeToFit":True, "headerClass":"BOB-header"}
                        ],
                        "wrapText":True, "autoHeight":True, "width":"200px",
                    },
                    {
                        "headerName":"Best Region",
                        # "headerClass": "center-header",# "headerClass": "ag-header-cell-label-center",
                        # "headerClass": "BOB-header",
                        "headerClass": "BOB-center-header",
                        "suppressStickyLabel": True,
                        "children":[
                            {"field":"BOB_Center", "headerName":"Center", "headerClass":"BOB-header", "cellStyle":{"backgroundColor": "#6ff95a30"}},# "valueFormatter":{"function": "d3.format(',.2f')(params.value)"}
                            {"field":"BOB_5th", "headerName":"5th", "headerClass":"BOB-header", "cellStyle":{"backgroundColor": "#6ff95a30"}},
                            {"field":"BOB_95th", "headerName":"95th", "headerClass":"BOB-header", "cellStyle":{"backgroundColor": "#6ff95a30"}}
                        ],
                    },
                    {
                        "headerName":"Worst Region",
                        # "headerClass": "center-header",# "headerClass": "ag-header-cell-label-center",
                        # "headerClass": "WOW-header",
                        "headerClass": "WOW-center-header",
                        "suppressStickyLabel": True,
                        "children":[
                            # {"field":"WOW_Center", "headerName":"Center", "headerClass":"WOW-header", "cellStyle":{"backgroundColor": "#d359593d"}, "valueFormatter":{"function": "d3.format(',')(params.value)"}},
                            {"field":"WOW_Center", "headerName":"Center", "headerClass":"WOW-header", "cellStyle":{"backgroundColor": "#d359593d"}},
                            {"field":"WOW_5th", "headerName":"5th", "headerClass":"WOW-header", "cellStyle":{"backgroundColor": "#d359593d"}},
                            {"field":"WOW_95th", "headerName":"95th", "headerClass":"WOW-header", "cellStyle":{"backgroundColor": "#d359593d"}}
                        ]
                    },

                ],
                defaultColDef={
                    "resizable": True, 
                    # "cellStyle": {"styleConditions": [], "defaultStyle": {"textAlign": "center","fontSize":"14px"}},
                    "cellStyle": {'whiteSpace':'pre-line','lineHeight':'1.2',"textAlign": "center","fontSize":"14px",'display':'inline-flex', 'alignItems':'center', 'justifyContent':'center',},
                    "headerClass": "center-header",
                    "dangerously_allow_code": True,
                    "wrapText":True, "autoHeight":True,
                    # "cellRenderer":"markdown",

                },
                # defaultColDef={
                #     "cellStyle": {"styleConditions": [], "defaultStyle": {"textAlign": "center"}},
                #     "cellRenderer":"markdown",
                #     "wrapText":True, "autoHeight":True,
                # },

                columnSize="sizeToFit",#"sizeToFit" #autoSize
                style={"height":"100%", "width":"750px", "display":"inline-block", "verticalAlign":"top"}# width:40%
            ),

            html.Div([

                dcc.Graph(
                    id="optimization-line-chart",
                    style={"position":"relative","height":"40vh","left":"0%","width":"100%","marginTop":"3vh","border":"0px solid","padding":"5px", 
                        "boxSizing":"border-box"},
                    config={'displayModeBar':False}
                ),

                dcc.Graph(
                    id="optimization-scatter-chart",
                    style={"position":"relative","height":"40vh","left":"0%","width":"100%","marginTop":"3vh","border":"0px solid","padding":"5px", 
                        "boxSizing":"border-box"},
                    config={'displayModeBar':False}
                ),
            ], style={"marginLeft":"1%","width":"calc(98% - 750px)","display":"inline-block"})#width:58%

        ], id="optimization-comparison-modal-body"),
        # dbc.ModalFooter(dbc.Button("Save", id="save-actions", class_name="me-1"))
    ], id="optimization-comparison-modal", is_open=False, fullscreen=True, zIndex=99999, autoFocus=False, centered=True),#style={"position":"absolute"} ## size="lg",, #size="xl"


    dcc.Store(id="BOB-values-store", data=[]),
    dcc.Store(id="WOW-values-store", data=[]),



    ###################### Data Preparation #####################
    html.Button("Data Preparation", id="data-preparation-btn",
                style={"position":"absolute","border":"none","right":"10vw","top":"0vh","color":"transparent"}),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select Analysis Type")),
        dbc.ModalBody([
            dcc.RadioItems(
                id="analysis-type-radio-btn",
                options=["Good vs Bad", "Unplanned Stop Data"],
                value="Unplanned Stop Data",
                inputStyle={"marginRight":"8px"}
            )
        ]),
        dbc.ModalFooter(dbc.Button("Next", id="analysis-type-next-btn", class_name="me-1"))
    ], id="analysis-type-modal", is_open=False, zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),


    ################## Unplanned Stop Data UI ######################
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Unplanned Stop Data")),
        dbc.ModalBody([
            html.Div("Site"),
            dcc.Dropdown(id="select-site-dropdown",multi=False,clearable=False,placeholder="Select Site...",
                options=["Carbondale", "Danville"],value="Carbondale",optionHeight=40,disabled=False,
                style={"fontSize":"14px", "cursor":"pointer"},
            ),
            html.Div("Machine", style={"marginTop":"2vh"}),
            dcc.Dropdown(id="select-machine-dropdown",multi=False,clearable=False,placeholder="Select Machine...",
                options=["Coater 44", "Coater 51", "Coater 52"],value="Coater 44",optionHeight=40,disabled=False,
                style={"fontSize":"14px", "cursor":"pointer"},
            ),
            html.Div("Product (Optional)", style={"marginTop":"2vh"}),
            dcc.Dropdown(id="select-product-dropdown",multi=False,clearable=True,placeholder="Select Product...",
                options=["RG300", "RG342"],value="RG300",optionHeight=40,disabled=False,
                style={"fontSize":"14px", "cursor":"pointer"},
            ),
            html.Div("Start Date", style={"marginTop":"2vh"}),
            dcc.DatePickerSingle(id='start-date',min_date_allowed=date(2024, 8, 1),max_date_allowed=date(2027, 12, 31),
                initial_visible_month=date(2026, 1, 1),date=date(2026, 1, 1),style={"fontSize":10}
            ),
            html.Div("End Date", style={"marginTop":"2vh"}),
            dcc.DatePickerSingle(id='end-date',min_date_allowed=date(2024, 8, 1),max_date_allowed=date(2027, 12, 31),
                initial_visible_month=date.today(),date=date.today(),style={"fontSize":10}
            ),
            html.Div(dbc.Button("Prepare Data", id="prepare-UPS-data-btn", class_name="me-1", style={"height":"auto","marginTop":"2vh","width":"100%"})),
            dcc.Download(id="download-UPS-data"),

        ]),
        # dbc.ModalFooter(dbc.Button("Done", id="UPS-data-query-done-btn", class_name="me-1"))
    ], id="UPS-data-query-modal", is_open=False, size="sm", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),



    ######################## Good vs Bad UI ####################
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Good vs Bad")),
        dbc.ModalBody([
            html.Div("Good Data"),
            dcc.Upload(id="upload-good-data",
                       children=html.Div([
                           "Drag and Drop or ",
                           html.A("Select File", style={"cursor":"pointer","color":"#800080","fontWeight":"600"})
                        ]),
                        multiple=False,
                        style={'height':'60px','lineHeight':'60px','borderWidth':'1px','borderStyle':'dashed',
                               'borderRadius':'5px','textAlign':'center','margin':'0px 10px 10px 10px'},
            ),
            html.Div("Bad Data", style={"marginTop":"4vh"}),
            dcc.Upload(id="upload-bad-data",
                       children=html.Div([
                           "Drag and Drop or ",
                           html.A("Select File", style={"cursor":"pointer","color":"#800080","fontWeight":"600"})
                        ]),
                        multiple=False,
                        style={'height':'60px','lineHeight':'60px','borderWidth':'1px','borderStyle':'dashed',
                               'borderRadius':'5px','textAlign':'center','margin':'0px 10px 10px 10px'},
            ),

            html.Div(dbc.Button("Prepare Data", id="prepare-good-vs-bad-data-btn", class_name="me-1", style={"height":"auto","marginTop":"2vh","width":"100%"})),
            dcc.Download(id="download-good-vs-bad-data"),
        ]),
        # dbc.ModalFooter(dbc.Button("Done", id="UPS-data-query-done-btn", class_name="me-1"))
    ], id="good-vs-bad-data-query-modal", is_open=False, zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),




    ###################### Settings ##############################
    html.Button("Settings", id="settings-btn",
                style={"position":"absolute","border":"none","right":"0vw","top":"0vh","color":"transparent"}),
    
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("1. Import Data")),
        dbc.ModalBody([
            
            dcc.Upload(id="upload-data",
                       children=html.Div([
                           "Drag and Drop or ",
                           html.A("Select File", style={"cursor":"pointer","color":"#800080","fontWeight":"600"})
                        ]),
                        multiple=False,
                        style={'height':'60px','lineHeight':'60px','borderWidth':'1px','borderStyle':'dashed',
                               'borderRadius':'5px','textAlign':'center','margin':'10px'},

            ),

            html.Div(id="output-data-upload"),

        ]),
        dbc.ModalFooter(dbc.Button("Next", id="get-data-next-btn", class_name="me-1"))
    ], id="get-data-modal", is_open=False, size="xl", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),


    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("2. Select Output")),
        dbc.ModalBody([
            
            dcc.Dropdown(
                id="select-output-dropdown",
                multi=False,
                clearable=False,
                placeholder="Select Output...",
                optionHeight=40,
                disabled=False,
                style={"fontSize":"14px", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                # className="regular-dropdown"
            ),

        ]),
        dbc.ModalFooter(dbc.Button("Next", id="save-output-next-btn", class_name="me-1"))
    ], id="select-output-modal", is_open=False, zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),#size="lg",


    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("3. Select Inputs")),
        dbc.ModalBody([
            tags_selection_UI
        ]),
        dbc.ModalFooter(dbc.Button("Next", id="save-input-next-btn", class_name="me-1"))
    ], id="select-input-modal", is_open=False, size="lg", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),



    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("4. Does The Order Of Samples Matter?")),
        dbc.ModalBody([
            
            dcc.Dropdown(
                id="sample-order-dropdown",
                multi=False,
                clearable=False,
                options=["No","Yes"],
                value="No",
                # placeholder="Select Output...",
                optionHeight=40,
                disabled=False,
                style={"fontSize":"14px", "cursor":"pointer"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                # className="regular-dropdown"
            ),

        ]),
        dbc.ModalFooter(dbc.Button("Next", id="sample-order-next-btn", class_name="me-1"))
    ], id="select-sample-order-modal", is_open=False, zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),#size="lg"


    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("5. Train Model")),
        dbc.ModalBody([
            
            html.Div([
                html.Div("PARAMETERS", style={"fontWeight":"700"}),
                html.Div(style={"height":"10px"}),
                html.Div("Number of Trees"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=1, max=500, step=1, marks=None,
                    value=200,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="n-estimators",
                    className="slider-class",#
                ),
                html.Div("Learning Rate"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=0.01, max=1.0, step=0.01, marks=None,
                    # step=None, marks={val: "" for val in learning_rate_slider_values},
                    value=0.1,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="learning-rate",
                    className="slider-class",#
                ),
                html.Div("Tree Height"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=1, max=20, step=1, marks=None,
                    value=2,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="max-depth",
                    className="slider-class",#
                ),
                html.Div("Fraction of Columns"),
                html.Div(style={"height":"5px"}),
                dcc.Slider(
                    min=0.1, max=1.0, step=0.1, marks=None,
                    value=0.1,
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"14px"}},
                    id="colsample-bynode",
                    className="slider-class",#
                ),
                html.Div(style={"height":"50px"}),
                dbc.Button("Train", id="train-model", class_name="me-1", style={"width":"100%","height":"80px"}),

            ], style={"border":"1px solid","display":"inline-block","width":"200px","padding":"5px 10px 10px 10px","borderRadius":"10px"}),

            html.Div([
                html.Div("RESULTS", style={"fontWeight":"700"}),
                html.Div(style={"height":"10px"}),
                html.Div([
                    dcc.Graph(
                        id="model-accuracy-gauge",
                        figure=get_indicator_KPI(0, 0, "Accuracy %"),
                        # style={"height":"100%","width":"100%","padding":"0px", "border":"1px solid",
                        #     "boxSizing":"border-box"},
                        style={"border":"0px solid","boxSizing":"border-box","display":"inline-block"},
                        config={'displayModeBar': False}
                    ),
                    html.Div(style={"display":"inline-block","border":"0px solid","height":"180px","width":"200px","boxSizing":"border-box"}),
                ]),
                
                dcc.Graph(
                    id="cv-graph",
                    style={"height":"150px","width":"520px","padding":"5px", "border":"0px solid",
                        "boxSizing":"border-box"},
                    config={'displayModeBar': False}
                ),

            ], id="something", style={"display":"inline-block","width":"550px","border":"1px solid","verticalAlign":"top",
                                      "marginLeft":"10px","height":"390px","padding":"5px 10px 10px 10px","borderRadius":"10px"}),
        
        ]),
        dbc.ModalFooter(dbc.Button("Done", id="train-model-close-btn", class_name="me-1"))
    ], id="train-model-modal", is_open=False, size="lg", zIndex=99999, autoFocus=False, centered=True, backdrop="static", keyboard=False),

    dcc.Store(id="previous-model-score", data=None),

], id="main-window2", style={"position":"relative","width":"100%","height":"100vh","backgroundColor":"#f1f1f1","border":"0px solid"})



def get_var_explained_chart(explained_variations, n_components=3):
    explained_variations = explained_variations[0: n_components]
    x_names = []
    formatted_text = []
    for i in range(0, n_components):
        x_names.append(f"{i+1}")
        # formatted_text.append(f"{(explained_variations[i] * 100)}%")
        formatted_text.append(f'{explained_variations[i] * 100:.0f}%')
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_names,
        y=explained_variations,
        name="Source Variation",
    ))
    fig.add_trace(go.Scatter(
        x=x_names,
        y=explained_variations.cumsum(),
        mode='lines+markers',
        name="Cummulative Var Sum",
        # line=dict(color='lightgray'),## lightgreen
        # marker=dict(color=colors, size=sizes),
        #  marker=dict(color=colors),
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=15, b=10),#t=35
        title=dict(text="Variance Explained by Source", font=dict(size=16, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=14, color="black"),
        showlegend=False,
        xaxis = dict(tickangle=0),
        yaxis=dict(tickformat=".0%"),
        plot_bgcolor="white",
        xaxis_title = "Source"
    )
    return fig


def get_PCA_chart(features, Z_contributions, x_max=None, n_iterations=1, fontSize=12, markerSize=15):
    # features = ["feature1", "feature2",...]
    # features = X.columns
    # max_Xs = [6, -1, 1.5, 1.8]
    # max_Xs = contributions[1,:]
    # initial_Xs = np.zeros(len(features))
    initial_Xs = Z_contributions

    initial_datas = []
    for i in range(0, len(features)):
        initial_datas.append(go.Scatter(x=[initial_Xs[i]], y=[features[i]], mode='markers', name=features[i], marker=dict(color="rgb(0, 136, 255)", size=markerSize)))

    for i in range(0, len(features)):
        initial_datas.append(go.Scatter(x=[-Z_contributions[i], Z_contributions[i]], y=[features[i], features[i]], mode='lines', name=features[i] + "line", line=dict(color="rgb(0, 136, 255)", width=1)))
        
    frames = []
    for _ in range(0, n_iterations):
        datas = []
        for i in range(0, len(features)):
            datas.append(go.Scatter(x=[-Z_contributions[i]], y=[features[i]], mode='markers', name=features[i]))
        frames.append(go.Frame(data=datas))

        datas = []
        for i in range(0, len(features)):
            datas.append(go.Scatter(x=[Z_contributions[i]], y=[features[i]], mode='markers', name=features[i]))
        frames.append(go.Frame(data=datas))

    if x_max is None:
        x_max = abs(Z_contributions).max()*1.2
        # max_Z_contribution = abs(Z_contributions).max()*1.2
    fig = go.Figure(
        data=initial_datas,
        layout=go.Layout(
            margin=dict(l=0, r=10, t=0, b=10),#t=35
            xaxis=dict(range=[-x_max, x_max], autorange=False),
            yaxis=dict(range=[-1, len(features)], autorange=False),
            # title=dict(text="Start Title"),
            showlegend=False,
            updatemenus=[dict(
                type="buttons",
                # x=0,y=1.01,
                # x=-0.1,y=1,
                x=1,y=1,
                xanchor="right",
                yanchor="bottom",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 2000, "redraw": False}, "fromcurrent": True, "transition": {"duration": 5000, "easing":"linear"}}],
                            # args=[None, {"frame": {"duration": 1000, "redraw": False}, "fromcurrent": True, "transition": {"duration": 5000, "easing":"linear"}}],
                        )
                ]
            )],
            autosize=True,
            plot_bgcolor="white",
            font=dict(size=fontSize)
        ),
        frames=frames
    )
    return fig



def get_multi_trace_graph(xs, ys, names, x_title, y_title, title, show_legend=True):
    fig = go.Figure()
    for i in range(0, len(ys)):
        fig.add_trace(go.Scatter(
            x=xs[i],
            y=ys[i],
            name=names[i],
            mode='lines+markers',
        ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=0),#t=35
        title=dict(text=title, font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=14, color="black"),
        showlegend=show_legend,
        # xaxis=dict(title=dict(text=x_title)),
        yaxis=dict(title=dict(text=y_title, font=dict(size=14)), range=[0,1]),
        hovermode="x unified",
    )
    return fig


def get_both_graphs(x, y, feature, x_title, y_title, title, curr_x=None, curr_y=None):
    frac = 0.1
    min_samples = 0
    df_sorted = df[[feature, target]].dropna().sort_values(feature).reset_index(drop=True)
    if is_classification:
        df_sorted[target] = np.where(df_sorted[target] == target_map[0], 0, 1)
    exp_y = df_sorted[target].rolling(window=int(len(df_sorted)*frac), center=True, min_periods=min_samples).mean()
    df_sorted[target] = exp_y
    df_agg = df_sorted.groupby(feature)[target].mean().reset_index()
    if len(df_agg) > 5000:
        step_size = int(np.ceil(len(df_agg)/5000))
        df_agg = df_agg.iloc[::step_size, :]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_agg[feature],
        y=df_agg[target],
        name="Univariate",
        mode='markers+lines',
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+lines',
        hovertemplate=(
                f"{x_title}: %{{x}}<br>" +
                f"{y_title}: %{{y}}<br>" +
                "<extra></extra>"
        ),
        name="Multivariate",
        # line=dict(shape="spline", smoothing=1.3)
    ))
    if curr_x != None:
        fig.add_trace(go.Scatter(
            x=[curr_x],
            y=[curr_y],
            mode='markers',
            marker=dict(
                color='gray', # Set a single color
                size=10
            ),
            hovertemplate=(
                f"{x_title}: %{{x}}<br>" +
                f"{y_title}: %{{y}}<br>" +
                "<extra></extra>"
            ),
            name="Current",
        ))
    # if threshold != None:
    #     fig.add_hrect(y0=threshold, y1=1, line_width=0, fillcolor="#599ad3", opacity=0.2, showlegend=showlegend, name=above_threshold_label)
    #     fig.add_hrect(y0=0, y1=threshold, line_width=0, fillcolor="#f9a75a", opacity=0.2, showlegend=showlegend, name=below_threshold_label)
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        title=dict(text=title, font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=16, color="black"),
        showlegend=True, plot_bgcolor="white",
        xaxis = dict(title=dict(text=x_title), gridcolor="#f0f0f0"),
        yaxis = dict(title=dict(text=y_title), gridcolor="#f0f0f0"),
    )
    return fig



def get_prediction_graph_of_numerical_feature(df, feature, contrib_df, max_points=5000, frac=0.1, ytitle="Prediction"):
    predictions = contrib_df.sum(axis=1)
    if is_classification:
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions) * 100
    combined_df = pd.DataFrame({"feature":df[feature], "predictions":predictions}).dropna().sort_values("feature").reset_index(drop=True)

    # combined_df["predictions"] = combined_df["predictions"].rolling(window=int(len(combined_df)*frac), center=True, min_periods=0).mean()
    # combined_df = combined_df.groupby("feature")["predictions"].mean().reset_index()

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    combined_df["predictions"] = combined_df["predictions"].rolling(window=int(len(combined_df)*frac), center=True, min_periods=0).mean()
    combined_df = combined_df.groupby("feature")["predictions"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_df["feature"],
        y=combined_df["predictions"],
        mode='markers+lines',
        hovertemplate=(
                f"{feature}: %{{x}}<br>" +
                f"{ytitle}: %{{y}}<br>" +
                "<extra></extra>"
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=False,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=ytitle)),#, range=[0,1]
    )
    return fig


def get_prediction_graph_of_categorical_feature(df, feature, contrib_df, max_points=5000, max_categories=10, ytitle="Prediction"):
    predictions = contrib_df.sum(axis=1)
    if is_classification:
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions) * 100
    combined_df = pd.DataFrame({"feature":df[feature], "predictions":predictions}).dropna().reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    cat_counts = combined_df["feature"].value_counts()
    if len(cat_counts) > max_categories:
        categories = list(cat_counts.index[0:max_categories-1])
        combined_df["feature"] = np.where(combined_df["feature"].isin(categories), combined_df["feature"], "Other..")
        categories.append("Other..")
    else:
        categories = list(cat_counts.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_df["feature"],
        y=combined_df["predictions"],
        mode='markers',
        hovertemplate=(
                f"{feature}: %{{x}}<br>" +
                f"{ytitle}: %{{y}}<br>" +
                "<extra></extra>"
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=False,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=ytitle)),#, range=[0,1]
    )
    return fig


def get_prediction_graph_of_numerical_feature_by_category(df, feature, categorical_feature, contrib_df, max_points=5000, max_categories=5, frac=0.1, ytitle=""):
    predictions = contrib_df.sum(axis=1)
    if is_classification:
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions) * 100
    combined_df = pd.DataFrame({"feature":df[feature], "categorical_feature":df[categorical_feature], "contrib":predictions}).dropna().sort_values("feature").reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    cat_counts = combined_df["categorical_feature"].value_counts()
    if len(cat_counts) > max_categories:
        categories = list(cat_counts.index[0:max_categories-1])
        combined_df["categorical_feature"] = np.where(combined_df["categorical_feature"].isin(categories), combined_df["categorical_feature"], "Other..")
        categories.append("Other..")
    else:
        categories = list(cat_counts.index)

    fig = go.Figure()
    for i in range(0, len(categories)):
        category_df = combined_df.loc[combined_df["categorical_feature"] == categories[i], ["feature", "contrib"]].reset_index(drop=True)
        
        category_df["contrib"] = category_df["contrib"].rolling(window=int(len(category_df)*frac), center=True, min_periods=0).mean()
        category_df = category_df.groupby("feature")["contrib"].mean().reset_index()

        fig.add_trace(go.Scatter(
            x=category_df["feature"],
            y=category_df["contrib"],
            mode='markers+lines',
            name=categories[i],
            hovertemplate=(
                f"{feature}: %{{x}}<br>" +
                f"{ytitle}: %{{y}}<br>" +
                "%{fullData.name}<br>"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=ytitle)),#, range=[0,1]
    )
    return fig



def get_prediction_graph_of_numerical_feature_by_number(df, feature, feature2, contrib_df, max_points=5000, color_title="Predictions"):
    predictions = contrib_df.sum(axis=1)
    if is_classification:
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions) * 100
    combined_df = pd.DataFrame({feature:df[feature], feature2:df[feature2], color_title:predictions}).dropna().sort_values(feature).reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    combined_df = combined_df.groupby([feature, feature2])[color_title].mean().reset_index()
    fig = px.scatter(combined_df,
        x=feature,
        y=feature2,
        color=color_title,
        #labels={color_title:color_title.replace(" ", "<br>")}
    )
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=feature2)),#, range=[0,1]
    )
    return fig


def get_prediction_graph_of_categorical_feature_by_category(df, feature, feature2, contrib_df, max_points=5000, max_categories=5, color_title="Predictions"):
    predictions = contrib_df.sum(axis=1)
    if is_classification:
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions) * 100
    combined_df = pd.DataFrame({feature:df[feature], feature2:df[feature2], color_title:predictions}).dropna().reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    cat_counts_feature = combined_df[feature].value_counts()
    if len(cat_counts_feature) > max_categories:
        categories_feature = list(cat_counts_feature.index[0:max_categories-1])
        combined_df[feature] = np.where(combined_df[feature].isin(categories_feature), combined_df[feature], "Other..")
        categories_feature.append("Other..")
    else:
        categories_feature = list(cat_counts_feature.index)
    
    cat_counts_feature2 = combined_df[feature2].value_counts()
    if len(cat_counts_feature2) > max_categories:
        categories_feature2 = list(cat_counts_feature2.index[0:max_categories-1])
        combined_df[feature2] = np.where(combined_df[feature2].isin(categories_feature2), combined_df[feature2], "Other..")
        categories_feature2.append("Other..")
    else:
        categories_feature2 = list(cat_counts_feature2.index)

    combined_df = combined_df.groupby([feature, feature2])[color_title].mean().reset_index()
    fig = px.scatter(combined_df,
        x=feature,
        y=feature2,
        color=color_title,
    )
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=feature2)),#, range=[0,1]
    )
    return fig



def get_contrib_graph_of_numerical_feature(df, feature, contrib_df, max_points=5000, ytitle="Contribution"):
    if is_classification:
        predictions = contrib_df[feature] + contrib_df["_bias_"].iat[0]
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)
        baseline_prediction = contrib_df["_bias_"].iat[0]
        baseline_prediction = (1 / (1 + np.exp(-baseline_prediction)))
        baseline_prediction = calibrate_probability(baseline_prediction)
        contributions = (predictions - baseline_prediction) * 100
    else:
        contributions = contrib_df[feature]
    combined_df = pd.DataFrame({"feature":df[feature], "contrib":contributions}).dropna().sort_values("feature").reset_index(drop=True)
    # combined_df = pd.DataFrame({"feature":df[feature], "contrib":contrib_df[feature]}).dropna().sort_values("feature").reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_df["feature"],
        y=combined_df["contrib"],
        mode='markers+lines',
        hovertemplate=(
                f"{feature}: %{{x}}<br>" +
                f"{ytitle}: %{{y}}<br>" +
                "<extra></extra>"
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=False,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=ytitle)),#, range=[0,1]
    )
    return fig


def get_contrib_graph_of_categorical_feature(df, feature, contrib_df, max_points=5000, max_categories=10, ytitle="Contribution"):
    if is_classification:
        predictions = contrib_df[feature] + contrib_df["_bias_"].iat[0]
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)
        baseline_prediction = contrib_df["_bias_"].iat[0]
        baseline_prediction = (1 / (1 + np.exp(-baseline_prediction)))
        baseline_prediction = calibrate_probability(baseline_prediction)
        contributions = (predictions - baseline_prediction) * 100
    else:
        contributions = contrib_df[feature]
    combined_df = pd.DataFrame({"feature":df[feature], "contrib":contributions}).dropna().reset_index(drop=True)
    # combined_df = pd.DataFrame({"feature":df[feature], "contrib":contrib_df[feature]}).dropna().reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    cat_counts = combined_df["feature"].value_counts()
    if len(cat_counts) > max_categories:
        categories = list(cat_counts.index[0:max_categories-1])
        combined_df["feature"] = np.where(combined_df["feature"].isin(categories), combined_df["feature"], "Other..")
        categories.append("Other..")
    else:
        categories = list(cat_counts.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_df["feature"],
        y=combined_df["contrib"],
        mode='markers',
        hovertemplate=(
                f"{feature}: %{{x}}<br>" +
                f"{ytitle}: %{{y}}<br>" +
                "<extra></extra>"
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=False,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=ytitle)),#, range=[0,1]
    )
    return fig


def get_contrib_graph_of_numerical_feature_by_category(df, feature, categorical_feature, contrib_df, max_points=5000, max_categories=5, ytitle=""):
    if is_classification:
        predictions = contrib_df[feature] + contrib_df["_bias_"].iat[0]# predictions = contrib_df[feature] + contrib_df[categorical_feature] + contrib_df["_bias_"].iat[0]
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)
        baseline_prediction = contrib_df["_bias_"].iat[0]
        baseline_prediction = (1 / (1 + np.exp(-baseline_prediction)))
        baseline_prediction = calibrate_probability(baseline_prediction)
        contributions = (predictions - baseline_prediction) * 100
    else:
        contributions = contrib_df[feature]# contributions = contrib_df[feature] + contrib_df[categorical_feature]
    combined_df = pd.DataFrame({"feature":df[feature], "categorical_feature":df[categorical_feature], "contrib":contributions}).dropna().sort_values("feature").reset_index(drop=True)
    # combined_df = pd.DataFrame({"feature":df[feature], "categorical_feature":df[categorical_feature], "contrib":contrib_df[feature]+contrib_df[categorical_feature]}).dropna().sort_values("feature").reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    cat_counts = combined_df["categorical_feature"].value_counts()
    if len(cat_counts) > max_categories:
        categories = list(cat_counts.index[0:max_categories-1])
        combined_df["categorical_feature"] = np.where(combined_df["categorical_feature"].isin(categories), combined_df["categorical_feature"], "Other..")
        categories.append("Other..")
    else:
        categories = list(cat_counts.index)

    fig = go.Figure()
    for i in range(0, len(categories)):
        category_df = combined_df.loc[combined_df["categorical_feature"] == categories[i], ["feature", "contrib"]].reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=category_df["feature"],
            y=category_df["contrib"],
            mode='markers+lines',
            name=categories[i],
            hovertemplate=(
                f"{feature}: %{{x}}<br>" +
                f"{ytitle}: %{{y}}<br>" +
                "%{fullData.name}<br>"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=ytitle)),#, range=[0,1]
    )
    return fig


def get_contrib_graph_of_numerical_feature_by_number(df, feature, feature2, contrib_df, max_points=5000, color_title="Contrib"):
    if is_classification:
        predictions = contrib_df[feature] + contrib_df[feature2] + contrib_df["_bias_"].iat[0]
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)
        baseline_prediction = contrib_df["_bias_"].iat[0]
        baseline_prediction = (1 / (1 + np.exp(-baseline_prediction)))
        baseline_prediction = calibrate_probability(baseline_prediction)
        contributions = (predictions - baseline_prediction) * 100
    else:
        contributions = contrib_df[feature] + contrib_df[feature2]
    combined_df = pd.DataFrame({feature:df[feature], feature2:df[feature2], color_title:contributions}).dropna().sort_values(feature).reset_index(drop=True)
    # combined_df = pd.DataFrame({feature:df[feature], feature2:df[feature2], color_title:contrib_df[feature]+contrib_df[feature2]}).dropna().sort_values(feature).reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]
    
    combined_df = combined_df.groupby([feature, feature2])[color_title].mean().reset_index()# This should technically be before the max_points check. For performance purposes, I placed it here
    fig = px.scatter(combined_df,
        x=feature,
        y=feature2,
        color=color_title,
        #labels={color_title:color_title.replace(" ", "<br>")}
    )
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=feature2)),#, range=[0,1]
    )
    return fig


def get_contrib_graph_of_categorical_feature_by_category(df, feature, feature2, contrib_df, max_points=5000, max_categories=5, color_title="Contrib"):
    if is_classification:
        predictions = contrib_df[feature] + contrib_df[feature2] + contrib_df["_bias_"].iat[0]
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)
        baseline_prediction = contrib_df["_bias_"].iat[0]
        baseline_prediction = (1 / (1 + np.exp(-baseline_prediction)))
        baseline_prediction = calibrate_probability(baseline_prediction)
        contributions = (predictions - baseline_prediction) * 100
    else:
        contributions = contrib_df[feature] + contrib_df[feature2]
    combined_df = pd.DataFrame({feature:df[feature], feature2:df[feature2], color_title:contributions}).dropna().reset_index(drop=True)
    # combined_df = pd.DataFrame({feature:df[feature], feature2:df[feature2], color_title:contrib_df[feature]+contrib_df[feature2]}).dropna().sort_values(feature).reset_index(drop=True)

    if len(combined_df) > max_points:
        step_size = int(np.ceil(len(combined_df)/max_points))
        combined_df = combined_df.iloc[::step_size, :]

    cat_counts_feature = combined_df[feature].value_counts()
    if len(cat_counts_feature) > max_categories:
        categories_feature = list(cat_counts_feature.index[0:max_categories-1])
        combined_df[feature] = np.where(combined_df[feature].isin(categories_feature), combined_df[feature], "Other..")
        categories_feature.append("Other..")
    else:
        categories_feature = list(cat_counts_feature.index)
    
    cat_counts_feature2 = combined_df[feature2].value_counts()
    if len(cat_counts_feature2) > max_categories:
        categories_feature2 = list(cat_counts_feature2.index[0:max_categories-1])
        combined_df[feature2] = np.where(combined_df[feature2].isin(categories_feature2), combined_df[feature2], "Other..")
        categories_feature2.append("Other..")
    else:
        categories_feature2 = list(cat_counts_feature2.index)

    combined_df = combined_df.groupby([feature, feature2])[color_title].mean().reset_index()# 
    fig = px.scatter(combined_df,
        x=feature,
        y=feature2,
        color=color_title,
    )
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text=feature)),
        yaxis=dict(title=dict(text=feature2)),#, range=[0,1]
    )
    return fig



def get_train_test_graph(train_scores, test_scores, x_title, y_title, title, show_legend=True):
    fig = go.Figure()
    tree_counts = list(range(1, len(test_scores)+1))
    fig.add_trace(go.Scatter(
        x=tree_counts,
        y=train_scores,
        name="Train",
        mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=tree_counts,
        y=test_scores,
        name="Test",
        mode='lines',
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=0),#t=35
        title=dict(text=title, font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=14, color="black"),
        showlegend=show_legend,
        xaxis=dict(title=dict(text=x_title)),
        yaxis=dict(title=dict(text=y_title, font=dict(size=14))),#, range=[0,1]
        hovermode="x unified",
    )
    return fig


def get_scatter_color_map(x, y, color, x_title, y_title, color_title, title):
    graph_df = pd.DataFrame({x_title:x, y_title:y, color_title:color})
    fig = px.scatter(graph_df,
                 x=x_title,
                 y=y_title,
                 color=color_title,
                 #labels={color_title:color_title.replace(" ", "<br>")}
                 )
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        title=dict(text=title, font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=16, color="black"),
        showlegend=True,
        xaxis = dict(title=dict(text=x_title)),
        yaxis = dict(title=dict(text=y_title)),
        legend=dict(
            title=dict(
                text=color_title.replace(" ", "<br>")
            )
        )
    )
    return fig


def get_univariate_graph_of_numerical_feature(df, feature, target, frac=0.1, min_samples=0, max_points=5000):
    df_sorted = df[[feature, target]].dropna().sort_values(feature).reset_index(drop=True)
    if is_classification:
        df_sorted[target] = np.where(df_sorted[target] == target_map[0], 0, 1)
        if target_map[1] == 1:
            y_display = f"{target} (prob)"
        else:
            y_display = f"{target_map[1]} (prob)"
    else:
        y_display = target
    exp_y = df_sorted[target].rolling(window=int(len(df_sorted)*frac), center=True, min_periods=min_samples).mean()
    df_sorted[target] = exp_y
    df_agg = df_sorted.groupby(feature)[target].mean().reset_index()
    if len(df_agg) > max_points:
        step_size = int(np.ceil(len(df_agg)/max_points))
        df_agg = df_agg.iloc[::step_size, :]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_agg[feature],
        y=df_agg[target],
        # name="Univariate",
        mode='markers+lines',
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=16, color="black"),
        showlegend=False, plot_bgcolor="white",
        xaxis = dict(title=dict(text=feature), gridcolor="#f0f0f0"),
        yaxis = dict(title=dict(text=y_display), gridcolor="#f0f0f0"),
    )
    return fig


def get_univariate_graph_of_categorical_feature(df, feature, target, min_samples=0):
    df_category = df[[feature, target]].dropna().reset_index(drop=True)
    cat_counts = df_category[feature].value_counts()
    categories = cat_counts[cat_counts >= min_samples].index
    df_category[feature] = np.where(df_category[feature].isin(categories), df_category[feature], "OTHER")
    if is_classification:
        df_category[target] = np.where(df_category[target] == target_map[0], 0, 1)
        if target_map[1] == 1:
            y_display = f"{target} (prob)"
        else:
            y_display = f"{target_map[1]} (prob)"
    else:
        y_display = target

    df_agg = df_category.groupby(feature)[target].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_agg[feature],
        y=df_agg[target],
        # name="Univariate",
        mode='markers+lines',
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=16, color="black"),
        showlegend=False, plot_bgcolor="white",
        xaxis = dict(title=dict(text=feature), gridcolor="#f0f0f0"),
        yaxis = dict(title=dict(text=y_display), gridcolor="#f0f0f0"),
    )
    return fig


def get_univariate_graph_of_numerical_feature_by_category(df, feature, categorical_feature, target, frac=0.1, min_samples=0, max_points=5000, max_categories=5):
    df_sorted = df[[feature, categorical_feature, target]].dropna().sort_values(feature).reset_index(drop=True)
    cat_counts = df_sorted[categorical_feature].value_counts()
    if len(cat_counts) > max_categories:
        categories = list(cat_counts.index[0:max_categories-1])
        df_sorted[categorical_feature] = np.where(df_sorted[categorical_feature].isin(categories), df_sorted[categorical_feature], "OTHER")
        categories.append("OTHER")
    else:
        categories = list(cat_counts.index)
    if is_classification:
        df_sorted[target] = np.where(df_sorted[target] == target_map[0], 0, 1)
        if target_map[1] == 1:
            y_display = f"{target} (prob)"
        else:
            y_display = f"{target_map[1]} (prob)"
    else:
        y_display = target

    fig = go.Figure()
    for i in range(0, len(categories)):
        df_category = df_sorted.loc[df_sorted[categorical_feature] == categories[i], [feature, target]].reset_index(drop=True)
        exp_y = df_category[target].rolling(window=int(len(df_category)*frac), center=True, min_periods=min_samples).mean()
        df_category[target] = exp_y
        df_agg = df_category.groupby(feature)[target].mean().reset_index()
        if len(df_agg) > max_points:
            step_size = int(np.ceil(len(df_agg)/max_points))
            df_agg = df_agg.iloc[::step_size, :]
        fig.add_trace(go.Scatter(
            x=df_agg[feature],
            y=df_agg[target],
            mode='markers+lines',
            name=categories[i],
        ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=16, color="black"),
        showlegend=True, plot_bgcolor="white",
        xaxis = dict(title=dict(text=feature), gridcolor="#f0f0f0"),
        yaxis = dict(title=dict(text=y_display), gridcolor="#f0f0f0"),
    )
    return fig



def get_univariate_graph_of_categorical_feature_by_category(df, feature, categorical_feature, target, min_samples=0, max_categories=5):
    df_category = df[[feature, categorical_feature, target]].dropna().reset_index(drop=True)
    cat_counts = df_category[feature].value_counts()
    categories = cat_counts[cat_counts >= min_samples].index
    df_category[feature] = np.where(df_category[feature].isin(categories), df_category[feature], "OTHER")

    cat_counts = df_category[categorical_feature].value_counts()
    if len(cat_counts) > max_categories:
        categories = list(cat_counts.index[0:max_categories-1])
        df_category[categorical_feature] = np.where(df_category[categorical_feature].isin(categories), df_category[categorical_feature], "OTHER")
        categories.append("OTHER")
    else:
        categories = list(cat_counts.index)

    if is_classification:
        df_category[target] = np.where(df_category[target] == target_map[0], 0, 1)
        if target_map[1] == 1:
            y_display = f"{target} (prob)"
        else:
            y_display = f"{target_map[1]} (prob)"
    else:
        y_display = target

    fig = go.Figure()
    for i in range(0, len(categories)):
        df_filtered = df_category.loc[df_category[categorical_feature] == categories[i], [feature, target]].reset_index(drop=True)
        df_agg = df_filtered.groupby(feature)[target].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=df_agg[feature],
            y=df_agg[target],
            mode='markers+lines',
            name=categories[i],
        ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=0, b=10),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        font=dict(family="arial", size=16, color="black"),
        showlegend=True, plot_bgcolor="white",
        xaxis = dict(title=dict(text=feature), gridcolor="#f0f0f0"),
        yaxis = dict(title=dict(text=y_display), gridcolor="#f0f0f0"),
    )
    return fig


def get_BOB_WOW_line_chart(BOB_df, WOW_df, feature, y_display, max_points=5000):
    if len(BOB_df) > max_points/2:
        BOB_df = BOB_df.head(max_points)
        WOW_df = WOW_df.tail(max_points)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(BOB_df)+1),
        y=BOB_df[feature],
        mode='markers+lines',
        name="Best",
        customdata=BOB_df["Pred =>"],
        hovertemplate=(
                f"{feature}: %{{y}}<br>" +
                f"{y_display}: %{{customdata}}<br>" +
                f"{'Sample'}: %{{x}}<br>" +
                "<extra></extra>"
        ),
        marker=dict(color="green"),
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(BOB_df)+1, len(BOB_df)+len(WOW_df)+1),#np.arange(101, 201),
        y=WOW_df[feature],
        mode='markers+lines',
        name="Worst",
        customdata=WOW_df["Pred =>"],
        hovertemplate=(
                f"{feature}: %{{y}}<br>" +
                f"{y_display}: %{{customdata}}<br>" +
                f"{'Sample'}: %{{x}}<br>" +
                "<extra></extra>"
        ),
        marker=dict(color="red"),
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text="Ranked Samples")),
        yaxis=dict(title=dict(text=feature)),#, range=[0,1]
    )
    return fig


def get_BOB_WOW_scatter_chart(BOB_df, WOW_df, feature, y_display, max_points=5000):
    if len(BOB_df) > max_points/2:
        BOB_df = BOB_df.head(max_points)
        WOW_df = WOW_df.tail(max_points)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=BOB_df["Pred =>"],
        y=BOB_df[feature],
        mode='markers',
        name="Best",
        customdata=np.arange(1, BOB_df.shape[0]+1),
        hovertemplate=(
                f"{feature}: %{{y}}<br>" +
                f"{y_display}: %{{x}}<br>" +
                f"{'Sample'}: %{{customdata}}<br>" +
                "<extra></extra>"
        ),
        marker=dict(color="green"),
    ))
    fig.add_trace(go.Scatter(
        x=WOW_df["Pred =>"],
        y=WOW_df[feature],
        mode='markers',
        name="Worst",
        customdata=np.arange(len(BOB_df)+1, len(BOB_df)+len(WOW_df)+1),#np.arange(101, 201),
        hovertemplate=(
                f"{feature}: %{{y}}<br>" +
                f"{y_display}: %{{x}}<br>" +
                f"{'Sample'}: %{{customdata}}<br>" +
                "<extra></extra>"
        ),
        marker=dict(color="red"),
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=35, b=0),#t=35
        title=dict(text="", font=dict(size=25, color="black"), automargin=True, yref='paper'),
        # font=dict(family="arial", size=14, color="black"),
        showlegend=True,
        xaxis=dict(title=dict(text=y_display)),
        yaxis=dict(title=dict(text=feature)),
    )
    return fig



def calibrate_probability(prob):
    # adjusted_odds = (prob / (1 - prob)) / balancing_ratio
    # prob_calibrated = adjusted_odds / (1 + adjusted_odds)
    prob_calibrated = prob / (prob + balancing_ratio * (1-prob))
    return prob_calibrated




def format_SM_data_to_N_minutes_before_UPS(df, date_feature, running_good_feature, event_flag, MINUTES_BEFORE_EVENT, BUFFER_MINUTES_FOR_STARTUP, BUFFER_MINUTES_FOR_STOP):
    ###### Format datetime to datetime and sort
    df[date_feature] = pd.to_datetime(df[date_feature])
    df = df.sort_values(date_feature, ascending=True).reset_index(drop=True)
    print(df[date_feature].min())
    print(df[date_feature].max())

    ####### Filter out based on production status #############
    filtered_df = df.loc[(df[running_good_feature] == 1) | (df[event_flag] == 1), :].reset_index(drop=True)
    print(filtered_df.shape)

    ####### Main Logic ########################################
    status = [np.nan]*len(filtered_df)
    event_indices = list(np.where(filtered_df[event_flag] == 1)[0])
    if event_indices[0] < 2:
        status[0] = "Buffer"
        status[1] = "Buffer"
    else:
        event_indices.insert(0, 0)

    dates = list(filtered_df[date_feature])
    for i in range(0, len(event_indices) - 1):
        status[event_indices[i]] = "Buffer"
        status[event_indices[i + 1]] = "Buffer"
        startup_index = event_indices[i] + 1
        event_index = event_indices[i + 1] - 1
        startup_time = dates[startup_index]
        event_time = dates[event_index]
        for j in range(startup_index, event_index + 1):
            time_diff_startup = dates[j] - startup_time
            minutes_since_startup = time_diff_startup.days * 1440 + time_diff_startup.seconds / 60
            time_diff_event = event_time - dates[j]
            minutes_until_event = time_diff_event.days * 1440 + time_diff_event.seconds / 60
            if (minutes_since_startup < BUFFER_MINUTES_FOR_STARTUP) or (minutes_until_event < BUFFER_MINUTES_FOR_STOP):
                status[j] = "Buffer"
            elif minutes_until_event < MINUTES_BEFORE_EVENT + BUFFER_MINUTES_FOR_STOP:
                status[j] = f"UPS in {MINUTES_BEFORE_EVENT} min"

    # Handle the values after the last event occurrence
    if event_indices[-1] > len(filtered_df) - 3:
        status[-2] = "Buffer"
        status[-1] = "Buffer"
    else:
        startup_index = event_indices[-1] + 1
        last_sample_index = len(filtered_df) - 1
        startup_time = dates[startup_index]
        last_sample_time = dates[last_sample_index]
        for j in range(startup_index, last_sample_index + 1):
            time_diff_startup = dates[j] - startup_time
            minutes_since_startup = time_diff_startup.days * 1440 + time_diff_startup.seconds / 60
            time_diff_last_sample = last_sample_time - dates[j]
            minutes_until_last_sample = time_diff_last_sample.days * 1440 + time_diff_last_sample.seconds / 60
            if (minutes_since_startup < BUFFER_MINUTES_FOR_STARTUP) or (minutes_until_last_sample < MINUTES_BEFORE_EVENT + BUFFER_MINUTES_FOR_STOP):
                status[j] = "Buffer"
    
    ######## Insert Status values into new Column ##################
    filtered_df.insert(loc=0, column="Unplanned Stop Status", value=status)
    filtered_df["Unplanned Stop Status"] = filtered_df["Unplanned Stop Status"].fillna("Steady State")

    ######## Filter out Buffer time and remove useless columns ########
    final_df = filtered_df.loc[filtered_df["Unplanned Stop Status"] != "Buffer", :].drop(columns=[running_good_feature, event_flag])
    print(final_df.shape)
    return final_df



# source-pca-animation-dropdown
@app.callback(
    dash.dependencies.Output(component_id="pca-modal", component_property='is_open'),
    dash.dependencies.Output(component_id="pca-animation", component_property='figure'),
    dash.dependencies.Output(component_id="explained-var-chart", component_property='figure'),
    dash.dependencies.Input(component_id="pca-data", component_property='data'),
    dash.dependencies.Input(component_id="source-pca-animation-dropdown", component_property='value'),
    prevent_initial_call=True,
)
def update_pca_animation(pca_data, source):
    print("in update_pca_animation")
    # numerical_columns =  df.dtypes[df.dtypes != "object"].index.sort_values(ascending=False)
    # model_df_nona = df.dropna()
    # X_ranks = model_df_nona.rank(axis=0, method="average", numeric_only=True) / model_df_nona.shape[0]
    # X_ranks = X_ranks.sort_index(axis=1, ascending=False)

    # n_components = 10
    # pca = PCA(n_components=n_components)#n_components=10)
    # pca.fit(X_ranks)
    
    # Z = pca.transform(X_ranks)
    # maxZ = np.abs(Z).max(axis=0)
    # diagMaxZ = np.diag(maxZ)
    # contributions = np.dot(diagMaxZ, pca.components_)

    contributions = np.array(pca_data["contributions"])
    var_explained = np.array(pca_data["var_explained"])
    numerical_columns = pca_data["features"]

    pca_chart = get_PCA_chart(features=numerical_columns, Z_contributions=contributions[int(source[7:])-1,:], x_max=abs(contributions).max()*1.2, n_iterations=2, fontSize=12, markerSize=15)
    explained_var_chart = get_var_explained_chart(explained_variations=var_explained, n_components=10)

    return True, pca_chart, explained_var_chart



@app.callback(
    dash.dependencies.Output(component_id="pca-data", component_property='data'),
    dash.dependencies.Input(component_id="pca-button", component_property='n_clicks'),
    prevent_initial_call=True,
)
def update_pca_data(n_clicks):
    print("in update_pca_data")

    processed_df = df[selected_features]
    numerical_columns =  processed_df.dtypes[processed_df.dtypes != "object"].index.sort_values(ascending=False)
    model_df_nona = processed_df[numerical_columns].dropna()
    # numerical_columns =  df.dtypes[df.dtypes != "object"].index.sort_values(ascending=False)
    # model_df_nona = df[numerical_columns].dropna()

    X_ranks = model_df_nona.rank(axis=0, method="average") / model_df_nona.shape[0]
    # X_ranks = X_ranks.sort_index(axis=1, ascending=False)

    n_components = 10
    pca = PCA(n_components=n_components)#n_components=10)
    pca.fit(X_ranks)

    Z = pca.transform(X_ranks)
    maxZ = np.abs(Z).max(axis=0)
    diagMaxZ = np.diag(maxZ)
    contributions = np.dot(diagMaxZ, pca.components_)
    return {"contributions":contributions.tolist(), "var_explained":pca.explained_variance_ratio_.tolist(), "features":numerical_columns.tolist()}#contributions.tolist()




@app.callback(
    dash.dependencies.Output(component_id="contributions-table", component_property='rowData'),
    dash.dependencies.Output(component_id="contributions-table", component_property='columnDefs'),
    # dash.dependencies.Output(component_id="optimization-comparison-table", component_property='columnDefs'),
    # dash.dependencies.Output(component_id="comparison-mode", component_property='data', allow_duplicate=True),
    dash.dependencies.Input(component_id="left-arrow", component_property='n_clicks'),
    dash.dependencies.Input(component_id="right-arrow", component_property='n_clicks'),
    dash.dependencies.State(component_id="contributions-table", component_property='columnDefs'),
    # dash.dependencies.State(component_id="optimization-comparison-modal", component_property='is_open'),
    prevent_initial_call=True,
)
def click_next_sample(left_arrow_clicks, right_arrow_clicks, columnDefs):
    print("in click_next_sample")
    if left_arrow_clicks == None:
        left_arrow_clicks = 0
    if right_arrow_clicks == None:
        right_arrow_clicks = 0

    sample_df = df[selected_features].dropna(subset=target)
    curr_index = (right_arrow_clicks - left_arrow_clicks) % sample_df.shape[0]
    sample_df = sample_df.iloc[[curr_index], :].drop(columns=target)
    numeric_cols =  sample_df.dtypes[sample_df.dtypes != "object"].index

    # sample_df[numeric_cols] = sample_df[numeric_cols].applymap(lambda x: f"{x:.2f}")
    sample_df[numeric_cols] = sample_df[numeric_cols].astype(str)
    curr_X = sample_df.iloc[0, :]#.to_numpy(dtype=object)

    curr_contribs = contributions_df.iloc[curr_index, 0:-1]


    # Get Feature Contributions back to probability scale if model is a classifier
    if is_classification:
        bias = contributions_df["_bias_"].iat[0]
        predictions = curr_contribs + bias
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)
        baseline_prediction = (1 / (1 + np.exp(-bias)))
        baseline_prediction = calibrate_probability(baseline_prediction)
        curr_contribs = (predictions - baseline_prediction) * 100


    # result_df = pd.DataFrame({"values":curr_X, "contrib":curr_contribs}).reset_index(names="feature")
    result_df = pd.DataFrame({"values":curr_X, "contrib":curr_contribs}).reindex(index=ordered_features).reset_index(names="feature")
    # result_df = result_df.reindex(index=ordered_features).reset_index(names="feature")

    # result_df["feature"] = "Review files \n<span style='font-weight:normal; color:white; backgroundColor:black; padding:4px 5px 5px 5px; border-radius:5px; display:inline-block; margin-top:5px;'>25.67</span>"
    result_df["feature"] = result_df["feature"] + "\n<span style='font-weight:normal; color:white; backgroundColor:#616161; padding:4px 5px 5px 5px; border-radius:5px; display:inline-block; margin-top:5px;'>" + result_df["values"].map(str) + "</span>"

    result_df["pos"] = np.where(result_df["contrib"] > 0, result_df["contrib"], 0)
    result_df["neg"] = np.where(result_df["contrib"] < 0, result_df["contrib"], 0)

    # result_df["pos"] = result_df["pos"].map("{:.2f}".format)
    result_df["pos"] = result_df["pos"].map(f"{{:.{contrib_decimals}f}}".format)
    # result_df["neg"] = result_df["neg"].map("{:.2f}".format)
    result_df["neg"] = result_df["neg"].map(f"{{:.{contrib_decimals}f}}".format)


    if is_classification:
        prediction = contributions_df.iloc[curr_index, :].sum()
        prediction = (1 / (1 + np.exp(-prediction)))
        prediction = calibrate_probability(prediction)*100
        if target_map[1] == 1:
            y_display = f"{target} (prob)\n{prediction:.{predict_decimals}f}%"
        else:
            y_display = f"{target_map[1]} (prob)\n{prediction:.{predict_decimals}f}%"
    else:
        prediction = contributions_df.iloc[curr_index, :].sum()
        y_display = f"{target}\n{prediction:.{predict_decimals}f}"
    columnDefs[1]["headerName"] = y_display

    return result_df.to_dict("records"), columnDefs



@app.callback(
    # dash.dependencies.Output(component_id="edit-numerical-value-modal", component_property='is_open'),
    dash.dependencies.Output(component_id="feature-contrib-graph", component_property='figure'),
    dash.dependencies.Output(component_id="feature-prediction-graph", component_property='figure'),

    dash.dependencies.Output(component_id="selected-feature-data", component_property='data'),


    dash.dependencies.Input(component_id="contributions-table", component_property='cellClicked'),
    dash.dependencies.State(component_id="mode-selection-radio", component_property='value'),
    dash.dependencies.State(component_id="additional-feature-dropdown", component_property='value'),
    prevent_initial_call=True,
)
def click_contributions_table(cell_data, mode_selection, additional_feature):
    # ALL values are string now so I have to update code
    print("in click_contributions_table")
    cell_value = cell_data.get("value")
    if "\n" not in cell_value:
        return dash.no_update, dash.no_update, dash.no_update#, dash.no_update, dash.no_update
    
    selected_feature, css = cell_value.split("\n")
    print(selected_feature)

    if mode_selection == "Edit":
        value = css.split(">")[1].split("<")[0]
        print(value)
        return dash.no_update, dash.no_update, {"feature":selected_feature, "value":value}


    if is_classification:
        if target_map[1] == 1:
            ytitle = f"{target} (prob)"
        else:
            ytitle = f"{target_map[1]} (prob)"
    else:
        ytitle = target

    if additional_feature:
        if df[selected_feature].dtype == object and df[additional_feature].dtype == object:
            prediction_fig = get_prediction_graph_of_categorical_feature_by_category(df, selected_feature, additional_feature, contributions_df, max_points=5000, max_categories=5, color_title=ytitle)
            contrib_fig = get_contrib_graph_of_categorical_feature_by_category(df, selected_feature, additional_feature, contributions_df, max_points=5000, max_categories=5, color_title="Contrib")
        elif df[selected_feature].dtype == object and df[additional_feature].dtype != object:
            prediction_fig = get_prediction_graph_of_numerical_feature_by_category(df, additional_feature, selected_feature, contributions_df, max_points=5000, max_categories=5, frac=0.1, ytitle=ytitle)
            contrib_fig = get_contrib_graph_of_numerical_feature_by_category(df, additional_feature, selected_feature, contributions_df, max_points=5000, max_categories=5, ytitle="Contributions")
        elif df[selected_feature].dtype != object and df[additional_feature].dtype == object:
            prediction_fig = get_prediction_graph_of_numerical_feature_by_category(df, selected_feature, additional_feature, contributions_df, max_points=5000, max_categories=5, frac=0.1, ytitle=ytitle)
            contrib_fig = get_contrib_graph_of_numerical_feature_by_category(df, selected_feature, additional_feature, contributions_df, max_points=5000, max_categories=5, ytitle="Contributions")
        else:
            prediction_fig = get_prediction_graph_of_numerical_feature_by_number(df, selected_feature, additional_feature, contributions_df, max_points=5000, color_title=ytitle)
            contrib_fig = get_contrib_graph_of_numerical_feature_by_number(df, selected_feature, additional_feature, contributions_df, max_points=5000, color_title="Contrib")
    else:
        if df[selected_feature].dtype == object:
            prediction_fig = get_prediction_graph_of_categorical_feature(df, selected_feature, contributions_df, max_points=5000, max_categories=10, ytitle=ytitle)
            contrib_fig = get_contrib_graph_of_categorical_feature(df, selected_feature, contributions_df, max_points=5000, max_categories=10, ytitle="Contributions")
        else:
            prediction_fig = get_prediction_graph_of_numerical_feature(df, selected_feature, contributions_df, max_points=5000, frac=0.1, ytitle=ytitle)
            contrib_fig = get_contrib_graph_of_numerical_feature(df, selected_feature, contributions_df, max_points=5000, ytitle="Contributions")

    return contrib_fig, prediction_fig, dash.no_update#, dash.no_update, dash.no_update


# selected-feature-numerical-modal
@app.callback(
    dash.dependencies.Output(component_id="edit-numerical-value-modal", component_property='is_open'),
    dash.dependencies.Output(component_id="selected-feature-numerical-modal", component_property='children'),
    dash.dependencies.Output(component_id="numerical-slider", component_property='min'),
    dash.dependencies.Output(component_id="numerical-slider", component_property='max'),
    dash.dependencies.Output(component_id="numerical-slider", component_property='value'),

    dash.dependencies.Output(component_id="edit-categorical-value-modal", component_property='is_open'),
    dash.dependencies.Output(component_id="selected-feature-categorical-modal", component_property='children'),
    dash.dependencies.Output(component_id="categorical-dropdown", component_property='value'),
    dash.dependencies.Output(component_id="categorical-dropdown", component_property='options'),

    dash.dependencies.Input(component_id="selected-feature-data", component_property='data'),
    prevent_initial_call=True,
)
def edit_modal(selected_data):
    print("in edit_modal")
    selected_feature = selected_data["feature"]
    value = selected_data["value"]# this is already a string

    if df.dtypes[selected_feature] == object:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, selected_feature, value, categorical_features[selected_feature]
    else:
        min_val = df[selected_feature].min().round(2)
        max_val = df[selected_feature].max().round(2)
        return True, selected_feature, min_val, max_val, float(value), dash.no_update, dash.no_update, dash.no_update, dash.no_update,
    




@app.callback(
    dash.dependencies.Output(component_id="contributions-table", component_property='rowData', allow_duplicate=True),
    dash.dependencies.Output(component_id="contributions-table", component_property='columnDefs', allow_duplicate=True),
    dash.dependencies.Input(component_id="numerical-slider", component_property='value'),
    dash.dependencies.Input(component_id="categorical-dropdown", component_property='value'),
    dash.dependencies.State(component_id="contributions-table", component_property='rowData'),
    # dash.dependencies.State(component_id="mode-selection-radio", component_property='value'),
    # dash.dependencies.State(component_id="additional-feature-dropdown", component_property='value'),
    dash.dependencies.State(component_id="contributions-table", component_property='columnDefs'),
    dash.dependencies.State(component_id="selected-feature-data", component_property='data'),
    prevent_initial_call=True,
)
def update_predictions(slider_value, categorical_value, rowData, columnDefs, selected_data):
    # ALL values are string
    print("in update_predictions")
    # selected_feature = "Area_Between_2nd_Cross_Over_to_3rd_C.Pa"

    if ctx.triggered_id == "numerical-slider":
        print("called by numerical slider")
        updated_value = slider_value
    else:
        print("called by categorical dropdown")
        updated_value = categorical_value

    selected_feature = selected_data["feature"]
    # value = selected_data["value"]

    rowData_df = pd.DataFrame(rowData)
    features = rowData_df["feature"].str.split("\n", expand=True).iloc[:, 0]
    features_list = features.to_list()
    ordered_values = rowData_df["values"].to_list()
    # rowData_df.loc[rowData_df["feature"] == selected_feature, ""]

    local_ordered_features = []
    ordered_features_with_category = []
    for i in range(0, len(features_list)):
        feature = features_list[i]
        if feature == selected_feature:
            ordered_values[i] = updated_value#sets the selected feature value to the value in the slider
        if feature in categorical_features:
            ordered_features_with_category.append(feature + "?" + ordered_values[i])
            ordered_values[i] = 1
        else:
            ordered_features_with_category.append(feature)
        local_ordered_features.append(feature)
    # print(ordered_features)

    # Form ordered_X dataframe
    ordered_X = pd.DataFrame(np.reshape(ordered_values, (1,-1)), columns=ordered_features_with_category)
    # print(ordered_X.dtypes)

    # Get model ready X
    model_ready_X = ordered_X.reindex(columns=model_features, fill_value=0)
    # print(model_ready_X)

    # Cast model_ready_X numerical columns to floats
    numerical_columns =  model_features_dtypes[model_features_dtypes != "object"].index# I think that model_features_dtypes will never be object because it is defined after X is encoded
    model_ready_X[numerical_columns] = model_ready_X[numerical_columns].astype(float)

    # Cast model_ready_X categorical columns to category
    # categorical_columns =  model_features_dtypes[model_features_dtypes == "category"].index
    # model_ready_X[categorical_columns] = model_ready_X[categorical_columns].astype("category")

    # Get prediction contributions
    contribs = model.get_booster().predict(xgb.DMatrix(model_ready_X), pred_contribs=True)
    # contribs = model.get_booster().predict(xgb.DMatrix(model_ready_X, enable_categorical=True), pred_contribs=True)

    bias = contribs[0, -1]
    contribs = contribs[:, :-1]
    
    # Store prediction contributions in a dataframe
    contrib_df = pd.DataFrame(contribs, columns=model_ready_X.columns)
    # print(contrib_df)

    # Aggregate SHAP values for categorical features
    contrib_df = contrib_df.groupby(lambda x: x.split('?')[0], axis=1).sum()
    
    # Reorder the dataframe columns to match the web app and store into array
    ordered_contribs = np.asarray(contrib_df[local_ordered_features])[0]
    # print(ordered_contribs)

    # Get predicted value and set that to columnDef header
    if is_classification:
        tot_contribs = ordered_contribs.sum() + bias
        prediction = (1 / (1 + np.exp(-tot_contribs)))
        prediction = calibrate_probability(prediction)*100
        if target_map[1] == 1:
            y_display = f"{target} (prob)\n{prediction:.{predict_decimals}f}%"
        else:
            y_display = f"{target_map[1]} (prob)\n{prediction:.{predict_decimals}f}%"
    else:
        prediction = ordered_contribs.sum() + bias
        y_display = f"{target}\n{prediction:.{predict_decimals}f}"
    columnDefs[1]["headerName"] = y_display

    # Get Feature Contributions back to probability scale if model is a classifier
    if is_classification:
        predictions = ordered_contribs + bias
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)
        baseline_prediction = (1 / (1 + np.exp(-bias)))
        baseline_prediction = calibrate_probability(baseline_prediction)
        ordered_contribs = (predictions - baseline_prediction) * 100

    # set the contributions in Pos and Neg columns
    rowData_df["pos"] = np.where(ordered_contribs > 0, ordered_contribs, 0)
    rowData_df["neg"] = np.where(ordered_contribs < 0, ordered_contribs, 0)
    # rowData_df["pos"] = rowData_df["pos"].map("{:.2f}".format)
    rowData_df["pos"] = rowData_df["pos"].map(f"{{:.{contrib_decimals}f}}".format)
    # rowData_df["neg"] = rowData_df["neg"].map("{:.2f}".format)
    rowData_df["neg"] = rowData_df["neg"].map(f"{{:.{contrib_decimals}f}}".format)

    # set the selected feature and value fields to the corresponding inline css feature value
    if df[selected_feature].dtype == object:
        rowData_df.loc[features == selected_feature, "values"] = updated_value
        rowData_df.loc[features == selected_feature, "feature"] = selected_feature + "\n<span style='font-weight:normal; color:white; backgroundColor:#616161; padding:4px 5px 5px 5px; border-radius:5px; display:inline-block; margin-top:5px;'>" + updated_value + "</span>"
    else:
        # rowData_df.loc[features == selected_feature, "values"] = f"{updated_value:.2f}"
        # rowData_df.loc[features == selected_feature, "feature"] = selected_feature + "\n<span style='font-weight:normal; color:white; backgroundColor:#616161; padding:4px 5px 5px 5px; border-radius:5px; display:inline-block; margin-top:5px;'>" + f"{updated_value:.2f}" + "</span>"
        rowData_df.loc[features == selected_feature, "values"] = f"{updated_value}"
        rowData_df.loc[features == selected_feature, "feature"] = selected_feature + "\n<span style='font-weight:normal; color:white; backgroundColor:#616161; padding:4px 5px 5px 5px; border-radius:5px; display:inline-block; margin-top:5px;'>" + f"{updated_value}" + "</span>"


    return rowData_df.to_dict("records"), columnDefs




@app.callback(
    dash.dependencies.Output(component_id="optimization-line-chart", component_property='figure'),
    dash.dependencies.Output(component_id="optimization-scatter-chart", component_property='figure'),
    # dash.dependencies.Output(component_id="optimization-comparison-table", component_property='columnDefs'),
    # dash.dependencies.Output(component_id="comparison-mode", component_property='data', allow_duplicate=True),
    dash.dependencies.Input(component_id="optimization-comparison-table", component_property='cellClicked'),
    
    dash.dependencies.Input(component_id="BOB-values-store", component_property='data'),
    dash.dependencies.Input(component_id="WOW-values-store", component_property='data'),

    # dash.dependencies.Input(component_id="optimization-groupby-feature-dropdown", component_property='value'),
    dash.dependencies.State(component_id="optimization-comparison-modal", component_property='is_open'),
    prevent_initial_call=True,
)
def update_optimization_charts(cell_data, BOB_data, WOW_data, is_open):
    print("in update_optimization_line_chart")
    if not is_open:
        return dash.no_update, dash.no_update
    
    feature = cell_data.get("value")
    if feature in df.columns:
        BOB_df = pd.DataFrame(BOB_data)
        WOW_df = pd.DataFrame(WOW_data)
        if is_classification:
            if target_map[1] == 1:
                y_display = f"{target} (prob)"
            else:
                y_display = f"{target_map[1]} (prob)"
        else:
            y_display = f"{target}*"

        if df[feature].dtype == object:
            BOB_WOW_line_chart = get_BOB_WOW_line_chart(BOB_df, WOW_df, feature, y_display, max_points=5000)
            BOB_WOW_scatter_chart = get_BOB_WOW_scatter_chart(BOB_df, WOW_df, feature, y_display, max_points=5000)
        else:
            BOB_WOW_line_chart = get_BOB_WOW_line_chart(BOB_df, WOW_df, feature, y_display, max_points=5000)
            BOB_WOW_scatter_chart = get_BOB_WOW_scatter_chart(BOB_df, WOW_df, feature, y_display, max_points=5000)
        return BOB_WOW_line_chart, BOB_WOW_scatter_chart
    
    return dash.no_update, dash.no_update



############################## show the selected data UI
@app.callback(
    dash.dependencies.Output(component_id="analysis-type-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="UPS-data-query-modal", component_property='is_open'),
    dash.dependencies.Output(component_id="good-vs-bad-data-query-modal", component_property='is_open'),
    dash.dependencies.Input(component_id="analysis-type-next-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="analysis-type-radio-btn", component_property='value'),
    prevent_initial_call=True,
)
def show_selected_data_UI(n_clicks, analysis_type):
    if analysis_type == "Good vs Bad":
        return False, False, True
    elif analysis_type == "Unplanned Stop Data":
        return False, True, False
    return False, False, False


############################## show Data Preparation Modal
@app.callback(
    dash.dependencies.Output(component_id="analysis-type-modal", component_property='is_open'),
    dash.dependencies.Input(component_id="data-preparation-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def show_data_preparation_modal(n_clicks):
    return True



############################## prepare UPS data
@app.callback(
    dash.dependencies.Output(component_id="download-UPS-data", component_property='data'),
    dash.dependencies.Input(component_id="prepare-UPS-data-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="select-site-dropdown", component_property='value'),
    dash.dependencies.State(component_id="select-machine-dropdown", component_property='value'),
    dash.dependencies.State(component_id="select-product-dropdown", component_property='value'),
    dash.dependencies.State(component_id="start-date", component_property='date'),
    dash.dependencies.State(component_id="end-date", component_property='date'),
    prevent_initial_call=True,
)
def prepare_UPS_data(n_clicks, plant, machine, product, start_date, end_date):
    import psycopg2
    database = 'tenant_storage'
    SQL_info = {
        "Carbondale":{
            "user":"979c439c-5f75-44b6-9e88-e01cc1a673c6",
            "password":"sma_bz56zylvceeFk6SflyDrIL4g9ywayalE78CzzJdjZAWY_",
            "host":"ipg-carbondale.sightmachine.io",
            "machines":{
                "Coater 44":{
                    "table":"sightmachine.cycle_mt_coater44"
                },
                "Coater 51":{
                    "table":"sightmachine.cycle_mt_coater51"
                },
                "Coater 52":{
                    "table":"sightmachine.cycle_mt_coater52"
                }
            }
        }
    }
    user = SQL_info[plant]["user"]
    password = SQL_info[plant]["password"]
    host = SQL_info[plant]["host"]
    table = SQL_info[plant]["machines"][machine]["table"]

    start_date_feature = "Cycle Start Time"
    end_date_feature = "Cycle End Time"
    # date_feature = "Cycle Start Time"
    running_good_feature = "RUNNING GOOD PRODUCT_SM"
    event_flag = "Unplanned Stop Flag"
    product_feature = "Production Product Description"
    # query = f"""
    # select * from {table}
    # where "Cycle Start Time" >= '2026-01-01' and "Cycle Start Time" < '2026-02-01'
    # limit 100000
    # """
    query = f"""
    select * from {table}
    where "{end_date_feature}" >= '{start_date}' and "{end_date_feature}" < '{end_date}'
    and ("{running_good_feature}" = 1 or "{event_flag}" = 1)
    limit 100000
    """
    print(query)

    with psycopg2.connect(database=database, user=user, password=password, host=host, port=5432, connect_timeout=60, sslmode="require") as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        query_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        print(query_df.shape)

    MINUTES_BEFORE_EVENT = 10
    BUFFER_MINUTES_FOR_STARTUP = 3
    BUFFER_MINUTES_FOR_STOP = 3
    final_df = format_SM_data_to_N_minutes_before_UPS(query_df, start_date_feature, running_good_feature, event_flag, MINUTES_BEFORE_EVENT, BUFFER_MINUTES_FOR_STARTUP, BUFFER_MINUTES_FOR_STOP)
    if product == None:
        file_name = f"{plant}_{machine}_UPS_{start_date}-{end_date}.csv"
    else:
        final_df = final_df.loc[final_df[product_feature] == product, :]
        file_name = f"{plant}_{machine}_UPS_{product}_{start_date}-{end_date}.csv"
    return dcc.send_data_frame(final_df.to_csv, file_name, index=False)




############################## prepare good vs bad data #############
@app.callback(
    dash.dependencies.Output(component_id="download-good-vs-bad-data", component_property='data'),
    dash.dependencies.Input(component_id="prepare-good-vs-bad-data-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="upload-good-data", component_property='contents'),
    dash.dependencies.State(component_id="upload-good-data", component_property='filename'),
    dash.dependencies.State(component_id="upload-bad-data", component_property='contents'),
    dash.dependencies.State(component_id="upload-bad-data", component_property='filename'),
    # dash.dependencies.State(component_id="upload-bad-data", component_property='last_modified'),
    prevent_initial_call=True,
)
def prepare_Good_vs_Bad_data(n_clicks, good_contents, good_filename, bad_contents, bad_filename):
    if good_contents is None or bad_contents is None:
        return dash.no_update
    
    good_content_type, good_content_string = good_contents.split(',')
    good_decoded = base64.b64decode(good_content_string)
    if "csv" in good_filename:
        good_df = pd.read_csv(io.StringIO(good_decoded.decode('utf-8')))
    elif "xls" in good_filename:
        good_df = pd.read_excel(io.BytesIO(good_decoded), sheet_name=0, engine="openpyxl")
    good_df = good_df[good_df.columns[~good_df.isna().all()]]
    
    bad_content_type, bad_content_string = bad_contents.split(',')
    bad_decoded = base64.b64decode(bad_content_string)
    if "csv" in bad_filename:
        bad_df = pd.read_csv(io.StringIO(bad_decoded.decode('utf-8')))
    elif "xls" in bad_filename:
        bad_df = pd.read_excel(io.BytesIO(bad_decoded), sheet_name=0, engine="openpyxl")
    bad_df = bad_df[bad_df.columns[~bad_df.isna().all()]]

    good_df.insert(0, "Good vs Bad", "Good")
    bad_df.insert(0, "Good vs Bad", "Bad")
    combined_df = pd.concat([good_df, bad_df], axis=0, ignore_index=True)

    date_string = datetime.now().strftime("%Y-%m-%d %H%M%S")
    file_name = f"Good_vs_Bad {date_string}.csv"
    return dcc.send_data_frame(combined_df.to_csv, file_name, index=False)



############################## show optimization-modal
@app.callback(
    dash.dependencies.Output(component_id="interpolated-options-div", component_property='style'),
    dash.dependencies.Output(component_id="historical-options-div", component_property='style'),
    dash.dependencies.Input(component_id="optimization-method-radio", component_property='value'),
    prevent_initial_call=True,
)
def optimization_method_options_selection(method_selected):
    print("in optimization_method_options_selection")
    if method_selected == "Interpolated":
        return {"display":"block"}, {"display":"none"}
    else:
        return {"display":"none"}, {"display":"block"}


############################## show optimization-modal
@app.callback(
    dash.dependencies.Output(component_id="optimization-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="show-optimization-modal-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def show_optimization_modal(n_clicks):
    return True




############################## Optimization Interpolated and Historical
@app.callback(
    dash.dependencies.Output(component_id="optimization-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="download-optimization-data", component_property='data'),

    dash.dependencies.Output(component_id="BOB-values-store", component_property='data'),
    dash.dependencies.Output(component_id="WOW-values-store", component_property='data'),
    

    dash.dependencies.Input(component_id="optimize-btn", component_property='n_clicks'),
    
    dash.dependencies.State(component_id="exluded-optmization-features-dropdown", component_property='value'),
    dash.dependencies.State(component_id="higher-is-better-dropdown", component_property='value'),
    
    dash.dependencies.State(component_id="optimization-method-radio", component_property='value'),
    dash.dependencies.State(component_id="top-n", component_property='value'),
    dash.dependencies.State(component_id="percentile-slider", component_property='value'),
    dash.dependencies.State(component_id="contributions-table", component_property='rowData'),
    
    prevent_initial_call=True,
)
def find_optimal_values(n_clicks, static_features, optimization_direction, optimization_method, top_n, percentiles, rowData):
    print("in find_optimal_values")
    if static_features is None:
        static_features = []
    
    # Grab only the selected columns of df
    # X = df[selected_features].drop(columns=target)
    X = df[selected_features].copy()
    
    # Get predictions
    predictions = contributions_df.sum(axis=1)
    if is_classification:
        predictions = (1 / (1 + np.exp(-predictions)))
        predictions = calibrate_probability(predictions)

    # Filter dataframe by the static values and static features
    if len(static_features) > 0:
        rowData_df = pd.DataFrame(rowData)
        features = rowData_df["feature"].str.split("\n", expand=True).iloc[:, 0]
        static_features_mask = features.isin(static_features)
        static_features = features.loc[static_features_mask].to_list()
        static_feature_values = rowData_df.loc[static_features_mask, "values"].to_list()

        features_dtypes = df.dtypes
        filter_mask = np.repeat(True, X.shape[0])
        for j in range(0, len(static_features)):
            if features_dtypes[static_features[j]] == "object":
                filter_mask = filter_mask & (X[static_features[j]] == static_feature_values[j])
            else:
                static_feature_value = float(static_feature_values[j])
                tol = X[static_features[j]].std(ddof=0) * 0.15
                filter_mask = filter_mask & (X[static_features[j]].between(static_feature_value - tol, static_feature_value + tol))
        X = X.loc[filter_mask, :].reset_index(drop=True)
        predictions = predictions.loc[filter_mask].reset_index(drop=True)
        print(f"After filtering shape : {X.shape}")


    if optimization_method == "Interpolated":
        return True, dash.no_update
    else:# Optimization method = Historical
        # find sorted prediction indices
        if optimization_direction == "Yes":
            sorted_indices = np.argsort(-predictions)
        else:
            sorted_indices = np.argsort(predictions)

        if "." in top_n:
            top_n = int(float(top_n) * X.shape[0])
        else:
            top_n = int(top_n)#20
        top_n = min(top_n, X.shape[0])
        best_indices = sorted_indices[0:top_n]
        best_samples = X.iloc[best_indices, :][ordered_features]
        best_predictions = predictions.iloc[best_indices]
        best_samples.insert(0, "Pred =>", best_predictions)
        best_samples.insert(0, "Actual =>", X[target].iloc[best_indices])

        worst_indices = sorted_indices[-top_n:]
        worst_samples = X.iloc[worst_indices, :][ordered_features]
        worst_predictions = predictions.iloc[worst_indices]
        worst_samples.insert(0, "Pred =>", worst_predictions)
        worst_samples.insert(0, "Actual =>", X[target].iloc[worst_indices])

        best_worst_df = pd.concat([best_samples, worst_samples], axis=0, ignore_index=True)
        best_worst_df.index = range(1, len(best_worst_df) + 1)



    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # return False, dcc.send_data_frame(best_worst_df.T.to_csv, filename=f"optimization{timestamp}.csv", index=True), best_samples.to_dict("records"), worst_samples.to_dict("records")
    return False, dash.no_update, best_samples.to_dict("records"), worst_samples.to_dict("records")




############################## Optimization Interpolated and Historical
@app.callback(
    dash.dependencies.Output(component_id="optimization-comparison-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="optimization-comparison-table", component_property='rowData', allow_duplicate=True),
    dash.dependencies.Output(component_id="optimization-comparison-table", component_property='columnDefs'),

    dash.dependencies.Input(component_id="BOB-values-store", component_property='data'),
    dash.dependencies.Input(component_id="WOW-values-store", component_property='data'),
    # dash.dependencies.Input(component_id="optimize-btn", component_property='n_clicks'),
    # dash.dependencies.State(component_id={"type":"update-x-values", "index":ALL}, component_property='value'),
    # dash.dependencies.State(component_id="exluded-optmization-features-dropdown", component_property='value'),
    # dash.dependencies.State(component_id="higher-is-better-dropdown", component_property='value'),
    # dash.dependencies.State(component_id="feature-slider-container", component_property='children'),
    # dash.dependencies.State(component_id="optimization-method-radio", component_property='value'),
    # dash.dependencies.State(component_id="top-n", component_property='value'),
    # dash.dependencies.State(component_id="percentile-slider", component_property='value'),
    dash.dependencies.State(component_id="optimization-comparison-table", component_property='columnDefs'),
    prevent_initial_call=True,
)
def update_comparison_modal(BOB_data, WOW_data, columnDefs):
    print("in update_comparison_modal")
    BOB_df = pd.DataFrame(BOB_data)#.drop(columns=["Pred =>"])
    WOW_df = pd.DataFrame(WOW_data)#.drop(columns=["Pred =>"])

    # display the best means and worst means as headers
    if is_classification:
        BOB_actual_mean = np.where(BOB_df["Actual =>"] == target_map[0], 0, 1).mean()
        WOW_actual_mean = np.where(WOW_df["Actual =>"] == target_map[0], 0, 1).mean()
        if target_map[1] == 1:
            target_label = f"{target} (prob)"
        else:
            target_label = f"{target_map[1]} (prob)"
        columnDefs[0]["headerName"] = f"{target_label} =>"
        columnDefs[1]["headerName"] = f"Best: {BOB_actual_mean*100:.{predict_decimals}f}%"
        columnDefs[2]["headerName"] = f"Worst: {WOW_actual_mean*100:.{predict_decimals}f}%"
        # columnDefs[1]["headerName"] = f"Best: {BOB_actual_mean*100:.1f}%"
        # columnDefs[2]["headerName"] = f"Worst: {WOW_actual_mean*100:.1f}%"
    else:
        BOB_actual_mean = BOB_df["Actual =>"].mean()
        WOW_actual_mean = WOW_df["Actual =>"].mean()
        columnDefs[0]["headerName"] = f"{target} =>"
        columnDefs[1]["headerName"] = f"Best: {BOB_actual_mean:.{predict_decimals}f}"
        columnDefs[2]["headerName"] = f"Worst: {WOW_actual_mean:.{predict_decimals}f}"
        # columnDefs[1]["headerName"] = f"Best: {BOB_actual_mean:.1f}"
        # columnDefs[2]["headerName"] = f"Worst: {WOW_actual_mean:.1f}"

    # Drop Actual and Predicted columns
    BOB_df = BOB_df.drop(columns=["Actual =>","Pred =>"])
    WOW_df = WOW_df.drop(columns=["Actual =>","Pred =>"])

    # Get BOB Summary Statistics for numerical columns
    BOB_summary_df = BOB_df.describe(percentiles=[0.05, 0.5, 0.95]).T
    BOB_summary_df.columns = ["Samples","Mean","Stdev","Min","BOB_5th","BOB_Center","BOB_95th","Max"]
    BOB_summary_df.index.name = "Variable"

    # Get BOB Summary Statistics for categorical columns
    categorical_columns =  BOB_df.dtypes[BOB_df.dtypes == "object"].index
    BOB_categorical_summary = pd.DataFrame(BOB_df[categorical_columns].mode(dropna=True).iloc[0])
    BOB_categorical_summary.columns = ["BOB_Center"]
    BOB_categorical_summary.index.name = "Variable"
    BOB_summary_df = pd.concat([BOB_summary_df, BOB_categorical_summary], axis=0)

    # Get WOW Summary Statistics for numerical columns
    WOW_summary_df = WOW_df.describe(percentiles=[0.05, 0.5, 0.95]).T#WOW_df.drop(columns=["Actual =>","Pred =>"]).describe(percentiles=[0.05, 0.5, 0.95]).T
    WOW_summary_df.columns = ["Samples2","Mean2","Stdev2","Min2","WOW_5th","WOW_Center","WOW_95th","Max2"]
    WOW_summary_df.index.name = "Variable"
    
    # Get WOW Summary Statistics for categorical columns
    categorical_columns =  WOW_df.dtypes[WOW_df.dtypes == "object"].index
    WOW_categorical_summary = pd.DataFrame(WOW_df[categorical_columns].mode(dropna=True).iloc[0])
    WOW_categorical_summary.columns = ["WOW_Center"]
    WOW_categorical_summary.index.name = "Variable"
    WOW_summary_df = pd.concat([WOW_summary_df, WOW_categorical_summary], axis=0)

    # Combine BOB and WOW dataframes
    combined_df = pd.concat([BOB_summary_df, WOW_summary_df], axis=1).loc[ordered_features, :].reset_index(drop=False)
    combined_df.to_clipboard()

    return True, combined_df.to_dict("records"), columnDefs



############################## Closes the Train Model UI
@app.callback(
    dash.dependencies.Output(component_id="train-model-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="train-model-close-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def close_train_model_modal(n_clicks):
    return False


############################## Handles the Train Model UI
@app.callback(
    dash.dependencies.Output(component_id="model-accuracy-gauge", component_property='figure'),
    dash.dependencies.Output(component_id="cv-graph", component_property='figure'),
    dash.dependencies.Output(component_id="update-page-btn", component_property='n_clicks'),
    dash.dependencies.Output(component_id="previous-model-score", component_property='data'),
    
    dash.dependencies.Input(component_id="train-model", component_property='n_clicks'),
    dash.dependencies.State(component_id="n-estimators", component_property='value'),
    dash.dependencies.State(component_id="learning-rate", component_property='value'),
    dash.dependencies.State(component_id="max-depth", component_property='value'),
    dash.dependencies.State(component_id="colsample-bynode", component_property='value'),
    dash.dependencies.State(component_id="update-page-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="previous-model-score", component_property='data'),
    prevent_initial_call=True,
)
def train_model(n_clicks, n_estimators, learning_rate, max_depth, colsample_bynode, update_page_n_clicks, prev_model_score):
    print("in train_model")
    global model
    global balancing_ratio
    processed_df = pd.concat([df], axis=1).loc[:, selected_features]

    if is_classification:
        if order_matters:
            cv = TimeSeriesSplit(n_splits=5)
            df_ordered = processed_df
        else:
            cv = StratifiedKFold(n_splits=5)
            df_ordered = processed_df.sample(frac=1, random_state=0)
        print("classification")
        df_clean_target = df_ordered.dropna(subset=target)
        y = np.where(df_clean_target[target] == target_map[0], 0, 1)
        X = df_clean_target.drop(columns=target)
        encoded_X = pd.get_dummies(X, prefix_sep="?")
        print(encoded_X.shape)
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        balancing_ratio = neg_count / pos_count
        model = xgb.XGBClassifier(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  colsample_bynode=colsample_bynode,
                                  base_score=y.mean(),
                                  scale_pos_weight=balancing_ratio,
                                  random_state=0)
        results = cross_validate(model, encoded_X, y, cv=cv, scoring="roc_auc", return_train_score=True)
        print(results["test_score"])
        model.fit(encoded_X, y)
        # dtrain = xgb.DMatrix(encoded_X, label=y)
        # params = {
        #     "objective": "binary:logistic",
        #     "eval_metric": "logloss",
        #     "eta": learning_rate,
        #     "max_depth": max_depth,
        #     "colsample_bynode": colsample_bynode,
        #     "scale_pos_weight": balancing_ratio,
        #     "base_score": y.mean(),# nnnnnnnnnnnnnneeeeeeeeeeewwwwwwwwwwwwwww
        #     "gamma": 1,
        #     "seed":0,
        # }
        # Perform cross-validation
        # cv = StratifiedKFold(n_splits=5)
        # cv_results = xgb.cv(
        #     params=params,
        #     dtrain=dtrain,
        #     num_boost_round=n_estimators,
        #     folds=cv,
        #     metrics="auc",# logloss
        #     early_stopping_rounds=40, # Stop if performance doesn't improve for 40 rounds
        #     seed=0,             # for reproducibility
        #     # callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)] # Print evaluation results
        # )
        # model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))

        folds = [f"Group {i+1}" for i in range(0, 5)]# 5
        train_scores = np.maximum(results["train_score"], 0)
        test_scores = np.maximum(results["test_score"], 0)# added this #+ 0.007, 
        curr_model_score = test_scores.mean()*100

        # curr_model_score = cv_results["test-auc-mean"].iat[-1] * 100
        print(curr_model_score)
        if prev_model_score == None:
            prev_model_score = curr_model_score
        model_accuracy_gauge = get_indicator_KPI(curr_value=curr_model_score, prev_value=prev_model_score, title="Accuracy %")
        cv_graph = get_multi_trace_graph(xs=[folds, folds], ys=[train_scores,test_scores], names=["Train","Test"], x_title="", y_title="Accuracy", title="")
        # cv_graph = get_train_test_graph(train_scores=cv_results["train-auc-mean"], test_scores=cv_results["test-auc-mean"], x_title="Number of Trees", y_title="Accuracy", title="", show_legend=True)

    else:
        if order_matters:
            cv = TimeSeriesSplit(n_splits=5)
            df_ordered = processed_df
        else:
            cv = KFold(n_splits=5)
            df_ordered = processed_df.sample(frac=1, random_state=0)
        print("regression")
        df_clean_target = df_ordered.dropna(subset=target)
        y = df_clean_target[target]
        X = df_clean_target.drop(columns=target)
        encoded_X = pd.get_dummies(X, prefix_sep="?")
        print(encoded_X.shape)
        model = xgb.XGBRegressor(n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 max_depth=max_depth,
                                 colsample_bynode=colsample_bynode,
                                 base_score=y.mean(),
                                 random_state=0)
        results = cross_validate(model, encoded_X, y, cv=cv, scoring="r2", return_train_score=True)# 0.6208767361111109
        print(results["test_score"])
        model.fit(encoded_X, y)

        # dtrain = xgb.DMatrix(encoded_X, label=y)
        # params = {
        #     "objective": "reg:squarederror",
        #     "eval_metric": "rmse",
        #     "eta": learning_rate,
        #     "max_depth": max_depth,
        #     "colsample_bynode": colsample_bynode,
        #     "base_score": y.mean(),# nnnnnnnnnnnnnneeeeeeeeeeewwwwwwwwwwwwwww
        #     "gamma": 1,
        #     "seed":0,
        # }
        # Perform cross-validation
        # cv = StratifiedKFold(n_splits=5)
        # cv_results = xgb.cv(
        #     params=params,
        #     dtrain=dtrain,
        #     num_boost_round=n_estimators,
        #     folds=cv,
        #     metrics="rmse",   # Evaluation metrics
        #     early_stopping_rounds=40, # Stop if performance doesn't improve for 40 rounds
        #     seed=0,             # for reproducibility
        #     # callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)] # Print evaluation results
        # )
        # model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))
        # var_y = np.var(y, ddof=0)
        # train_r2 = 1 - ((cv_results["train-rmse-mean"] ** 2 ) / var_y)
        # test_r2 = 1 - ((cv_results["test-rmse-mean"] ** 2 ) / var_y)
        # print(cv_results)

        folds = [f"Group {i+1}" for i in range(0, 5)]# 5
        train_scores = np.maximum(results["train_score"], 0)
        test_scores = np.maximum(results["test_score"], 0)# added this #+ 0.007, 
        curr_model_score = test_scores.mean()*100
        
        # curr_model_score = test_r2.iat[-1] * 100
        print(curr_model_score)
        if prev_model_score == None:
            prev_model_score = curr_model_score
        model_accuracy_gauge = get_indicator_KPI(curr_value=curr_model_score, prev_value=prev_model_score, title="Accuracy %")
        cv_graph = get_multi_trace_graph(xs=[folds, folds], ys=[train_scores,test_scores], names=["Train","Test"], x_title="", y_title="R2", title="")
        # cv_graph = get_train_test_graph(train_scores=train_r2, test_scores=test_r2, x_title="Number of Trees", y_title="R Squared", title="", show_legend=True)


    update_model_metadata(model, df_clean_target)
    print(f"Avg target: {y.mean()}")

    # curr_model_score = cv_results["test-auc-mean"].iat[-1] * 100
    # print(curr_model_score)
    # if prev_model_score == None:
    #     prev_model_score = curr_model_score
    # model_accuracy_gauge = get_indicator_KPI(curr_value=curr_model_score, prev_value=prev_model_score, title="Accuracy %")
    # cv_graph = get_train_test_graph(train_scores=cv_results["train-auc-mean"], test_scores=cv_results["test-auc-mean"], x_title="Number of Trees", y_title="Accuracy", title="", show_legend=True)

    return model_accuracy_gauge, cv_graph, update_page_n_clicks + 1, curr_model_score


############################## Handles the Sample Order UI and Opens the Train Model UI
@app.callback(
    dash.dependencies.Output(component_id="train-model-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="select-sample-order-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="sample-order-next-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="sample-order-dropdown", component_property='value'),
    prevent_initial_call=True,
)
def saves_sample_order_and_opens_train_model_modal(n_clicks, sample_order_matters):
    print("inside open_train_model_modal")
    global order_matters

    if sample_order_matters == "No":
        order_matters = False
    else:
        order_matters = True
    return True, False


# ############################## Handles the Sample Order UI
# @app.callback(
#     dash.dependencies.Output(component_id="select-sample-order-modal", component_property='is_open', allow_duplicate=True),
#     dash.dependencies.Input(component_id="sample-order-dropdown", component_property='value'),
#     prevent_initial_call=True,
# )
# def update_sample_order_variable(sample_order_matters):
#     print("inside update_sample_order_variable")
#     global order_matters

#     if sample_order_matters == "No":
#         order_matters = False
#     else:
#         order_matters = True
#     return dash.no_update



# ############################## Opens the Sample Order UI
# @app.callback(
#     dash.dependencies.Output(component_id="select-sample-order-modal", component_property='is_open', allow_duplicate=True),
#     dash.dependencies.Output(component_id="select-input-modal", component_property='is_open'),
#     dash.dependencies.Input(component_id="save-input-next-btn", component_property='n_clicks'),
#     prevent_initial_call=True,
# )
# def open_sample_order_modal(n_clicks):
#     return True, False


############################## Handles the Select Input UI and Opens the Sample Order UI
@app.callback(
    dash.dependencies.Output(component_id="select-sample-order-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="select-input-modal", component_property='is_open'),
    dash.dependencies.Input(component_id="save-input-next-btn", component_property='n_clicks'),
    dash.dependencies.State(component_id="final-features-checklist", component_property='value'),
    prevent_initial_call=True,
)
def save_inputs_and_open_sample_order_modal(n_clicks, right_panel_features):
    global selected_features
    selected_features = right_panel_features.copy()
    if target not in selected_features:
        selected_features.append(target)
    return True, False


############################## Opens the Select Input UI
@app.callback(
    dash.dependencies.Output(component_id="select-input-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="select-output-modal", component_property='is_open'),
    dash.dependencies.Input(component_id="save-output-next-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def open_select_input_modal(n_clicks):
    return True, False


############################## Handles the Select Output UI
@app.callback(
    dash.dependencies.Output(component_id="select-output-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id='model-accuracy-gauge', component_property='figure', allow_duplicate=True),
    dash.dependencies.Output(component_id="previous-model-score", component_property='data', allow_duplicate=True),
    dash.dependencies.Input(component_id="select-output-dropdown", component_property='value'),
    prevent_initial_call=True,
)
def update_output_variable(target_):
    print("inside update_output_variable")
    global target
    global is_classification
    global target_map

    if target_ is None:
        return dash.no_update, dash.no_update, dash.no_update
    processed_target = pd.concat([df], axis=1).loc[:, target_]

    print(target_)
    print(processed_target.shape)
    # print(df.shape)
    target = target_
    # unique_vals = np.unique(df[target_].dropna())
    unique_vals = np.unique(processed_target.dropna())
    if len(unique_vals) == 2:
        is_classification = True
        target_map = {0:unique_vals[0], 1:unique_vals[1]}
    elif (len(unique_vals) > 2) and is_numeric_dtype(processed_target):#is_numeric_dtype(df[target_])
        is_classification = False
    return dash.no_update, get_indicator_KPI(0, 0, "Accuracy %"), None


############################## Opens the Select Output UI
@app.callback(
    dash.dependencies.Output(component_id="select-output-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Output(component_id="get-data-modal", component_property='is_open'),
    dash.dependencies.Input(component_id="get-data-next-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def open_select_output_modal(n_clicks):
    return True, False


# get_indicator_KPI(0, 0, "Accuracy %")
############################## Handles the Import Data UI
@app.callback(
    dash.dependencies.Output(component_id="output-data-upload", component_property='children'),
    dash.dependencies.Output(component_id="select-output-dropdown", component_property='options'),
    dash.dependencies.Output(component_id="all-features-checklist", component_property='options'),
    dash.dependencies.Output(component_id="all-features-checklist", component_property='value'),
    dash.dependencies.Output(component_id="final-features-checklist", component_property='options'),
    dash.dependencies.Output(component_id="final-features-checklist", component_property='value'),
    dash.dependencies.Output(component_id='total-tags', component_property='children'),
    dash.dependencies.Input(component_id="upload-data", component_property='contents'),
    dash.dependencies.Input(component_id="upload-data", component_property='filename'),
    dash.dependencies.Input(component_id="upload-data", component_property='last_modified'),
    prevent_initial_call=True,
)
def show_uploaded_data(contents, filename, modified_date):
    global df
    print("inside show_uploaded_data")
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded), sheet_name=0, engine="openpyxl")
        df = df[df.columns[~df.isna().all()]]
        n_row, n_col = df.shape
        df_head = df.head(10)
        children = html.Div([
            html.H5(filename),
            html.Div([
                html.H6("Data Preview...", style={"display":"inline-block"}),
                html.H6(f"Rows = {n_row:,} | Columns = {n_col:,}", style={"display":"inline-block","float":"right"}),
            ]),
            dag.AgGrid(
                rowData=df_head.to_dict('records'),
                columnDefs=[{'field': i} for i in df.columns],
                defaultColDef = {"resizable": True},
                columnSize="autoSize",#"sizeToFit"
            ),
        ])
        return children, df.columns, df.columns, df.columns, df.columns, df.columns, f"{len(df.columns)} Total Variables"


############################## Opens the Import Data UI
@app.callback(
    dash.dependencies.Output(component_id="get-data-modal", component_property='is_open', allow_duplicate=True),
    dash.dependencies.Input(component_id="settings-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def open_get_data_modal(n_clicks):
    print("inside open_settings")
    return True



@app.callback(
    dash.dependencies.Output(component_id="feature-slider-container", component_property='children'),
    dash.dependencies.Output(component_id="positive-contribution-container", component_property='children'),
    dash.dependencies.Output(component_id="negative-contribution-container", component_property='children'),

    dash.dependencies.Output(component_id="exluded-optmization-features-dropdown", component_property='options'),
    dash.dependencies.Output(component_id="add-another-feature-dropdown", component_property='options'),
    dash.dependencies.Output(component_id="optimization-direction", component_property='children'),
    dash.dependencies.Output(component_id="chart-title-target", component_property='children'),
    dash.dependencies.Output(component_id="additional-feature-dropdown", component_property='options'),
    dash.dependencies.Input(component_id="update-page-btn", component_property='n_clicks'),
    prevent_initial_call=True,
)
def clicked_button(n_clicks):
    print("inside clicked_button")
    processed_df = pd.concat([df], axis=1).loc[:, selected_features].dropna(subset=target)

    df_clean_target = processed_df# df_clean_target = df.dropna(subset=target)
    
    feature_sliders = []
    positive_contributions_containers = []
    negative_contributions_containers = []
    
    # for i in range(0, len(df_clean_target.columns)):
    for i in range(0, len(ordered_features)):
        # feature = df_clean_target.columns[i]
        feature = ordered_features[i]
        # print(feature)
        if df_clean_target[feature].dtype == object:
            categories = np.unique(df_clean_target[feature].dropna())
            feature_slider = html.Div([
                html.Div(feature, title=feature, id={"type":"click-features", "index":feature}, n_clicks=0,
                         style={"lineHeight":"1.5","marginBottom":"0px","marginTop":"2px","whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis","cursor":"pointer"}),
                dcc.Dropdown(
                    id={"type":"update-x-values", "index":feature},
                    options=categories,
                    value=df_clean_target[feature].mode().iloc[0],
                    multi=False,
                    clearable=False,
                    placeholder="",
                    optionHeight=40,
                    disabled=False,
                    style={"fontSize":"12px", "border":"none"},#,"textOrientation":"upright","writingMode":"vertical-lr"
                    className="custom-dropdown"
                )
            ], style={"width":"15vw","border":"1px solid #cccccc","fontSize":"14px","height":"60px","paddingLeft":"10px","borderRadius":"5px","backgroundColor":"#ffffff"})
            feature_sliders.append(feature_slider)



            # contrib = 10
            positive_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "border":"1px solid","borderLeft":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "border":"0px solid","borderLeft":"none","backgroundColor":"#599ad3"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","left":"0vw"})
            ], id={"type":"positive-contributions-div", "index":feature},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderLeft":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            positive_contributions_containers.append(positive_container)


            # contrib = 10
            negative_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "float":"right", "border":"1px solid","borderRight":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "right":"0px", "border":"0px solid","borderRight":"none","backgroundColor":"#f9a65a"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","right":"0vw"})
            ], id={"type":"negative-contributions-div", "index":feature},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderRight":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            negative_contributions_containers.append(negative_container)
        else:
            # feature_min = df_clean_target[feature].quantile(0.01)
            # feature_max = df_clean_target[feature].quantile(0.99)
            feature_min, feature_max = df_clean_target[feature].quantile([0.01, 0.99])
            # feature_min = df[feature].min()
            # feature_max = df[feature].max()
            precision_size = (feature_max - feature_min) / 50
            if precision_size >= 1:
                rounded_places = 0
                feature_step = 1
            elif precision_size >= 0.1:
                rounded_places = 1
                feature_step = 0.1
            elif precision_size >= 0.01:
                rounded_places = 2
                feature_step = 0.01
            else:
                rounded_places = 3
                feature_step = 0.001
            feature_min = round(feature_min, rounded_places)
            feature_max = round(feature_max, rounded_places)
            # feature_min = feature_min.round(rounded_places)
            # feature_max = feature_max.round(rounded_places)


            feature_slider = html.Div([
                html.Div(feature, title=feature, id={"type":"click-features", "index":feature}, n_clicks=0,
                         style={"lineHeight":"1.5","marginBottom":"11px","marginTop":"2px","whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis","cursor":"pointer"}),
                dcc.Slider(
                    min=feature_min, max=feature_max, step=feature_step, marks=None,
                    value=df_clean_target[feature].median().round(rounded_places),
                    tooltip={"placement": "right", "always_visible": True, "style":{"fontSize":"13px"}},#,"lineHeight":"1"
                    id={"type":"update-x-values", "index":feature},
                    className="slider-class",#
                )
            ], style={"width":"15vw","border":"1px solid #cccccc","fontSize":"14px","height":"60px","paddingLeft":"10px","borderRadius":"5px","backgroundColor":"#ffffff"})
            feature_sliders.append(feature_slider)



            # contrib = 10
            positive_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "border":"1px solid","borderLeft":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "border":"0px solid","borderLeft":"none","backgroundColor":"#599ad3"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","left":"0vw"})
            ], id={"type":"positive-contributions-div", "index":feature},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderLeft":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            positive_contributions_containers.append(positive_container)


            # contrib = 10
            negative_container = html.Div([
                html.Div(
                    # style={"width":f"{contrib}vw","height":"40px", "marginTop":"10px", "float":"right", "border":"1px solid","borderRight":"none"}),
                    style={"position":"absolute","width":"0vw","height":"40px", "top":"10px", "right":"0px", "border":"0px solid","borderRight":"none","backgroundColor":"#f9a65a"}),
                html.Div("",
                         style={"position":"absolute","top":"20px","right":"0vw"})
            ], id={"type":"negative-contributions-div", "index":feature},
            style={"position":"relative", "width":"15vw","border":"1px solid transparent","borderRight":"none","fontSize":"14px",
                   "height":"60px","fontWeight":"600"})
            negative_contributions_containers.append(negative_container)

    if is_classification:
        if target_map[1] == 1:
            target_label = f"{target} (prob)"
            # page_title = f"{target.upper()} - SIMULATION"
        else:
            target_label = f"{target_map[1]} (prob)"
            # page_title = f"{target_map[1].upper()} - SIMULATION"
    else:
        target_label = target
        # page_title = f"{target.upper()} - SIMULATION"

    print("end clicked button")
    # return feature_sliders, positive_contributions_containers, negative_contributions_containers, f"{target_label.upper()} - INSIGHTS", ordered_features, ordered_features, f"Are Higher Values of {target_label} Better?", f"{target_label} vs "
    return feature_sliders, positive_contributions_containers, negative_contributions_containers, ordered_features, ordered_features, f"Are Higher Values of {target_label} Better?", f"{target_label} vs ", ordered_features




# @app.callback(
#     # dash.dependencies.Output(component_id={"type":"positive-contributions-div", "index":ALL}, component_property='children'),
#     # dash.dependencies.Output(component_id={"type":"negative-contributions-div", "index":ALL}, component_property='children'),
#     # dash.dependencies.Output(component_id="model-prediction", component_property='children'),
#     # dash.dependencies.Output(component_id="modal", component_property='is_open'),
#     # dash.dependencies.Output(component_id="contrib-graph", component_property='figure'),
#     # dash.dependencies.Output(component_id="chart-title-feature", component_property='children'),
#     # dash.dependencies.Output(component_id="add-another-feature-dropdown", component_property='value'),

#     dash.dependencies.Output(component_id={"type":"update-x-values", "index":ALL}, component_property='value', allow_duplicate=True),# NEWWWWWWW
    
#     dash.dependencies.Input(component_id={"type":"update-x-values", "index":ALL}, component_property='value'),
#     # dash.dependencies.Input(component_id={"type":"click-features", "index":ALL}, component_property='n_clicks'),
#     # dash.dependencies.State(component_id={"type":"positive-contributions-div", "index":ALL}, component_property='children'),
#     # dash.dependencies.State(component_id={"type":"negative-contributions-div", "index":ALL}, component_property='children'),
    
#     dash.dependencies.State(component_id="synced-movement-checkbox", component_property='value'),
#     prevent_initial_call=True,
# )
# def update_synced_movement(ordered_values, synced_movement):
#     print("inside update_synced_movement")
#     if (ctx.triggered_id["type"] == "update-x-values") and len(synced_movement) > 0:
#         print("doing synced stuff")
#     else:
#         return dash.no_update



@app.callback(
    dash.dependencies.Output(component_id={"type":"positive-contributions-div", "index":ALL}, component_property='children'),
    dash.dependencies.Output(component_id={"type":"negative-contributions-div", "index":ALL}, component_property='children'),

    # dash.dependencies.Output(component_id="model-prediction", component_property='children'),
    dash.dependencies.Output(component_id="page-title", component_property='children'),

    # dash.dependencies.Output(component_id="modal", component_property='is_open'),
    # dash.dependencies.Output(component_id="contrib-graph", component_property='figure'),

    dash.dependencies.Output(component_id="chart-title-feature", component_property='children'),
    dash.dependencies.Output(component_id="add-another-feature-dropdown", component_property='value'),

    dash.dependencies.Output(component_id={"type":"update-x-values", "index":ALL}, component_property='value', allow_duplicate=True),# NEWWWWWWW
    
    dash.dependencies.Output(component_id="feature-prediction-graph", component_property='figure', allow_duplicate=True),
    dash.dependencies.Output(component_id="feature-contrib-graph", component_property='figure', allow_duplicate=True),
    

    dash.dependencies.Input(component_id={"type":"update-x-values", "index":ALL}, component_property='value'),
    dash.dependencies.Input(component_id={"type":"click-features", "index":ALL}, component_property='n_clicks'),
    dash.dependencies.State(component_id={"type":"positive-contributions-div", "index":ALL}, component_property='children'),
    dash.dependencies.State(component_id={"type":"negative-contributions-div", "index":ALL}, component_property='children'),
    
    dash.dependencies.State(component_id="synced-movement-checkbox", component_property='value'),

    dash.dependencies.State(component_id="additional-feature-dropdown", component_property='value'),
    prevent_initial_call=True,
)
def update_prediction(ordered_values, n_clicks, positive_contributions_div, negative_contributions_div, synced_movement, additional_feature):
    print("inside update_prediction")
    # print(ctx.triggered_id["type"])
    # print(n_clicks)


    # Get update feature and index
    selected_feature = ctx.triggered_id["index"]
    value = ctx.triggered[0]["value"]
    print(f"Selected Feature: {selected_feature} and value: {value}")


    # Define all values and features
    input_info = ctx.inputs_list[0]


    synced_values = []
    if (ctx.triggered_id["type"] == "update-x-values") and len(synced_movement) > 0:
        print("in synced movement!!!!!!!!!!!!!!")
        X = pd.concat([df], axis=1).loc[:, selected_features].drop(columns=target)
        if selected_feature in categorical_features:
            filter_mask = X[selected_feature] == value
        else:
            tol = X[selected_feature].std(ddof=0) * 0.15
            filter_mask = X[selected_feature].between(value - tol, value + tol)
        filtered_X = X.loc[filter_mask, :].reset_index(drop=True)
        # synced_values = []
        for i in range(0, len(input_info)):
            feature = input_info[i]["id"]["index"]
            if feature == selected_feature:
                synced_values.append(value)
            else:
                if feature in categorical_features:
                    synced_values.append(filtered_X[feature].mode().iat[0])
                else:
                    synced_values.append(filtered_X[feature].median())
        
        # Check if ordered_values has changed
        if synced_values != ordered_values:
            print("In this call the synced values are the DIFFERENT from ordered values. This means I will store the new synced values in ordered_values")
            # print(synced_values)
            # print(ordered_values)
            # print()
            # print()
            ordered_values = synced_values.copy()
            # return [dash.no_update]*len(positive_contributions_div), [dash.no_update]*len(negative_contributions_div), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, synced_values
        else:
            print("In this call the synced values are the SAME as ordered values. This means I will update the predictions")
        # There is a forseable problem when we do optimization. The optimization UI will trigger update-x-values.
        # We have to find a way for this if-statement condition to be false in this scenario


    # Get all features...............Consider setting an id to the feature div, and use it's children as a state. No need to loop
    local_ordered_features = []
    ordered_features_with_category = []
    for i in range(0, len(input_info)):
        feature = input_info[i]["id"]["index"]
        if feature == selected_feature:
            selected_feature_value = ordered_values[i]
        if feature in categorical_features:
            ordered_features_with_category.append(feature + "?" + ordered_values[i])
            ordered_values[i] = 1
        else:
            ordered_features_with_category.append(feature)
        local_ordered_features.append(feature)
    # print(ordered_features)

    # Form ordered_X dataframe
    ordered_X = pd.DataFrame(np.reshape(ordered_values, (1,-1)), columns=ordered_features_with_category)
    # print(ordered_X.dtypes)
    # print(ordered_X)

    # Get model ready X
    model_ready_X = ordered_X.reindex(columns=model_features, fill_value=0)
    # print(model_ready_X)


    if (ctx.triggered_id["type"] == "click-features") and (np.any(n_clicks)):
        print("doing click feature stuff")

        if is_classification:
            if target_map[1] == 1:
                ytitle = f"{target} (prob)"
            else:
                ytitle = f"{target_map[1]} (prob)"
        else:
            ytitle = target

        if additional_feature:
            if df[selected_feature].dtype == object and df[additional_feature].dtype == object:
                pass
            elif df[selected_feature].dtype == object and df[additional_feature].dtype != object:
                prediction_fig = get_prediction_graph_of_numerical_feature_by_category(df, additional_feature, selected_feature, contributions_df, max_points=5000, max_categories=5, frac=0.1, ytitle=ytitle)
                contrib_fig = get_contrib_graph_of_numerical_feature_by_category(df, additional_feature, selected_feature, contributions_df, max_points=5000, max_categories=5, ytitle="Contribution")
            elif df[selected_feature].dtype != object and df[additional_feature].dtype == object:
                prediction_fig = get_prediction_graph_of_numerical_feature_by_category(df, selected_feature, additional_feature, contributions_df, max_points=5000, max_categories=5, frac=0.1, ytitle=ytitle)
                contrib_fig = get_contrib_graph_of_numerical_feature_by_category(df, selected_feature, additional_feature, contributions_df, max_points=5000, max_categories=5, ytitle="Contribution")
            else:
                prediction_fig = get_prediction_graph_of_numerical_feature_by_number(df, selected_feature, additional_feature, contributions_df, max_points=5000, color_title=ytitle)
                contrib_fig = get_contrib_graph_of_numerical_feature_by_number(df, selected_feature, additional_feature, contributions_df, max_points=5000, color_title="Contrib")
        else:
            if df[selected_feature].dtype == object:
                prediction_fig = get_prediction_graph_of_categorical_feature(df, selected_feature, contributions_df, max_points=5000, max_categories=10, ytitle=ytitle)
                contrib_fig = get_contrib_graph_of_categorical_feature(df, selected_feature, contributions_df, max_points=5000, max_categories=10, ytitle="Contribution")
            else:
                prediction_fig = get_prediction_graph_of_numerical_feature(df, selected_feature, contributions_df, max_points=5000, frac=0.1, ytitle=ytitle)
                contrib_fig = get_contrib_graph_of_numerical_feature(df, selected_feature, contributions_df, max_points=5000, ytitle="Contribution")
        
        return [dash.no_update]*len(positive_contributions_div), [dash.no_update]*len(negative_contributions_div), dash.no_update, selected_feature, None, [dash.no_update]*len(ordered_values), prediction_fig, contrib_fig




    # Get prediction contributions
    # print(model_ready_X.dtypes)
    contribs = model.get_booster().predict(xgb.DMatrix(model_ready_X), pred_contribs=True)
    # contribs = model.predict(xgb.DMatrix(model_ready_X), pred_contribs=True)

    bias = contribs[0, -1]
    contribs = contribs[:, :-1]
    
    # Store prediction contributions in a dataframe
    contrib_df = pd.DataFrame(contribs, columns=model_ready_X.columns)
    # print(contrib_df)

    # Aggregate SHAP values for categorical features
    contrib_df = contrib_df.groupby(lambda x: x.split('?')[0], axis=1).sum()
    
    # Reorder the dataframe columns to match the web app and store into array
    ordered_contribs = np.asarray(contrib_df[local_ordered_features])[0]



    # Get feature locations for positive contribution div and negative contributions div
    # This step needs to be done because update-x-values and positive-contrib-div and negative-contrib-div do not have same order of features
    positive_elements = ctx.states_list[0]
    negative_elements = ctx.states_list[1]
    positive_feature_index_mapping = {}
    negative_feature_index_mapping = {}
    for j in range(0, len(positive_elements)):
        positive_feature_index_mapping[positive_elements[j]["id"]["index"]] = j
        negative_feature_index_mapping[negative_elements[j]["id"]["index"]] = j



    # Update contribution bars and labels
    max_width = 15# max width of container
    for i in range(0, len(ordered_contribs)):
        pos_feature_index = positive_feature_index_mapping[local_ordered_features[i]]
        neg_feature_index = negative_feature_index_mapping[local_ordered_features[i]]
        if ordered_contribs[i] >= 0:
            new_contrib = ordered_contribs[i]
            new_width = (new_contrib / max_contrib) * max_width

            positive_contributions_div[pos_feature_index][0]["props"]["style"]["width"] = f"{new_width}vw"
            positive_contributions_div[pos_feature_index][1]["props"]["children"] = f"+{new_contrib:.{sig_figs}f}"
            positive_contributions_div[pos_feature_index][1]["props"]["style"]["left"] = f"{new_width}vw"

            negative_contributions_div[neg_feature_index][0]["props"]["style"]["width"] = "0vw"
            negative_contributions_div[neg_feature_index][1]["props"]["children"] = ""
            negative_contributions_div[neg_feature_index][1]["props"]["style"]["right"] = "0"
        else:
            new_contrib = -ordered_contribs[i]
            new_width = (new_contrib / max_contrib) * max_width

            positive_contributions_div[pos_feature_index][0]["props"]["style"]["width"] = "0vw"
            positive_contributions_div[pos_feature_index][1]["props"]["children"] = ""
            positive_contributions_div[pos_feature_index][1]["props"]["style"]["left"] = "0"

            negative_contributions_div[neg_feature_index][0]["props"]["style"]["width"] = f"{new_width}vw"
            negative_contributions_div[neg_feature_index][1]["props"]["children"] = f"-{new_contrib:.{sig_figs}f}"
            negative_contributions_div[neg_feature_index][1]["props"]["style"]["right"] = f"{new_width}vw"


    # Get prediction
    if is_classification:
        tot_contribs = ordered_contribs.sum() + bias
        prediction = (1 / (1 + np.exp(-tot_contribs)))
        prediction = calibrate_probability(prediction)*100
        if target_map[1] == 1:
            class1_display = f"{target} (prob)"
        else:
            class1_display = f"{target_map[1]} (prob)"
        # prediction_div = [
        #     html.Div([
        #         html.Div([
        #             html.Span(f"{prediction:.{1}f}", style={"border":"0px solid","lineHeight":"1"}, title=f"{prediction}%"), 
        #             html.Span("%",style={"border":"0px solid","fontSize":"2.5vh","marginBottom":"1vh"}),
        #         ], style={"width":"100%","border":"0px solid","marginTop":"1vh"}),
                
        #         html.Div(class1_display, style={"border":"0px solid","fontSize":"2vh","lineHeight":"1","paddingBottom":"1vh"}),
        #     ], style={"position":"relative","width":"100%","border":"0px solid","textAlign":"center","color":"#333333"})
        # ]
        prediction_div = f"{class1_display} : {prediction:.{1}f}"
    else:
        prediction = ordered_contribs.sum() + bias
        # prediction_div = [
        #     html.Div([
        #         html.Div([
        #             html.Span(f"{prediction:.{sig_figs}f}", style={"border":"0px solid","lineHeight":"1"}, title=str(prediction)), 
        #         ], style={"width":"100%","border":"0px solid","marginTop":"1vh"}),
                
        #         html.Div(target, style={"border":"0px solid","fontSize":"2vh","lineHeight":"1","paddingBottom":"1vh"}),
        #     ], style={"position":"relative","width":"100%","border":"0px solid","textAlign":"center","color":"#333333"}),
        # ]
        prediction_div = f"{target} : {prediction:.{sig_figs}f}"


    print("end prediction")
    if len(synced_values) > 0:
        return positive_contributions_div, negative_contributions_div, prediction_div, dash.no_update, dash.no_update, synced_values, dash.no_update, dash.no_update
    else:
        return positive_contributions_div, negative_contributions_div, prediction_div, dash.no_update, dash.no_update, [dash.no_update]*len(ordered_values), dash.no_update, dash.no_update
    # return positive_contributions_div, negative_contributions_div, prediction_div, str(prediction), False, dash.no_update
    # return positive_contributions_div, negative_contributions_div, f"{prediction:.{sig_figs}f}", str(prediction), False, dash.no_update



@app.callback(
    dash.dependencies.Output(component_id="contrib-graph", component_property='figure', allow_duplicate=True),
    dash.dependencies.Input(component_id="add-another-feature-dropdown", component_property='value'),
    dash.dependencies.State(component_id={"type":"update-x-values", "index":ALL}, component_property='value'),
    dash.dependencies.State(component_id="chart-title-feature", component_property='children'),
    # dash.dependencies.Input(component_id={"type":"click-features", "index":ALL}, component_property='n_clicks'),
    # dash.dependencies.State(component_id={"type":"positive-contributions-div", "index":ALL}, component_property='children'),
    # dash.dependencies.State(component_id={"type":"negative-contributions-div", "index":ALL}, component_property='children'),
    prevent_initial_call=True,
)
def graph_3D(additional_feature, ordered_values, selected_feature):
    print("inside graph_3D")
    # print(ctx.triggered_id)
    if additional_feature == None:
        return dash.no_update

    # selected_feature = ctx.triggered_id["index"]
    # selected_feature = "LAM to JJ Wait Time (Days)"
    # print(ordered_values)
    # value = ctx.triggered[0]["value"]


    # Get all features...............Consider setting an id to the feature div, and use it's children as a state. No need to loop
    input_info = ctx.states_list[0]# input_info = ctx.inputs_list[0]
    local_ordered_features = []
    ordered_features_with_category = []
    for i in range(0, len(input_info)):
        # feature = input_info[i]["id"]["index"].split("-")[1]
        feature = input_info[i]["id"]["index"]
        if feature in categorical_features:
            ordered_features_with_category.append(feature + "?" + ordered_values[i])
            ordered_values[i] = 1
        else:
            ordered_features_with_category.append(feature)
        local_ordered_features.append(feature)
    # print(ordered_features)

    # Form ordered_X dataframe
    ordered_X = pd.DataFrame(np.reshape(ordered_values, (1,-1)), columns=ordered_features_with_category)
    # print(ordered_X.dtypes)
    # print(ordered_X)

    # Get model ready X
    model_ready_X = ordered_X.reindex(columns=model_features, fill_value=0)
    # print(model_ready_X)


    if True:
        print("doing click feature stuff")
        feature_values = pd.concat([df], axis=1).loc[:, [selected_feature, additional_feature]].dropna().drop_duplicates()
        # feature_values = df[[selected_feature, additional_feature]].dropna().drop_duplicates()
        feature_values_w_dummies = pd.get_dummies(feature_values, prefix_sep="?")

        # if len(feature_values) > 1000:
        #     feature_values = np.linspace(start=feature_values[0], stop=feature_values[-1], num=1000)
        model_ready_X = model_ready_X.loc[model_ready_X.index.repeat(len(feature_values))].reset_index(drop=True)
        # model_ready_X[[selected_feature, additional_feature]] = feature_values
        model_ready_X[feature_values_w_dummies.columns] = feature_values_w_dummies.to_numpy().tolist()

        if is_classification:
            predicted_vals = model.predict_proba(model_ready_X)[:,1]
            # predicted_vals = model.predict(xgb.DMatrix(model_ready_X))
            predicted_vals = calibrate_probability(predicted_vals)
            if target_map[1] == 1:
                y_display = f"{target} (prob)"
            else:
                y_display = f"{target_map[1]} (prob)"
        else:
            predicted_vals = model.predict(model_ready_X)
            # predicted_vals = model.predict(xgb.DMatrix(model_ready_X))
            y_display = f"{target}*"

        if (feature_values[selected_feature].dtype != object) and (feature_values[additional_feature].dtype != object):
            contrib_3D_fig = get_scatter_color_map(x=feature_values[selected_feature], y=feature_values[additional_feature], color=predicted_vals, x_title=selected_feature, y_title=additional_feature, color_title=y_display, title="")
        elif (feature_values[selected_feature].dtype != object) and (feature_values[additional_feature].dtype == object):
            contrib_3D_fig = get_scatter_color_map(x=feature_values[selected_feature], y=predicted_vals, color=feature_values[additional_feature], x_title=selected_feature, y_title=y_display, color_title=additional_feature, title="")
        elif (feature_values[selected_feature].dtype == object) and (feature_values[additional_feature].dtype != object):
            contrib_3D_fig = get_scatter_color_map(x=feature_values[additional_feature], y=predicted_vals, color=feature_values[selected_feature], x_title=additional_feature, y_title=y_display, color_title=selected_feature, title="")
        else:
            contrib_3D_fig = get_scatter_color_map(x=feature_values[selected_feature], y=feature_values[additional_feature], color=predicted_vals, x_title=selected_feature, y_title=additional_feature, color_title=y_display, title="")
        



        # feature_index = np.where(model_ready_X.columns == selected_feature)[0][0]
        # contribs = model.get_booster().predict(xgb.DMatrix(model_ready_X), pred_contribs=True)
        # contrib_fig = get_graph(x=feature_values, y=contribs[:,feature_index], x_title=selected_feature, y_title="Contribution", title=f"{selected_feature} Contributions")
        return contrib_3D_fig
        



    



if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)