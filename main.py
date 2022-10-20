# Run this app with `python main.py` and
# visit http://localhost:8050/ in your web browser.

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from src.inference import infer, load_model

import cv2
import warnings
import plotly.express as px
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(
    __name__,
    title="Wood Defect Detection",
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # external_stylesheets
)
app.config.suppress_callback_exceptions = True
# server = app.server


# Load Configs and Model
args, model = load_model()
# Choose Image
images_list = [
    "test_color_001.png",
    "test_combined_005.png",
    "test_good_000.png",
    "test_hole_001.png",
    "test_liquid_008.png",
    "test_scratch_004.png",
]


app.layout = html.Div(
    [
        html.Div(
            children=[
                html.P(
                    children="🕵️",
                    className="header-emoji",
                ),
                html.H1("Wood Defect Detection", className="header-title"),
                html.P(
                    children="Automated visual evaluation AI system to control wood quality.",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(children="Model:", className="menu-title"),
                dcc.Dropdown(
                    options=["PaDiM"],
                    value="PaDiM",
                    clearable=False,
                    id="model-name",
                ),
                html.Div(children="Choose an image:", className="menu-title"),
                dcc.Dropdown(
                    id="image-dropdown",
                    options=[{"label": i, "value": i} for i in images_list],
                    # initially display the first entry in the list
                    value=images_list[0],
                    clearable=False,
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(id="image"),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    Output("image", "children"),
    Input("image-dropdown", "value"),
    Input("model-name", "value"),
)
def image_inference(image_name, model_name):
    model_name = model_name
    image = "assets/images/wood/" + image_name

    original_image = cv2.imread(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    fig_original = px.imshow(original_image, title="Original Image")

    output, score = infer(model, image)
    fig_output = px.imshow(output, title="Anomaly Detected")

    layout = html.Div(
        [
            html.Div(
                children=[
                    dcc.Graph(
                        id="original-image",
                        figure=fig_original,
                        style={
                            "width": "60vh",
                            "height": "60vh",
                            "display": "inline-block",
                        },
                    ),
                    dcc.Graph(
                        id="detect-image",
                        figure=fig_output,
                        style={
                            "width": "60vh",
                            "height": "60vh",
                            "display": "inline-block",
                        },
                    ),
                ],
                className="row",
            ),
            dcc.Markdown(
                f"""
               Model Confidence Score: {round(score*100, 1)}%
            """,
                style={
                    "text-align": "center",
                },
                className="conf-score",
            ),
        ]
    )
    return layout


if __name__ == "__main__":
    app.run_server(debug=True)
