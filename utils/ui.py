# utils/ui.py
"""
UI helpers for Dash applications.
Includes functions for creating overlays, banners, and sliders.
"""

from dash import html, dcc

_OVERLAY_BASE = {
    "position": "absolute",
    "left": "50%",
    "transform": "translateX(-50%)",
    "zIndex": 1000,
    "borderRadius": "12px",
    "backdropFilter": "blur(4px)",
    "WebkitBackdropFilter": "blur(4px)",
}

def overlay_banner(text: str, id_: str, visible=True):
    """
    Creates a banner overlay with the specified text.
    """
    style = {
        **_OVERLAY_BASE,
        "top": "5%",                     # 5 % Luft nach oben
        "left": "50%",
        "transform": "translateX(-50%)",
        "zIndex": 1000,

        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "gap": "0.75rem",

        "minWidth": "260px",
        "width": "40%",
        "padding": "0.75rem 1.25rem",
        "background": "rgba(0,0,0,.55)",
        "backdropFilter": "blur(4px)",
        "color": "#fff",
        "fontSize": "clamp(16px,1.25vw,24px)",
        "fontWeight": 600,
        "borderRadius": "12px",
        "boxShadow": "0 2px 6px rgba(0,0,0,.25)",

        "display": "block" if visible else "none",
    }
    return html.Div(text, id=id_, style=style, className="overlay-banner")



def yesno_banner(id_: str, question: str):
    """
    Creates a yes/no confirmation banner overlay.
    Args:
        id_: Unique ID for the banner
        question: Question to display in the banner 
    Returns:
        HTML Div component with the banner
    """
    style = {
        **_OVERLAY_BASE,
        "top": "calc(5% - 12px)",      
        "left": "50%",     

        "padding": "14px 22px",
        "background": "rgba(0,0,0,0.60)",
        "color": "#fff",
        "display": "none",
    }
    return html.Div(
        [
            html.Span(question, style={"marginRight": "12px"}),
            html.Button("Yes", id=f"{id_}-yes",   n_clicks=0, className="overlay-btn"),
            html.Button("No",  id=f"{id_}-no",    n_clicks=0, className="overlay-btn", style={"marginLeft": "8px"}),
        ],
        id=id_,
        style=style,
    )


def slider_overlay(id_slider="fire-radius-slider", id_container="slider-overlay"):
    """
    Creates a slider overlay for adjusting the fire radius.
    """
    style = {
        **_OVERLAY_BASE,
        "left": "0",           
        "transform": "none",
        "width": "calc(100% - 2rem)",  
        "bottom": "24px",
        "padding": "16px 24px",
        "background": "rgba(255,255,255,0.9)",
        "display": "none",
    }
    return html.Div(
        dcc.Slider(
            id=id_slider,
            min=0, max=100, step=1, value=0,
            marks={i: f"{i} m" for i in range(0, 110, 10)},
            updatemode="drag",
        ),
        id=id_container,
        style=style,
        className="slider-overlay",
    )
