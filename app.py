"""
FireGPT - Fire Emergency Response Planning Application

A Dash web application that provides AI-assisted fire emergency response planning.
The app allows users to:
- Place a fire location on a map
- Adjust the fire size
- Get AI-generated recommendations for response strategies
- Visualize fire station routes and helicopter assistance
- Chat with an AI assistant about fire response tactics

The application integrates mapping, routing, and RAG-based AI to create
a comprehensive fire response planning tool.
"""
import sys
import pathlib
from dotenv import load_dotenv
import dash
from dash import html, dcc, Input, Output, State
import dash_leaflet as dl
from itertools import cycle
from math import cos, radians, log2, floor
from utils.ui import overlay_banner, slider_overlay
from dash.exceptions import PreventUpdate
from utils.icons import get_icon
from utils.routing import osrm_route, COLORS
from utils.extract_metadata import diagnostics
from utils.stations import sorted_stations        
from geopy.distance import geodesic  
from utils.helpers import gdf_to_records, strip_fire_graphics, serialize_history, deserialize_history, parse_rag_recommendations, simplify_path, densify_path
from rag_core.interface import rag_first_turn, rag_chat_turn
import pprint
from utils.status_bus import set_status as set_rag_status, get_status

load_dotenv()                              
sys.path.append(str(pathlib.Path(__file__).parent))

RANK_LABELS = [
    "fastest", "second fastest", "third fastest",
    "fourth fastest", "fifth fastest", "sixth fastest"
]

app = dash.Dash(__name__, title="FireGPT", suppress_callback_exceptions=True)

app.layout = html.Div(
    className="container",
    children=[
        # Chat-Panel
        html.Div(
            id="chat-panel",
            className="panel chat-panel",
            style={"display": "none"},
            children=[
                html.H1("FireGPT Chat"),
                html.Div(
                    id="chat-log",
                    className="chat-log",  
                    children=[
                        html.Div(
                            id="chat-spinner",
                            className="spinner",
                            style={
                                "display": "none",
                                "position": "absolute",
                                "top": "40%",   
                                "left": "50%",
                                "transform": "translate(-50%, -50%)",
                                "zIndex": 10
                            }
                        ),
                        html.Div(
                            id="loading-status",
                            className="status-text",
                            children="", 
                            style={
                                "display": "none",
                                "position": "absolute",
                                "top": "55%",   
                                "left": "50%",
                                "transform": "translateX(-50%)",
                                "zIndex": 10
                            }
                        ),
                        html.Div(id="message-container", children=[])
                    ]
                ),

                html.Div(
                    [
                        dcc.Textarea(
                            id="user-msg",
                            className="chat-input",
                            placeholder="Type your question here..."
                        ),
                        html.Button("Send", id="send-btn", className="chat-btn", disabled=False),
                    ],
                    className="chat-input-group"
                ),
            ],
        ),
        html.Div(
            className="panel map-panel",
            children=[
                dl.Map(
                   id="map",                  
                   center=[51.2, 10.5],
                   zoom=6,
                    children=[
                        dl.TileLayer(),
                        dl.LayerGroup(id="route-layer"),   
                        dl.LayerGroup(id="marker-layer"), 
                    ],
                    style={"height": "100%", "width": "100%"},
                ),
                overlay_banner("Click on the map to place the fire location.", "instruction-banner", visible=True),
                slider_overlay(),                         
                html.Div(id="coord-display", className="coord-display"),
                html.Div(
                    "This region is not supported, we are currently limited to Germany.",
                    id="region-error-banner",
                    className="error-banner",
                    style={"display": "none"}
                ),
            ],
        ),
        html.Button(
            "Reset",
            id="reset-btn",
            className="reset-btn"
        ),

        dcc.Store(id="coord-store", data={"fire": None, "station": None}),
        dcc.Store(id="chat-store",  data=[]),
        dcc.Store(id="ui-store", data={
            "step": 1,
            "add_info": None,
            "fire_radius": 0.5,
            "slider_done": False    
        }),
        dcc.Store(id="diag-store", data={}),        
        dcc.Store(id="pending-msg-store", data=""),
        dcc.Interval(id="anim-tick", interval=200, n_intervals=0, disabled=True),
        dcc.Store(id="anim-data"),
        dcc.Interval(id="poll-status", interval=3_000, n_intervals=0),
        dcc.Interval(id="error-banner-timer", interval=5000, n_intervals=0, disabled=True),
    ],
)

@app.callback(
    Output("marker-layer",  "children", allow_duplicate=True),
    Output("route-layer",   "children", allow_duplicate=True),
    Output("coord-store",   "data", allow_duplicate=True),
    Output("coord-display", "children", allow_duplicate=True),
    Output("ui-store",      "data", allow_duplicate=True),
    Output("slider-overlay","style", allow_duplicate=True),
    Output("instruction-banner",  "children", allow_duplicate=True),
    Output("map",           "viewport", allow_duplicate=True),
    Output("region-error-banner", "style", allow_duplicate=True),  # New output
    Output("error-banner-timer", "disabled", allow_duplicate=True),  # New output
    Input("map", "clickData"),
    State("coord-store", "data"),
    State("ui-store",    "data"),
    State("marker-layer","children"),
    prevent_initial_call=True,
)
def on_map_click(clickData, coord_store, ui, markers_in):
    """
    Handle map clicks to place the fire location.
    
    Places a fire marker at the clicked location and updates the UI state
    to progress to the fire size adjustment step. Restricts clicks to Germany only.
    
    Args:
        clickData: Map click location data
        coord_store: Current coordinates data store
        ui: Current UI state data
        markers_in: Current markers on the map
        
    Returns:
        Updated markers, route layer, coordinate store, coordinate display,
        UI state, slider overlay style, instruction banner text, map viewport,
        region error banner style, and error timer state
    """
    if not clickData or ui["step"] >= 4:
        raise PreventUpdate
    print("Map click detected", flush=True)
    fire_lat = clickData["latlng"]["lat"]
    fire_lng = clickData["latlng"]["lng"]
    
    # Check if click is within Germany's boundaries
    is_in_germany = (47.3 <= fire_lat <= 55.1) and (5.9 <= fire_lng <= 15.0)
    
    if not is_in_germany:
        # Return defaults and show error banner
        return (
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            {"display": "block"}, False  # Show error banner and enable timer
        )
    
    # Original function logic for valid clicks within Germany
    coord_store["fire"] = (fire_lat, fire_lng)
    radius = ui["fire_radius"]
    base_markers = strip_fire_graphics(markers_in)

    new_fire = [
        dl.Marker(id="fire-marker",
                  position=[fire_lat, fire_lng],
                  icon=get_icon("fire")),
        dl.Circle(
            id="fire-radius-circle",
            center=[fire_lat, fire_lng],
            radius=radius,
            color="red", fillColor="red",
            fillOpacity=0.25, weight=2,
        ),
    ]

    markers_out = base_markers + new_fire
    next_step = 2 if ui["step"] == 1 else ui["step"]
    ui_out = {**ui, "step": next_step}
    slider_style = {"display": "block"}

    mpp = 1800 / (0.6*800)
    zoom = int(round(max(0, min(20,
                 log2(156543.03392 * cos(radians(fire_lat)) / mpp)))))
    view = {"center": [fire_lat, fire_lng], "zoom": zoom}

    banner_text = "Use the slider to adjust the fire size."

    return (
        markers_out, [], coord_store, "", ui_out, slider_style, banner_text, view,
        {"display": "none"}, True  # Hide error banner and disable timer
    )

@app.callback(
    Output("marker-layer",       "children", allow_duplicate=True),
    Output("slider-overlay",     "style",    allow_duplicate=True),
    Output("instruction-banner", "children", allow_duplicate=True),
    Output("ui-store",           "data",     allow_duplicate=True),
    Output("instruction-banner", "style",    allow_duplicate=True),
    Input("fire-radius-slider",  "value"),
    State("coord-store",  "data"),
    State("marker-layer", "children"),
    State("ui-store",     "data"),
    prevent_initial_call=True,
)
def adjust_radius(radius, coord_store, markers_in, ui):
    """
    Update the fire radius based on slider input.
    
    Adjusts the visual representation of the fire area on the map and
    updates the UI to allow for additional information input.
    
    Args:
        radius: The fire radius value from the slider
        coord_store: Current coordinates data store
        markers_in: Current markers on the map
        ui: Current UI state data
        
    Returns:
        Updated markers, slider overlay style, instruction banner text,
        UI state data, and instruction banner style
    """
    if ui["step"] not in (2, 3) or coord_store["fire"] is None:
        raise PreventUpdate

    lat, lng = coord_store["fire"]
    markers  = strip_fire_graphics(markers_in)   
    markers += [                                   
        dl.Marker(id="fire-marker",
                  position=[lat, lng],
                  icon=get_icon("fire")),
        dl.Circle(
            id="fire-radius-circle",
            center=[lat, lng],
            radius=radius,
            color="red", fillColor="red",
            fillOpacity=0.25, weight=2,
        ),
    ]

    ui_out = {**ui, "step": 3, "fire_radius": radius}
    instr_children = [
        "Add additional information? ",
        html.Button("Yes", id="addinfo-yes", n_clicks=0, className="overlay-btn"),
        html.Button("No",  id="addinfo-no",  n_clicks=0, className="overlay-btn",
                    style={"marginLeft": "8px"}),
    ]

    return markers, {"display": "block"}, instr_children, ui_out, dash.no_update

@app.callback(
    Output("chat-panel",         "style"),
    Output("chat-panel",         "className"),
    Output("instruction-banner", "style",     allow_duplicate=True),
    Output("ui-store",           "data",      allow_duplicate=True),
    Output("slider-overlay",     "style",     allow_duplicate=True),
    Output("chat-spinner",       "style",     allow_duplicate=True),
    Output("loading-status",     "style",     allow_duplicate=True),
    Output("loading-status",     "children",  allow_duplicate=True),
    Output("send-btn",           "disabled",  allow_duplicate=True),
    Output("send-btn",           "className", allow_duplicate=True),
    Input("addinfo-yes", "n_clicks"),
    Input("addinfo-no",  "n_clicks"),
    State("ui-store",     "data"),
    prevent_initial_call=True,
)
def handle_yes_no(yes_clicks, no_clicks, ui):
    """
    Handle user's choice about providing additional information.
    
    Shows the chat panel and updates UI components based on whether
    the user chooses to provide additional information.
    
    Args:
        yes_clicks: Number of clicks on the "Yes" button
        no_clicks: Number of clicks on the "No" button
        ui: Current UI state data
        
    Returns:
        Updated chat panel style and class, instruction banner style,
        UI state data, slider overlay style, chat spinner style,
        loading status style and text, and send button state
    """
    if (yes_clicks or 0) == 0 and (no_clicks or 0) == 0:
        raise PreventUpdate

    choice   = "yes" if (yes_clicks or 0) > (no_clicks or 0) else "no"
    ui_out   = {**ui, "step": 4, "add_info": choice}

    chat_style  = {"display": "flex", "flexDirection": "column"}
    chat_class  = "panel chat-panel visible"
    slider_hide = {"display": "none"} if choice == "no" else dash.no_update
    banner_hide = {"display": "none"}

    busy          = choice == "no"
    spinner_style = {"display": "block"} if busy else {"display": "none"}
    status_style  = {"display": "block"} if busy else {"display": "none"}
    status_text   = "Answer is being generated..." if busy else ""

    btn_disabled  = busy
    btn_class     = "chat-btn disabled" if busy else "chat-btn"

    return (chat_style, chat_class, banner_hide, ui_out, slider_hide,
            spinner_style, status_style, status_text,
            btn_disabled, btn_class)


@app.callback(
    Output("marker-layer", "children",     allow_duplicate=True),
    Output("route-layer",  "children",     allow_duplicate=True),
    Output("diag-store",   "data",         allow_duplicate=True),
    Output("map",          "viewport"),
    Output("ui-store",     "data",         allow_duplicate=True),   #  ← NEU
    Input("ui-store",      "data"),
    State("coord-store",   "data"),
    State("marker-layer",  "children"),
    prevent_initial_call=True,
)
def build_routes(ui, coord, markers):
    """
    Calculate and build routes from fire stations to the fire location.
    
    Triggered when the UI state progresses to step 4. Calculates driving routes
    from nearby fire stations to the fire location and retrieves additional
    diagnostic information about the fire site.
    
    Args:
        ui: Current UI state data
        coord: Coordinates data store containing fire location
        markers: Current markers on the map
        
    Returns:
        Updated markers, route lines, diagnostic data, map viewport, and UI state
    """
    if ui.get("step") != 4 or ui.get("routes_done"):
        raise PreventUpdate
    if ui.get("add_info") == "no":
        set_rag_status("Ground Information retrieved, routes are being calculated...")
    markers_out = list(markers) if markers and markers is not dash.no_update else []
    print("Calculating routes...", flush=True)
    fire_lat, fire_lng = coord["fire"]
    route_children = []
    station_info   = []

    lats, lons = [fire_lat], [fire_lng]

    stations = sorted_stations(fire_lng, fire_lat, n=5)
    for colour, row in zip(cycle(COLORS), stations.itertuples(index=False)):
        st_lat, st_lng = row.geometry.y, row.geometry.x
        name = getattr(row, "name", "Station")

        try:
            pts, dur_s, dist_m = osrm_route(st_lat, st_lng, fire_lat, fire_lng)

            offroad_m = geodesic((pts[-1][0], pts[-1][1]),
                                 (fire_lat, fire_lng)).meters
            station_info.append({
                "name":        name,
                "station_lat": st_lat,
                "station_lng": st_lng,
                "drive_sec":   dur_s,
                "drive_min":   round(dur_s / 60, 1),
                "drive_m":     dist_m,
                "offroad_m":   offroad_m,
                "offroad_min": round((offroad_m / 1.8) / 60,1),
                "total_time_min": round(((offroad_m / 1.8) / 60) + dur_s / 60, 1), 
            })
        except Exception:
            print("======================================OSRM Not Available!======================================")
            pass

        lats.append(st_lat)
        lons.append(st_lng)

    station_info_sorted = sorted(station_info, key=lambda s: s["total_time_min"])
    for idx, s in enumerate(station_info_sorted):
        label = RANK_LABELS[idx] if idx < len(RANK_LABELS) else f"{idx+1}th fastest"
        s["rank_label"] = label
    station_info = station_info_sorted
    diag = diagnostics(fire_lng, fire_lat, n_stations=5)
    diag["stations"]       = gdf_to_records(diag["stations"])
    diag["heli"]           = gdf_to_records(diag["heli"])
    diag["stations_route"] = station_info
    diag["fire_radius"] = ui.get("fire_radius")

    heli_list = diag.get("heli", [])
    if heli_list:
        base = heli_list[0]
        b_lat, b_lon = base["Latitude"], base["Longitude"]
        w_lat = diag["water"]["water_lat"]
        w_lon = diag["water"]["water_lon"]
        d_bw = geodesic((b_lat, b_lon), (w_lat, w_lon)).meters
        d_wf = geodesic((w_lat, w_lon), (fire_lat, fire_lng)).meters
        speed_ms = 250_000 / 3600
        diag["heli_route"] = {
            "dist_base_water_m":    d_bw,
            "time_base_water_s":    d_bw / speed_ms,
            "dist_water_fire_m":    d_wf,
            "time_water_fire_s":    d_wf / speed_ms,
            "total_time_min":       ((d_bw + d_wf) / speed_ms) / 60,
        }

        lats += [b_lat, w_lat]
        lons += [b_lon, w_lon]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    px_w, px_h = 800, 600
    lat_diff, lon_diff = max_lat - min_lat, max_lon - min_lon
    zoom_lon = log2((px_w * 360) / (256 * lon_diff)) if lon_diff > 0 else 18
    zoom_lat = log2((px_h * 360) / (256 * lat_diff)) if lat_diff > 0 else 18
    zoom = floor(min(zoom_lon, zoom_lat))
    zoom = max(0, min(18, zoom))

    chat_frac = 0.4
    map_frac  = 1 - chat_frac
    lon_span  = max_lon - min_lon
    lon_shift = (chat_frac/2) / map_frac * lon_span
    center_lon += lon_shift

    viewport = {"center": [center_lat, center_lon], "zoom": zoom}
    ui_next = {**ui, "routes_done": True}

    print("\n=== Diagnostics ===")
    pprint.pprint(diag, sort_dicts=False)
    return markers_out, [], diag, viewport,  ui_next


@app.callback(
    Output("message-container", "children",   allow_duplicate=True),
    Output("chat-store",        "data",       allow_duplicate=True),
    Output("marker-layer",      "children",   allow_duplicate=True),
    Output("route-layer",       "children",   allow_duplicate=True),
    Output("map",               "viewport",   allow_duplicate=True),
    Output("chat-spinner",      "style",      allow_duplicate=True),
    Output("loading-status",    "style",      allow_duplicate=True),
    Output("loading-status",    "children",   allow_duplicate=True),
    Output("send-btn",          "disabled",   allow_duplicate=True),
    Output("send-btn",          "className",  allow_duplicate=True),
    Output("anim-data",    "data",  allow_duplicate=True),          
    Output("anim-tick",    "disabled",  allow_duplicate=True),     
    Input("diag-store",         "data"),
    State("ui-store", "data"), 
    State("message-container",  "children"),
    State("coord-store",        "data"),
    State("marker-layer",       "children"),     
    prevent_initial_call=True,
)
def run_rag_after_routes(diag, ui, chatlog_prev,
                         coord_store, marker_layer_prev):
    """
    Generate initial AI response after routes are calculated.
    
    Processes diagnostic data through the RAG system to generate an initial
    response and recommendation for fire response. Creates route visualizations
    based on the AI recommendations.
    
    Args:
        diag: Diagnostic data about fire location and resources
        ui: Current UI state data
        chatlog_prev: Previous chat messages
        coord_store: Coordinates data store containing fire location
        marker_layer_prev: Current markers on the map
        
    Returns:
        Updated chat messages, chat store data, map markers, route lines,
        map viewport, animation data, animation state, spinner status,
        loading status, and send button state
    """
    if not diag or ui.get("add_info") != "no":
        raise PreventUpdate

    llm_answer, history = rag_first_turn("", diag)  

    selected_stations, heli_required = parse_rag_recommendations(
            llm_answer,
            diag["stations_route"]
    ) 
    print(llm_answer)
    print(selected_stations, heli_required)

    markers_out     = list(marker_layer_prev or [])
    route_children  = []
    colour_cycle    = cycle(COLORS)

    fire_lat, fire_lng = coord_store["fire"]
    lats, lons = [fire_lat], [fire_lng]

    truck_paths = []   

    for s in reversed(selected_stations):          
        pts, _, _ = osrm_route(s["station_lat"], s["station_lng"],
                            fire_lat, fire_lng)
        colour = next(colour_cycle)

        pts_simple = simplify_path(pts, step=5)  
        truck_paths.append(pts_simple)

        markers_out.append(
            dl.Marker(
                position=[s["station_lat"], s["station_lng"]],
                icon=get_icon("firestation"),
                children=dl.Tooltip(s["name"])
            )
        )
        route_children.append(
            dl.Polyline(positions=pts, color=colour, weight=4)
        )
        lats += [s["station_lat"]]
        lons += [s["station_lng"]]


    heli_path = None
    if heli_required and diag.get("heli"):
        base = diag["heli"][0]
        b_lat, b_lon = base["Latitude"], base["Longitude"]
        w_lat = diag["water"]["water_lat"]
        w_lon = diag["water"]["water_lon"]
        heli_raw  = [(b_lat, b_lon), (w_lat, w_lon), (fire_lat, fire_lng)]
        heli_path = densify_path(heli_raw, seg_points=60)


        markers_out += [
            dl.Marker(position=[b_lat, b_lon], icon=get_icon("hangar"),
                    children=dl.Tooltip(f"Heli-Base: {base['Standort']}")),
            dl.Marker(position=[w_lat, w_lon], icon=get_icon("water"),
                    children=dl.Tooltip("Water retrieval")),
        ]
        route_children += [
            dl.Polyline(positions=[[b_lat, b_lon], [w_lat, w_lon]],
                        color="black", weight=3, dashArray="5,10"),
            dl.Polyline(positions=[[w_lat, w_lon], [fire_lat, fire_lng]],
                        color="black", weight=3, dashArray="5,10"),
        ]
        lats += [b_lat, w_lat]
        lons += [b_lon, w_lon]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    px_w, px_h = 800, 600
    lat_diff, lon_diff = max_lat - min_lat, max_lon - min_lon
    zoom_lon = log2((px_w * 360) / (256 * lon_diff)) if lon_diff else 18
    zoom_lat = log2((px_h * 360) / (256 * lat_diff)) if lat_diff else 18
    zoom = floor(min(zoom_lon, zoom_lat))
    zoom = max(0, min(18, zoom))
    viewport = {"center": [center_lat, center_lon], "zoom": zoom}
    # ---------------------------------------------------------------


    chatlog_prev = chatlog_prev or []
    chatlog_prev.append(
        html.Div(
            [html.Span("👨‍🚒", className="role"),
            dcc.Markdown(llm_answer, link_target="_blank", style={'white-space': 'pre-wrap'})],
        className="bot-msg",
        )
    )

    hidden = {"display": "none"}
    history_serial = serialize_history(history)
    
    anim_payload = {
        "trucks": [
            {"id": s["name"], "path": p}     
            for s, p in zip(selected_stations, truck_paths)
        ],
        "heli": {"path": heli_path} if heli_path else None
    }

    set_rag_status("")

    return (
        chatlog_prev, history_serial,
        markers_out, route_children, viewport,
        hidden, hidden, "",               
        False, "chat-btn",
        anim_payload,                     
        False                              
    )



@app.callback(
    Output("message-container", "children",   allow_duplicate=True),
    Output("pending-msg-store", "data",       allow_duplicate=True),
    Output("user-msg",          "value"),
    Output("chat-spinner",      "style",      allow_duplicate=True),
    Output("loading-status",    "style",      allow_duplicate=True),
    Output("send-btn",          "disabled",   allow_duplicate=True),
    Output("send-btn",          "className",  allow_duplicate=True),
    Input("send-btn",           "n_clicks"),
    State("user-msg",           "value"),
    State("message-container",  "children"),
    State("ui-store",           "data"),     
    prevent_initial_call=True,
)
def queue_user_msg(_, msg, chatlog_prev, ui):   
    """
    Add user message to the chat and prepare for AI response.
    
    Adds the user's message to the chat log and displays a spinner while
    waiting for the AI to respond.
    
    Args:
        _: Number of clicks on send button (not used directly)
        msg: User's message text
        chatlog_prev: Previous chat messages
        ui: Current UI state data
        
    Returns:
        Updated chat messages, pending message store, cleared input field,
        spinner and loading status style, and send button state
    """
    if not msg:
        raise PreventUpdate

    chatlog_prev = chatlog_prev or []

    user_chunk    = html.Div([html.Span("👤", className="role"), msg],
                             className="user-msg")
    spinner_chunk = html.Div(                        
        [
            html.Div(className="spinner-inline"),    
            html.Div(                                
                "",
                className="status-inline"
            ),
        ],
        id={"type": "inline-spinner", "idx": len(chatlog_prev)},
        className="spinner-inline-wrapper"
    )

    chatlog_now = chatlog_prev + [user_chunk, spinner_chunk]

    show_big = ui.get("add_info") == "no" and not ui.get("routes_done")
    big_spinner = {"display": "block"} if show_big else {"display": "none"}

    return (chatlog_now, msg, "",
            big_spinner, big_spinner,
            True, "chat-btn disabled")




@app.callback(
    Output("message-container", "children",   allow_duplicate=True),
    Output("chat-store",        "data",       allow_duplicate=True),
    Output("marker-layer",      "children", allow_duplicate=True),
    Output("route-layer",       "children", allow_duplicate=True),
    Output("map",               "viewport", allow_duplicate=True),
    Output("anim-data",         "data",     allow_duplicate=True),
    Output("anim-tick",         "disabled", allow_duplicate=True),
    Output("chat-spinner",      "style",      allow_duplicate=True),
    Output("loading-status",    "style",      allow_duplicate=True),
    Output("send-btn",          "disabled",   allow_duplicate=True),
    Output("send-btn",          "className",  allow_duplicate=True),
    Output("pending-msg-store", "data",allow_duplicate=True),
    Input("pending-msg-store",  "data"),
    State("chat-store",         "data"),
    State("diag-store",         "data"),
    State("message-container",  "children"),
    State("coord-store",  "data"),         
    State("marker-layer", "children"),   
    State("ui-store",     "data"), 
    prevent_initial_call=True,
)
def run_rag_and_reply(pending_msg, history_json, diag, chatlog_prev,
                      coord_store, marker_layer_prev, ui):
    """
    Process user message through RAG and generate AI response.
    
    Takes the user's message, processes it through the RAG system, and
    generates an AI response. Updates chat messages and visualization
    if necessary based on the response.
    
    Args:
        pending_msg: User's message to process
        history_json: Chat history in serialized format
        diag: Diagnostic data about fire location and resources
        chatlog_prev: Previous chat messages
        coord_store: Coordinates data store containing fire location
        marker_layer_prev: Current markers on the map
        ui: Current UI state data
        
    Returns:
        Updated chat messages, chat history, map markers, route lines,
        map viewport, animation data, animation state, spinner status,
        loading status, send button state, and cleared pending message
    """
    if not pending_msg:
        raise PreventUpdate

    for el in reversed(chatlog_prev):
        if not isinstance(el, dict):
            continue                     

        el_props = el.get("props", {})
        el_id    = el_props.get("id")

        if isinstance(el_id, dict) and el_id.get("type") == "inline-spinner":
            style = el_props.get("style", {})
            style.update({"display": "none"})
            el_props["style"] = style
            el["props"] = el_props
            break

    history = deserialize_history(history_json) if history_json else []

    if not history:                                     
        set_rag_status("Ground Information retrieved, routes are being calculated...")
        print("Running RAG first turn with info", flush=True)
        answer, history = rag_first_turn(pending_msg, diag)
        selected_stations, heli_required = parse_rag_recommendations(
            answer,
            diag["stations_route"]
        ) 
        print(answer)
        print(selected_stations, heli_required)

        markers_out     = list(marker_layer_prev or [])
        route_children  = []
        colour_cycle    = cycle(COLORS)

        fire_lat, fire_lng = coord_store["fire"]
        lats, lons = [fire_lat], [fire_lng]

        truck_paths = []   

        for s in reversed(selected_stations):          
            pts, _, _ = osrm_route(s["station_lat"], s["station_lng"],
                                fire_lat, fire_lng)
            colour = next(colour_cycle)

            pts_simple = simplify_path(pts, step=5)  
            truck_paths.append(pts_simple)

            markers_out.append(
                dl.Marker(
                    position=[s["station_lat"], s["station_lng"]],
                    icon=get_icon("firestation"),
                    children=dl.Tooltip(s["name"])
                )
            )
            route_children.append(
                dl.Polyline(positions=pts, color=colour, weight=4)
            )
            lats += [s["station_lat"]]
            lons += [s["station_lng"]]


        heli_path = None
        if heli_required and diag.get("heli"):
            base = diag["heli"][0]
            b_lat, b_lon = base["Latitude"], base["Longitude"]
            w_lat = diag["water"]["water_lat"]
            w_lon = diag["water"]["water_lon"]
            heli_raw  = [(b_lat, b_lon), (w_lat, w_lon), (fire_lat, fire_lng)]
            heli_path = densify_path(heli_raw, seg_points=60)


            markers_out += [
                dl.Marker(position=[b_lat, b_lon], icon=get_icon("hangar"),
                        children=dl.Tooltip(f"Heli-Base: {base['Standort']}")),
                dl.Marker(position=[w_lat, w_lon], icon=get_icon("water"),
                        children=dl.Tooltip("Water retrieval")),
            ]
            route_children += [
                dl.Polyline(positions=[[b_lat, b_lon], [w_lat, w_lon]],
                            color="black", weight=3, dashArray="5,10"),
                dl.Polyline(positions=[[w_lat, w_lon], [fire_lat, fire_lng]],
                            color="black", weight=3, dashArray="5,10"),
            ]
            lats += [b_lat, w_lat]
            lons += [b_lon, w_lon]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        px_w, px_h = 800, 600
        lat_diff, lon_diff = max_lat - min_lat, max_lon - min_lon
        zoom_lon = log2((px_w * 360) / (256 * lon_diff)) if lon_diff else 18
        zoom_lat = log2((px_h * 360) / (256 * lat_diff)) if lat_diff else 18
        zoom = floor(min(zoom_lon, zoom_lat))
        zoom = max(0, min(18, zoom))
        viewport = {"center": [center_lat, center_lon], "zoom": zoom}
        anim_payload = {
            "trucks": [
                {"id": s["name"], "path": p}
                for s, p in zip(selected_stations, truck_paths)
            ],
            "heli": {"path": heli_path} if heli_path else None
        }
    else:                                                
        answer, history = rag_chat_turn(pending_msg, diag, history)
        print("Running RAG answer", flush=True)
        markers_out = dash.no_update
        route_children = dash.no_update
        viewport = dash.no_update
        anim_payload = dash.no_update


    history_serial = serialize_history(history)      


    bot_chunk = html.Div(
        [html.Span("👨‍🚒", className="role"),
            dcc.Markdown(answer, link_target="_blank", style={'white-space': 'pre-wrap'})],
        className="bot-msg",
    )

    set_rag_status("")

    for i in range(len(chatlog_prev) - 1, -1, -1):
        el_id = chatlog_prev[i].get("props", {}).get("id", {})
        if isinstance(el_id, dict) and el_id.get("type") == "inline-spinner":
            chatlog_prev.pop(i)
            break
    chatlog_final = chatlog_prev + [bot_chunk]

    hidden = {"display": "none"}

    set_rag_status("")



    return (chatlog_final, history_serial,
            markers_out, route_children, viewport,
            anim_payload, False,
            hidden, hidden,
            False, "chat-btn",
            "")



@app.callback(
    Output("marker-layer", "children", allow_duplicate=True),
    Output("route-layer",  "children", allow_duplicate=True),
    Output("coord-store",  "data",     allow_duplicate=True),
    Output("chat-store",   "data",     allow_duplicate=True),
    Output("ui-store",     "data",     allow_duplicate=True),
    Output("chat-panel",   "className", allow_duplicate=True),
    Output("chat-panel",   "style",     allow_duplicate=True),
    Output("slider-overlay","style",    allow_duplicate=True),
    Output("instruction-banner", "style"),
    Output("instruction-banner", "children", allow_duplicate=True),
    Output("message-container", "children", allow_duplicate=True),
    Output("pending-msg-store", "data",     allow_duplicate=True),
    Output("chat-spinner",      "style",    allow_duplicate=True),
    Output("loading-status",    "style",    allow_duplicate=True),
    Output("loading-status",    "children", allow_duplicate=True),
    Output("send-btn",          "disabled", allow_duplicate=True),
    Output("send-btn",          "className",allow_duplicate=True),
    Output("anim-data",         "data",     allow_duplicate=True),
    Output("anim-tick",         "disabled", allow_duplicate=True),
    Output("map",           "viewport",allow_duplicate=True),
    Output("anim-tick",         "n_intervals", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_app(n_clicks):
    """
    Reset the application to its initial state.
    
    Clears all markers, routes, chat history, and UI state when the
    reset button is clicked.
    
    Args:
        n_clicks: Number of clicks on reset button
        
    Returns:
        Reset values for all application components and state stores
    """
    if not n_clicks:
        raise PreventUpdate

    init_ui = {
        "step": 1,
        "add_info": None,
        "fire_radius": 0.5,
        "slider_done": False,
    }

    default_banner_style = overlay_banner("", "instruction-banner", visible=True).style
    default_banner_style["display"] = "block"

    return (
        [], [],                                   # marker-/route-layer
        {"fire": None, "station": None},          # coord-store
        [],                                       # chat-store
        init_ui,                                  # ui-store
        "panel chat-panel", {"display":"none"},   # Chat-Panel aus
        {"display":"none"},                       # Slider aus
        default_banner_style,
        "Click on the map to place the fire location.",

        [],       # message-container leer
        "",       # pending-msg-store
        {"display":"none"},  # chat-spinner
        {"display":"none"},  # loading-status style
        "",                  # loading-status text
        False, "chat-btn",   # Send-Button 
        {},                  # anim-data geleert
        True,                # anim-tick disabled
        {"center": [51.2, 10.5], "zoom": 6}, # Viewport 
        0                    # anim-tick n_intervals reset
    )

@app.callback(
    Output("marker-layer", "children", allow_duplicate=True),
    Input("anim-tick", "n_intervals"),
    State("anim-data", "data"),
    State("marker-layer", "children"),
    prevent_initial_call=True,
)
def animate(n, data, static_markers):
    """
    Animate moving markers (fire trucks and helicopter) on the map.
    
    Updates the positions of moving markers on each animation tick to
    create the appearance of movement along routes.
    
    Args:
        n: Current animation interval count
        data: Animation data containing paths for trucks and helicopter
        static_markers: Static markers currently on the map
        
    Returns:
        Updated markers including both static and moving elements
    """
    if not data or not data.get("trucks"):        
        raise PreventUpdate

    max_len = max(len(t["path"]) for t in data["trucks"])
    f_truck = n % max_len

    moving = []
    for t in data["trucks"]:
        lat, lon = t["path"][f_truck % len(t["path"])]
        moving.append(
            dl.Marker(id=f"truck-{t['id']}",
                      position=[lat, lon],
                      icon=get_icon("firetruck"))
        )

    if data.get("heli") and data["heli"]["path"]:
        f_heli = n % len(data["heli"]["path"])
        h_lat, h_lon = data["heli"]["path"][f_heli]
        moving.append(
            dl.Marker(id="heli-moving",
                      position=[h_lat, h_lon],
                      icon=get_icon("heli"))
        )

    stat = [
        m for m in static_markers
        if m["props"].get("id") not in
           {"heli-moving", *[f"truck-{t['id']}" for t in data["trucks"]]}
    ]
    return stat + moving

@app.callback(
    Output("loading-status", "children"),
    Output("loading-status", "style"),
    Input("poll-status", "n_intervals"),
)
def update_loading_status(_):
    """
    Update the loading status message.
    
    Polls the current status message and updates the loading status display.
    
    Args:
        _: Interval tick count (not used directly)
        
    Returns:
        Current status message and display style
    """
    msg = get_status()
    style = {"display": "block"} if msg else {"display": "none"}
    return msg, style

@app.callback(
    Output("region-error-banner", "style", allow_duplicate=True),
    Output("error-banner-timer", "disabled", allow_duplicate=True),
    Input("error-banner-timer", "n_intervals"),
    State("region-error-banner", "style"),
    prevent_initial_call=True
)
def hide_error_banner(n_intervals, banner_style):
    """
    Hide the error banner after 5 seconds.
    
    Args:
        n_intervals: Number of intervals elapsed
        banner_style: Current banner style
        
    Returns:
        Updated banner style and timer state
    """
    if n_intervals >= 1:
        return {"display": "none"}, True
    
    return dash.no_update, dash.no_update

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
