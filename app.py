from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import csv
from functools import wraps
from datetime import timedelta
import pandas as pd
import matplotlib
import re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uuid import uuid4
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import seaborn as sns


from torch_geometric.nn import GATConv


class GAT_DGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout):
        super(GAT_DGNN, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=heads, dropout=dropout)
        self.conv2 = GATConv(32 * heads, 16, heads=4, dropout=dropout)
        self.conv3 = GATConv(16 * 4, out_channels, heads=1, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(32 * heads)
        self.bn2 = torch.nn.BatchNorm1d(16 * 4)

    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


best_hyperparameters = {  # Initialize model with the best hyperparameters
    "lr": 0.00370187450731649,
    "dropout": 0.49984325307071925,
    "heads": 11,
    "k_dynamic": 6,
}
input_features = 29
num_classes = 2
model = GAT_DGNN(
    in_channels=input_features,
    out_channels=num_classes,
    heads=best_hyperparameters["heads"],
    dropout=best_hyperparameters["dropout"],
)
try:
    model.load_state_dict(
        torch.load(
            "models/gat_dynamic_temporal_fraud_model.pth",
            map_location=torch.device("cpu"),
        )
    )
    model.eval()
except FileNotFoundError:
    print(
        "Error: 'gat_dynamic_temporal_fraud_model.pth' not found. Fraud detection will not function."
    )
    model_loaded = False
else:
    model_loaded = True

scaler = StandardScaler()
scaler_fitted = False


TIME_THRESHOLD = 10

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_123")

app.config["TEMPLATES_AUTO_RELOAD"] = True
DATABASE_FILE = "users.json"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
GRAPH_DIR = "static/fraud_graphs"  # Directory for generated graphs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
app.config.update(
    SESSION_COOKIE_NAME="my_session",
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)
app.permanent_session_lifetime = timedelta(days=7)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_users():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_users(users):
    with open(DATABASE_FILE, "w") as f:
        json.dump(users, f, indent=4)


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"message": "You must be logged in."}), 401
            return redirect(url_for("index"))
        return f(*args, **kwargs)

    return wrapper


def save_plot(fig, filename):
    path = os.path.join(GRAPH_DIR, filename)
    print(f"[DEBUG - save_plot] Saving plot to: {path}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"/{path}"


def generate_fraud_analysis_graphs(df):
    graph_paths = []

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numerical_cols) >= 2:
        x_col = numerical_cols[0]
        y_col = numerical_cols[1]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Fraud Analysis: Scatter Plot of {x_col} vs {y_col}")
        scatter_path = save_plot(fig, f"{uuid4().hex}_fraud_scatter.png")
        print(f"[DEBUG - generate_graphs] Saved scatter plot to: {scatter_path}")
        graph_paths.append(scatter_path)

    corr_matrix = df[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    ax.set_title("Fraud Analysis: Correlation Heatmap")
    heatmap_path = save_plot(fig, f"{uuid4().hex}_fraud_heatmap.png")
    print(f"[DEBUG - generate_graphs] Saved heatmap to: {heatmap_path}")
    graph_paths.append(heatmap_path)

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    if categorical_cols:
        cat_col = categorical_cols[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, y=cat_col)
        ax.set_xlabel("Count")
        ax.set_ylabel(cat_col)
        ax.set_title(f"Fraud Analysis: Distribution of {cat_col}")
        bar_path = save_plot(fig, f"{uuid4().hex}_fraud_bar.png")
        print(f"[DEBUG - generate_graphs] Saved bar graph to: {bar_path}")
        graph_paths.append(bar_path)

    if "Time" in df.columns and "Amount" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x="Time", y="Amount")
        ax.set_xlabel("Time")
        ax.set_ylabel("Transaction Amount")
        ax.set_title("Fraud Analysis: Time Series of Transaction Amount")
        time_series_path = save_plot(fig, f"{uuid4().hex}_fraud_timeseries.png")
        print(
            f"[DEBUG - generate_graphs] Saved time series plot to: {time_series_path}"
        )
        graph_paths.append(time_series_path)

    return graph_paths


#def cleanup_graphs():
    
 #   for filename in os.listdir(GRAPH_DIR):
  #      file_path = os.path.join(GRAPH_DIR, filename)
   #     try:
    #        if os.path.isfile(file_path):
     #           os.unlink(file_path)
      #  except Exception as e:
       #     print(f"Error deleting {file_path}: {e}")


# @app.before_request
# def before_request():
# """Cleanup graphs at the start of each session (or request)."""
# if "username" in session:
# cleanup_graphs()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"message": "Please provide all required information."}), 400

    # Email validation
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(email_regex, email):
        return jsonify({"message": "Invalid email format."}), 400

    users = load_users()
    if username in users:
        return jsonify({"message": "Username already exists."}), 409

    hashed_password = generate_password_hash(password)
    users[username] = {"email": email, "password": hashed_password}
    save_users(users)
    return jsonify({"message": "Signup successful!"}), 200

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"message": "Please provide username and password."}), 400

    users = load_users()
    if username in users and check_password_hash(users[username]["password"], password):
        session.clear()
        session["username"] = username
        session["email"] = users[username]["email"]
        session.permanent = True
        print(f"[DEBUG] Session after login: {session}")
        return jsonify({"message": "Login successful!", "username": username}), 200

    elif username in users:
        return jsonify({"message": "Invalid password."}), 401
    else:
        return jsonify({"message": "Username not found."}), 401


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html", username=session["username"], email=session["email"]
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/analyze")
@login_required
def analyze():
    return render_template("analyze.html", username=session["username"])


@app.route("/graphs")
@login_required
def graphs_page():
    graph_files = [
        f"/static/fraud_graphs/{f}" for f in os.listdir(GRAPH_DIR) if f.endswith(".png")
    ]
    print(f"[DEBUG - graphs_page] Graph files found: {graph_files}")
    return render_template(
        "graphs.html",
        username=session["username"],
        email=session["email"],
        graph_files=graph_files,
    )


@app.route("/get-csv-headers", methods=["POST"])
@login_required
def get_csv_headers():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            reader = csv.reader(file.stream.read().decode("utf-8").splitlines())
            headers = next(reader)
            return jsonify({"columns": headers}), 200
        except Exception as e:
            return jsonify({"error": f"Error reading CSV: {str(e)}"}), 500
    return jsonify({"error": "Invalid file format"}), 400




@app.route("/detect-fraud", methods=["POST"])
@login_required
def detect_fraud():
    if not model_loaded:
        return jsonify(
            {"success": False, "message": "Fraud detection model not loaded."}
        )

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No selected file"})

    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file)
            original_df = df.copy()  # Keep a copy for analysis if fraud is detected
            required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
            if not all(col in df.columns for col in required_cols):
                return jsonify(
                    {
                        "success": False,
                        "message": f"File must contain the following columns: {', '.join(required_cols)}",
                    }
                )

            features = df[[f"V{i}" for i in range(1, 29)] + ["Amount"]].values
            time_values = df["Time"].values

            global scaler
            global scaler_fitted
            numerical_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
            if not scaler_fitted:
                scaler.fit(df[numerical_cols])
                scaler_fitted = True
            scaled_features = scaler.transform(features)

            num_nodes = len(df)
            edge_index = []
            global TIME_THRESHOLD
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if abs(time_values[i] - time_values[j]) <= TIME_THRESHOLD:
                        edge_index.append([i, j])
                        edge_index.append([j, i])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            x = torch.tensor(scaled_features, dtype=torch.float32)
            graph_data = Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                out = model(graph_data.x, graph_data.edge_index)
                pred = out.argmax(dim=1)
                prob = torch.exp(out)[:, 1].numpy()

            fraud_indices = np.where(pred.numpy() == 1)[0]
            fraud_count = len(fraud_indices)
            fraud_probabilities = prob[fraud_indices]
            sample_frauds = (
                df.iloc[fraud_indices[:5]].to_dict("records") if fraud_count > 0 else []
            )

            fraud_analysis_graphs = []
            if fraud_count > 0:
                fraud_df = original_df.iloc[fraud_indices]
                fraud_analysis_graphs = generate_fraud_analysis_graphs(fraud_df)

            return jsonify(
                {
                    "success": True,
                    "fraud_count": fraud_count,
                    "total_transactions": num_nodes,
                    "fraud_percentage": (
                        (fraud_count / num_nodes) * 100 if num_nodes > 0 else 0
                    ),
                    "sample_frauds": sample_frauds,
                    "fraud_probabilities": fraud_probabilities.tolist(),
                    "all_predictions": pred.numpy().tolist(),
                    "fraud_analysis_graphs": [
                        os.path.basename(path) for path in fraud_analysis_graphs
                    ],
                    "redirect_url": url_for("graphs_page"),
                }
            )

        except Exception as e:
            return jsonify(
                {"success": False, "message": f"Error analyzing file: {str(e)}"}
            )

    return jsonify({"success": False, "message": "Invalid file format"})


# ------------------- Main -------------------

if __name__ == "__main__":
    app.run(debug=True)
