# 🚦 Traffic Route Optimizer (AI + OSM + Streamlit)

A smart routing system that computes and compares **fastest** and **shortest** routes using real-world map data, simulated traffic, and machine learning.

---

## 🧠 Overview

This project combines:

* OpenStreetMap (OSM) road networks
* Vehicle-specific routing constraints
* Simulated traffic behavior
* Machine Learning delay prediction
* Optional terrain-based adjustments
* Interactive map visualization

The goal is simple:
👉 Give smarter route decisions than basic navigation systems.

---

## 🚀 Key Features

### 1. Dual Routing Engine

* **Time-priority route** → Minimum travel time
* **Distance-priority route** → Minimum distance
* Automatically detects and handles identical routes

---

### 2. Vehicle Intelligence

Each vehicle behaves differently:

* Speed variations
* Road accessibility rules
* Size-based restrictions

**Supported Vehicles:**

* Car
* Bike
* Auto
* Truck
* Bus

---

### 3. Traffic Simulation

* Traffic signals based on road type
* Random delay: **5–40 seconds per signal**
* Max **2 signals per km**

---

### 4. Machine Learning Delay Prediction

* Model: Random Forest
* Trained on synthetic traffic data

**Inputs:**

* Hour of day
* Road type
* Distance

---

### 5. Terrain Awareness (Optional)

* Uses `.tif` elevation files
* Adjusts travel time based on slope

---

### 6. Interactive Map UI

* Google Maps–like interface

* Color-coded traffic:

  * 🟢 Low delay
  * 🟠 Medium delay
  * 🔴 High delay

* Start / End / Waypoints

* Auto zoom

---

### 7. HERE API Comparison (Optional)

* Compare model vs real-world routing
* Displays % difference in travel time

---

## 📁 Project Structure

```
Traffic Delay Prediction/

├── run_app.py
├── osm_data.py
├── train_model.py
├── here_api.py
├── traffic_model.pkl
├── model_columns.pkl
├── elevation.tif
├── requirements.txt
└── venv/
```

👉 You can store **multiple `.tif` files** — the app will automatically detect and use them.

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd "Traffic Delay Prediction"
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Train the Model

```bash
python train_model.py
```

Generates:

* `traffic_model.pkl`
* `model_columns.pkl`

---

### 5. Run the App

```bash
streamlit run run_app.py
```

---

## ⚙️ How It Works

### Step 1: Road Network

* Fetches drivable roads using OSMnx

### Step 2: Vehicle Rules

* Applies restrictions and penalties

### Step 3: Traffic Simulation

* Adds signals + delays

### Step 4: ML Prediction

* Predicts additional delay

### Step 5: Routing

* Time-priority → fastest route
* Distance-priority → shortest route

### Step 6: Visualization

* Rendered using Folium

---

## 📊 Output Includes

* ETA (minutes)
* Distance (km)
* Traffic stops count
* Traffic delay
* ML delay
* Segment breakdown
* Route comparison

---

## ⚠️ Limitations

* Traffic is **simulated (not real-time)**
* Elevation depends on `.tif` availability
* HERE API requires API key
* Large routes may increase computation time

---

## 🌍 Elevation Data

* Place `.tif` files in the project folder
* If absent → app still works (no terrain adjustments)

---

## 💡 Final Note

This isn’t just a route finder.

It’s a **decision engine** balancing:

> Time vs Distance vs Traffic vs Vehicle Constraints
