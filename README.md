⚡ SpectraGrid – AI-Powered FRA Diagnostics Platform for Transformer Health Monitoring
🧠 Smart India Hackathon (Problem Statement ID: SIH25190)
👥 Team:The Fun-gineers (Team ID: 71285)

Welcome to SpectraGrid, an intelligent diagnostic platform designed to transform transformer health monitoring through AI-driven Frequency Response Analysis (FRA).

SpectraGrid enables power engineers to analyze, visualize, and interpret FRA data from multiple vendors in a unified interface — helping detect mechanical and electrical faults like winding deformation, core displacement, or insulation degradation with speed and precision. ⚙️

📜 Problem Statement

Traditional FRA analysis is complex and time-consuming, requiring expert interpretation and comparison with baseline data.
Moreover, lack of data format standardization across OEMs (Omicron, Doble, Megger, etc.) makes analysis inefficient.

The challenge:

To design a unified software solution that can ingest multi-format FRA data, perform AI-based analysis, and deliver automated fault classification and maintenance recommendations — even without baseline signatures.

🚀 Our Solution — SpectraGrid

SpectraGrid bridges the gap between electrical diagnostics and artificial intelligence.
It’s a web-based platform that performs FRA data parsing, preprocessing, visualization, and intelligent fault analysis.

Key objectives achieved:

🔹 Standardization of multi-vendor FRA formats (CSV, XML, proprietary binaries)

🔹 Automated feature extraction and classification using ML models

🔹 Fault interpretation through explainable AI logic

🔹 Intuitive dashboards for engineers and asset managers

By leveraging machine learning and signal analytics, SpectraGrid reduces reliance on expert judgment and enables predictive, data-driven transformer maintenance.

⚙️ Tech Stack
Technology	Purpose	Icon
React.js	Frontend UI & Visualization	<img src="https://github.com/pianist22/Images/blob/main/React.png" width="180" height="90">
Flask (Python)	Backend Microservice for Model Inference	<img src="https://github.com/pianist22/Images/blob/main/Flask.png" width="180" height="90">
Python	Signal Processing & AI Modeling	<img src="https://github.com/pianist22/Images/blob/main/Python-2.png" width="180" height="90">
Scikit-Learn	ML Model Training & Fault Classification	<img src="https://github.com/pianist22/Images/blob/main/Scikit-learn.png" width="180" height="90">
Pandas / NumPy	Data Cleaning & Feature Engineering	<img src="https://github.com/pianist22/Images/blob/main/pandas.png" width="180" height="90">
Plotly / Matplotlib	Interactive FRA Signature Visualization	<img src="https://github.com/pianist22/Images/blob/main/plotly.png" width="180" height="90">
Node.js + Express.js	API Gateway Integration	<img src="https://github.com/pianist22/Images/blob/main/Nodejs.png" width="180" height="90">
MongoDB	Storage of Test Records & Predictions	<img src="https://github.com/pianist22/Images/blob/main/Mongodb.png" width="180" height="90">
💡 Key Features

📁 Multi-Format Data Parsing
Reads and standardizes FRA data from major OEM formats.

🧠 AI-Based Fault Classification
Uses trained ML models (CNN/Random Forest) to detect probable faults.

📊 Interactive Visualization
Plots frequency response curves and highlights deviations automatically.

🧾 Fault Summary Reports
Generates detailed reports with confidence scores and maintenance insights.

🔍 Explainable AI (XAI)
Offers interpretable fault reasoning for transparent decision-making.

🌐 Web-Based Platform
Runs seamlessly across browsers for engineers and asset managers.

🧠 AI Model Workflow

1. Data Ingestion: Load FRA test data from multiple vendors
2. Feature Extraction: Amplitude ratio, phase response, resonance peaks
3. Classification: AI model identifies likely fault types
4. Visualization: FRA curve plotted with fault indicators
5. Insights: System suggests probable root causes and maintenance steps

📊 System Architecture
<p align="center"> <img src="https://github.com/pianist22/Images/blob/main/SpectraGrid_FRA_Architecture.png" alt="SpectraGrid FRA Architecture" width="800"> </p>
🖥️ Platform Preview
<p> <img src="https://github.com/pianist22/Images/blob/main/SpectraGrid_Home.png" width="260" height="480"> <img src="https://github.com/pianist22/Images/blob/main/SpectraGrid_Analytics.png" width="260" height="480"> <img src="https://github.com/pianist22/Images/blob/main/SpectraGrid_Report.png" width="260" height="480"> </p>

Empowering Power Utilities with Intelligent FRA Diagnostics and Predictive Maintenance. ⚡

🧩 Future Scope

Integration with real-time FRA test instruments via IoT gateways

Deployment of deep learning models (1D-CNNs) for waveform interpretation

Development of mobile-friendly diagnostic dashboards

Integration with SCADA / asset management systems for proactive alerts

🤝 Developed By

Team Paranoid Coderz
Smart India Hackathon 2025

📩 For collaboration or technical queries:
Email: paranoidcoderz.dev@gmail.com
