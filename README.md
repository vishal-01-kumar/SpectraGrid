⚡ SpectraGrid – AI-Powered FRA Diagnostics Platform for Transformer Health Monitoring
🧠 Smart India Hackathon (Problem Statement ID: SIH25190)
👥 Team: The Fun-gineers (Team ID: 71285)

Welcome to SpectraGrid, an intelligent diagnostic platform designed to transform transformer health monitoring through AI-driven Frequency Response Analysis (FRA).

SpectraGrid enables power engineers to analyze, visualize, and interpret FRA data from multiple vendors in a unified interface — helping detect mechanical and electrical faults like winding deformation, core displacement, or insulation degradation with speed and precision. ⚙️

📜 Problem Statement

Traditional FRA analysis is complex and time-consuming, requiring expert interpretation and comparison with baseline data.
Moreover, the lack of data format standardization across OEMs (Omicron, Doble, Megger, etc.) makes analysis inefficient.

The challenge:
To design a unified software solution that can ingest multi-format FRA data, perform AI-based analysis, and deliver automated fault classification and maintenance recommendations — even without baseline signatures.

🚀 Our Solution — SpectraGrid

SpectraGrid bridges the gap between electrical diagnostics and artificial intelligence.
It’s a web-based platform that performs FRA data parsing, preprocessing, visualization, and intelligent fault analysis.

Key Objectives Achieved:

Standardization of multi-vendor FRA formats (CSV, XML, proprietary binaries)

Automated feature extraction and classification using ML models

Fault interpretation through explainable AI logic

Intuitive dashboards for engineers and asset managers

By leveraging machine learning and signal analytics, SpectraGrid reduces reliance on expert judgment and enables predictive, data-driven transformer maintenance.

## ⚙️ Tech Stack

| Technology           | Purpose                                 |
|----------------------|-----------------------------------------|
| React.js             | Frontend UI & Visualization             |
| Flask (Python)       | Backend Microservice for Model Inference|
| Python               | Signal Processing & AI Modeling         |
| Scikit-Learn         | ML Model Training & Fault Classification|
| Pandas / NumPy       | Data Cleaning & Feature Engineering     |
| Plotly / Matplotlib  | Interactive FRA Signature Visualization |
| Node.js + Express.js | API Gateway Integration                  |
| MongoDB              | Storage of Test Records & Predictions   |

💡 Key Features

Multi-Format Data Parsing: Reads and standardizes FRA data from major OEM formats.

AI-Based Fault Classification: Uses trained ML models (CNN/Random Forest) to detect probable faults.

Interactive Visualization: Plots frequency response curves and highlights deviations automatically.

Fault Summary Reports: Generates detailed reports with confidence scores and maintenance insights.

Explainable AI (XAI): Offers interpretable fault reasoning for transparent decision-making.

Web-Based Platform: Runs seamlessly across browsers for engineers and asset managers.

🧠 AI Model Workflow

Data Ingestion: Load FRA test data from multiple vendors

Feature Extraction: Amplitude ratio, phase response, resonance peaks

Classification: AI model identifies likely fault types

Visualization: FRA curve plotted with fault indicators

Insights: System suggests probable root causes and maintenance steps

📊 System Architecture

Core Modules:

Data Ingestion and Validation

Vendor Detection and Format Standardization

Feature Extraction and Engineering

AI Model Inference (CNN + Random Forest)

Explainable AI Visualization

Result Dashboard and Report Generation

🖥️ Platform Overview

Unified dashboard for uploading and analyzing FRA data

Real-time prediction with confidence levels

Interactive plots showing deviation and fault trends

AI-powered chatbot for contextual assistance

Exportable PDF reports for maintenance documentation

🧩 Future Scope

Integration with real-time FRA test instruments via IoT gateways

Deployment of deep learning models (1D-CNNs) for waveform interpretation

Development of mobile-friendly diagnostic dashboards

Integration with SCADA / asset management systems for proactive alerts

🤝 Developed By

Team: The Fun-gineers
Smart India Hackathon 2025
