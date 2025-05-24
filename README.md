# Real-time-AI-Powered-Network-Intrusion-Detection-Automated-Defense-System
Real-time AI-Powered Network Intrusion Detection &amp; Automated Defense System

This project presents a robust, real-time solution for identifying and responding to network intrusions using advanced machine learning and AI. Developed as a proactive cybersecurity measure, this system continuously monitors network traffic flows, leveraging an AI-powered detection engine to classify potential threats and trigger automated defense mechanisms.

Key Features:
Real-time Intrusion Detection: Utilizes a pre-trained machine learning model (Random Forest Classifier, along with an Anomaly Detector) to analyze incoming network traffic flows in real time, classifying them as benign or various attack types (e.g., DoS, Exploits, Reconnaissance, SQL Injection).
Automated Defense & Response: Implements a sophisticated, rule-based defense system that triggers immediate, simulated mitigation actions (e.g., IP blocking, rate limiting, host isolation, process termination) based on the detected attack type and confidence level.
Geographical Threat Intelligence: Integrates GeoIP lookup to identify the origin of suspicious traffic, providing critical location context for detected intrusions.
AI-Powered Threat Analysis (Gemini LLM): Incorporates a Large Language Model (Google Gemini) to provide on-demand, contextual explanations of detected attacks, including their relevance to the specific threat landscape (e.g., in Egypt), and suggests immediate practical mitigation steps for security analysts.
Intuitive Streamlit Dashboard: Features an interactive web-based interface built with Streamlit for real-time monitoring of detection logs, visualization of attack trends, and comprehensive reporting of simulation results.
Email Alerting System: Notifies security personnel via email upon detection of critical threats, ensuring timely awareness and response.
Technical Stack:
Core Logic: Python
Machine Learning: scikit-learn (Random Forest Classifier, Isolation Forest for Anomaly Detection, StandardScaler, LabelEncoder)
Data Handling: pandas, numpy
Real-time Monitoring & UI: Streamlit
Geographical Data: geoip2
AI/LLM Integration: google-generativeai (Gemini API)
Visualization: matplotlib, seaborn
Email Notifications: smtplib
This project demonstrates a comprehensive approach to modern cybersecurity, combining predictive AI with automated response capabilities and intelligent contextual insights, making it a valuable tool for enhancing network security posture.
