💳 Credit Card Fraud Detection

A Django + Machine Learning web application that detects fraudulent credit card transactions in real-time.
This project uses scikit-learn models with a clean, production-ready Django backend and is fully deployable on platforms like Render, PythonAnywhere, or Heroku.

🚀 Features

✅ Machine Learning – Uses scikit-learn models to predict fraudulent transactions.

✅ Django Web Interface – Simple web UI for uploading transaction data or entering details.

✅ Interactive Predictions – Enter details and get instant fraud probability.

✅ Admin Dashboard – Django Admin for managing datasets and user activity.

✅ Scalable Deployment – Easily deployable on Render/Heroku with PostgreSQL or SQLite.



⚡ Tech Stack
Component	Technology
Backend	Django 4.2 / Python 3.10+
Machine Learning	scikit-learn, pandas, numpy
Database	SQLite (dev) / PostgreSQL (prod)
Deployment	Render / Heroku / PythonAnywhere
🧑‍💻 Local Setup
1️⃣ Clone the Repository
git clone https://github.com/Rahulkumar160602/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Run Migrations
python manage.py migrate

5️⃣ Start the Development Server
python manage.py runserver


Then visit: http://127.0.0.1:8000/

🌐 Deployment (Render Example)

Push code to GitHub.

Create a new Render Web Service.

Set:

Runtime: python-3.10.13 (use runtime.txt)

Build Command:

pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput


Start Command:

gunicorn fraud_detection.wsgi


Add environment variables (SECRET_KEY, DEBUG=False, DB URL if using PostgreSQL).

📊 Dataset

The model is trained on the Kaggle Credit Card Fraud Detection dataset
.
It is highly imbalanced (fraud cases are rare), and techniques like SMOTE or imbalanced-learn are used for better performance.

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License – feel free to use and modify it.

🏆 Author

Developed by Rahul Kumar

⭐ Support

If you like this project, give it a star ⭐ on GitHub to show your support!
