# Deployment Guide: Hosting Your Big Data Sentiment Analysis Dashboard

This project is configured for cloud deployment on both **Render** and **Vercel** with zero extra setup.

---

## Option 1: Deploy on Render (Recommended for Python Web Apps)

[Render](https://render.com) provides continuous deployment for Python Flask/Gunicorn applications.

### Step-by-Step Instructions:
1. **Push your code to GitHub / GitLab:**
   ```bash
   cd d:\Projects\BDA
   git init
   git add .
   git commit -m "Initial commit of BDA sentiment analysis system"
   git remote add origin https://github.com/<your-username>/bda-sentiment-analysis.git
   git push -u origin main
   ```

2. **Log in to [Render Dashboard](https://dashboard.render.com/):**
   - Click **"New +"** &rarr; Select **"Web Service"**.
   - Connect your GitHub repository.

3. **Configure the Web Service Settings:**
   - **Name:** `bda-sentiment-analysis` (or your choice)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn web.app:app`
   - **Instance Type:** `Free`

4. **Click "Create Web Service"**:
   - Render will build and deploy the app.
   - Once finished, you will receive a public HTTPS URL (e.g. `https://bda-sentiment-analysis.onrender.com`).

*(Note: The repository includes `render.yaml` and `Procfile`, which automatically pre-configure these settings if you use Render's Blueprint feature).*

---

## Option 2: Deploy on Vercel

[Vercel](https://vercel.com) supports Python Serverless Functions via the included `vercel.json` and `api/index.py`.

### Step-by-Step Instructions:

#### Method A: Using the Vercel CLI (Fastest)
1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. In the project directory, run:
   ```bash
   cd d:\Projects\BDA
   vercel
   ```
3. Follow the interactive prompts (accept defaults). Your live URL will be generated immediately.

#### Method B: Using the Vercel Web Dashboard
1. Push your project to GitHub.
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) &rarr; Click **"Add New..."** &rarr; **"Project"**.
3. Import your GitHub repository.
4. Leave root directory as `./` and click **"Deploy"**.
5. Vercel automatically detects `vercel.json` and `api/index.py` and deploys the Flask application.

---

## Features Available on Your Deployed Live Dashboard
- **Interactive Sentiment Inference:** Type any customer review to test the live ML classifier.
- **Dataset KPI Metrics:** Total reviews, average rating, positive/neutral/negative percentage share.
- **Visual Analytics:** View 4 core interactive analytical charts.
- **Ingested Review Data Explorer:** Browse and filter review records by product category and sentiment.
- **System Architecture:** Detailed breakdown of HDFS, MapReduce, Spark, MLlib, and Graph layers.
