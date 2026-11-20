# Deployment Guide: Cloud Hosting Your Big Data Sentiment Analysis Dashboard

This project is fully configured for cloud deployment on **Render** (via `render.yaml` & `Procfile`) and **Vercel** (via `vercel.json` & `api/index.py`) with zero manual server configuration.

---

## Option 1: Deploy on Render (Recommended for Web Applications)

[Render](https://render.com) provides continuous deployment with automatic Git triggers for Python Flask/Gunicorn applications.

### Step-by-Step Instructions:

1. **Log in to [Render Dashboard](https://dashboard.render.com/):**
   - Sign in using your **GitHub account** (`pratyush-barik`).

2. **Create Web Service:**
   - Click the blue **"New +"** button &rarr; Select **"Web Service"**.
   - Under *Connect a repository*, choose **`pratyush-barik/BDA-Sentiment-Analysis`** and click **"Connect"**.

3. **Configure Service Settings:**
   - **Name:** `bda-sentiment-analysis` (or your preferred service name)
   - **Region:** `Oregon (US West)` (or your closest region)
   - **Branch:** `main`
   - **Language / Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn web.app:app`
   - **Instance Type:** `Free` ($0 / month)

4. **Configure Environment Variables (Optional for MongoDB Atlas):**
   - Scroll down to the **"Environment Variables"** section and click **"Add Environment Variable"**:
     - **Key:** `MONGO_URI`
     - **Value:** `mongodb+srv://<username>:<password>@cluster0.xxx.mongodb.net/?retryWrites=true&w=majority`
     - **Key:** `PYTHON_VERSION`
     - **Value:** `3.11.9`
   *(Note: If you omit `MONGO_URI`, the system automatically falls back to its built-in Local JSON Document Store with zero errors).*

5. **Deploy & Launch:**
   - Click **"Deploy Web Service"**.
   - Render installs packages, starts Gunicorn, and marks the service **`Live`** within 1–2 minutes.
   - Your public HTTPS URL will be available at the top:  
     👉 **`https://bda-sentiment-analysis.onrender.com`**

---

## Option 2: Deploy on Vercel (Serverless WSGI)

[Vercel](https://vercel.com) supports Python Serverless Functions via the included `vercel.json` and `api/index.py`.

### Step-by-Step Instructions:

1. Push your repository to GitHub (`https://github.com/pratyush-barik/BDA-Sentiment-Analysis`).
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) &rarr; Click **"Add New..."** &rarr; **"Project"**.
3. Import `pratyush-barik/BDA-Sentiment-Analysis`.
4. Leave root directory as `./` and click **"Deploy"**.
5. Vercel automatically detects `vercel.json` and `api/index.py` and deploys the application to edge functions.

---

## Features Available on Your Deployed Live Dashboard

- **Dynamic Evaluation Sample Size Controller ($n \le 25,000$):** Continuously scrub or select any analysis sample size below 25,000 with real-time KPI recalculations.
- **Empirical Figure Inspector (10-Graph Picker):** Interactive selector and domain filter pills exploring all 10 publication-quality analytical charts with formal scientific captions and methodology notes.
- **Real-Time Sentiment Inference Protocol:** Test live text against the 6-stage Spark MLlib classification pipeline.
- **Corpus Document Store Explorer:** Filter and browse review records by sentiment, product category, and sample limit.
- **Curricular Architecture & Specifications:** Exhaustive mapping across Units 1–6 and Course Outcomes CO3 & CO4.
