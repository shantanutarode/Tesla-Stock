# 🚀 Deployment Guide

## Deploy Your Tesla Stock Prediction App Online

This guide shows how to deploy your Streamlit app to various cloud platforms so others can access it via a public URL.

---

## Option 1: Streamlit Community Cloud (Easiest & Free) ⭐

### Requirements:
- GitHub account
- This project code pushed to a GitHub repository

### Steps:

1. **Push to GitHub**
```bash
cd "c:\Users\pruth\OneDrive\Desktop\Tesla Stock Prediction"

# Initialize git (if not already)
git init
git add .
git commit -m "Tesla Stock Prediction Project"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/tesla-stock-prediction.git
git push -u origin main
```

2. **Deploy to Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click "New app"
- Select your repository
- Set main file: `streamlit_app.py`
- Click "Deploy"

3. **Your app will be live at:**
```
https://YOUR_USERNAME-tesla-stock-prediction.streamlit.app
```

### Streamlit Cloud Features:
✅ Free hosting
✅ Automatic updates from GitHub
✅ HTTPS enabled
✅ Easy to use
✅ Good for portfolios

---

## Option 2: Heroku (Advanced)

### Requirements:
- Heroku account
- Heroku CLI installed

### Additional Files Needed:

**1. Create `Procfile`:**
```bash
web: sh setup.sh && streamlit run streamlit_app.py
```

**2. Create `setup.sh`:**
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

**3. Update `requirements.txt`:**
(Already created - no changes needed)

### Deploy Commands:
```bash
heroku login
heroku create tesla-stock-prediction
git push heroku main
heroku open
```

---

## Option 3: Railway.app (Modern & Simple)

### Steps:

1. **Push code to GitHub** (same as Streamlit Cloud)

2. **Deploy on Railway:**
- Go to [railway.app](https://railway.app)
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose your repository
- Railway auto-detects Streamlit
- Click "Deploy"

3. **Configure:**
- Go to Settings → Networking
- Generate Domain
- Your app is live!

### Railway Features:
✅ $5/month free tier
✅ Automatic deployments
✅ Good performance
✅ Easy scaling

---

## Option 4: Render.com

### Requirements:
- Render account
- GitHub repository

### Additional Files:

**Create `render.yaml`:**
```yaml
services:
  - type: web
    name: tesla-stock-prediction
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

### Deploy:
1. Push to GitHub
2. Go to [render.com](https://render.com)
3. Click "New +" → "Blueprint"
4. Connect repository
5. Deploy

---

## Option 5: Google Cloud Run (Professional)

### Requirements:
- Google Cloud account
- Docker knowledge

### Steps:

**1. Create `Dockerfile`:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run streamlit_app.py --server.port 8080 --server.address 0.0.0.0
```

**2. Deploy:**
```bash
# Install Google Cloud SDK first
gcloud init
gcloud builds submit --tag gcr.io/PROJECT_ID/tesla-prediction
gcloud run deploy --image gcr.io/PROJECT_ID/tesla-prediction --platform managed
```

---

## Comparison Table

| Platform | Free Tier | Ease | Speed | Best For |
|----------|-----------|------|-------|----------|
| **Streamlit Cloud** | ✅ Yes | ⭐⭐⭐⭐⭐ | Medium | Beginners, Demos |
| **Heroku** | ✅ Limited | ⭐⭐⭐ | Medium | Projects |
| **Railway** | ✅ $5/mo | ⭐⭐⭐⭐ | Fast | Modern Apps |
| **Render** | ✅ Yes | ⭐⭐⭐⭐ | Fast | Full-stack |
| **Google Cloud** | ⚠️ Credits | ⭐⭐ | Very Fast | Enterprise |

---

## 🎯 Recommended for This Project

### For Assignment Submission:
**Use Streamlit Community Cloud**
- Easiest setup
- Free forever
- Perfect for portfolios
- Shareable link

### For Professional Portfolio:
**Use Railway or Render**
- Better performance
- Custom domains
- More control

---

## 📝 Pre-Deployment Checklist

Before deploying to any platform:

- [ ] Test app locally: `streamlit run streamlit_app.py`
- [ ] Ensure `requirements.txt` is complete
- [ ] Check file paths (use relative paths)
- [ ] Verify TSLA.csv is in repository
- [ ] Remove any API keys or secrets
- [ ] Test with sample data
- [ ] Optimize for performance

---

## 🔧 Common Deployment Issues

### Issue 1: App won't start
**Solution**: Check requirements.txt has all dependencies

### Issue 2: Data file not found
**Solution**: Ensure TSLA.csv is in same directory as streamlit_app.py

### Issue 3: Memory errors
**Solution**: Use smaller dataset or upgrade plan

### Issue 4: Module not found
**Solution**: Add missing package to requirements.txt

---

## 🌐 After Deployment

### Share Your App:
1. Copy the public URL
2. Add to your README.md
3. Share on LinkedIn
4. Include in resume/portfolio

### Monitor Performance:
- Check app logs
- Monitor uptime
- Track user engagement
- Fix any errors

### Update Your App:
```bash
# Make changes locally
git add .
git commit -m "Updated predictions"
git push origin main
# Most platforms auto-deploy on GitHub push!
```

---

## 💡 Pro Tips

1. **Add a custom domain** (for professional look)
2. **Enable analytics** (Google Analytics)
3. **Add authentication** (if needed)
4. **Optimize loading speed** (cache data)
5. **Add error handling** (user-friendly messages)

---

## 📊 Sample README Badge

After deploying, add this to your README:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL)
```

Shows as: ![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

---

## 🎓 For Your Assignment

**Recommended Approach:**
1. Complete local development ✅ (Done!)
2. Test thoroughly locally ✅
3. Push to GitHub
4. Deploy to Streamlit Cloud
5. Include live URL in submission
6. Record video showing live app

**Bonus Points:**
- ✨ Live deployed app
- ✨ Custom domain
- ✨ Professional README
- ✨ Video demo with live link

---

## 📚 Additional Resources

- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Railway Deployment](https://docs.railway.app/deploy/deployments)
- [Render Documentation](https://render.com/docs)

---

**Ready to deploy? Start with Streamlit Cloud - it's the easiest!** 🚀

Your app is production-ready and will impress evaluators with a live deployment!
