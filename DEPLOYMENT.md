# Deployment Guide for AskADS RAG Bot

This guide will help you deploy your Streamlit app to Streamlit Community Cloud so anyone can access it via a public URL.

## Prerequisites

1. **GitHub Account**: You need a GitHub account (free)
2. **Streamlit Community Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io) (free)
3. **Git Repository**: Your code should be in a GitHub repository

## Step 1: Prepare Your Repository

### 1.1 Ensure All Required Files Are Committed

Make sure these files are in your repository:
- ✅ `app.py` - Main application file
- ✅ `requirements.txt` - Python dependencies
- ✅ `rag_index/` - Vector database directory (must be committed!)
  - `rag_index/chroma_db/` - ChromaDB files
  - `rag_index/meta.jsonl` - Metadata file
  - `rag_index/faiss_e5.index` - FAISS index (if used)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Git ignore file

### 1.2 Verify rag_index is NOT in .gitignore

Check your `.gitignore` file. The `rag_index/` folder **MUST be committed** to GitHub for the app to work. If it's ignored, the deployed app won't have the vector database.

**Important**: If `rag_index/` is in `.gitignore`, remove it or comment it out.

### 1.3 Check File Sizes

GitHub has file size limits:
- Individual files: 100 MB
- Repository: 1 GB (free tier)

If your `rag_index/` folder is very large, you may need to:
- Use Git LFS (Large File Storage) for large files
- Or compress the index files

## Step 2: Push to GitHub

### 2.1 Initialize Git (if not already done)

```bash
cd /path/to/MS-ADS-BOT
git init
git add .
git commit -m "Initial commit for deployment"
```

### 2.2 Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click "New repository"
3. Name it (e.g., "ms-ads-rag-bot")
4. **Don't** initialize with README (if you already have files)
5. Click "Create repository"

### 2.3 Push Your Code

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name.

## Step 3: Deploy to Streamlit Community Cloud

### 3.1 Sign In to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign in" and authorize with your GitHub account
3. You'll be redirected to your Streamlit dashboard

### 3.2 Deploy Your App

1. Click "New app" button
2. Fill in the deployment form:
   - **Repository**: Select your repository from the dropdown
   - **Branch**: Select `main` (or your default branch)
   - **Main file path**: Enter `app.py`
   - **App URL**: Choose a custom subdomain (e.g., `ms-ads-bot`)
3. Click "Deploy"

### 3.3 Wait for Deployment

- Streamlit will install dependencies from `requirements.txt`
- This may take 5-10 minutes on first deployment
- You'll see build logs in real-time
- If there are errors, check the logs

## Step 4: Configure Secrets (API Keys)

### 4.1 Add OpenAI API Key as Secret

1. In your Streamlit dashboard, click on your deployed app
2. Click "⋮" (three dots) → "Settings" → "Secrets"
3. Add your OpenAI API key:

```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"
```

4. Click "Save"

**Important**: Never commit API keys to your repository! Always use Streamlit's secrets feature.

### 4.2 Update app.py (if needed)

Your app should already check for the environment variable. The secrets you add will be available as environment variables automatically.

## Step 5: Test Your Deployed App

1. Visit your app URL: `https://YOUR_APP_NAME.streamlit.app`
2. Test the functionality:
   - Enter a question about MS-ADS
   - Verify the RAG system works
   - Check that citations appear

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Check `requirements.txt` includes all dependencies. Some packages might need specific versions.

### Issue: "FileNotFoundError: Missing ChromaDB directory"

**Solution**: 
- Verify `rag_index/` folder is committed to GitHub
- Check `.gitignore` doesn't exclude `rag_index/`
- Ensure all files in `rag_index/` are committed

### Issue: "OpenAI API key not found"

**Solution**:
- Go to app Settings → Secrets
- Add `OPENAI_API_KEY` in the secrets file
- Redeploy the app

### Issue: App is slow to load

**Solution**:
- First load downloads models (E5, reranker) - this is normal
- Subsequent loads use cached models
- Consider using `@st.cache_resource` (already in your code)

### Issue: Build fails

**Solution**:
- Check build logs for specific errors
- Verify Python version compatibility (Streamlit Cloud uses Python 3.9-3.11)
- Ensure all dependencies in `requirements.txt` are valid

## Alternative Deployment Options

If Streamlit Community Cloud doesn't work for you, consider:

### 1. **Render** (render.com)
- Free tier available
- Supports Streamlit apps
- More control over environment

### 2. **Railway** (railway.app)
- Free tier with credits
- Easy deployment from GitHub
- Good for production apps

### 3. **Heroku** (heroku.com)
- Paid service now
- More complex setup
- Good for production apps

### 4. **AWS/GCP/Azure**
- More complex setup
- Requires cloud account
- Better for enterprise apps

## Updating Your Deployed App

Whenever you push changes to your GitHub repository:

1. Streamlit Community Cloud will automatically detect changes
2. It will redeploy your app
3. You can also manually trigger redeploy from the dashboard

## Security Best Practices

1. ✅ Never commit API keys or secrets
2. ✅ Use Streamlit secrets for sensitive data
3. ✅ Keep dependencies updated
4. ✅ Review your `.gitignore` regularly
5. ✅ Don't commit large files unnecessarily

## Cost

- **Streamlit Community Cloud**: Free
- **OpenAI API**: Pay-per-use (very affordable for small apps)
- **GitHub**: Free for public repos

## Support

- Streamlit Community Cloud Docs: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- Streamlit Forums: [discuss.streamlit.io](https://discuss.streamlit.io)

---

**Ready to deploy?** Follow the steps above and you'll have a public URL in about 10 minutes! 🚀

