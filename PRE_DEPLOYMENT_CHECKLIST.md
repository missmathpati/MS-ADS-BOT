# Pre-Deployment Checklist

Use this checklist before deploying your app to ensure everything is ready.

## ✅ Files Required for Deployment

- [ ] `app.py` exists and is working locally
- [ ] `requirements.txt` exists with all dependencies
- [ ] `.streamlit/config.toml` exists (created automatically)
- [ ] `.gitignore` exists (created automatically)
- [ ] `rag_index/` folder exists and contains:
  - [ ] `rag_index/chroma_db/` directory with database files
  - [ ] `rag_index/meta.jsonl` file
  - [ ] `rag_index/faiss_e5.index` (if used)

## ✅ Git Repository Setup

- [ ] Git repository initialized (`git init`)
- [ ] All files added (`git add .`)
- [ ] Initial commit made (`git commit -m "Initial commit"`)
- [ ] GitHub repository created
- [ ] Remote added (`git remote add origin <url>`)
- [ ] Code pushed to GitHub (`git push -u origin main`)

## ✅ Verify rag_index is NOT Ignored

- [ ] Check `.gitignore` - `rag_index/` should be **commented out** (with `#`)
- [ ] Verify `rag_index/` folder will be committed:
  ```bash
  git status
  ```
  You should see `rag_index/` in the list of files to be committed

## ✅ Test Locally

- [ ] App runs locally without errors (`streamlit run app.py`)
- [ ] RAG system loads successfully
- [ ] Can ask questions and get responses
- [ ] Citations appear correctly
- [ ] No console errors

## ✅ Requirements.txt Check

- [ ] All dependencies listed in `requirements.txt`
- [ ] Version numbers are compatible
- [ ] No missing packages

## ✅ API Key Setup

- [ ] Have OpenAI API key ready
- [ ] Know how to add it to Streamlit secrets (Settings → Secrets)
- [ ] **Never commit API key to repository**

## ✅ File Size Check

- [ ] `rag_index/` folder size is reasonable (< 100 MB per file)
- [ ] Total repository size is manageable
- [ ] If files are large, consider Git LFS

## ✅ Ready to Deploy?

Once all items above are checked:

1. **Push to GitHub** (if not already done)
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with GitHub
4. **Deploy** your app
5. **Add API key** in Settings → Secrets
6. **Test** your deployed app
7. **Share** the URL with others!

---

**Quick Command Reference:**

```bash
# Check what will be committed
git status

# Add all files
git add .

# Commit
git commit -m "Ready for deployment"

# Push to GitHub
git push origin main
```

---

**Need help?** See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

