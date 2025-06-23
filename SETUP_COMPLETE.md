# StockTrader Development Environment

## ✅ **Setup Complete!** 

### 🔧 **What was configured:**

1. **Node.js PATH** - Added to `~/.bash_profile` and `~/.bashrc`
2. **Automatic verification** - Checks npm availability on shell startup
3. **Development scripts** - Complete npm run commands for full-stack development

### 🚀 **Start Development:**

```bash
# Single command to start both frontend and backend
npm run dev

# Or start individually:
npm run dev:backend   # FastAPI server
npm run dev:frontend  # Next.js dev server
```

### 📂 **Files Modified:**

- `~/.bash_profile` - Node.js PATH export
- `~/.bashrc` - Node.js PATH export (GitBash compatibility)
- `package.json` - Full-stack development scripts
- `setup-dev.sh` - Environment verification script

### 🛠 **Available Commands:**

| Command | Description |
|---------|-------------|
| `npm run dev` | Start both services with concurrently |
| `npm run dev:backend` | Start FastAPI backend only |
| `npm run dev:frontend` | Start Next.js frontend only |
| `npm run install:all` | Install all dependencies |
| `npm run docs` | Open API documentation |
| `./setup-dev.sh` | Verify development environment |

### 🔄 **For New Terminal Sessions:**

Every new GitBash terminal will now automatically:
1. ✅ Add Node.js to PATH
2. ✅ Verify npm is available
3. ✅ Show confirmation message

### 💡 **Next Steps:**

1. Open a new terminal to test the automatic setup
2. Run `npm run dev` to start the full application
3. Access:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs
