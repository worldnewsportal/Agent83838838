# 🤖 Gemma 4 AI Agent — GitHub Actions

> A powerful AI coding agent powered by **Gemma 4 31B** via Google AI Studio,  
> running entirely on GitHub Actions. Inspired by Claude Code.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Gemma 4 31B** | Google's most powerful open model, 256K context |
| 💭 **Thinking Mode** | Built-in chain-of-thought reasoning (HIGH level) |
| 🔧 **13 Tools** | Read/write files, run commands, search code, create PRs |
| 🔄 **ReAct Loop** | Reason → Act → Observe (up to 30 iterations) |
| 🆓 **100% Free** | Google AI Studio free tier |
| ⚡ **GitHub Native** | Runs in Actions, no external servers |
| 🔍 **Code Review** | Auto-reviews every PR |
| 🛠️ **Auto-Fix** | Fixes failing tests automatically |
| 📊 **Health Reports** | Weekly code quality reports |

---

## 🚀 Setup (5 minutes)

### Step 1 — Get free Gemini API key
1. Go to **[aistudio.google.com](https://aistudio.google.com)**
2. Click **"Get API Key"** → **"Create API key"**
3. Copy the key

### Step 2 — Add to GitHub Secrets
1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Name: `GEMINI_API_KEY`
4. Value: paste your key

### Step 3 — Copy agent files
Copy these files to your repository:
```
.github/
  workflows/
    ai-agent.yml          ← Main agent workflow
    code-health.yml       ← Weekly code reviews
    auto-fix-tests.yml    ← Auto-fix failing tests
agent/
  agent.py                ← Core agent brain
```

### Step 4 — That's it! 🎉

---

## 💬 How to Use

### Trigger via Issue/PR Comments

Simply comment with `/ai` followed by your request:

```
/ai Fix the authentication bug in auth.py

/ai Add unit tests for the UserService class

/ai Refactor the database module to use async/await

/ai Add TypeScript types to all function signatures

/ai Review this code and suggest performance improvements

/ai Implement pagination for the /api/users endpoint
```

### Trigger via Issue Title

Create an issue with `[AI]` prefix:

```
[AI] Implement dark mode toggle
[AI] Fix memory leak in WebSocket handler
[AI] Add input validation to all API endpoints
```

### Trigger via Labels

Add label `ai-fix` or `ai-implement` to any issue.

### Trigger Manually

Go to **Actions** → **Gemma 4 AI Agent** → **Run workflow** → enter your task.

### Auto PR Review

Every opened/updated PR is automatically reviewed by the agent.

---

## 🔧 Available Tools

The agent has access to these 13 tools:

| Tool | Description |
|---|---|
| `read_file` | Read any file in the repo |
| `write_file` | Create or overwrite files |
| `list_directory` | Explore project structure |
| `run_command` | Execute shell commands, tests, builds |
| `search_code` | grep across the entire codebase |
| `delete_file` | Remove files |
| `git_diff` | View current changes |
| `create_pull_request` | Open a PR with all changes |
| `post_comment` | Comment on issues/PRs |
| `get_pr_diff` | Read a PR's full diff |
| `get_issue` | Read issue details & comments |
| `apply_patch` | Apply unified diff patches |
| `finish` | Mark task complete with summary |

---

## 📊 Workflows

### 1. Main Agent (`ai-agent.yml`)
- **Triggers:** Issue comments, PR comments, labeled issues, `[AI]` issues, manual
- **Model:** `gemma-4-31b-it` with thinking mode HIGH
- **Max iterations:** 30

### 2. Code Health (`code-health.yml`)
- **Triggers:** Every Monday 9 AM, or manual
- **Scopes:** `full`, `security`, `performance`, `tests`
- Posts results as a GitHub Issue

### 3. Auto-Fix Tests (`auto-fix-tests.yml`)
- **Triggers:** After CI failure, or manual
- **Flow:** Run tests → capture failures → Gemma 4 fixes → creates PR

---

## 🎛️ Configuration

Edit `agent/agent.py` to customize:

```python
# Change model
MODEL = "gemma-4-31b-it"         # 31B Dense, best quality
# MODEL = "gemma-4-26b-a4b-it"   # 26B MoE, faster & efficient

# Adjust thinking level
thinking_level="HIGH"   # HIGH | MEDIUM | LOW | NONE

# Max iterations (default 30)
max_iterations = 30

# Temperature (0 = deterministic, 1 = creative)
temperature=0.2
```

---

## 💡 Example Tasks

```bash
# Bug fixes
/ai The login endpoint returns 500 when email has special characters, fix it

# New features  
/ai Add rate limiting middleware (100 req/min per IP) to all API routes

# Refactoring
/ai Refactor the utils.py file to split it into separate modules by functionality

# Testing
/ai Add comprehensive tests for the payment processing module, aim for 90% coverage

# Documentation
/ai Generate API documentation in OpenAPI 3.0 format for all endpoints

# Code review
/ai Review the changes in this PR and check for security vulnerabilities

# Performance
/ai Profile the database queries in models.py and optimize the slow ones
```

---

## 🔒 Security Notes

- The `GITHUB_TOKEN` is automatically provided by GitHub Actions
- Only `GEMINI_API_KEY` needs to be added manually
- The agent runs in an isolated GitHub Actions runner
- All changes go through PRs — no direct pushes to main
- Free tier: 1,500 requests/day, 1M tokens/min

---

## 📈 Free Tier Limits (Google AI Studio)

| Limit | Value |
|---|---|
| Requests per day | 1,500 |
| Tokens per minute | 1,000,000 |
| Context window | 256K tokens |
| Cost | **$0** |

---

<div align="center">

**Powered by Gemma 4 31B · Google AI Studio · GitHub Actions**

*Free, Open, and Powerful*

</div>
