#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         GEMMA 4 AI AGENT — Powered by Google AI Studio  ║
║         ReAct Loop · Function Calling · GitHub Native    ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any

import requests
from google import genai
from google.genai import types

# ─── CONFIG ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN   = os.environ["GITHUB_TOKEN"]
REPO_OWNER     = os.environ.get("REPO_OWNER", "")
REPO_NAME      = os.environ.get("REPO_NAME", "")
TASK           = os.environ.get("TASK", "")
ISSUE_NUMBER   = os.environ.get("ISSUE_NUMBER", "")
PR_NUMBER      = os.environ.get("PR_NUMBER", "")
TRIGGER_TYPE   = os.environ.get("TRIGGER_TYPE", "manual")
BRANCH         = os.environ.get("BRANCH", "main")

# Gemma 4 model — 31B with full 256K context
MODEL = "gemma-4-31b-it"

# Output directory for logs
OUTPUT_DIR = Path("agent_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── GITHUB CLIENT ───────────────────────────────────────
GITHUB_API = "https://api.github.com"
GH_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def gh(method: str, path: str, **kwargs) -> dict | list | None:
    url = f"{GITHUB_API}{path}"
    resp = requests.request(method, url, headers=GH_HEADERS, **kwargs)
    if resp.status_code in (200, 201, 204):
        return resp.json() if resp.content else {}
    log(f"⚠️ GitHub API {method} {path} → {resp.status_code}: {resp.text[:300]}")
    return None

# ─── LOGGING ─────────────────────────────────────────────
log_lines = []

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_lines.append(line)

# ─── ALL TOOLS DEFINITIONS ───────────────────────────────

TOOLS = [
    types.Tool(function_declarations=[

        # 1. Read file
        types.FunctionDeclaration(
            name="read_file",
            description="Read the contents of any file in the repository. Use this to understand code before modifying it.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="File path relative to repo root"),
                },
                required=["path"]
            )
        ),

        # 2. Write / create file
        types.FunctionDeclaration(
            name="write_file",
            description="Write or create a file with the given content. Overwrites existing files. Use for code generation, fixes, and new files.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path":    types.Schema(type=types.Type.STRING, description="File path relative to repo root"),
                    "content": types.Schema(type=types.Type.STRING, description="Full file content to write"),
                },
                required=["path", "content"]
            )
        ),

        # 3. List directory
        types.FunctionDeclaration(
            name="list_directory",
            description="List files and folders in a directory. Use to explore project structure.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="Directory path (default: '.' for root)"),
                },
                required=[]
            )
        ),

        # 4. Run bash command
        types.FunctionDeclaration(
            name="run_command",
            description="Execute a shell command. Use for running tests, linting, building, installing packages, git operations, etc.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "command": types.Schema(type=types.Type.STRING, description="Shell command to run"),
                    "timeout": types.Schema(type=types.Type.INTEGER, description="Timeout in seconds (default 60)"),
                },
                required=["command"]
            )
        ),

        # 5. Search code
        types.FunctionDeclaration(
            name="search_code",
            description="Search for a pattern across the codebase using grep. Find functions, classes, usages, imports, etc.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "pattern":   types.Schema(type=types.Type.STRING, description="Regex or text pattern to search"),
                    "file_glob": types.Schema(type=types.Type.STRING, description="File pattern e.g. '*.py' or '*.ts' (default: all files)"),
                    "case_sensitive": types.Schema(type=types.Type.BOOLEAN, description="Case sensitive search (default false)"),
                },
                required=["pattern"]
            )
        ),

        # 6. Delete file
        types.FunctionDeclaration(
            name="delete_file",
            description="Delete a file from the repository.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING, description="File path to delete"),
                },
                required=["path"]
            )
        ),

        # 7. Git diff
        types.FunctionDeclaration(
            name="git_diff",
            description="Show git diff to review changes made so far.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "staged": types.Schema(type=types.Type.BOOLEAN, description="Show staged diff (default false = unstaged)"),
                },
                required=[]
            )
        ),

        # 8. Create Pull Request
        types.FunctionDeclaration(
            name="create_pull_request",
            description="Create a GitHub Pull Request with all staged changes. Commits everything and opens a PR.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title":       types.Schema(type=types.Type.STRING, description="PR title"),
                    "body":        types.Schema(type=types.Type.STRING, description="PR description in Markdown"),
                    "branch_name": types.Schema(type=types.Type.STRING, description="New branch name for the PR"),
                    "base_branch": types.Schema(type=types.Type.STRING, description="Base branch (default: main)"),
                },
                required=["title", "body", "branch_name"]
            )
        ),

        # 9. Comment on issue/PR
        types.FunctionDeclaration(
            name="post_comment",
            description="Post a comment on a GitHub Issue or Pull Request.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "number":  types.Schema(type=types.Type.INTEGER, description="Issue or PR number"),
                    "comment": types.Schema(type=types.Type.STRING, description="Comment text in Markdown"),
                },
                required=["number", "comment"]
            )
        ),

        # 10. Get PR diff
        types.FunctionDeclaration(
            name="get_pr_diff",
            description="Get the full code diff of a Pull Request to review it.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "pr_number": types.Schema(type=types.Type.INTEGER, description="Pull Request number"),
                },
                required=["pr_number"]
            )
        ),

        # 11. Get issue details
        types.FunctionDeclaration(
            name="get_issue",
            description="Get full details of a GitHub Issue including title, body, labels, and comments.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "issue_number": types.Schema(type=types.Type.INTEGER, description="Issue number"),
                },
                required=["issue_number"]
            )
        ),

        # 12. Apply patch
        types.FunctionDeclaration(
            name="apply_patch",
            description="Apply a unified diff patch to files. Useful for targeted code edits.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "patch": types.Schema(type=types.Type.STRING, description="Unified diff patch content"),
                },
                required=["patch"]
            )
        ),

        # 13. Finish
        types.FunctionDeclaration(
            name="finish",
            description="Call this when the task is fully complete. Provide a final summary of everything done.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "summary": types.Schema(type=types.Type.STRING, description="Summary of all actions taken and results"),
                },
                required=["summary"]
            )
        ),
    ])
]

# ─── TOOL EXECUTOR ───────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    log(f"🔧 Tool: {name}({json.dumps(args, ensure_ascii=False)[:120]})")

    # 1. read_file
    if name == "read_file":
        path = Path(args["path"])
        if not path.exists():
            return f"❌ File not found: {args['path']}"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            if len(lines) > 500:
                content = "\n".join(lines[:500]) + f"\n... [truncated, {len(lines)-500} more lines]"
            return content
        except Exception as e:
            return f"❌ Error reading file: {e}"

    # 2. write_file
    elif name == "write_file":
        path = Path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"✅ Written {len(args['content'])} chars to {args['path']}"

    # 3. list_directory
    elif name == "list_directory":
        dir_path = Path(args.get("path", "."))
        if not dir_path.exists():
            return f"❌ Directory not found: {dir_path}"
        items = []
        for item in sorted(dir_path.iterdir()):
            if item.name.startswith(".git"):
                continue
            prefix = "📁 " if item.is_dir() else "📄 "
            items.append(f"{prefix}{item.relative_to('.')}")
        return "\n".join(items) if items else "Empty directory"

    # 4. run_command
    elif name == "run_command":
        cmd = args["command"]
        timeout = int(args.get("timeout", 60))
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=timeout
            )
            out = result.stdout[-3000:] if result.stdout else ""
            err = result.stderr[-1500:] if result.stderr else ""
            code = result.returncode
            return f"Exit code: {code}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        except subprocess.TimeoutExpired:
            return f"❌ Command timed out after {timeout}s"
        except Exception as e:
            return f"❌ Error: {e}"

    # 5. search_code
    elif name == "search_code":
        pattern = args["pattern"]
        glob    = args.get("file_glob", "")
        ci_flag = "" if args.get("case_sensitive") else "-i"
        include = f"--include='{glob}'" if glob else ""
        cmd = f"grep -rn {ci_flag} {include} '{pattern}' . --exclude-dir=.git 2>/dev/null | head -60"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout or "No matches found"

    # 6. delete_file
    elif name == "delete_file":
        path = Path(args["path"])
        if path.exists():
            path.unlink()
            return f"✅ Deleted {args['path']}"
        return f"❌ File not found: {args['path']}"

    # 7. git_diff
    elif name == "git_diff":
        flag = "--cached" if args.get("staged") else ""
        result = subprocess.run(f"git diff {flag}", shell=True, capture_output=True, text=True)
        diff = result.stdout
        if len(diff) > 5000:
            diff = diff[:5000] + "\n... [diff truncated]"
        return diff or "No changes"

    # 8. create_pull_request
    elif name == "create_pull_request":
        title       = args["title"]
        body        = args["body"]
        branch_name = args["branch_name"]
        base        = args.get("base_branch", "main")

        # Git setup
        subprocess.run("git config user.email 'gemma-agent@github-actions.bot'", shell=True)
        subprocess.run("git config user.name 'Gemma 4 AI Agent'", shell=True)
        subprocess.run(f"git checkout -b {branch_name}", shell=True)
        subprocess.run("git add -A", shell=True)
        commit_msg = f"🤖 {title}\n\nGenerated by Gemma 4 AI Agent"
        subprocess.run(f'git commit -m "{commit_msg}"', shell=True, capture_output=True)
        push = subprocess.run(
            f"git push origin {branch_name}",
            shell=True, capture_output=True, text=True
        )
        if push.returncode != 0:
            return f"❌ Push failed: {push.stderr}"

        pr = gh("POST", f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls", json={
            "title": title, "body": body,
            "head": branch_name, "base": base
        })
        if pr:
            return f"✅ PR created: {pr['html_url']}"
        return "❌ Failed to create PR"

    # 9. post_comment
    elif name == "post_comment":
        number  = args["number"]
        comment = args["comment"]
        result  = gh("POST", f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{number}/comments",
                     json={"body": comment})
        return f"✅ Comment posted" if result else "❌ Failed to post comment"

    # 10. get_pr_diff
    elif name == "get_pr_diff":
        pr_num = args["pr_number"]
        resp   = requests.get(
            f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_num}",
            headers={**GH_HEADERS, "Accept": "application/vnd.github.diff"}
        )
        diff = resp.text
        if len(diff) > 8000:
            diff = diff[:8000] + "\n... [diff truncated]"
        return diff

    # 11. get_issue
    elif name == "get_issue":
        issue_num = args["issue_number"]
        issue = gh("GET", f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_num}")
        comments_data = gh("GET", f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_num}/comments")
        if not issue:
            return "❌ Issue not found"
        out = f"**#{issue_num}: {issue['title']}**\n\n{issue['body'] or 'No description'}\n\n"
        out += f"Labels: {', '.join(l['name'] for l in issue.get('labels', []))}\n\n"
        if comments_data:
            out += "**Comments:**\n"
            for c in comments_data[:5]:
                out += f"\n[@{c['user']['login']}]: {c['body'][:500]}\n"
        return out

    # 12. apply_patch
    elif name == "apply_patch":
        patch_file = Path("_agent.patch")
        patch_file.write_text(args["patch"])
        result = subprocess.run("git apply _agent.patch", shell=True, capture_output=True, text=True)
        patch_file.unlink(missing_ok=True)
        if result.returncode == 0:
            return "✅ Patch applied successfully"
        return f"❌ Patch failed: {result.stderr}"

    # 13. finish
    elif name == "finish":
        return f"__DONE__:{args['summary']}"

    return f"❌ Unknown tool: {name}"

# ─── SYSTEM PROMPT ───────────────────────────────────────

SYSTEM_PROMPT = """You are an elite AI coding agent — think of yourself as a combination of Claude Code, GitHub Copilot, and a senior software engineer. You operate inside a GitHub Actions runner with full access to the repository.

## Your capabilities:
- Read, write, create, delete any file in the repository
- Run any shell command (tests, builds, linting, installs)
- Search across the entire codebase
- Create Pull Requests with your changes
- Review Pull Requests and post detailed feedback
- Fix bugs, implement features, refactor code
- Debug failing tests
- Write documentation

## Your workflow (ReAct pattern):
1. **Reason** — think step by step about what needs to be done
2. **Act** — use the appropriate tool
3. **Observe** — analyze the tool output
4. **Repeat** — continue until task is complete
5. **Finish** — call `finish()` with a comprehensive summary

## Rules:
- Always explore the codebase first with `list_directory` and `read_file` before making changes
- Run tests after making changes to verify correctness
- Write clean, idiomatic code that matches the project style
- Provide clear explanations in PRs and comments
- If you create files, always verify they were written correctly
- Never truncate or leave code incomplete — write full, working implementations
- When reviewing PRs, be thorough and constructive

## Code quality standards:
- Follow the existing coding conventions and style
- Add proper error handling
- Write or update tests when implementing features
- Add docstrings/comments for complex logic
- Keep functions focused and well-named

You have access to the full repository. Think carefully, act methodically, and deliver exceptional results."""


# ─── MAIN AGENT LOOP ─────────────────────────────────────

def run_agent():
    log("=" * 60)
    log(f"🚀 Gemma 4 Agent Starting")
    log(f"📋 Task: {TASK[:200]}")
    log(f"🔗 Repo: {REPO_OWNER}/{REPO_NAME}")
    log(f"🎯 Trigger: {TRIGGER_TYPE}")
    log("=" * 60)

    # Initialize Gemma 4 client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build initial context
    context_info = f"""
Repository: {REPO_OWNER}/{REPO_NAME}
Branch: {BRANCH}
Trigger: {TRIGGER_TYPE}
"""
    if ISSUE_NUMBER:
        context_info += f"Issue #: {ISSUE_NUMBER}\n"
    if PR_NUMBER:
        context_info += f"PR #: {PR_NUMBER}\n"

    initial_message = f"""
{context_info}

## Task:
{TASK}

Please begin by exploring the repository structure, then proceed to complete the task. Think step by step.
"""

    # Conversation history
    messages = [{"role": "user", "parts": [{"text": initial_message}]}]

    max_iterations = 30
    final_summary  = None
    all_tool_calls = []

    for iteration in range(max_iterations):
        log(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")

        try:
            # Call Gemma 4 with thinking mode ON
            response = client.models.generate_content(
                model=MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=TOOLS,
                    thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )
        except Exception as e:
            log(f"❌ API error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue

        # Extract response
        candidate = response.candidates[0]
        parts      = candidate.content.parts if candidate.content else []

        # Add model response to history
        # Add model response to history - استخدم content مباشرة
messages.append(response.candidates[0].content)

# ...

tool_results.append(
    types.Part.from_function_response(
        name=name,
        response={"result": result}
    )
)

# Feed tool results back
if tool_results:
    messages.append(
        types.Content(role="tool", parts=tool_results)
    )
        elif not has_tool_call:
            # No tool call and no finish — agent is done reasoning
            log("✅ Agent completed reasoning")
            # Extract text as summary
            for part in parts:
                if hasattr(part, "text") and part.text:
                    final_summary = part.text
                    break
            break

    # ─── POST FINAL RESULTS ──────────────────────────────
    if not final_summary:
        final_summary = "Task completed. See agent logs for details."

    log(f"\n✅ DONE: {final_summary[:300]}")

    # Save full log
    log_file = OUTPUT_DIR / "agent.log"
    log_file.write_text("\n".join(log_lines))

    # Save tool call summary
    tool_summary = OUTPUT_DIR / "tools_used.json"
    tool_summary.write_text(json.dumps(all_tool_calls, indent=2, ensure_ascii=False))

    # Post final comment to GitHub
    number = ISSUE_NUMBER or PR_NUMBER
    if number:
        tools_used = "\n".join(
            f"- `{t['tool']}`" for t in all_tool_calls[:15]
        )
        comment = f"""## 🤖 Gemma 4 Agent — Task Complete

{final_summary}

---
<details>
<summary>🔧 Tools Used ({len(all_tool_calls)} calls)</summary>

{tools_used}
{"..." if len(all_tool_calls) > 15 else ""}
</details>

<sub>Powered by **Gemma 4 31B** via Google AI Studio · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</sub>
"""
        gh("POST", f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{number}/comments",
           json={"body": comment})

    log("🎉 Agent finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(run_agent())
