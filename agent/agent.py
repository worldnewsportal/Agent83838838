#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║        GEMMA 4 AI AGENT — Powered by Google AI Studio   ║
║        ReAct Loop · Function Calling · GitHub Native    ║
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

# ─── CONFIG ────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN   = os.environ["GITHUB_TOKEN"]
REPO_OWNER     = os.environ.get("REPO_OWNER", "")
REPO_NAME      = os.environ.get("REPO_NAME", "")
TASK           = os.environ.get("TASK", "")
ISSUE_NUMBER   = os.environ.get("ISSUE_NUMBER", "")
PR_NUMBER      = os.environ.get("PR_NUMBER", "")
TRIGGER_TYPE   = os.environ.get("TRIGGER_TYPE", "manual")
BRANCH         = os.environ.get("BRANCH", "main")

MODEL = "gemma-4-31b-it"

OUTPUT_DIR = Path("agent_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── GITHUB CLIENT ─────────────────────────────────────────
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

# ─── LOGGING ───────────────────────────────────────────────
log_lines = []
def log(msg: str):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_lines.append(line)

# ─── TOOLS DEFINITIONS ─────────────────────────────────────
TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="read_file",
            description="Read the contents of any file in the repository.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"path": types.Schema(type=types.Type.STRING, description="File path relative to repo root")},
                required=["path"]
            )
        ),
        types.FunctionDeclaration(
            name="write_file",
            description="Write or create a file with the given content.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path":    types.Schema(type=types.Type.STRING, description="File path relative to repo root"),
                    "content": types.Schema(type=types.Type.STRING, description="Full file content to write"),
                },
                required=["path", "content"]
            )
        ),
        types.FunctionDeclaration(
            name="list_directory",
            description="List files and folders in a directory.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"path": types.Schema(type=types.Type.STRING, description="Directory path (default: '.')")},
                required=[]
            )
        ),
        types.FunctionDeclaration(
            name="run_command",
            description="Execute a shell command.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "command": types.Schema(type=types.Type.STRING, description="Shell command to run"),
                    "timeout": types.Schema(type=types.Type.INTEGER, description="Timeout in seconds (default 60)"),
                },
                required=["command"]
            )
        ),
        types.FunctionDeclaration(
            name="search_code",
            description="Search for a pattern across the codebase using grep.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "pattern":        types.Schema(type=types.Type.STRING, description="Regex or text pattern"),
                    "file_glob":      types.Schema(type=types.Type.STRING, description="File pattern e.g. '*.py'"),
                    "case_sensitive": types.Schema(type=types.Type.BOOLEAN, description="Case sensitive (default false)"),
                },
                required=["pattern"]
            )
        ),
        types.FunctionDeclaration(
            name="delete_file",
            description="Delete a file from the repository.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"path": types.Schema(type=types.Type.STRING, description="File path to delete")},
                required=["path"]
            )
        ),
        types.FunctionDeclaration(
            name="git_diff",
            description="Show git diff to review changes made so far.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"staged": types.Schema(type=types.Type.BOOLEAN, description="Show staged diff")},
                required=[]
            )
        ),
        types.FunctionDeclaration(
            name="create_pull_request",
            description="Create a GitHub Pull Request with all staged changes.",
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
        types.FunctionDeclaration(
            name="post_comment",
            description="Post a comment on a GitHub Issue or Pull Request.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "number":  types.Schema(type=types.Type.INTEGER, description="Issue or PR number"),
                    "comment": types.Schema(type=types.Type.STRING,  description="Comment text in Markdown"),
                },
                required=["number", "comment"]
            )
        ),
        types.FunctionDeclaration(
            name="get_pr_diff",
            description="Get the full code diff of a Pull Request.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"pr_number": types.Schema(type=types.Type.INTEGER, description="Pull Request number")},
                required=["pr_number"]
            )
        ),
        types.FunctionDeclaration(
            name="get_issue",
            description="Get full details of a GitHub Issue.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"issue_number": types.Schema(type=types.Type.INTEGER, description="Issue number")},
                required=["issue_number"]
            )
        ),
        types.FunctionDeclaration(
            name="apply_patch",
            description="Apply a unified diff patch to files.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"patch": types.Schema(type=types.Type.STRING, description="Unified diff patch content")},
                required=["patch"]
            )
        ),
        types.FunctionDeclaration(
            name="finish",
            description="Call this when the task is fully complete.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"summary": types.Schema(type=types.Type.STRING, description="Summary of all actions taken")},
                required=["summary"]
            )
        ),
    ])
]

# ─── TOOL EXECUTOR ─────────────────────────────────────────
def execute_tool(name: str, args: dict) -> str:
    log(f"🔧 Tool: {name}({json.dumps(args, ensure_ascii=False)[:120]})")

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

    elif name == "write_file":
        path = Path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"✅ Written {len(args['content'])} chars to {args['path']}"

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

    elif name == "run_command":
        cmd     = args["command"]
        timeout = int(args.get("timeout", 60))
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            out  = result.stdout[-3000:] if result.stdout else ""
            err  = result.stderr[-1500:] if result.stderr else ""
            code = result.returncode
            return f"Exit code: {code}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        except subprocess.TimeoutExpired:
            return f"❌ Command timed out after {timeout}s"
        except Exception as e:
            return f"❌ Error: {e}"

    elif name == "search_code":
        pattern  = args["pattern"]
        glob     = args.get("file_glob", "")
        ci_flag  = "" if args.get("case_sensitive") else "-i"
        include  = f"--include='{glob}'" if glob else ""
        cmd      = f"grep -rn {ci_flag} {include} '{pattern}' . --exclude-dir=.git 2>/dev/null | head -60"
        result   = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout or "No matches found"

    elif name == "delete_file":
        path = Path(args["path"])
        if path.exists():
            path.unlink()
            return f"✅ Deleted {args['path']}"
        return f"❌ File not found: {args['path']}"

    elif name == "git_diff":
        flag   = "--cached" if args.get("staged") else ""
        result = subprocess.run(f"git diff {flag}", shell=True, capture_output=True, text=True)
        diff   = result.stdout
        if len(diff) > 5000:
            diff = diff[:5000] + "\n... [diff truncated]"
        return diff or "No changes"

    elif name == "create_pull_request":
        title       = args["title"]
        body        = args["body"]
        branch_name = args["branch_name"]
        base        = args.get("base_branch", "main")
        subprocess.run("git config user.email 'gemma-agent@github-actions.bot'", shell=True)
        subprocess.run("git config user.name 'Gemma 4 AI Agent'", shell=True)
        subprocess.run(f"git checkout -b {branch_name}", shell=True)
        subprocess.run("git add -A", shell=True)
        commit_msg = f"🤖 {title}\n\nGenerated by Gemma 4 AI Agent"
        subprocess.run(f'git commit -m "{commit_msg}"', shell=True, capture_output=True)
        push = subprocess.run(f"git push origin {branch_name}", shell=True, capture_output=True, text=True)
        if push.returncode != 0:
            return f"❌ Push failed: {push.stderr}"
        pr = gh("POST", f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls", json={
            "title": title, "body": body, "head": branch_name, "base": base
        })
        if pr:
            return f"✅ PR created: {pr['html_url']}"
        return "❌ Failed to create PR"

    elif name == "post_comment":
        number  = args["number"]
        comment = args["comment"]
        result  = gh("POST", f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{number}/comments", json={"body": comment})
        return "✅ Comment posted" if result else "❌ Failed to post comment"

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

    elif name == "get_issue":
        issue_num    = args["issue_number"]
        issue        = gh("GET", f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_num}")
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

    elif name == "apply_patch":
        patch_file = Path("_agent.patch")
        patch_file.write_text(args["patch"])
        result = subprocess.run("git apply _agent.patch", shell=True, capture_output=True, text=True)
        patch_file.unlink(missing_ok=True)
        if result.returncode == 0:
            return "✅ Patch applied successfully"
        return f"❌ Patch failed: {result.stderr}"

    elif name == "finish":
        return f"__DONE__:{args['summary']}"

    return f"❌ Unknown tool: {name}"

# ─── SYSTEM PROMPT ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an elite AI coding agent. You operate inside a GitHub Actions runner with full access to the repository.

## Your workflow (ReAct pattern):
1. Reason — think step by step
2. Act — use the appropriate tool
3. Observe — analyze the tool output
4. Repeat — continue until task is complete
5. Finish — call finish() with a comprehensive summary

## Rules:
- Always explore the codebase first before making changes
- Run tests after making changes to verify correctness
- Write clean, idiomatic code
- Never truncate or leave code incomplete
- When reviewing PRs, be thorough and constructive
"""

# ─── MAIN AGENT LOOP ───────────────────────────────────────
def run_agent():
    log("=" * 60)
    log(f"🚀 Gemma 4 Agent Starting")
    log(f"📋 Task: {TASK[:200]}")
    log(f"🔗 Repo: {REPO_OWNER}/{REPO_NAME}")
    log(f"🎯 Trigger: {TRIGGER_TYPE}")
    log("=" * 60)

    client = genai.Client(api_key=GEMINI_API_KEY)

    context_info = f"Repository: {REPO_OWNER}/{REPO_NAME}\nBranch: {BRANCH}\nTrigger: {TRIGGER_TYPE}\n"
    if ISSUE_NUMBER:
        context_info += f"Issue #: {ISSUE_NUMBER}\n"
    if PR_NUMBER:
        context_info += f"PR #: {PR_NUMBER}\n"

    initial_message = f"{context_info}\n## Task:\n{TASK}\n\nBegin by exploring the repository structure, then complete the task step by step."

    # ✅ الطريقة الصحيحة لبناء المحادثة
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=initial_message)]
        )
    ]

    max_iterations = 30
    final_summary  = None
    all_tool_calls = []

    for iteration in range(max_iterations):
        log(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")

        try:
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

        candidate = response.candidates[0]
        parts     = candidate.content.parts if candidate.content else []

        # ✅ الإصلاح الرئيسي: استخدم content مباشرة بدون ._pb
        messages.append(response.candidates[0].content)

        tool_results  = []
        has_tool_call = False
        done          = False

        for part in parts:
            if hasattr(part, "text") and part.text:
                log(f"💭 Agent: {part.text[:300]}")

            if hasattr(part, "function_call") and part.function_call:
                has_tool_call = True
                fc   = part.function_call
                name = fc.name
                args = dict(fc.args)
                all_tool_calls.append({"tool": name, "args": args})

                result = execute_tool(name, args)
                log(f"   → {str(result)[:200]}")

                if result.startswith("__DONE__:"):
                    final_summary = result[9:]
                    done = True
                    break

                # ✅ الإصلاح الثاني: استخدم Part.from_function_response
                tool_results.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result}
                    )
                )

        if done:
            break

        # ✅ الإصلاح الثالث: استخدم types.Content مع role="tool"
        if tool_results:
            messages.append(
                types.Content(role="tool", parts=tool_results)
            )
        elif not has_tool_call:
            log("✅ Agent completed reasoning")
            for part in parts:
                if hasattr(part, "text") and part.text:
                    final_summary = part.text
                    break
            break

    # ─── POST FINAL RESULTS ────────────────────────────────
    if not final_summary:
        final_summary = "Task completed. See agent logs for details."

    log(f"\n✅ DONE: {final_summary[:300]}")

    log_file = OUTPUT_DIR / "agent.log"
    log_file.write_text("\n".join(log_lines))

    tool_summary = OUTPUT_DIR / "tools_used.json"
    tool_summary.write_text(json.dumps(all_tool_calls, indent=2, ensure_ascii=False))

    number = ISSUE_NUMBER or PR_NUMBER
    if number:
        tools_used = "\n".join(f"- `{t['tool']}`" for t in all_tool_calls[:15])
        comment = f"""## 🤖 Gemma 4 Agent — Task Complete

{final_summary}

---
🔧 Tools Used ({len(all_tool_calls)} calls)
{tools_used}
{"..." if len(all_tool_calls) > 15 else ""}

Powered by **Gemma 4 31B** via Google AI Studio · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""
        gh("POST", f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{number}/comments", json={"body": comment})

    log("🎉 Agent finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(run_agent())
