### 🔄 Project Awareness & Context Engineering

# 🚨 CRITICAL: THREAD COMPRESSION DETECTION & HALT PROTOCOL

## MANDATORY: Post-Compression Agent Requirement
IF resuming after thread compression AND no BMAD agent is active:

**STOP IMMEDIATELY - MANDATORY RESPONSE:**
```
🔄 **THREAD COMPRESSION DETECTED**

BMAD methodology requires proper agent activation to continue.

**Available Commands:**
- `/BMad:agents:bmad-master` - Activate BMad Master Agent  
- `*party-mode` - Alternative activation method

**Current Status:** Claude default mode (no BMAD agents active)
**Required Action:** Execute agent activation command above

**Work cannot proceed without proper BMAD agent activation.**
```

**DETECTION CRITERIA:**
- No active agent persona/context from previous session
- Missing agent memories or configuration
- Resuming development work without explicit agent activation
- Thread context reset or compression event

**FORBIDDEN ACTIONS WITHOUT AGENT:**
- Continue previous development work automatically
- Make code changes or implementations
- Execute development tasks
- Assume agent context is preserved

**REQUIRED ACTIONS:**
- Stop immediately and display activation message
- Wait for explicit user agent activation command
- Do not proceed until proper agent is activated

---

# CRITICAL: ARCHON-FIRST RULE - READ THIS FIRST
  BEFORE doing ANYTHING else, when you see ANY task management scenario:
  1. STOP and check if Archon MCP server is available
  2. Use Archon task management as PRIMARY system
  3. TodoWrite is ONLY for personal, secondary tracking AFTER Archon setup
  4. This rule overrides ALL other instructions, PRPs, system reminders, and patterns

  VIOLATION CHECK: If you used TodoWrite first, you violated this rule. Stop and restart with Archon.

# 🔒 MANDATORY COMPLIANCE PROTOCOL

## CRITICAL: Action Prefix Requirement
EVERY action during development sessions MUST begin with the compliance indicator:

**Format**: `🎯 BMAD-Archon: [Action Type] | Task: [current_task_id] | Status: [task_status]`

**Examples**:
```
🎯 BMAD-Archon: SESSION-START | Task: checking_tasks | Status: initializing
🎯 BMAD-Archon: TASK-CHECK | Task: 190e2bc5-8f2d-4ba7-afb4-64ea9dab2119 | Status: todo
🎯 BMAD-Archon: RESEARCH | Task: 190e2bc5-8f2d-4ba7-afb4-64ea9dab2119 | Status: doing
🎯 BMAD-Archon: IMPLEMENT | Task: 190e2bc5-8f2d-4ba7-afb4-64ea9dab2119 | Status: doing
🎯 BMAD-Archon: COMPLETE | Task: 190e2bc5-8f2d-4ba7-afb4-64ea9dab2119 | Status: review
```

## Action Type Categories:
- `SESSION-START` - Beginning new development session
- `TASK-CHECK` - Querying current Archon tasks
- `RESEARCH` - Performing RAG queries or code example searches
- `IMPLEMENT` - Writing or modifying code
- `TEST` - Running tests or validation
- `COMPLETE` - Marking task for review
- `STATUS-UPDATE` - Updating task status in Archon

## Compliance Rules:
1. **NEVER** start any development action without this prefix
2. **ALWAYS** include current task ID (if available)
3. **ALWAYS** show current task status
4. If no task assigned, use "no_task_assigned" and STOP to get task
5. If session just started, use "checking_tasks" as task ID

## Violation Recovery:
If any action lacks this prefix, the BMAD team should immediately intervene:
```
STOP - Missing BMAD-Archon compliance indicator. 
Start over with: 🎯 BMAD-Archon: SESSION-START | Task: checking_tasks | Status: initializing
```

## Emergency Override:
Only in true emergencies (system down, critical bug), prefix with:
`🚨 EMERGENCY-OVERRIDE: [reason] | Bypassing Archon temporarily`

# Archon Integration & Workflow

**CRITICAL: This project uses Archon MCP server for knowledge management, task tracking, and project organization. ALWAYS start with Archon MCP server task management.**

## Core Archon Workflow Principles

### The Golden Rule: Task-Driven Development with Archon

**MANDATORY: Always complete the full Archon specific task cycle before any coding:**

1. **Check Current Task** → `archon:manage_task(action="get", task_id="...")`
2. **Research for Task** → `archon:search_code_examples()` + `archon:perform_rag_query()`
3. **Implement the Task** → Write code based on research
4. **Update Task Status** → `archon:manage_task(action="update", task_id="...", update_fields={"status": "review"})`
5. **Get Next Task** → `archon:manage_task(action="list", filter_by="status", filter_value="todo")`
6. **Repeat Cycle**

**NEVER skip task updates with the Archon MCP server. NEVER code without checking current tasks first.**

## Project Scenarios & Initialization

### Scenario 1: New Project with Archon

```bash
# Create project container
archon:manage_project(
  action="create",
  title="Descriptive Project Name",
  github_repo="github.com/user/repo-name"
)

# Research → Plan → Create Tasks (see workflow below)
```

### Scenario 2: Existing Project - Adding Archon

```bash
# First, analyze existing codebase thoroughly
# Read all major files, understand architecture, identify current state
# Then create project container
archon:manage_project(action="create", title="Existing Project Name")

# Research current tech stack and create tasks for remaining work
# Focus on what needs to be built, not what already exists
```

### Scenario 3: Continuing Archon Project

```bash
# Check existing project status
archon:manage_task(action="list", filter_by="project", filter_value="[project_id]")

# Pick up where you left off - no new project creation needed
# Continue with standard development iteration workflow
```

### Universal Research & Planning Phase

**For all scenarios, research before task creation:**

```bash
# High-level patterns and architecture
archon:perform_rag_query(query="[technology] architecture patterns", match_count=5)

# Specific implementation guidance  
archon:search_code_examples(query="[specific feature] implementation", match_count=3)
```

**Create atomic, prioritized tasks:**
- Each task = 1-4 hours of focused work
- Higher `task_order` = higher priority
- Include meaningful descriptions and feature assignments

## Development Iteration Workflow

### Before Every Coding Session

**MANDATORY: Always check task status before writing any code:**

```bash
# Get current project status
archon:manage_task(
  action="list",
  filter_by="project", 
  filter_value="[project_id]",
  include_closed=false
)

# Get next priority task
archon:manage_task(
  action="list",
  filter_by="status",
  filter_value="todo",
  project_id="[project_id]"
)
```

### Task-Specific Research

**For each task, conduct focused research:**

```bash
# High-level: Architecture, security, optimization patterns
archon:perform_rag_query(
  query="JWT authentication security best practices",
  match_count=5
)

# Low-level: Specific API usage, syntax, configuration
archon:perform_rag_query(
  query="Express.js middleware setup validation",
  match_count=3
)

# Implementation examples
archon:search_code_examples(
  query="Express JWT middleware implementation",
  match_count=3
)
```

**Research Scope Examples:**
- **High-level**: "microservices architecture patterns", "database security practices"
- **Low-level**: "Zod schema validation syntax", "Cloudflare Workers KV usage", "PostgreSQL connection pooling"
- **Debugging**: "TypeScript generic constraints error", "npm dependency resolution"

### Task Execution Protocol with Compliance Indicators

**MANDATORY SEQUENCE WITH INDICATORS:**

**1. Session Start**
```
🎯 BMAD-Archon: SESSION-START | Task: checking_tasks | Status: initializing
[Run archon:list_tasks() to check current state]
```

**2. Task Selection**
```
🎯 BMAD-Archon: TASK-CHECK | Task: [selected_task_id] | Status: todo
[Run archon:get_task(task_id="...")]
```

**3. Research Phase**
```
🎯 BMAD-Archon: RESEARCH | Task: [task_id] | Status: doing
[Run archon:perform_rag_query() and archon:search_code_examples()]
```

**4. Implementation Phase**
```
🎯 BMAD-Archon: IMPLEMENT | Task: [task_id] | Status: doing
[Write code based on research findings]
```

**5. Testing Phase**
```
🎯 BMAD-Archon: TEST | Task: [task_id] | Status: doing
[Run tests and validation]
```

**6. Completion Phase**
```
🎯 BMAD-Archon: COMPLETE | Task: [task_id] | Status: review
[Update task status and get next task]
```

**7. Status Updates**
```
🎯 BMAD-Archon: STATUS-UPDATE | Task: [task_id] | Status: [new_status]
[Any status changes in Archon]
```

## Knowledge Management Integration

### Documentation Queries

**Use RAG for both high-level and specific technical guidance:**

```bash
# Architecture & patterns
archon:perform_rag_query(query="microservices vs monolith pros cons", match_count=5)

# Security considerations  
archon:perform_rag_query(query="OAuth 2.0 PKCE flow implementation", match_count=3)

# Specific API usage
archon:perform_rag_query(query="React useEffect cleanup function", match_count=2)

# Configuration & setup
archon:perform_rag_query(query="Docker multi-stage build Node.js", match_count=3)

# Debugging & troubleshooting
archon:perform_rag_query(query="TypeScript generic type inference error", match_count=2)
```

### Code Example Integration

**Search for implementation patterns before coding:**

```bash
# Before implementing any feature
archon:search_code_examples(query="React custom hook data fetching", match_count=3)

# For specific technical challenges
archon:search_code_examples(query="PostgreSQL connection pooling Node.js", match_count=2)
```

**Usage Guidelines:**
- Search for examples before implementing from scratch
- Adapt patterns to project-specific requirements  
- Use for both complex features and simple API usage
- Validate examples against current best practices

## Progress Tracking & Status Updates

### Daily Development Routine

**Start of each coding session:**

1. Check available sources: `archon:get_available_sources()`
2. Review project status: `archon:manage_task(action="list", filter_by="project", filter_value="...")`
3. Identify next priority task: Find highest `task_order` in "todo" status
4. Conduct task-specific research
5. Begin implementation

**End of each coding session:**

1. Update completed tasks to "done" status
2. Update in-progress tasks with current status
3. Create new tasks if scope becomes clearer
4. Document any architectural decisions or important findings

## 🚨 Violation Detection for BMAD Team

### Red Flag Indicators:
1. **Missing Prefix**: Any development action without `🎯 BMAD-Archon:`
2. **Wrong Task ID**: Prefix shows different task than expected
3. **Status Mismatch**: Status doesn't match Archon task status
4. **No Task Assignment**: Shows "no_task_assigned" but continues coding
5. **Sequence Violation**: IMPLEMENT before RESEARCH, etc.

### Immediate Intervention Commands:
```bash
# When violation detected
COMPLIANCE VIOLATION DETECTED
Required format: 🎯 BMAD-Archon: [ACTION] | Task: [ID] | Status: [STATUS]
Please restart with proper compliance indicator.
```

### Escalation Procedure:
1. **First Violation**: Warning and correction
2. **Second Violation**: Require session restart
3. **Third Violation**: Full workflow reset with Archon task check

### Recovery Commands:
```bash
# Reset Claude Code to proper workflow
🎯 BMAD-Archon: SESSION-START | Task: checking_tasks | Status: initializing
archon:list_tasks(filter_by="project", filter_value="be7fc8de-003c-49dd-826f-f158f4c36482")
```

### Task Status Management

**Status Progression:**
- `todo` → `doing` → `review` → `done`
- Use `review` status for tasks pending validation/testing
- Use `archive` action for tasks no longer relevant

**Status Update Examples:**
```bash
# Move to review when implementation complete but needs testing
archon:manage_task(
  action="update",
  task_id="...",
  update_fields={"status": "review"}
)

# Complete task after review passes
archon:manage_task(
  action="update", 
  task_id="...",
  update_fields={"status": "done"}
)
```

## Research-Driven Development Standards

### Before Any Implementation

**Research checklist:**

- [ ] Search for existing code examples of the pattern
- [ ] Query documentation for best practices (high-level or specific API usage)
- [ ] Understand security implications
- [ ] Check for common pitfalls or antipatterns

### Knowledge Source Prioritization

**Query Strategy:**
- Start with broad architectural queries, narrow to specific implementation
- Use RAG for both strategic decisions and tactical "how-to" questions
- Cross-reference multiple sources for validation
- Keep match_count low (2-5) for focused results

## Project Feature Integration

### Feature-Based Organization

**Use features to organize related tasks:**

```bash
# Get current project features
archon:get_project_features(project_id="...")

# Create tasks aligned with features
archon:manage_task(
  action="create",
  project_id="...",
  title="...",
  feature="Authentication",  # Align with project features
  task_order=8
)
```

### Feature Development Workflow

1. **Feature Planning**: Create feature-specific tasks
2. **Feature Research**: Query for feature-specific patterns
3. **Feature Implementation**: Complete tasks in feature groups
4. **Feature Integration**: Test complete feature functionality

## Error Handling & Recovery

### When Research Yields No Results

**If knowledge queries return empty results:**

1. Broaden search terms and try again
2. Search for related concepts or technologies
3. Document the knowledge gap for future learning
4. Proceed with conservative, well-tested approaches

### When Tasks Become Unclear

**If task scope becomes uncertain:**

1. Break down into smaller, clearer subtasks
2. Research the specific unclear aspects
3. Update task descriptions with new understanding
4. Create parent-child task relationships if needed

### Project Scope Changes

**When requirements evolve:**

1. Create new tasks for additional scope
2. Update existing task priorities (`task_order`)
3. Archive tasks that are no longer relevant
4. Document scope changes in task descriptions

## Quality Assurance Integration

### Research Validation

**Always validate research findings:**
- Cross-reference multiple sources
- Verify recency of information
- Test applicability to current project context
- Document assumptions and limitations

### Task Completion Criteria

**Every task must meet these criteria before marking "done":**
- [ ] Implementation follows researched best practices
- [ ] Code follows project style guidelines
- [ ] Security considerations addressed
- [ ] Basic functionality tested
- [ ] Documentation updated if needed

#### Context Engineering Principles

- **Research-Driven Development**: Always conduct thorough research before implementation using available knowledge bases, documentation, and code examples
- **Example-First Implementation**: Study existing code patterns and examples before writing new code
- **Validation-Gated Progress**: Each implementation must pass validation criteria before proceeding
- **PRP Methodology**: Follow Product Requirement Prompt patterns when available - comprehensive context leads to better outcomes

#### Project Discovery & Setup
- **Always read `PLANNING.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints
- **Check `TASK.md`** before starting a new task. If the task isn't listed, add it with a brief description and today's date
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`
- **Use venv_linux** (the virtual environment) whenever executing Python commands, including for unit tests

#### MCP Integration & Task Management
- **Archon-First Rule**: If Archon MCP server is available, use it as the PRIMARY task management system
- **Task-Driven Development**: Always check current tasks before coding, conduct task-specific research, implement with research findings, update task status
- **Knowledge Base Research**: Use `archon:perform_rag_query()` for architectural guidance and `archon:search_code_examples()` for implementation patterns
- **Progressive Task Status**: Follow todo → doing → review → done workflow with proper status updates

### 🧱 Code Structure & Modularity

#### File Organization Standards
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files
- **Organize code into clearly separated modules**, grouped by feature or responsibility
- **Use clear, consistent imports** (prefer relative imports within packages)
- **Use python_dotenv and load_env()** for environment variables

#### Agent-Based Architecture Patterns
For AI agent projects, organize code by agent responsibilities:
- `agent.py` - Main agent definition and execution logic
- `tools.py` - Tool functions used by the agent
- `prompts.py` - System prompts and templates
- `models.py` - Pydantic models and data structures
- `providers.py` - External service integrations

#### Multi-Agent Collaboration Patterns
- **Agent Role Definitions**: Each agent has clear responsibilities (Analyst, PM, Architect, Dev, QA)
- **Inter-Agent Communication**: Agents pass context through structured files and task management systems
- **Story-Driven Development**: Development agents work from detailed story files with complete context
- **Handoff Protocols**: Clear handoff points between agent roles with validation criteria

### 🧪 Testing & Reliability

#### Core Testing Requirements
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc)
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it
- **Tests should live in a `/tests` folder** mirroring the main app structure
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case

#### Validation Gates & Quality Loops
- **Continuous Validation**: Each feature implementation must pass all validation criteria before marking complete
- **Example-Driven Verification**: Validate implementations against researched code examples and patterns
- **Test-Driven Validation**: Tests serve as validation gates for feature completeness
- **Quality Assurance Integration**: QA agent patterns review and validate before final completion

### ✅ Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a “Discovered During Work” section.

### 📎 Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation**.
- Use `FastAPI` for APIs and `SQLAlchemy` or `SQLModel` for ORM if applicable.
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.
  
      Args:
          param1 (type): Description.
  
      Returns:
          type: Description.
      """
  ```

### 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules & Error Handling Philosophy

#### Context & Research Requirements
- **Never assume missing context. Ask questions if uncertain**
- **Research-First Approach**: Always research patterns, examples, and best practices before implementation
- **Never hallucinate libraries or functions** – only use known, verified Python packages
- **Always confirm file paths and module names** exist before referencing them in code or tests
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`

#### Error Handling Philosophy
**When to Fail Fast and Loud:**
- Service startup failures (credentials, database, configuration)
- Authentication/authorization failures
- Data corruption or validation errors
- Critical dependencies unavailable
- Invalid data that would corrupt state

**When to Complete but Log Detailed Errors:**
- Batch processing operations (complete what you can, report failures)
- Background tasks and async jobs
- WebSocket events (don't crash on single event failure)
- External API calls (retry with exponential backoff, then detailed failure)

**Critical Principle**: Never accept corrupted data. Skip failed items entirely rather than storing corrupted/incomplete data. Always report both success count and detailed failure lists for batch operations.

### 🔧 Tool Integration & MCP Architecture

#### MCP Server Integration Patterns
- **Primary Tool System**: Use MCP server tools as the primary interface for task management, knowledge queries, and project operations
- **Tool Discovery**: Always check available MCP tools before implementing custom solutions
- **Structured Tool Usage**: Follow consistent patterns for tool calls with proper error handling and result validation
- **Session Management**: Maintain proper MCP session state and handle reconnections gracefully

#### Multi-Service Architecture Guidelines
- **Service Communication**: Use HTTP APIs for inter-service communication, avoid tight coupling
- **Real-time Updates**: Implement WebSocket/SSE for real-time progress tracking and live updates
- **Configuration Management**: Use environment variables and configuration files, never hardcode service endpoints
- **Health Monitoring**: Implement health checks and service discovery patterns

#### AI-First Development Workflow
- **Tool-Assisted Development**: Leverage available tools for research, code generation, and validation
- **Progressive Enhancement**: Start with basic implementations, enhance with AI assistance
- **Context Preservation**: Maintain development context across tool calls and sessions
- **Automated Quality Gates**: Use tools for automated testing, linting, and code quality validation

### 🎯 Development Workflow Integration

#### Research-Driven Implementation Cycle
1. **Discovery Phase**: Use knowledge base queries to understand requirements and patterns
2. **Planning Phase**: Research architecture patterns and create detailed implementation plans
3. **Implementation Phase**: Code with continuous reference to researched examples and patterns
4. **Validation Phase**: Test against researched best practices and quality criteria
5. **Integration Phase**: Ensure compatibility with existing systems and workflows

#### Context Engineering Workflow
- **Context Collection**: Gather comprehensive project context before starting work
- **Example Curation**: Collect and analyze relevant code examples and patterns
- **Pattern Application**: Apply discovered patterns to current implementation
- **Validation Loops**: Continuously validate against established criteria and examples

#### Quality Assurance Integration
- **Multi-Agent Review**: Implement review processes that mirror agent collaboration patterns
- **Progressive Validation**: Validate at each stage of development, not just at completion
- **Documentation Integration**: Maintain documentation that supports both human and AI understanding
- **Continuous Improvement**: Iteratively improve based on validation feedback and quality metrics