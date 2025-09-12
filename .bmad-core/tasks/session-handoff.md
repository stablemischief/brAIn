# Session Handoff Generator Task

## Purpose
Generate comprehensive handoff documentation before ending development sessions to ensure smooth recovery.

## Execution Protocol

### 1. Assess Current State
- Review all active work and in-progress changes
- Check Archon task synchronization
- Validate project health and stability
- Document any architectural decisions made

### 2. Generate SESSION-HANDOFF.md
Create detailed handoff documentation including:

```markdown
# SESSION HANDOFF - [DATE]

## Session Summary
**Duration**: [start_time] - [end_time]
**Tasks Completed**: [list of completed tasks with IDs]
**Tasks In Progress**: [current task status]
**Major Accomplishments**: [key features/fixes implemented]

## Current Project State
**Git Status**: [clean/pending changes/branch info]
**Active Branch**: [branch_name]
**Last Commit**: [commit_hash] - [commit_message]
**Test Status**: [all passing/X failing/not run]
**Build Status**: [successful/failed/not run]

## Next Session Priorities
1. **Immediate Priority**: [next task with Archon ID]
   - Status: [todo/doing/review]
   - Context: [what needs to be done]
   - Dependencies: [any blockers or requirements]

2. **Following Tasks**: [ordered list]
   - Task: [description] (ID: [archon_task_id])
   - Task: [description] (ID: [archon_task_id])

## Architectural Context
**Recent Decisions**: [any architecture changes or patterns established]
**Code Patterns**: [new patterns introduced or modified]
**Dependencies**: [new packages or services integrated]
**Configuration Changes**: [any env vars, configs, or settings modified]

## Technical Notes
**Known Issues**: [any bugs or technical debt identified]
**Performance Notes**: [any performance observations]
**Security Considerations**: [any security-related changes or concerns]
**Refactoring Needs**: [areas identified for future cleanup]

## Validation Checklist
- [ ] All tests passing
- [ ] No linting violations
- [ ] Git status clean
- [ ] Archon tasks synchronized
- [ ] Documentation updated
- [ ] No broken functionality

## Archon Synchronization
**Project ID**: [archon_project_id]
**Active Tasks**:
- [task_id]: [task_name] - Status: [status]
- [task_id]: [task_name] - Status: [status]

**Task Priority Order**: [list tasks by priority/task_order]

## Environment Info
**Python Version**: [version]
**Key Dependencies**: [critical package versions]
**Environment**: [dev/staging/prod context]
```

### 3. Validate Handoff Quality
- Ensure all critical context is captured
- Verify next session can start immediately
- Check that no context is lost
- Confirm technical state is accurately documented

### 4. Final Pre-Session-End Checks
- Commit all changes
- Update task statuses in Archon
- Run final test suite
- Ensure clean project state

## Success Criteria
- Next session can start with zero ramp-up time
- All context preserved in documentation
- Project state is stable and validated
- Archon synchronization is complete
- Technical debt is documented